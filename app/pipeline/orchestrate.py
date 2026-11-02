"""Run a whole video through the pipeline: one action for the seller, every stage still logged.

Each stage runs once and is never retried. If a required stage fails the run stops with a safe message, so a fault
cannot keep spending. Optional stages only add quality, so a draft survives without them.
"""
import logging
from collections.abc import Callable
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor

from app import catalogue
from app.config import Settings
from app.db import connect, init_db, iso, utcnow
from app.logging_setup import configure_logging
from app.models.base import MetaClient, ProviderError
from app.pipeline.stages import run_stage

log = logging.getLogger("sts.pipeline")

STEPS = [("transcribe", "Listening to your narration"), ("prices", "Picking out the prices you said"),
         ("track", "Finding and tracking your products"), ("keyframes", "Choosing product photos"),
         ("identify", "Reading product labels"), ("link", "Matching prices to products"),
         ("associate", "Building your draft shop"), ("cutouts", "Cutting out product photos"),
         ("overlay", "Making your tracking video")]
OPTIONAL = {"link", "cutouts", "overlay"}
ESTIMATED_COST_USD = 0.13
MAX_ATTEMPTS = 2


def can_process(s: Settings) -> bool:
    """Paid runs need an access code, so a public server can never spend credits by accident."""
    return s.allow_live_model_calls and s.api_key_present and bool(s.upload_code)


def processing_block(conn, s: Settings, run_id: str) -> str | None:
    """Why this run must not spend money right now, or None if it may."""
    spent = conn.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM model_calls WHERE created_at LIKE ?",
                         (iso(utcnow())[:10] + "%",)).fetchone()[0]
    if spent + ESTIMATED_COST_USD > s.max_daily_usd:
        return "Today's AI budget has been reached. Please try again tomorrow."
    tries = conn.execute("SELECT COUNT(*) FROM model_calls WHERE run_id = ? AND stage = 'transcribe'",
                         (run_id,)).fetchone()[0]
    if tries >= MAX_ATTEMPTS:
        return "This video has been tried too many times. Please upload it again."
    return None


def recover_interrupted(conn) -> int:
    """A restart kills any run in progress. Mark those failed so they are not stuck forever."""
    count = conn.execute("UPDATE runs SET status = 'failed', error = 'The server restarted. Please try again.'"
                         " WHERE status = 'processing'").rowcount
    conn.commit()
    return count


def make_starter(s: Settings) -> Callable[[str], None]:
    """Run videos one at a time in a separate process, because CPU work here would stall the website."""
    pool: list[ProcessPoolExecutor] = []

    def finished(run_id: str, future) -> None:
        if future.exception() is None:
            return
        log.error("pipeline process failed", extra={"ctx": {"run_id": run_id}})
        conn = connect(s.db_path)
        try:
            catalogue.set_run_status(conn, run_id, "failed", None, "Processing stopped unexpectedly. Please try again.")
        finally:
            conn.close()

    def start(run_id: str) -> None:
        if not pool:
            pool.append(ProcessPoolExecutor(max_workers=1))
        try:
            future = pool[0].submit(run_pipeline, run_id, s)
        except BrokenExecutor:
            pool[0] = ProcessPoolExecutor(max_workers=1)
            future = pool[0].submit(run_pipeline, run_id, s)
        future.add_done_callback(lambda f: finished(run_id, f))

    return start


def step_states(run: dict) -> list[dict]:
    """For the progress list: each step is done, active or waiting."""
    names = [name for name, _ in STEPS]
    current = names.index(run["stage"]) if run["stage"] in names else -1
    return [{"label": label, "state": "done" if i < current else "active" if i == current else "waiting"}
            for i, (_, label) in enumerate(STEPS)]


def _mark(conn, run_id: str, stage: str) -> None:
    """Once a draft exists the seller may be editing or publishing, so never move the run back."""
    status = catalogue.get_run(conn, run_id)["status"]
    catalogue.set_run_status(conn, run_id, status if status in ("draft_ready", "published") else "processing", stage)


def run_pipeline(run_id: str, s: Settings, client_factory: Callable | None = None) -> bool:
    """Returns True when a draft exists. Runs in its own process, so it opens its own connection."""
    configure_logging()
    conn = connect(s.db_path)
    init_db(conn)
    try:
        run = catalogue.get_run(conn, run_id)
        client = client_factory(conn) if client_factory else MetaClient(s, conn)
        for stage, _ in STEPS:
            _mark(conn, run_id, stage)
            try:
                run_stage(stage, run, s, conn, client)
            except Exception as exc:
                log.exception("stage failed", extra={"ctx": {"run_id": run_id, "stage": stage}})
                if stage not in OPTIONAL:
                    message = exc.safe_message if isinstance(exc, ProviderError) else "This step could not finish."
                    catalogue.set_run_status(conn, run_id, "failed", stage, f"Stopped at {stage}: {message}")
                    return False
        _mark(conn, run_id, "done")
        return True
    finally:
        conn.close()
