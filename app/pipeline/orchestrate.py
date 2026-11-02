"""Run a whole video through the pipeline: one action for the seller, every stage still logged.

Each stage runs once and is never retried. If a required stage fails the run stops with a safe message, so a fault
cannot keep spending. Optional stages only add quality, so a draft survives without them.
"""
import logging
from collections.abc import Callable

from app import catalogue
from app.config import Settings
from app.db import connect, init_db
from app.models.base import MetaClient, ProviderError
from app.pipeline.stages import run_stage

log = logging.getLogger("sts.pipeline")

STEPS = [("transcribe", "Listening to your narration"), ("prices", "Picking out the prices you said"),
         ("track", "Finding and tracking your products"), ("keyframes", "Choosing product photos"),
         ("identify", "Reading product labels"), ("link", "Matching prices to products"),
         ("associate", "Building your draft shop"), ("cutouts", "Cutting out product photos"),
         ("overlay", "Making your tracking video")]
OPTIONAL = {"link", "cutouts", "overlay"}
ESTIMATED_COST_USD = 0.10


def can_process(s: Settings) -> bool:
    return s.allow_live_model_calls and s.api_key_present


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
    """Returns True when a draft exists. Runs on a background thread, so it opens its own connection."""
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
