"""Run one pipeline stage at a time. Model stages need --live AND STS_ALLOW_LIVE_MODEL_CALLS=true.

    python -m app.pipeline.cli ingest     <run_id>
    python -m app.pipeline.cli transcribe <run_id> --live
    python -m app.pipeline.cli prices     <run_id> --live | --regex
    python -m app.pipeline.cli track      <run_id> --live
    python -m app.pipeline.cli keyframes  <run_id>
    python -m app.pipeline.cli identify   <run_id> --live
    python -m app.pipeline.cli link       <run_id> --live   # optional Muse Spark price matching
    python -m app.pipeline.cli associate  <run_id>
    python -m app.pipeline.cli cutouts    <run_id>
    python -m app.pipeline.cli overlay    <run_id>

Artefacts in runs/{run_id}/artefacts/: transcript.json, price_mentions.json, tracks.json,
keyframes.json, guesses.json, association.json, cutouts.json. Each stage can be re-run alone.
`track` sends ONE request per prompt (each billed per frame); the default is STS_SAM_PROMPTS.
"""
import argparse
import json
import sys

from app import catalogue
from app.artefacts import run_dir
from app.config import get_settings
from app.db import connect, init_db
from app.logging_setup import configure_logging
from app.models.base import AdapterNotVerified, LiveCallsDisabled, MetaClient, ProviderError
from app.pipeline.stages import LIVE_STAGES, StageError, run_stage


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["ingest", "transcribe", "prices", "track", "keyframes", "identify", "associate",
                                      "cutouts", "overlay", "link"])
    ap.add_argument("run_id")
    ap.add_argument("--live", action="store_true", help="allow this stage to make a paid model call")
    ap.add_argument("--regex", action="store_true", help="prices: use the regex fallback, no model call")
    ap.add_argument("--prompts", default=None, help="track: ';'-separated SAM noun phrases")
    args = ap.parse_args(argv)

    s = get_settings()
    conn = connect(s.db_path)
    init_db(conn)
    run = catalogue.get_run(conn, args.run_id)
    if not run:
        raise SystemExit("unknown run")
    if run["source"] != "real":
        raise SystemExit("refusing to run the pipeline on a fixture run")
    run_dir(s.runs_dir, args.run_id, create=True)
    if args.stage in ("ingest", "track") and not run["video_path"]:
        raise SystemExit("run has no uploaded video")
    needs_live = args.stage in LIVE_STAGES or (args.stage == "prices" and not args.regex)
    if needs_live and not args.live:
        raise SystemExit(f"{args.stage} makes a paid model call: pass --live to confirm")
    try:
        run_stage(args.stage, run, s, conn, MetaClient(s, conn), args.prompts, args.regex)
        catalogue.set_run_status(conn, args.run_id, catalogue.get_run(conn, args.run_id)["status"], args.stage)
    except StageError as exc:
        raise SystemExit(str(exc))
    except (LiveCallsDisabled, AdapterNotVerified) as exc:
        print(f"{args.stage}: {exc}", file=sys.stderr)
        return 2
    except ProviderError as exc:
        catalogue.set_run_status(conn, args.run_id, "failed", args.stage, exc.safe_message)
        print(f"{args.stage}: provider error {exc.status} {exc.error_type}: {exc.safe_message} "
              f"(raw: {exc.raw_path}). Not retried.", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
