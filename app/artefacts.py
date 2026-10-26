"""Run IDs and safe paths under runs/{run_id}/."""
import re
import secrets
from datetime import datetime
from pathlib import Path

from app.db import utcnow

RUN_ID_RE = re.compile(r"^r_\d{8}T\d{6}_[0-9a-f]{8}$")
RUN_SUBDIRS = ("input", "frames", "raw", "keyframes", "cutouts", "artefacts")


class UnsafePathError(ValueError):
    pass


def new_run_id(now: datetime | None = None) -> str:
    now = now or utcnow()
    return f"r_{now.strftime('%Y%m%dT%H%M%S')}_{secrets.token_hex(4)}"


def validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not RUN_ID_RE.match(run_id):
        raise UnsafePathError("invalid run id")
    return run_id


def safe_join(base: Path, *parts: str) -> Path:
    """Join parts under base, refusing anything that resolves outside it."""
    base_resolved = base.resolve()
    for part in parts:
        if not part or Path(part).is_absolute() or "\x00" in part:
            raise UnsafePathError("unsafe path component")
    target = base_resolved.joinpath(*parts).resolve()
    if not target.is_relative_to(base_resolved):
        raise UnsafePathError("path escapes base directory")
    return target


def run_dir(runs_root: Path, run_id: str, create: bool = False) -> Path:
    path = safe_join(runs_root, validate_run_id(run_id))
    if create:
        for sub in RUN_SUBDIRS:
            (path / sub).mkdir(parents=True, exist_ok=True)
    return path


def resolve_artefact(runs_root: Path, stored: str | None) -> Path | None:
    """Resolve a path stored in the DB (relative to runs_root); None if missing or unsafe."""
    if not stored:
        return None
    try:
        path = safe_join(runs_root, *Path(stored).parts)
    except UnsafePathError:
        return None
    return path if path.is_file() else None


def relative_to_runs(runs_root: Path, path: Path) -> str:
    return path.resolve().relative_to(runs_root.resolve()).as_posix()
