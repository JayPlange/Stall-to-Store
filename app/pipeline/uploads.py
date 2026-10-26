"""Upload validation: extension, size (streamed), then ffprobe for container, duration and audio.

The client filename is never used on disk; we store input/source{ext} only.
"""
import shutil
from pathlib import Path
from typing import BinaryIO

from app.config import Settings
from app.pipeline.ingest import IngestError, VideoMeta, probe

CHUNK = 1024 * 1024


class UploadError(ValueError):
    pass


def safe_extension(filename: str | None, allowed: tuple[str, ...]) -> str:
    ext = Path(filename or "").suffix.lower()
    if ext not in allowed:
        raise UploadError(f"unsupported file type; allowed: {', '.join(allowed)}")
    return ext


def save_stream(src: BinaryIO, dst: Path, max_bytes: int) -> int:
    written = 0
    with dst.open("wb") as out:
        while chunk := src.read(CHUNK):
            written += len(chunk)
            if written > max_bytes:
                out.close()
                dst.unlink(missing_ok=True)
                raise UploadError(f"file is larger than {max_bytes // (1024 * 1024)} MB")
            out.write(chunk)
    if written == 0:
        dst.unlink(missing_ok=True)
        raise UploadError("file is empty")
    return written


def validate_video(path: Path, settings: Settings) -> VideoMeta:
    try:
        meta = probe(path, settings.ffprobe_bin)
    except IngestError as exc:
        raise UploadError(str(exc)) from exc
    if meta.duration_s < settings.min_duration_s:
        raise UploadError(f"video is shorter than {settings.min_duration_s:g}s")
    if meta.duration_s > settings.max_duration_s:
        raise UploadError(f"video is longer than {settings.max_duration_s:g}s")
    if not meta.has_audio:
        raise UploadError("video has no audio; prices are read from the narration")
    return meta


def accept_upload(src: BinaryIO, filename: str | None, run_path: Path, settings: Settings) -> tuple[Path, VideoMeta]:
    """Save and validate. On any failure the run directory is removed."""
    try:
        ext = safe_extension(filename, settings.allowed_extensions)
        dst = run_path / "input" / f"source{ext}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        save_stream(src, dst, settings.max_upload_bytes)
        meta = validate_video(dst, settings)
        return dst, meta
    except UploadError:
        shutil.rmtree(run_path, ignore_errors=True)
        raise
