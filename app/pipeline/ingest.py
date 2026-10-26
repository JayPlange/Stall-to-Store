"""Stage 1: ffprobe metadata, audio.wav (16 kHz mono PCM) and frames at 2 fps. Local only, no model calls."""
import json
import logging
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

log = logging.getLogger("sts.ingest")

FFMPEG_TIMEOUT_S = 300


class IngestError(RuntimeError):
    pass


@dataclass
class VideoMeta:
    duration_s: float
    width: int
    height: int
    fps: float | None
    has_audio: bool
    format_name: str
    video_codec: str
    audio_codec: str | None


def tools_available(ffmpeg_bin: str = "ffmpeg", ffprobe_bin: str = "ffprobe") -> bool:
    return bool(shutil.which(ffmpeg_bin) and shutil.which(ffprobe_bin))


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, capture_output=True, timeout=FFMPEG_TIMEOUT_S, check=False)
    except FileNotFoundError as exc:
        raise IngestError(f"{cmd[0]} not found") from exc
    except subprocess.TimeoutExpired as exc:
        raise IngestError(f"{Path(cmd[0]).name} timed out") from exc


def _parse_rate(rate: str | None) -> float | None:
    if not rate or rate == "0/0":
        return None
    num, _, den = rate.partition("/")
    try:
        return float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        return None


def probe(path: Path, ffprobe_bin: str = "ffprobe") -> VideoMeta:
    proc = _run([ffprobe_bin, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)])
    if proc.returncode != 0:
        raise IngestError("file is not a readable video")
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise IngestError("ffprobe returned unreadable output") from exc
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"
                  and not s.get("disposition", {}).get("attached_pic")), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if video is None:
        raise IngestError("file has no video stream")
    try:
        duration = float(data.get("format", {}).get("duration") or video.get("duration") or 0)
    except ValueError:
        duration = 0.0
    return VideoMeta(
        duration_s=duration,
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        fps=_parse_rate(video.get("avg_frame_rate")),
        has_audio=audio is not None,
        format_name=data.get("format", {}).get("format_name", ""),
        video_codec=video.get("codec_name", ""),
        audio_codec=audio.get("codec_name") if audio else None,
    )


def extract_audio(src: Path, dst: Path, ffmpeg_bin: str = "ffmpeg") -> Path:
    proc = _run([ffmpeg_bin, "-nostdin", "-y", "-v", "error", "-i", str(src),
                 "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst)])
    if proc.returncode != 0 or not dst.is_file():
        raise IngestError("audio extraction failed")
    return dst


def extract_frames(src: Path, frames_dir: Path, fps: float = 2.0, ffmpeg_bin: str = "ffmpeg") -> list[dict]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    pattern = frames_dir / "%05d.jpg"
    proc = _run([ffmpeg_bin, "-nostdin", "-y", "-v", "error", "-i", str(src),
                 "-vf", f"fps={fps}", "-q:v", "3", str(pattern)])
    if proc.returncode != 0:
        raise IngestError("frame extraction failed")
    files = sorted(frames_dir.glob("*.jpg"))
    if not files:
        raise IngestError("no frames extracted")
    # ffmpeg numbers from 1; frame i (0-based) samples t = i / fps.
    return [{"frame_idx": i, "t": round(i / fps, 3), "file": f.name} for i, f in enumerate(files)]


def ingest(video_path: Path, run_path: Path, fps: float = 2.0,
           ffmpeg_bin: str = "ffmpeg", ffprobe_bin: str = "ffprobe") -> dict:
    """Write audio.wav, frames/*.jpg, frames.json and metadata.json into run_path."""
    meta = probe(video_path, ffprobe_bin)
    audio_path = None
    if meta.has_audio:
        audio_path = extract_audio(video_path, run_path / "audio.wav", ffmpeg_bin)
    frames = extract_frames(video_path, run_path / "frames", fps, ffmpeg_bin)
    (run_path / "frames.json").write_text(json.dumps({"fps": fps, "frames": frames}, indent=2), encoding="utf-8")
    metadata = {**asdict(meta), "frame_count": len(frames), "frame_fps": fps,
                "audio": audio_path.name if audio_path else None}
    (run_path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    log.info("ingested", extra={"ctx": {"run_dir": run_path.name, "frames": len(frames),
                                        "duration_s": meta.duration_s}})
    return metadata
