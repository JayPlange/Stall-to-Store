"""Stage 1: ffprobe metadata, audio.wav, a normalised video and its frames. Local only, no model calls.

normalised.mp4 is what SAM 3.1 receives: 2 fps, long side <= 1280 px, H.264, faststart, no audio.
frames/ are extracted from that same file, so SAM frame N == our frame N at t = N / fps, and SAM's
pixel boxes are in the same coordinate space as our frames. (SAM bills per frame: 60 s = 120 frames.)
audio.wav is mono 16-bit PCM at 24 kHz, Muse Voice Transcribe's documented native rate.
"""
import json
import logging
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

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


AUDIO_RATE = 24000
MAX_SIDE = 1280


def extract_audio(src: Path, dst: Path, ffmpeg_bin: str = "ffmpeg") -> Path:
    proc = _run([ffmpeg_bin, "-nostdin", "-y", "-v", "error", "-i", str(src), "-vn", "-ac", "1",
                 "-ar", str(AUDIO_RATE), "-c:a", "pcm_s16le", "-map_metadata", "-1", str(dst)])
    if proc.returncode != 0 or not dst.is_file():
        raise IngestError("audio extraction failed")
    return dst


def normalise_video(src: Path, dst: Path, fps: float = 2.0, ffmpeg_bin: str = "ffmpeg") -> Path:
    scale = (f"scale='if(gte(iw,ih),trunc(min({MAX_SIDE},iw)/2)*2,-2)'"
             f":'if(gte(iw,ih),-2,trunc(min({MAX_SIDE},ih)/2)*2)'")
    proc = _run([ffmpeg_bin, "-nostdin", "-y", "-v", "error", "-i", str(src), "-an", "-vf", f"fps={fps},{scale}",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "20",
                 "-movflags", "+faststart", "-map_metadata", "-1", str(dst)])
    if proc.returncode != 0 or not dst.is_file():
        raise IngestError("video normalisation failed")
    return dst


def extract_frames(normalised: Path, frames_dir: Path, fps: float = 2.0, ffmpeg_bin: str = "ffmpeg") -> list[dict]:
    """Every frame of the normalised video, so frame index i is SAM's frame i."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    proc = _run([ffmpeg_bin, "-nostdin", "-y", "-v", "error", "-i", str(normalised),
                 "-fps_mode", "passthrough", "-q:v", "3", str(frames_dir / "%05d.jpg")])
    if proc.returncode != 0:
        raise IngestError("frame extraction failed")
    files = sorted(frames_dir.glob("*.jpg"))
    if not files:
        raise IngestError("no frames extracted")
    return [{"frame_idx": i, "t": round(i / fps, 3), "file": f.name} for i, f in enumerate(files)]


def ingest(video_path: Path, run_path: Path, fps: float = 2.0,
           ffmpeg_bin: str = "ffmpeg", ffprobe_bin: str = "ffprobe") -> dict:
    """Write audio.wav, frames/*.jpg, frames.json and metadata.json into run_path."""
    meta = probe(video_path, ffprobe_bin)
    audio_path = None
    if meta.has_audio:
        audio_path = extract_audio(video_path, run_path / "audio.wav", ffmpeg_bin)
    normalised = normalise_video(video_path, run_path / "normalised.mp4", fps, ffmpeg_bin)
    frames = extract_frames(normalised, run_path / "frames", fps, ffmpeg_bin)
    (run_path / "frames.json").write_text(json.dumps({"fps": fps, "frames": frames}, indent=2), encoding="utf-8")
    with Image.open(run_path / "frames" / frames[0]["file"]) as first:
        frame_w, frame_h = first.size
    metadata = {**asdict(meta), "frame_count": len(frames), "frame_fps": fps, "frame_width": frame_w,
                "frame_height": frame_h, "normalised": normalised.name,
                "audio": audio_path.name if audio_path else None}
    (run_path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    log.info("ingested", extra={"ctx": {"run_dir": run_path.name, "frames": len(frames),
                                        "duration_s": meta.duration_s}})
    return metadata
