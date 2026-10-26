"""Settings. Secrets are SecretStr and never logged or rendered."""
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="STS_", extra="ignore")

    model_api_key: SecretStr | None = Field(default=None, validation_alias="MODEL_API_KEY")
    allow_live_model_calls: bool = False
    meta_api_base: str = "https://api.meta.ai/v1"
    vision_model: str = "muse-spark-1.3"
    transcribe_model: str = "muse-voice-transcribe-1.0"
    segment_model: str = "sam-3.1"
    model_timeout_s: float = 120.0

    runs_dir: Path = Path("runs")
    db_path: Path = Path("var/stall_to_store.db")

    max_upload_mb: int = 200
    min_duration_s: float = 5.0
    max_duration_s: float = 90.0
    allowed_extensions: tuple[str, ...] = (".mp4", ".mov", ".m4v", ".3gp", ".webm", ".mkv")
    frame_fps: float = 2.0
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"

    dev_fixtures: bool = False
    proposal_ttl_minutes: int = 15
    click_to_chat_ttl_minutes: int = 1440
    max_order_lines: int = 30
    max_line_quantity: int = 50

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def api_key_present(self) -> bool:
        return bool(self.model_api_key and self.model_api_key.get_secret_value().strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()
