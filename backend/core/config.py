from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AEROLOCATE_", env_file=".env", extra="ignore")

    environment: str = "development"
    database_url: str = "sqlite:///./data/aerolocate.db"
    redis_url: str = "redis://localhost:6379/0"
    job_backend: str = Field(default="thread", pattern="^(thread|celery)$")
    data_dir: Path = Path("./data")
    max_upload_bytes: int = 2_147_483_648
    max_media_seconds: float = 7200.0
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"
    ffmpeg_timeout_s: float = 600.0
    ffmpeg_memory_limit_bytes: int = 4_294_967_296
    analysis_timeout_s: float = 3600.0
    max_concurrent_jobs_per_user: int = 2
    jwt_secret: SecretStr = SecretStr("")
    jwt_ttl_minutes: int = 60
    allow_registration: bool = True
    rate_limit_per_minute: int = 120
    retention_days: int = 30
    flight_cache_ttl_hours: int = 24
    cors_origins: str = "http://localhost:5173"
    classifier: str = Field(default="ast", pattern="^(spectral|ast)$")
    ast_model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
    torch_device: str = "auto"
    default_ground_elevation_m: float = 0.0
    default_temperature_c: float = 15.0
    fr24_api_key: SecretStr = SecretStr("")
    fr24_base_url: str = "https://fr24api.flightradar24.com/api"
    fr24_snapshot_step_s: int = 60
    adsbx_api_key: SecretStr = SecretStr("")
    adsbx_base_url: str = "https://gateway.adsbexchange.com/api/aircraft/v2"
    opensky_client_id: str = ""
    opensky_client_secret: SecretStr = SecretStr("")
    opensky_base_url: str = "https://opensky-network.org/api"
    opensky_token_url: str = (
        "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
    )
    local_flight_dataset_dir: Path = Path("./flight_datasets")

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def artifact_dir(self) -> Path:
        return self.data_dir / "artifacts"


@lru_cache
def get_settings() -> Settings:
    return Settings()
