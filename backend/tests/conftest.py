import os
import tempfile
from pathlib import Path

_root = Path(tempfile.mkdtemp(prefix="aerolocate-test-"))

os.environ.update(
    AEROLOCATE_DATABASE_URL=f"sqlite:///{(_root / 'test.db').as_posix()}",
    AEROLOCATE_DATA_DIR=str(_root),
    AEROLOCATE_LOCAL_FLIGHT_DATASET_DIR=str(_root / "flights"),
    AEROLOCATE_JOB_BACKEND="thread",
    AEROLOCATE_CLASSIFIER="spectral",
    AEROLOCATE_JWT_SECRET="test-secret-test-secret-test-secret-test",
    AEROLOCATE_RATE_LIMIT_PER_MINUTE="100000",
    AEROLOCATE_FR24_API_KEY="",
    AEROLOCATE_ADSBX_API_KEY="",
)
os.environ.setdefault("AEROLOCATE_FFMPEG_PATH", "ffmpeg")
os.environ.setdefault("AEROLOCATE_FFPROBE_PATH", "ffprobe")
