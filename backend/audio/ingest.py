from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from backend.core.config import Settings

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".wav", ".flac", ".mp3", ".m4a"}
ALLOWED_CONTAINERS = {"mov", "mp4", "m4a", "matroska", "webm", "avi", "wav", "flac", "mp3"}
ISO6709 = re.compile(r"^([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)")
TIMESTAMP_TAGS = ("creation_time", "com.apple.quicktime.creationdate", "date", "encoded_date")
LOCATION_TAGS = ("location", "com.apple.quicktime.location.iso6709", "location-eng")


class MediaRejected(Exception):
    pass


@dataclass(frozen=True)
class TimestampCandidate:
    tag: str
    value: str
    trust: str = "unverified_metadata"


@dataclass
class MediaInfo:
    container: str
    duration_s: float
    sample_rate: int
    channels: int
    codec: str
    bit_rate: int | None
    has_video: bool
    lossy: bool
    timestamp_candidates: list[TimestampCandidate] = field(default_factory=list)
    embedded_location_hint: tuple[float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sniff_signature(path: Path) -> str | None:
    with path.open("rb") as handle:
        header = handle.read(16)
    if header[4:8] == b"ftyp":
        return "isobmff"
    if header[:4] == b"RIFF" and header[8:12] == b"WAVE":
        return "wav"
    if header[:4] == b"RIFF" and header[8:12] == b"AVI ":
        return "avi"
    if header[:4] == b"fLaC":
        return "flac"
    if header[:4] == b"\x1a\x45\xdf\xa3":
        return "matroska"
    if header[:3] == b"ID3" or (len(header) > 1 and header[0] == 0xff and header[1] & 0xe0 == 0xe0):
        return "mp3"
    if header[4:8] in (b"moov", b"mdat", b"wide", b"free", b"skip"):
        return "isobmff"
    return None


EXPECTED_SIGNATURES = {
    ".mp4": {"isobmff"},
    ".mov": {"isobmff"},
    ".m4a": {"isobmff"},
    ".mkv": {"matroska"},
    ".avi": {"avi"},
    ".wav": {"wav"},
    ".flac": {"flac"},
    ".mp3": {"mp3"},
}


def _limit_resources(memory_bytes: int) -> Any:
    if sys.platform == "win32":
        return None

    def apply() -> None:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    return apply


def _run(command: list[str], settings: Settings) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            command,
            capture_output=True,
            timeout=settings.ffmpeg_timeout_s,
            check=False,
            stdin=subprocess.DEVNULL,
            preexec_fn=_limit_resources(settings.ffmpeg_memory_limit_bytes),
        )
    except subprocess.TimeoutExpired as error:
        raise MediaRejected("media processing exceeded the time limit") from error
    except FileNotFoundError as error:
        raise MediaRejected(f"required tool not found: {command[0]}") from error


def _parse_location(value: str) -> tuple[float, float] | None:
    match = ISO6709.match(value.strip())
    if not match:
        return None
    return float(match.group(1)), float(match.group(2))


def probe_media(path: Path, extension: str, settings: Settings) -> MediaInfo:
    extension = extension.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise MediaRejected(f"unsupported file type: {extension}")
    signature = sniff_signature(path)
    if signature not in EXPECTED_SIGNATURES[extension]:
        raise MediaRejected("file contents do not match the declared type")
    completed = _run(
        [
            settings.ffprobe_path,
            "-v", "error",
            "-protocol_whitelist", "file",
            "-show_format",
            "-show_streams",
            "-of", "json",
            str(path),
        ],
        settings,
    )
    if completed.returncode != 0:
        raise MediaRejected("the file could not be parsed as media")
    payload = json.loads(completed.stdout.decode("utf-8", errors="replace"))
    streams = payload.get("streams", [])
    media_format = payload.get("format", {})
    containers = set((media_format.get("format_name") or "").split(","))
    if not containers & ALLOWED_CONTAINERS:
        raise MediaRejected("unsupported container format")
    audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    if audio is None:
        raise MediaRejected("the file has no audio stream")
    duration = float(media_format.get("duration") or audio.get("duration") or 0.0)
    if duration <= 0:
        raise MediaRejected("the file reports no duration")
    if duration > settings.max_media_seconds:
        raise MediaRejected("the recording exceeds the maximum duration")
    tags: dict[str, str] = {}
    for scope in [media_format, *streams]:
        tags.update({str(key).lower(): str(value) for key, value in (scope.get("tags") or {}).items()})
    candidates = [TimestampCandidate(tag, tags[tag]) for tag in TIMESTAMP_TAGS if tag in tags]
    location = next((_parse_location(tags[tag]) for tag in LOCATION_TAGS if tag in tags), None)
    codec = str(audio.get("codec_name") or "unknown")
    bit_rate = audio.get("bit_rate") or media_format.get("bit_rate")
    return MediaInfo(
        container=str(media_format.get("format_name")),
        duration_s=duration,
        sample_rate=int(audio.get("sample_rate") or 0),
        channels=int(audio.get("channels") or 0),
        codec=codec,
        bit_rate=int(bit_rate) if bit_rate else None,
        has_video=any(
            stream.get("codec_type") == "video" and not (stream.get("disposition") or {}).get("attached_pic")
            for stream in streams
        ),
        lossy=not (codec.startswith("pcm") or codec in ("flac", "alac")),
        timestamp_candidates=candidates,
        embedded_location_hint=location,
    )


def extract_audio(path: Path, output: Path, settings: Settings) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = _run(
        [
            settings.ffmpeg_path,
            "-nostdin",
            "-hide_banner",
            "-loglevel", "error",
            "-protocol_whitelist", "file",
            "-y",
            "-i", str(path),
            "-vn",
            "-map", "0:a:0",
            "-c:a", "pcm_f32le",
            "-f", "wav",
            str(output),
        ],
        settings,
    )
    if completed.returncode != 0 or not output.exists():
        raise MediaRejected("audio extraction failed")
    return output


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    return data, int(rate)
