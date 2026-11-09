from __future__ import annotations

import hashlib
import json
import random
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping

import httpx

from backend.flight_data.types import (
    AircraftIdentity,
    BoundingBox,
    FlightRecord,
    ProviderCapability,
    ProviderStatus,
    TimeWindow,
)

MAX_RETRIES = 4
BASE_BACKOFF_S = 1.0


class ProviderError(Exception):
    pass


class ProviderNotConfigured(ProviderError):
    pass


class ProviderCapabilityError(ProviderError):
    pass


class ProviderRateLimited(ProviderError):
    def __init__(self, message: str, retry_after_s: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


class ProviderAccessDenied(ProviderError):
    pass


class ResponseCache:
    def __init__(self, directory: Path, ttl_s: float) -> None:
        self.directory = directory
        self.ttl_s = ttl_s

    def _path(self, key: str) -> Path:
        return self.directory / f"{hashlib.sha256(key.encode()).hexdigest()}.json"

    def get(self, key: str) -> Any | None:
        path = self._path(key)
        if not path.exists():
            return None
        if time.time() - path.stat().st_mtime > self.ttl_s:
            path.unlink(missing_ok=True)
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def put(self, key: str, value: Any) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self._path(key).write_text(json.dumps(value), encoding="utf-8")

    def purge_expired(self) -> int:
        if not self.directory.exists():
            return 0
        removed = 0
        for path in self.directory.glob("*.json"):
            if time.time() - path.stat().st_mtime > self.ttl_s:
                path.unlink(missing_ok=True)
                removed += 1
        return removed


class HttpSession:
    def __init__(
        self,
        provider: str,
        client: httpx.Client | None = None,
        cache: ResponseCache | None = None,
        sleep: Any = time.sleep,
    ) -> None:
        self.provider = provider
        self.client = client or httpx.Client(timeout=30.0, follow_redirects=False)
        self.cache = cache
        self.sleep = sleep
        self.requests_made = 0

    def get_json(
        self,
        url: str,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        retry_after_header: str = "Retry-After",
        not_found_value: Any = None,
    ) -> Any:
        key = f"{url}|{sorted((params or {}).items())}"
        if self.cache is not None:
            cached = self.cache.get(key)
            if cached is not None:
                return cached
        for attempt in range(MAX_RETRIES + 1):
            self.requests_made += 1
            try:
                response = self.client.get(url, params=params, headers=headers)
            except httpx.TransportError as error:
                if attempt == MAX_RETRIES:
                    raise ProviderError(f"{self.provider}: transport failure: {error}") from error
                self.sleep(BASE_BACKOFF_S * 2**attempt + random.random())
                continue
            if response.status_code == 429:
                retry_after = _retry_seconds(response.headers.get(retry_after_header))
                if attempt == MAX_RETRIES:
                    raise ProviderRateLimited(f"{self.provider}: rate limited", retry_after)
                self.sleep(retry_after if retry_after is not None else BASE_BACKOFF_S * 2**attempt)
                continue
            if response.status_code in (401, 402, 403):
                raise ProviderAccessDenied(
                    f"{self.provider}: HTTP {response.status_code}; the key or plan does not permit this request"
                )
            if response.status_code == 404:
                return not_found_value
            if response.status_code >= 500:
                if attempt == MAX_RETRIES:
                    raise ProviderError(f"{self.provider}: HTTP {response.status_code}")
                self.sleep(BASE_BACKOFF_S * 2**attempt + random.random())
                continue
            if response.status_code >= 400:
                raise ProviderError(f"{self.provider}: HTTP {response.status_code}: {response.text[:200]}")
            payload = response.json()
            if self.cache is not None:
                self.cache.put(key, payload)
            return payload
        raise ProviderError(f"{self.provider}: request failed")


def _retry_seconds(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


class FlightDataProvider(ABC):
    name: str
    capabilities: tuple[ProviderCapability, ...]

    @abstractmethod
    def is_configured(self) -> bool: ...

    @abstractmethod
    def check_availability(self) -> ProviderStatus: ...

    @abstractmethod
    def search_area(self, box: BoundingBox, window: TimeWindow) -> list[FlightRecord]: ...

    @abstractmethod
    def get_track(self, identity: AircraftIdentity, window: TimeWindow) -> FlightRecord | None: ...

    def estimate_requests(self, box: BoundingBox, window: TimeWindow) -> int:
        return 1

    def supports(self, capability: ProviderCapability) -> bool:
        return capability in self.capabilities
