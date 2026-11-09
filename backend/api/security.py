from __future__ import annotations

import secrets
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from pwdlib import PasswordHash
from redis import Redis
from redis.exceptions import RedisError
from sqlalchemy.orm import Session

from backend.core.config import Settings, get_settings
from backend.models.database import get_session
from backend.models.entities import AuditLog, User

ALGORITHM = "HS256"
_hasher = PasswordHash.recommended()
_ephemeral_secret = secrets.token_urlsafe(48)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")


def signing_key(settings: Settings) -> str:
    configured = settings.jwt_secret.get_secret_value()
    if configured:
        if len(configured) < 32:
            raise RuntimeError("AEROLOCATE_JWT_SECRET must be at least 32 characters")
        return configured
    if settings.environment == "production":
        raise RuntimeError("AEROLOCATE_JWT_SECRET is required in production")
    return _ephemeral_secret


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, stored: str) -> bool:
    return _hasher.verify(password, stored)


def create_token(user_id: str, settings: Settings) -> str:
    now = datetime.now(timezone.utc)
    claims = {"sub": user_id, "iat": now, "exp": now + timedelta(minutes=settings.jwt_ttl_minutes)}
    return jwt.encode(claims, signing_key(settings), algorithm=ALGORITHM)


def user_from_token(token: str, session: Session, settings: Settings) -> User | None:
    try:
        claims = jwt.decode(token, signing_key(settings), algorithms=[ALGORITHM])
    except jwt.PyJWTError:
        return None
    return session.get(User, claims.get("sub"))


def current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    session: Annotated[Session, Depends(get_session)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> User:
    user = user_from_token(token, session, settings)
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or expired token")
    return user


def audit(session: Session, request: Request | None, user_id: str | None, action: str, target: str | None = None, **detail: object) -> None:
    client = request.client.host if request is not None and request.client else None
    session.add(AuditLog(user_id=user_id, action=action, target=target, detail=detail, client=client))


class RateLimiter:
    def __init__(self, redis_url: str | None, limit: int, window_s: int = 60) -> None:
        self.limit = limit
        self.window_s = window_s
        self.redis = Redis.from_url(redis_url, socket_timeout=0.2, socket_connect_timeout=0.2) if redis_url else None
        self.local: dict[str, deque[float]] = defaultdict(deque)
        self.lock = threading.Lock()

    def allow(self, key: str) -> bool:
        if self.redis is not None:
            try:
                bucket = f"aerolocate:rate:{key}:{int(time.time()) // self.window_s}"
                count = self.redis.incr(bucket)
                if count == 1:
                    self.redis.expire(bucket, self.window_s * 2)
                return int(count) <= self.limit
            except RedisError:
                pass
        with self.lock:
            now = time.monotonic()
            hits = self.local[key]
            while hits and now - hits[0] > self.window_s:
                hits.popleft()
            if len(hits) >= self.limit:
                return False
            hits.append(now)
            return True
