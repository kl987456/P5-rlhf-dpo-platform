"""
Shared security utilities: API-key auth + rate limiting.
"""
from __future__ import annotations

import os
import time
from collections import defaultdict
from threading import Lock
from typing import Callable

from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse

# ── API Key ────────────────────────────────────────────────────────────────────

_API_KEY: str = os.getenv("API_KEY", "")


async def require_api_key(request: Request) -> None:
    """Dependency — pass to route or router. No-op when API_KEY env var is unset."""
    if not _API_KEY:
        return
    key = request.headers.get("X-API-Key", "")
    if key != _API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )


# ── Rate Limiter ───────────────────────────────────────────────────────────────

_MAX_REQUESTS: int = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
_WINDOW_SECONDS: int = int(os.getenv("RATE_LIMIT_WINDOW", "60"))

_lock = Lock()
_counters: dict[str, list[float]] = defaultdict(list)


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def make_rate_limit_middleware() -> Callable:
    async def middleware(request: Request, call_next: Callable):
        if request.url.path in ("/health", "/"):
            return await call_next(request)
        ip = _client_ip(request)
        now = time.monotonic()
        cutoff = now - _WINDOW_SECONDS
        with _lock:
            timestamps = _counters[ip]
            # Evict expired entries
            while timestamps and timestamps[0] < cutoff:
                timestamps.pop(0)
            if len(timestamps) >= _MAX_REQUESTS:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded. Try again later."},
                    headers={"Retry-After": str(_WINDOW_SECONDS)},
                )
            timestamps.append(now)
        return await call_next(request)
    return middleware
