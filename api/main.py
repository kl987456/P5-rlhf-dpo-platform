"""
FastAPI application for the RLHF/DPO platform.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.db.database import Base, engine
from api.routes import preferences, runs
from api.security import make_rate_limit_middleware

APP_VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="RLHF/DPO Platform API",
    description=(
        "Human preference annotation + SFT → Reward Model → DPO training pipeline "
        "for aligning language models with human feedback."
    ),
    version=APP_VERSION,
    lifespan=lifespan,
)

_ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000").split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
)

app.middleware("http")(make_rate_limit_middleware())


@app.middleware("http")
async def _security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


app.include_router(preferences.router, prefix="/api/v1")
app.include_router(runs.router, prefix="/api/v1")


@app.get("/health", tags=["ops"])
def health() -> dict:
    return {"status": "ok", "version": APP_VERSION}


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {"message": "RLHF Platform API", "docs": "/docs"}


@app.exception_handler(Exception)
async def _unhandled(request, exc: Exception) -> JSONResponse:
    import logging
    logging.getLogger(__name__).exception("Unhandled: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
