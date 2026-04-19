"""
ORM models for the RLHF platform.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class Prompt(Base):
    """A prompt shown to annotators for pairwise preference collection."""
    __tablename__ = "prompts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[Optional[str]] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class CompletionPair(Base):
    """A (chosen, rejected) completion pair for a prompt."""
    __tablename__ = "completion_pairs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    prompt_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    chosen: Mapped[str] = mapped_column(Text, nullable=False)
    rejected: Mapped[str] = mapped_column(Text, nullable=False)
    annotator_id: Mapped[Optional[str]] = mapped_column(String(128))
    confidence: Mapped[Optional[float]] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class TrainingRun(Base):
    """Record of an SFT/DPO/RM training run."""
    __tablename__ = "training_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    run_type: Mapped[str] = mapped_column(String(32), nullable=False)   # sft | dpo | rm
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    model_name: Mapped[str] = mapped_column(String(256), nullable=False)
    config_json: Mapped[Optional[str]] = mapped_column(Text)
    output_dir: Mapped[Optional[str]] = mapped_column(Text)
    final_loss: Mapped[Optional[float]] = mapped_column(Float)
    num_steps: Mapped[Optional[int]] = mapped_column(Integer)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
