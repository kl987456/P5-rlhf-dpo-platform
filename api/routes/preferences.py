"""
/preferences endpoints — manage human preference annotations.
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from api.db.database import get_db
from api.db.models import CompletionPair, Prompt
from api.models.schemas import (
    PreferenceCreate,
    PreferenceOut,
    PromptCreate,
    PromptOut,
)

router = APIRouter(prefix="/preferences", tags=["preferences"])


# ── Prompts ────────────────────────────────────────────────────────────────────

@router.post("/prompts", response_model=PromptOut, status_code=status.HTTP_201_CREATED)
def create_prompt(body: PromptCreate, db: Session = Depends(get_db)) -> PromptOut:
    prompt = Prompt(text=body.text, source=body.source)
    db.add(prompt)
    db.flush()
    return PromptOut.model_validate(prompt)


@router.get("/prompts", response_model=List[PromptOut])
def list_prompts(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> List[PromptOut]:
    prompts = db.query(Prompt).order_by(Prompt.created_at.desc()).offset(offset).limit(limit).all()
    return [PromptOut.model_validate(p) for p in prompts]


# ── Preference pairs ───────────────────────────────────────────────────────────

@router.post("/", response_model=PreferenceOut, status_code=status.HTTP_201_CREATED)
def create_preference(body: PreferenceCreate, db: Session = Depends(get_db)) -> PreferenceOut:
    if db.get(Prompt, body.prompt_id) is None:
        raise HTTPException(status_code=404, detail="Prompt not found")
    pair = CompletionPair(
        prompt_id=body.prompt_id,
        chosen=body.chosen,
        rejected=body.rejected,
        annotator_id=body.annotator_id,
        confidence=body.confidence,
    )
    db.add(pair)
    db.flush()
    return PreferenceOut.model_validate(pair)


@router.get("/", response_model=List[PreferenceOut])
def list_preferences(
    prompt_id: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
) -> List[PreferenceOut]:
    q = db.query(CompletionPair)
    if prompt_id:
        q = q.filter(CompletionPair.prompt_id == prompt_id)
    pairs = q.order_by(CompletionPair.created_at.desc()).limit(limit).all()
    return [PreferenceOut.model_validate(p) for p in pairs]


@router.get("/{preference_id}", response_model=PreferenceOut)
def get_preference(preference_id: str, db: Session = Depends(get_db)) -> PreferenceOut:
    pair = db.get(CompletionPair, preference_id)
    if pair is None:
        raise HTTPException(status_code=404, detail="Preference pair not found")
    return PreferenceOut.model_validate(pair)
