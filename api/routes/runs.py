"""
/runs endpoints — launch and monitor SFT/DPO/RM training runs.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from api.db.database import get_db
from api.db.models import TrainingRun
from api.models.schemas import TrainingRunCreate, TrainingRunOut

router = APIRouter(prefix="/runs", tags=["training"])


@router.post("/", response_model=TrainingRunOut, status_code=status.HTTP_202_ACCEPTED)
def create_run(body: TrainingRunCreate, db: Session = Depends(get_db)) -> TrainingRunOut:
    """Launch a training run asynchronously via Celery."""
    from worker.tasks import launch_training_run  # imported here to avoid circular

    run = TrainingRun(
        run_type=body.run_type,
        model_name=body.model_name,
        config_json=json.dumps(body.config),
        output_dir=body.output_dir,
        status="pending",
    )
    db.add(run)
    db.flush()

    launch_training_run.apply_async(
        kwargs={"run_id": run.id, "run_type": body.run_type, "config": body.config},
        task_id=run.id,
        queue="training",
    )

    return TrainingRunOut.model_validate(run)


@router.get("/", response_model=List[TrainingRunOut])
def list_runs(
    run_type: str = None,
    limit: int = 50,
    db: Session = Depends(get_db),
) -> List[TrainingRunOut]:
    q = db.query(TrainingRun)
    if run_type:
        q = q.filter(TrainingRun.run_type == run_type)
    runs = q.order_by(TrainingRun.created_at.desc()).limit(limit).all()
    return [TrainingRunOut.model_validate(r) for r in runs]


@router.get("/{run_id}", response_model=TrainingRunOut)
def get_run(run_id: str, db: Session = Depends(get_db)) -> TrainingRunOut:
    run = db.get(TrainingRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Training run not found")
    return TrainingRunOut.model_validate(run)
