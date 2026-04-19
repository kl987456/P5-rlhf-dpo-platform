"""
Celery tasks for launching SFT, DPO, and Reward Model training runs.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict

from celery.utils.log import get_task_logger

from worker.celery_app import celery_app

logger = get_task_logger(__name__)


def _update_run(run_id: str, **kwargs) -> None:
    from api.db.database import db_session
    from api.db.models import TrainingRun
    with db_session() as db:
        run = db.get(TrainingRun, run_id)
        if run:
            for k, v in kwargs.items():
                setattr(run, k, v)


@celery_app.task(
    name="worker.tasks.launch_training_run",
    bind=True,
    acks_late=True,
)
def launch_training_run(
    self,
    run_id: str,
    run_type: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Launch a training run of type sft | dpo | rm.
    Updates the TrainingRun record with status and final loss.
    """
    _update_run(
        run_id,
        status="running",
        started_at=datetime.now(tz=timezone.utc),
    )

    try:
        if run_type == "sft":
            result = _run_sft(config)
        elif run_type == "dpo":
            result = _run_dpo(config)
        elif run_type == "rm":
            result = _run_rm(config)
        else:
            raise ValueError(f"Unknown run_type: {run_type!r}")

        _update_run(
            run_id,
            status="completed",
            completed_at=datetime.now(tz=timezone.utc),
            final_loss=result.get("final_loss"),
            num_steps=result.get("num_steps"),
            output_dir=result.get("output_dir"),
        )
        logger.info("[%s] Training run completed: %s", run_id, result)
        return result

    except Exception as exc:
        logger.exception("[%s] Training run failed: %s", run_id, exc)
        _update_run(
            run_id,
            status="failed",
            completed_at=datetime.now(tz=timezone.utc),
            error_message=str(exc)[:2000],
        )
        raise


def _run_sft(config: Dict[str, Any]) -> Dict[str, Any]:
    from training.sft_trainer import SFTConfig, run_sft
    cfg = SFTConfig(**{k: v for k, v in config.items() if hasattr(SFTConfig, k)})
    run_sft(cfg)
    return {"output_dir": cfg.output_dir}


def _run_dpo(config: Dict[str, Any]) -> Dict[str, Any]:
    from training.dpo_trainer import DPORunConfig, run_dpo
    cfg = DPORunConfig(**{k: v for k, v in config.items() if hasattr(DPORunConfig, k)})
    run_dpo(cfg)
    return {"output_dir": cfg.output_dir}


def _run_rm(config: Dict[str, Any]) -> Dict[str, Any]:
    from training.reward_model import RewardModelConfig, train_reward_model
    cfg = RewardModelConfig(**{k: v for k, v in config.items() if hasattr(RewardModelConfig, k)})
    preference_data = config.get("preference_data", [])
    train_reward_model(preference_data, cfg)
    return {"output_dir": cfg.output_dir}
