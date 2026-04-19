"""
Bradley-Terry Reward Model
==========================
Models human preference via the Bradley-Terry pair comparison model:

    P(y_w > y_l | x) = sigmoid(r(x, y_w) - r(x, y_l))

Training loss (negative log-likelihood of the preference):
    L = -E[log sigmoid(r_chosen - r_rejected)]

The reward head is a single linear scalar output on top of a pretrained LM.
We freeze the base model and fine-tune only the classification head during
early training, then optionally unfreeze with a small LR.
"""

from __future__ import annotations

import os
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    get_cosine_schedule_with_warmup,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PreferenceDataset(Dataset):
    """
    Each item: {"prompt": str, "chosen": str, "rejected": str}
    Tokenizes both chosen and rejected responses for Bradley-Terry loss.
    """

    def __init__(
        self,
        data: List[Dict],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def _encode(self, prompt: str, response: str) -> Dict[str, torch.Tensor]:
        text = f"Human: {prompt}\n\nAssistant: {response}"
        return self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        chosen_enc = self._encode(item["prompt"], item["chosen"])
        rejected_enc = self._encode(item["prompt"], item["rejected"])
        return {
            "chosen_input_ids": chosen_enc["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_enc["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_enc["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_enc["attention_mask"].squeeze(0),
        }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class RewardModel(nn.Module):
    """
    Bradley-Terry reward model.

    Architecture: pretrained encoder + linear scalar head.
    The backbone is AutoModelForSequenceClassification with num_labels=1,
    which adds a (hidden_size -> 1) projection on the [CLS] token.

    Forward: returns loss, accuracy, and raw reward scalars for
    chosen/rejected pairs so callers can log and analyse both.
    """

    def __init__(self, base_model_name: str, freeze_base: bool = False):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=1,
            ignore_mismatched_sizes=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        if freeze_base:
            for name, param in self.model.named_parameters():
                if "classifier" not in name and "score" not in name:
                    param.requires_grad = False

    # ------------------------------------------------------------------
    # Core forward — Bradley-Terry loss
    # ------------------------------------------------------------------

    def forward(
        self,
        chosen_input_ids: torch.Tensor,
        chosen_attention_mask: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        rejected_attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Bradley-Terry loss.

        Loss = -log(sigmoid(r_chosen - r_rejected))
             = log(1 + exp(r_rejected - r_chosen))      [numerically stable]

        Returns dict with:
            loss       — scalar training loss
            accuracy   — fraction of pairs where r_chosen > r_rejected
            r_chosen   — (B,) reward logits for chosen responses
            r_rejected — (B,) reward logits for rejected responses
            margin     — (B,) r_chosen - r_rejected  (positive = correct ranking)
        """
        r_chosen = self.model(
            input_ids=chosen_input_ids,
            attention_mask=chosen_attention_mask,
        ).logits.squeeze(-1)  # (B,)

        r_rejected = self.model(
            input_ids=rejected_input_ids,
            attention_mask=rejected_attention_mask,
        ).logits.squeeze(-1)  # (B,)

        margin = r_chosen - r_rejected

        # Bradley-Terry NLL: -log sigmoid(margin)
        # Using log_sigmoid for numerical stability
        loss = -torch.nn.functional.logsigmoid(margin).mean()
        accuracy = (margin > 0).float().mean()

        return {
            "loss": loss,
            "accuracy": accuracy,
            "r_chosen": r_chosen,
            "r_rejected": r_rejected,
            "margin": margin,
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def score(self, prompt: str, response: str, device: str = "cpu") -> float:
        """
        Score a single (prompt, response) pair.
        Higher score = better response according to the reward model.
        """
        self.eval()
        self.model.to(device)
        text = f"Human: {prompt}\n\nAssistant: {response}"
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)
        logits = self.model(**inputs).logits.squeeze()
        return float(logits.item())

    @torch.no_grad()
    def score_batch(
        self, prompts: List[str], responses: List[str], device: str = "cpu"
    ) -> List[float]:
        """Batch scoring for efficiency."""
        self.eval()
        self.model.to(device)
        texts = [f"Human: {p}\n\nAssistant: {r}" for p, r in zip(prompts, responses)]
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(device)
        logits = self.model(**inputs).logits.squeeze(-1)
        return logits.tolist()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

@dataclass
class RewardModelConfig:
    base_model_name: str = "distilbert-base-uncased"
    num_epochs: int = 3
    batch_size: int = 8
    lr: float = 1e-4
    warmup_ratio: float = 0.1
    max_length: int = 512
    weight_decay: float = 0.01
    output_dir: str = "reward_model"
    freeze_base: bool = False
    eval_steps: int = 50
    save_steps: int = 200
    wandb_project: str = "rlhf-platform"
    gradient_clip: float = 1.0
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


def train_reward_model(
    preference_data: List[Dict],
    cfg: Optional[RewardModelConfig] = None,
) -> RewardModel:
    """
    Train a Bradley-Terry reward model on human preference pairs.

    Args:
        preference_data: list of {"prompt", "chosen", "rejected"} dicts
        cfg: training configuration

    Returns:
        Trained RewardModel instance
    """
    if cfg is None:
        cfg = RewardModelConfig()

    try:
        import wandb
        wandb.init(project=cfg.wandb_project, name="reward-model", config=cfg.__dict__)
        use_wandb = True
    except ImportError:
        logger.warning("wandb not installed — skipping logging")
        use_wandb = False

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # ---- Split train/eval (90/10) ----
    n_eval = max(1, int(len(preference_data) * 0.1))
    eval_data = preference_data[-n_eval:]
    train_data = preference_data[:-n_eval]

    model = RewardModel(cfg.base_model_name, freeze_base=cfg.freeze_base)
    model.to(cfg.device)

    train_ds = PreferenceDataset(train_data, model.tokenizer, cfg.max_length)
    eval_ds = PreferenceDataset(eval_data, model.tokenizer, cfg.max_length)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    eval_loader = DataLoader(eval_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    total_steps = len(train_loader) * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    global_step = 0
    best_eval_loss = float("inf")

    for epoch in range(cfg.num_epochs):
        model.train()
        epoch_losses: List[float] = []
        epoch_accs: List[float] = []

        for batch in train_loader:
            batch = {k: v.to(cfg.device) for k, v in batch.items()}
            outputs = model(**batch)

            loss = outputs["loss"]
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_losses.append(loss.item())
            epoch_accs.append(outputs["accuracy"].item())
            global_step += 1

            if global_step % 10 == 0:
                log = {
                    "train/loss": loss.item(),
                    "train/accuracy": outputs["accuracy"].item(),
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/step": global_step,
                }
                logger.info(f"Step {global_step}: {log}")
                if use_wandb:
                    wandb.log(log, step=global_step)

            # Eval checkpoint
            if global_step % cfg.eval_steps == 0:
                eval_metrics = _evaluate_reward_model(model, eval_loader, cfg.device)
                logger.info(f"Eval @ step {global_step}: {eval_metrics}")
                if use_wandb:
                    wandb.log({f"eval/{k}": v for k, v in eval_metrics.items()}, step=global_step)

                if eval_metrics["loss"] < best_eval_loss:
                    best_eval_loss = eval_metrics["loss"]
                    model.model.save_pretrained(f"{cfg.output_dir}/best")
                    model.tokenizer.save_pretrained(f"{cfg.output_dir}/best")
                    logger.info(f"New best saved: eval_loss={best_eval_loss:.4f}")
                model.train()

        logger.info(
            f"Epoch {epoch+1}/{cfg.num_epochs} — "
            f"loss={np.mean(epoch_losses):.4f}, acc={np.mean(epoch_accs):.4f}"
        )

    # Save final checkpoint
    model.model.save_pretrained(cfg.output_dir)
    model.tokenizer.save_pretrained(cfg.output_dir)
    with open(f"{cfg.output_dir}/config.json", "w") as f:
        json.dump(cfg.__dict__, f, indent=2)

    if use_wandb:
        wandb.finish()

    return model


def _evaluate_reward_model(
    model: RewardModel, loader: DataLoader, device: str
) -> Dict[str, float]:
    model.eval()
    losses, accs, margins = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            losses.append(out["loss"].item())
            accs.append(out["accuracy"].item())
            margins.extend(out["margin"].tolist())
    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(np.mean(accs)),
        "mean_margin": float(np.mean(margins)),
        "margin_std": float(np.std(margins)),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Bradley-Terry reward model")
    parser.add_argument("--data", required=True, help="Path to preference JSON")
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-dir", default="reward_model")
    args = parser.parse_args()

    with open(args.data) as f:
        data = json.load(f)

    cfg = RewardModelConfig(
        base_model_name=args.model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir,
    )
    train_reward_model(data, cfg)
