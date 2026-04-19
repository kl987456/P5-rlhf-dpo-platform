"""
Direct Preference Optimization (DPO)
======================================
DPO (Rafailov et al., 2023) eliminates the explicit reward model by
reparameterizing the RLHF objective directly in terms of the policy.

The optimal policy under the KL-constrained reward maximization:
    pi*(y|x) = pi_ref(y|x) * exp(r(x,y)/beta) / Z(x)

Substituting into the preference probability and solving gives the DPO loss:

    L_DPO = -E_{(x,y_w,y_l)} [
        log sigma( beta * (
            log pi(y_w|x) - log pi_ref(y_w|x) -
            log pi(y_l|x) + log pi_ref(y_l|x)
        ))
    ]

This is equivalent to training with an implicit reward:
    r_implicit(x,y) = beta * log(pi(y|x) / pi_ref(y|x))

No reward model needed — preference signal flows directly into the policy.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class DPORunConfig:
    sft_model_path: str = "sft_weights"
    preference_dataset_path: str = "data/preferences.json"
    output_dir: str = "dpo_weights"

    # DPO core hyperparameters
    beta: float = 0.1            # KL penalty coefficient (lower = more creative)
    loss_type: str = "sigmoid"   # "sigmoid" | "hinge" | "ipo"

    # Training
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    lr: float = 5e-5
    warmup_ratio: float = 0.1
    max_length: int = 1024
    max_prompt_length: int = 512
    weight_decay: float = 0.01

    # Logging
    wandb_project: str = "rlhf-platform"
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 100

    # QLoRA (optional — use if GPU memory is tight)
    use_qlora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32

    device_map: str = "auto"


def run_dpo(cfg: Optional[DPORunConfig] = None) -> None:
    """
    DPO fine-tuning pipeline:
    1. Load SFT model as both policy and frozen reference
    2. Load preference dataset
    3. Train with TRL DPOTrainer
    4. Save final policy (merged or adapter-only)
    """
    if cfg is None:
        cfg = DPORunConfig()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import DPOTrainer, DPOConfig
        from datasets import Dataset
        import wandb
    except ImportError as e:
        raise ImportError(
            f"Missing dependency: {e}. Run: pip install trl transformers datasets wandb"
        ) from e

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    wandb.init(project=cfg.wandb_project, name="dpo", config=cfg.__dict__)
    logger.info("Starting DPO training")
    logger.info(f"  beta={cfg.beta}, loss_type={cfg.loss_type}")

    # ---- Load preference data ----
    with open(cfg.preference_dataset_path) as f:
        raw_prefs = json.load(f)

    logger.info(f"Loaded {len(raw_prefs)} preference pairs from {cfg.preference_dataset_path}")

    dataset = Dataset.from_list([
        {
            "prompt": p["prompt"],
            "chosen": p["chosen"],
            "rejected": p["rejected"],
        }
        for p in raw_prefs
    ])

    # 90/10 train/eval split
    splits = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = splits["train"]
    eval_dataset = splits["test"]
    logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # ---- Load model ----
    if cfg.use_qlora:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg.sft_model_path,
            quantization_config=bnb_config,
            device_map=cfg.device_map,
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
        peft_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_cfg)
        # DPO with QLoRA: ref_model=None, DPOTrainer creates it from base
        ref_model = None
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.sft_model_path,
            torch_dtype=torch.bfloat16,
            device_map=cfg.device_map,
            trust_remote_code=True,
        )
        # Frozen reference policy — identical weights, no grad
        ref_model = AutoModelForCausalLM.from_pretrained(
            cfg.sft_model_path,
            torch_dtype=torch.bfloat16,
            device_map=cfg.device_map,
            trust_remote_code=True,
        )
        for param in ref_model.parameters():
            param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(cfg.sft_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # DPO works better with left-padding

    # ---- DPO config ----
    dpo_cfg = DPOConfig(
        output_dir=cfg.output_dir,
        beta=cfg.beta,
        loss_type=cfg.loss_type,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        max_length=cfg.max_length,
        max_prompt_length=cfg.max_prompt_length,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps,
        eval_strategy="steps",
        report_to="wandb",
        bf16=True,
        remove_unused_columns=False,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=dpo_cfg,
    )

    logger.info("DPO training started...")
    trainer.train()
    logger.info("DPO training complete. Saving model...")

    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    with open(f"{cfg.output_dir}/dpo_config.json", "w") as f:
        json.dump(cfg.__dict__, f, indent=2)

    wandb.finish()
    logger.info(f"DPO model saved to {cfg.output_dir}")


# ---------------------------------------------------------------------------
# Implicit reward analysis helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_implicit_reward(
    model: "AutoModelForCausalLM",
    ref_model: "AutoModelForCausalLM",
    tokenizer: "AutoTokenizer",
    prompt: str,
    response: str,
    beta: float = 0.1,
    device: str = "cpu",
) -> float:
    """
    r_implicit(x,y) = beta * (log pi(y|x) - log pi_ref(y|x))

    Useful for ranking responses at inference time without a reward model.
    """
    text = f"{prompt}\n\n{response}"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    input_ids = inputs["input_ids"]

    def _log_prob(m):
        logits = m(**inputs).logits  # (1, T, V)
        log_probs = torch.log_softmax(logits, dim=-1)
        # Gather the log prob of each actual token (teacher-forcing)
        token_log_probs = log_probs[:, :-1, :].gather(
            dim=-1, index=input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)  # (1, T-1)
        return token_log_probs.sum().item()

    log_pi = _log_prob(model.to(device))
    log_pi_ref = _log_prob(ref_model.to(device))
    return beta * (log_pi - log_pi_ref)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="DPO fine-tuning")
    parser.add_argument("--sft-model", required=True, help="Path to SFT checkpoint")
    parser.add_argument("--data", required=True, help="Path to preference JSON")
    parser.add_argument("--output-dir", default="dpo_weights")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--loss-type", default="sigmoid", choices=["sigmoid", "hinge", "ipo"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--use-qlora", action="store_true")
    args = parser.parse_args()

    cfg = DPORunConfig(
        sft_model_path=args.sft_model,
        preference_dataset_path=args.data,
        output_dir=args.output_dir,
        beta=args.beta,
        loss_type=args.loss_type,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_qlora=args.use_qlora,
    )
    run_dpo(cfg)
