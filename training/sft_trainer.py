"""
Supervised Fine-Tuning (SFT) with QLoRA
========================================
Uses 4-bit NF4 quantization (BitsAndBytes) + LoRA adapters via PEFT.
Trains only the LoRA parameters (~1% of weights) to fit on a single T4/A100.

Memory footprint (Phi-3-mini, 3.8B params):
  - 4-bit quantized base: ~2 GB
  - LoRA rank-16 adapters: ~80 MB
  - Activations + optimizer: ~2 GB
  Total: ~4-5 GB — fits on a T4 16 GB

Usage:
    python -m training.sft_trainer --model microsoft/Phi-3-mini-4k-instruct
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class SFTConfig:
    model_name: str = "microsoft/Phi-3-mini-4k-instruct"
    dataset_name: str = "yahma/alpaca-cleaned"
    output_dir: str = "sft_weights"

    # LoRA hyperparameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )

    # Training hyperparameters
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4   # effective batch = 16
    lr: float = 2e-4
    warmup_ratio: float = 0.03
    max_seq_length: int = 1024
    dataset_subset: str = "train[:5000]"

    # Quantization
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    use_double_quant: bool = True

    # Logging
    wandb_project: str = "rlhf-platform"
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 200

    # Hardware
    device_map: str = "auto"


ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)

ALPACA_NO_INPUT_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{output}"
)


def format_alpaca(example: dict) -> str:
    """Format an Alpaca-style example into a training string."""
    if example.get("input", "").strip():
        return ALPACA_TEMPLATE.format(
            instruction=example["instruction"],
            input=example["input"],
            output=example["output"],
        )
    return ALPACA_NO_INPUT_TEMPLATE.format(
        instruction=example["instruction"],
        output=example["output"],
    )


def run_sft(cfg: Optional[SFTConfig] = None) -> None:
    """
    Full QLoRA SFT pipeline:
    1. Load 4-bit quantized model
    2. Apply LoRA via PEFT
    3. Load + format dataset
    4. Train with TRL SFTTrainer
    5. Save LoRA adapter weights
    """
    if cfg is None:
        cfg = SFTConfig()

    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainingArguments,
        )
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from trl import SFTTrainer
        from datasets import load_dataset
        import wandb
    except ImportError as e:
        raise ImportError(
            f"Missing dependency: {e}. Run: pip install transformers peft trl datasets bitsandbytes wandb"
        ) from e

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    wandb.init(project=cfg.wandb_project, name="sft", config=cfg.__dict__)
    logger.info("Starting SFT training with QLoRA")

    # ---- 1. Load quantized model ----
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_use_double_quant=cfg.use_double_quant,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    logger.info(f"Loading {cfg.model_name} in 4-bit NF4...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=bnb_config,
        device_map=cfg.device_map,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False  # required for gradient checkpointing
    model.config.pretraining_tp = 1

    # Prepare for k-bit training: casts norms to float32, enables grad ckpt
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # ---- 2. Apply LoRA ----
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.target_modules,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---- 3. Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # avoid warning with Flash Attention

    # ---- 4. Dataset ----
    logger.info(f"Loading dataset {cfg.dataset_name} ({cfg.dataset_subset})...")
    raw_dataset = load_dataset(cfg.dataset_name, split=cfg.dataset_subset)

    def preprocess(example):
        example["text"] = format_alpaca(example)
        return example

    dataset = raw_dataset.map(preprocess, num_proc=1, remove_columns=raw_dataset.column_names)
    logger.info(f"Dataset size: {len(dataset)} examples")

    # ---- 5. Training args ----
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=cfg.warmup_ratio,
        optim="paged_adamw_32bit",  # memory-efficient optimizer for QLoRA
        fp16=False,
        bf16=True,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=2,
        report_to="wandb",
        gradient_checkpointing=True,
        group_by_length=True,  # group similar-length sequences for efficiency
        dataloader_num_workers=0,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=cfg.max_seq_length,
        packing=False,
    )

    logger.info("Training started...")
    trainer.train()
    logger.info("Training complete. Saving adapter weights...")

    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    # Save training config
    with open(f"{cfg.output_dir}/sft_config.json", "w") as f:
        json.dump(cfg.__dict__, f, indent=2)

    wandb.finish()
    logger.info(f"SFT adapter saved to {cfg.output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="QLoRA SFT fine-tuning")
    parser.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--dataset", default="yahma/alpaca-cleaned")
    parser.add_argument("--output-dir", default="sft_weights")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--subset", default="train[:5000]")
    args = parser.parse_args()

    cfg = SFTConfig(
        model_name=args.model,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_seq_length=args.max_seq_length,
        dataset_subset=args.subset,
    )
    run_sft(cfg)
