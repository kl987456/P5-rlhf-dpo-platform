"""
Inference server — loads a fine-tuned (SFT or DPO) model and serves completions.

Supports both LoRA-merged and full checkpoints.
Uses greedy decoding (temperature=0) by default for reproducibility;
sampling is available for interactive use.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 0.0        # 0 = greedy
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = False


class ModelServer:
    """
    Wraps a HuggingFace causal LM for low-latency inference.

    Parameters
    ----------
    model_path : str
        Path to the merged checkpoint directory or HF model ID.
    device : str, optional
        "cuda", "cpu", or None (auto-detect).
    load_in_4bit : bool
        Use BitsAndBytes 4-bit quantization for GPU inference.
    """

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        load_in_4bit: bool = True,
    ) -> None:
        self.model_path = model_path
        self.load_in_4bit = load_in_4bit
        self._model = None
        self._tokenizer = None

        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load model and tokenizer (call once before serving)."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        logger.info("Loading tokenizer from %s", self.model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        bnb_config = None
        if self.load_in_4bit and self.device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        logger.info("Loading model from %s (device=%s, 4bit=%s)", self.model_path, self.device, self.load_in_4bit)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=bnb_config,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )
        if self.device == "cpu":
            self._model = self._model.to("cpu")
        self._model.eval()
        logger.info("Model loaded.")

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        """Generate a single completion for *prompt*."""
        if self._model is None:
            raise RuntimeError("Call load() before generate()")

        cfg = config or GenerationConfig()
        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature if cfg.do_sample else None,
                top_p=cfg.top_p if cfg.do_sample else None,
                top_k=cfg.top_k if cfg.do_sample else None,
                repetition_penalty=cfg.repetition_penalty,
                do_sample=cfg.do_sample,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def generate_batch(
        self,
        prompts: List[str],
        config: Optional[GenerationConfig] = None,
    ) -> List[str]:
        return [self.generate(p, config) for p in prompts]

    def score_preference(self, prompt: str, chosen: str, rejected: str) -> dict:
        """
        Compute implicit DPO reward for chosen vs rejected using the loaded model.
        Returns dict with chosen_logprob, rejected_logprob, and preference_margin.
        """
        if self._model is None:
            raise RuntimeError("Call load() before score_preference()")
        import torch

        def _logprob(text: str) -> float:
            full = prompt + text
            inputs = self._tokenizer(full, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self._model(**inputs, labels=inputs["input_ids"])
            return float(-out.loss.item())

        chosen_lp = _logprob(chosen)
        rejected_lp = _logprob(rejected)
        return {
            "chosen_logprob": chosen_lp,
            "rejected_logprob": rejected_lp,
            "preference_margin": chosen_lp - rejected_lp,
            "prefers_chosen": chosen_lp > rejected_lp,
        }
