"""Frozen Qwen hidden-state encoder used as the semantic log representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from transformers import AutoModel, AutoTokenizer


@dataclass(frozen=True)
class EncoderMetadata:
    model_name: str
    revision: str
    pooling: str
    max_length: int
    dtype: str
    device: str
    hidden_size: int


class FrozenQwenEncoder:
    """Load a Qwen model once and expose frozen hidden-state embeddings."""

    def __init__(
        self,
        model_name: str,
        *,
        load_path: str | None = None,
        revision: str = "main",
        pooling: str = "mean",
        max_length: int = 512,
        dtype: str = "bfloat16",
        device: str | None = None,
    ) -> None:
        if pooling not in {"mean", "first_token", "last_token"}:
            raise ValueError("pooling must be one of: mean, first_token, last_token")
        self.model_name = model_name
        source = load_path or model_name
        self.requested_revision = revision
        self.pooling = pooling
        self.max_length = int(max_length)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.torch_dtype = self._resolve_dtype(dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(source, revision=revision, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.model = AutoModel.from_pretrained(
            source,
            revision=revision,
            dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.hidden_size = int(self.model.config.hidden_size)
        self.revision = str(getattr(self.model.config, "_commit_hash", None) or revision)

    def _resolve_dtype(self, dtype: str) -> torch.dtype:
        normalized = dtype.lower()
        if normalized in {"bfloat16", "bf16"}:
            if self.device.type == "cuda" and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float32
        if normalized in {"float16", "fp16"}:
            return torch.float16 if self.device.type == "cuda" else torch.float32
        if normalized in {"float32", "fp32"}:
            return torch.float32
        raise ValueError(f"unsupported dtype: {dtype}")

    def _pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "first_token":
            return hidden[:, 0]
        if self.pooling == "last_token":
            indices = attention_mask.sum(dim=1).clamp(min=1) - 1
            return hidden[torch.arange(hidden.size(0), device=hidden.device), indices]
        weights = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)

    @torch.inference_mode()
    def encode(self, texts: Sequence[str], *, batch_size: int = 4) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        for start in range(0, len(texts), batch_size):
            batch = [str(text) for text in texts[start : start + batch_size]]
            tokens = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            tokens = {key: value.to(self.device) for key, value in tokens.items()}
            hidden = self.model(**tokens, return_dict=True).last_hidden_state
            outputs.append(self._pool(hidden, tokens["attention_mask"]).float().cpu())
        if not outputs:
            return torch.empty((0, self.hidden_size), dtype=torch.float32)
        return torch.cat(outputs, dim=0)

    def metadata(self) -> EncoderMetadata:
        return EncoderMetadata(
            model_name=self.model_name,
            revision=self.revision,
            pooling=self.pooling,
            max_length=self.max_length,
            dtype=str(self.torch_dtype).replace("torch.", ""),
            device=str(self.device),
            hidden_size=self.hidden_size,
        )
