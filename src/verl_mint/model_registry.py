from __future__ import annotations

import os
import re
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ModelConfig:
    num_parameters: float
    is_moe: bool
    inference_tp: int
    inference_dp: int
    max_model_len: int
    train_tp: int = 1
    train_pp: int = 1
    train_ep: int = 1
    train_cp: int = 1
    train_etp: int | None = None
    gradient_checkpointing: bool = False
    gpu_memory_utilization: float | None = None
    max_loras: int | None = None
    max_cpu_loras: int | None = None
    max_lora_rank: int | None = None
    train_lora_rank: int | None = None
    train_lora_alpha: int | None = None
    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None
    vllm_engine: str = "verl_http"
    vllm_distributed_executor_backend: str = "mp"

    @property
    def inference_gpus(self) -> int:
        return self.inference_tp * self.inference_dp

    @property
    def train_gpus(self) -> int:
        etp = self.train_etp if self.train_etp is not None else self.train_tp
        if self.train_ep >= self.train_tp and etp < self.train_tp:
            return self.train_ep * self.train_pp * self.train_cp
        if self.train_ep > 1 and self.train_cp > 1:
            return self.train_tp * self.train_pp * max(self.train_ep, self.train_cp)
        return self.train_tp * self.train_pp * self.train_ep * self.train_cp


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "Qwen/Qwen3-0.6B": ModelConfig(
        num_parameters=0.6,
        is_moe=False,
        inference_tp=1,
        inference_dp=1,
        train_tp=1,
        train_ep=1,
        max_model_len=32768,
    ),
    "Qwen/Qwen3-4B": ModelConfig(
        num_parameters=4.0,
        is_moe=False,
        inference_tp=1,
        inference_dp=1,
        train_tp=1,
        train_ep=1,
        max_model_len=32768,
        gradient_checkpointing=True,
    ),
    "Qwen/Qwen3-30B-A3B-Instruct-2507": ModelConfig(
        num_parameters=30.0,
        is_moe=True,
        inference_tp=4,
        inference_dp=1,
        train_tp=4,
        train_ep=1,
        max_model_len=32768,
        max_num_seqs=704,
        max_num_batched_tokens=1024,
        gpu_memory_utilization=0.85,
        max_loras=21,
        max_cpu_loras=210,
        max_lora_rank=64,
        train_lora_rank=16,
        train_lora_alpha=32,
        gradient_checkpointing=True,
        vllm_engine="async",
        vllm_distributed_executor_backend="mp",
    ),
    "Qwen/Qwen3-30B-A3B": ModelConfig(
        num_parameters=30.0,
        is_moe=True,
        inference_tp=4,
        inference_dp=1,
        train_tp=4,
        train_ep=1,
        max_model_len=32768,
        train_lora_rank=16,
        train_lora_alpha=32,
        gradient_checkpointing=True,
    ),
    "Qwen/Qwen3-30B-A3B-Base": ModelConfig(
        num_parameters=30.0,
        is_moe=True,
        inference_tp=4,
        inference_dp=1,
        train_tp=4,
        train_ep=1,
        max_model_len=32768,
        train_lora_rank=16,
        train_lora_alpha=32,
        gradient_checkpointing=True,
    ),
    "Qwen/Qwen3-30B-A3B-Thinking-2507": ModelConfig(
        num_parameters=30.0,
        is_moe=True,
        inference_tp=4,
        inference_dp=1,
        train_tp=4,
        train_ep=1,
        max_model_len=32768,
        train_lora_rank=16,
        train_lora_alpha=32,
        gradient_checkpointing=True,
    ),
}


def normalize_model_name(model_name_or_path: str) -> str:
    if model_name_or_path in MODEL_CONFIGS:
        return model_name_or_path
    match = re.search(r"models--([^/]+)--([^/]+)", model_name_or_path)
    if match:
        org, model = match.groups()
        candidate = f"{org}/{model}"
        if candidate in MODEL_CONFIGS:
            return candidate
    raise ValueError(f"unsupported model: {model_name_or_path}")


def get_model_config(model_name_or_path: str) -> ModelConfig:
    cfg = MODEL_CONFIGS[normalize_model_name(model_name_or_path)]
    raw = os.environ.get("VERL_MINT_MODEL_CONFIG_OVERRIDES_JSON", "").strip()
    if not raw:
        return cfg
    import json

    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("VERL_MINT_MODEL_CONFIG_OVERRIDES_JSON must be a JSON object")
    override = data.get(normalize_model_name(model_name_or_path))
    if override is None:
        return cfg
    if not isinstance(override, dict):
        raise ValueError("model config override must be a JSON object")
    unknown = sorted(set(override) - set(ModelConfig.__dataclass_fields__))
    if unknown:
        raise ValueError(f"unknown model config override fields: {unknown}")
    return replace(cfg, **override)


def get_training_parallelism(model_name_or_path: str) -> tuple[int, int, int, int, int | None]:
    cfg = get_model_config(model_name_or_path)
    return cfg.train_tp, cfg.train_pp, cfg.train_ep, cfg.train_cp, cfg.train_etp
