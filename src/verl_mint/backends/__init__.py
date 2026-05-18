from __future__ import annotations

from typing import Any

from verl_mint.backends.base import Backend, InferenceBackend, TrainingBackend
from verl_mint.backends.mint_training import MinTTrainingAdapter, build_mint_grpo_datums

_LAZY_EXPORTS = {
    "QwenSFTTrainingBackend": ("verl_mint.backends.qwen_sft", "QwenSFTTrainingBackend"),
    "QwenTextInferenceBackend": ("verl_mint.backends.qwen_sft", "QwenTextInferenceBackend"),
    "VerlBatchAdapter": ("verl_mint.backends.verl", "VerlBatchAdapter"),
    "VerlTrainingBackend": ("verl_mint.backends.verl", "VerlTrainingBackend"),
    "VerlInferenceBackend": ("verl_mint.backends.verl", "VerlInferenceBackend"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        import importlib

        module_name, attr_name = _LAZY_EXPORTS[name]
        value = getattr(importlib.import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "Backend",
    "InferenceBackend",
    "TrainingBackend",
    "QwenSFTTrainingBackend",
    "QwenTextInferenceBackend",
    "MinTTrainingAdapter",
    "build_mint_grpo_datums",
    "VerlBatchAdapter",
    "VerlTrainingBackend",
    "VerlInferenceBackend",
]
