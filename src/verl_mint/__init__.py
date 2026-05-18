from __future__ import annotations

from typing import Any

from verl_mint.backends.base import Backend, InferenceBackend, TrainingBackend
from verl_mint.contracts import (
    ArtifactRef,
    BackendKind,
    BackendSpec,
    CheckpointRequest,
    ExperienceBatch,
    GenerateRequest,
    GenerateResult,
    RolloutSessionSpec,
    RolloutSessionView,
    SessionHandle,
    SessionSpec,
    TokenizerInfo,
    TrainOp,
    TrainOpRequest,
    TrainOpResult,
    TrainingCapabilities,
    TrainingSessionView,
    TrainState,
)
from verl_mint.errors import (
    BackendError,
    BackendKindMismatchError,
    UnknownBackendError,
    UnknownSessionError,
    UnsupportedOperationError,
)
from verl_mint.model_registry import ModelConfig, get_model_config, get_training_parallelism, normalize_model_name
from verl_mint.router import BackendRegistry, BackendRouter
from verl_mint.service import (
    BackendRegistryService,
    InferenceService,
    MintService,
    RolloutService,
    TrainingService,
)
from verl_mint.storage import LocalStorageRepo, default_storage_root, shared_storage_roots_env_name, storage_env_name

_LAZY_EXPORTS = {
    "VerlBatchAdapter": ("verl_mint.backends.verl", "VerlBatchAdapter"),
    "VerlTrainingBackend": ("verl_mint.backends.verl", "VerlTrainingBackend"),
    "VerlInferenceBackend": ("verl_mint.backends.verl", "VerlInferenceBackend"),
}


def create_app(*args: Any, **kwargs: Any):
    from verl_mint.app import create_app as _create_app

    return _create_app(*args, **kwargs)


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        import importlib

        module_name, attr_name = _LAZY_EXPORTS[name]
        value = getattr(importlib.import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "create_app",
    "ArtifactRef",
    "Backend",
    "BackendError",
    "BackendKind",
    "BackendKindMismatchError",
    "BackendRegistry",
    "BackendRegistryService",
    "BackendRouter",
    "BackendSpec",
    "CheckpointRequest",
    "ExperienceBatch",
    "GenerateRequest",
    "GenerateResult",
    "InferenceBackend",
    "InferenceService",
    "VerlBatchAdapter",
    "VerlTrainingBackend",
    "VerlInferenceBackend",
    "LocalStorageRepo",
    "default_storage_root",
    "shared_storage_roots_env_name",
    "MintService",
    "ModelConfig",
    "get_model_config",
    "get_training_parallelism",
    "normalize_model_name",
    "RolloutService",
    "RolloutSessionSpec",
    "RolloutSessionView",
    "SessionHandle",
    "SessionSpec",
    "TokenizerInfo",
    "TrainOp",
    "TrainOpRequest",
    "TrainOpResult",
    "TrainingBackend",
    "TrainingCapabilities",
    "TrainingService",
    "TrainingSessionView",
    "TrainState",
    "UnknownBackendError",
    "UnknownSessionError",
    "UnsupportedOperationError",
    "storage_env_name",
]
