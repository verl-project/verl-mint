from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze_value(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_value(v) for v in value)
    if isinstance(value, tuple):
        return tuple(_freeze_value(v) for v in value)
    if isinstance(value, set):
        return frozenset(_freeze_value(v) for v in value)
    try:
        return deepcopy(value)
    except Exception as exc:
        raise TypeError(f"value is not safely copyable: {type(value).__name__}") from exc


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return _freeze_value(dict(value))


class BackendKind(str, Enum):
    INFERENCE = "inference"
    TRAINING = "training"


class TrainOp(str, Enum):
    FORWARD = "forward"
    FORWARD_BACKWARD = "forward_backward"
    FORWARD_BACKWARD_PPO = "forward_backward_ppo"
    FORWARD_BACKWARD_REVERSE_KL = "forward_backward_reverse_kl"
    OPTIMIZER_STEP = "optimizer_step"
    TRAIN_STEP = "train_step"
    RESET_EXPERT_BIAS = "reset_expert_bias"
    CUSTOM = "custom"


@dataclass(frozen=True)
class BackendSpec:
    backend_id: str
    kind: BackendKind
    provider: str
    model_family: str
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", BackendKind(self.kind))
        object.__setattr__(self, "config", _freeze_mapping(self.config))


@dataclass(frozen=True)
class GenerateRequest:
    prompt: str
    sampling: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sampling", _freeze_mapping(self.sampling))


@dataclass(frozen=True)
class GenerateResult:
    text: str
    token_ids: tuple[int, ...] = ()
    token_logprobs: tuple[float, ...] = ()
    prompt_token_ids: tuple[int, ...] = ()
    prompt_logprobs: tuple[float | None, ...] = ()
    stop_reason: str = "length"
    raw: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "token_ids", tuple(int(tok) for tok in self.token_ids))
        object.__setattr__(
            self,
            "token_logprobs",
            tuple(float(lp) for lp in self.token_logprobs),
        )
        object.__setattr__(self, "prompt_token_ids", tuple(int(tok) for tok in self.prompt_token_ids))
        object.__setattr__(
            self,
            "prompt_logprobs",
            tuple(None if lp is None else float(lp) for lp in self.prompt_logprobs),
        )
        object.__setattr__(self, "raw", _freeze_mapping(self.raw))


@dataclass(frozen=True)
class SessionSpec:
    session_id: str
    backend_id: str
    batch_codec: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True)
class SessionHandle:
    backend_session_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True)
class TrainState:
    step: int
    extras: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "extras", _freeze_mapping(self.extras))


@dataclass(frozen=True)
class TrainingCapabilities:
    supports_forward: bool = False
    supports_train_step: bool = True
    supports_reverse_kl: bool = False
    supports_tokenizer_info: bool = True
    supports_reset_expert_bias: bool = False
    supports_checkpoint_load: bool = True
    supports_checkpoint_save: bool = True
    extras: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "extras", _freeze_mapping(self.extras))


@dataclass(frozen=True)
class TokenizerInfo:
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True)
class TrainingSessionView:
    session_id: str
    backend_id: str
    batch_codec: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    state: TrainState | None = None
    status: str = "active"
    last_error: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True)
class TrainOpRequest:
    op: TrainOp
    batch_codec: str
    batch_payload: Any
    extension_op: str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "batch_payload", _freeze_value(self.batch_payload))
        object.__setattr__(self, "options", _freeze_mapping(self.options))


@dataclass(frozen=True)
class TrainOpResult:
    state: TrainState
    outputs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "outputs", _freeze_mapping(self.outputs))


@dataclass(frozen=True)
class CheckpointRequest:
    uri: str
    include_optimizer: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True)
class ArtifactRef:
    uri: str
    format: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True)
class RolloutSessionSpec:
    rollout_session_id: str
    inference_backend_id: str
    training_session_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True)
class ExperienceBatch:
    rollout_session_id: str
    training_session_id: str
    batch_codec: str
    batch_payload: Any
    policy_backend_id: str
    policy_version: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "batch_payload", _freeze_value(self.batch_payload))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True)
class RolloutSessionView:
    rollout_session_id: str
    inference_backend_id: str
    training_session_id: str
    status: str = "active"
    last_error: str | None = None
