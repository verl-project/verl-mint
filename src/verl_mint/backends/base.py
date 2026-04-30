from __future__ import annotations

from abc import ABC, abstractmethod

from verl_mint.contracts import (
    ArtifactRef,
    CheckpointRequest,
    GenerateRequest,
    GenerateResult,
    SessionHandle,
    SessionSpec,
    TokenizerInfo,
    TrainOpRequest,
    TrainOpResult,
    TrainingCapabilities,
    TrainState,
)


class Backend(ABC):
    pass


class InferenceBackend(Backend):
    @abstractmethod
    def generate(self, req: GenerateRequest) -> GenerateResult:
        raise NotImplementedError


class TrainingBackend(Backend):
    @abstractmethod
    def open_session(self, spec: SessionSpec) -> SessionHandle:
        raise NotImplementedError

    @abstractmethod
    def close_session(self, handle: SessionHandle) -> None:
        raise NotImplementedError

    @abstractmethod
    def capabilities(self) -> TrainingCapabilities:
        raise NotImplementedError

    @abstractmethod
    def get_tokenizer_info(self, handle: SessionHandle) -> TokenizerInfo:
        raise NotImplementedError

    @abstractmethod
    def forward(self, handle: SessionHandle, req: TrainOpRequest) -> TrainOpResult:
        raise NotImplementedError

    @abstractmethod
    def run_train_op(self, handle: SessionHandle, req: TrainOpRequest) -> TrainOpResult:
        raise NotImplementedError

    @abstractmethod
    def forward_backward_ppo(
        self,
        handle: SessionHandle,
        req: TrainOpRequest,
    ) -> TrainOpResult:
        raise NotImplementedError

    @abstractmethod
    def forward_backward_reverse_kl(
        self,
        handle: SessionHandle,
        req: TrainOpRequest,
    ) -> TrainOpResult:
        raise NotImplementedError

    @abstractmethod
    def reset_expert_bias(self, handle: SessionHandle) -> TrainOpResult:
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self, handle: SessionHandle, req: CheckpointRequest) -> ArtifactRef:
        raise NotImplementedError

    @abstractmethod
    def load_checkpoint(self, handle: SessionHandle, req: CheckpointRequest) -> TrainState:
        raise NotImplementedError
