from __future__ import annotations

import socket
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import uvicorn

from verl_mint import create_app
from verl_mint.backends.base import InferenceBackend, TrainingBackend
from verl_mint.backends.qwen_sft import QwenSFTTrainingBackend, QwenTextInferenceBackend
from verl_mint.contracts import (
    ArtifactRef,
    BackendKind,
    BackendSpec,
    CheckpointRequest,
    GenerateRequest,
    GenerateResult,
    SessionHandle,
    SessionSpec,
    TokenizerInfo,
    TrainOp,
    TrainOpRequest,
    TrainOpResult,
    TrainingCapabilities,
    TrainState,
)
from verl_mint.defaults import DEFAULT_BASE_MODEL_ID
from verl_mint.service import MintService
from verl_mint.storage import LocalStorageRepo


class ServerThread(threading.Thread):
    def __init__(self, *, app, host: str, port: int) -> None:
        super().__init__(daemon=True)
        config = uvicorn.Config(app, host=host, port=port, log_level="warning")
        self.server = uvicorn.Server(config)

    def run(self) -> None:
        self.server.run()

    def stop(self) -> None:
        self.server.should_exit = True


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_port(host: str, port: int, timeout_s: float = 30.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex((host, port)) == 0:
                return
        time.sleep(0.1)
    raise TimeoutError(f"server did not start on {host}:{port} within {timeout_s}s")


class FakeInferenceBackend(InferenceBackend):
    def generate(self, req: GenerateRequest) -> GenerateResult:
        return GenerateResult(text=req.prompt.upper(), raw={"provider": "fake-vllm"})


class NoopInferenceBackend(InferenceBackend):
    def generate(self, req: GenerateRequest) -> GenerateResult:
        return GenerateResult(text=req.prompt, raw={"provider": "noop"})


@dataclass
class _Session:
    step: int = 0


class FakeTrainingBackend(TrainingBackend):
    def __init__(self, checkpoint_root: Path | None = None) -> None:
        self.sessions: dict[str, _Session] = {}
        self.checkpoint_root = checkpoint_root or Path(
            tempfile.mkdtemp(prefix="verl-mint-fake-ckpt-")
        )

    def open_session(self, spec: SessionSpec) -> SessionHandle:
        self.sessions[spec.session_id] = _Session()
        return SessionHandle(backend_session_id=spec.session_id)

    def close_session(self, handle: SessionHandle) -> None:
        self.sessions.pop(handle.backend_session_id, None)

    def capabilities(self) -> TrainingCapabilities:
        return TrainingCapabilities(
            supports_forward=True,
            supports_train_step=True,
            supports_reverse_kl=False,
            supports_tokenizer_info=True,
            supports_reset_expert_bias=False,
            supports_checkpoint_load=True,
            supports_checkpoint_save=True,
            extras={"loss_fns": ["cross_entropy", "ppo"]},
        )

    def get_tokenizer_info(self, handle: SessionHandle) -> TokenizerInfo:
        return TokenizerInfo(metadata={"bos": 1, "eos": 2})

    def forward(self, handle: SessionHandle, req: TrainOpRequest) -> TrainOpResult:
        s = self.sessions[handle.backend_session_id]
        return TrainOpResult(state=TrainState(step=s.step), outputs={"mode": "forward"})

    def run_train_op(self, handle: SessionHandle, req: TrainOpRequest) -> TrainOpResult:
        s = self.sessions[handle.backend_session_id]
        if req.op in {TrainOp.FORWARD_BACKWARD, TrainOp.TRAIN_STEP, TrainOp.CUSTOM}:
            s.step += 1
        elif req.op is TrainOp.OPTIMIZER_STEP:
            s.step += 10
        return TrainOpResult(state=TrainState(step=s.step), outputs={"op": req.op.value})

    def forward_backward_ppo(self, handle: SessionHandle, req: TrainOpRequest) -> TrainOpResult:
        s = self.sessions[handle.backend_session_id]
        s.step += 1
        return TrainOpResult(state=TrainState(step=s.step), outputs={"op": req.op.value})

    def forward_backward_reverse_kl(self, handle: SessionHandle, req: TrainOpRequest) -> TrainOpResult:
        s = self.sessions[handle.backend_session_id]
        s.step += 1
        return TrainOpResult(state=TrainState(step=s.step), outputs={"op": req.op.value})

    def reset_expert_bias(self, handle: SessionHandle) -> TrainOpResult:
        return TrainOpResult(
            state=TrainState(step=self.sessions[handle.backend_session_id].step),
            outputs={"reset": True},
        )

    def save_checkpoint(self, handle: SessionHandle, req: CheckpointRequest) -> ArtifactRef:
        path = self._checkpoint_path(req.uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("checkpoint", encoding="utf-8")
        return ArtifactRef(uri=req.uri, format="pt", metadata={})

    def load_checkpoint(self, handle: SessionHandle, req: CheckpointRequest) -> TrainState:
        self.sessions[handle.backend_session_id].step = 7
        return TrainState(step=7)

    def _checkpoint_path(self, uri: str) -> Path:
        parsed = urlsplit(uri)
        if not parsed.scheme or parsed.scheme == "file":
            return Path(parsed.path if parsed.scheme == "file" else uri)
        rel = Path(parsed.netloc, parsed.path.lstrip("/"))
        return self.checkpoint_root / parsed.scheme / rel


def fake_app(storage_root: Path):
    service = MintService(storage=LocalStorageRepo(storage_root))
    service.backends.register(
        BackendSpec(
            backend_id="vllm-http",
            kind=BackendKind.INFERENCE,
            provider="vllm",
            model_family="qwen",
        ),
        FakeInferenceBackend(),
    )
    service.backends.register(
        BackendSpec(
            backend_id="megatron-http",
            kind=BackendKind.TRAINING,
            provider="megatron",
            model_family="qwen",
        ),
        FakeTrainingBackend(),
    )
    return create_app(service)


def qwen_app(storage_root: Path, *, inference: bool = True):
    service = MintService(storage=LocalStorageRepo(storage_root))
    if inference:
        service.backends.register(
            BackendSpec(
                backend_id="qwen-infer-local",
                kind=BackendKind.INFERENCE,
                provider="transformers",
                model_family="qwen",
            ),
            QwenTextInferenceBackend(model_id=DEFAULT_BASE_MODEL_ID),
        )
    service.backends.register(
        BackendSpec(
            backend_id="qwen-sft-local",
            kind=BackendKind.TRAINING,
            provider="transformers-peft",
            model_family="qwen",
        ),
        QwenSFTTrainingBackend(model_id=DEFAULT_BASE_MODEL_ID),
    )
    return create_app(service)


def verl_app(
    storage_root: Path,
    *,
    inference: bool = False,
    model_id: str = DEFAULT_BASE_MODEL_ID,
):
    from verl_mint.backends.verl import VerlInferenceBackend, VerlTrainingBackend

    service = MintService(storage=LocalStorageRepo(storage_root))
    if inference:
        service.backends.register(
            BackendSpec(
                backend_id="verl-infer",
                kind=BackendKind.INFERENCE,
                provider="verl",
                model_family="qwen",
            ),
            VerlInferenceBackend(
                backend_kwargs={"model_id": model_id},
            ),
        )
    service.backends.register(
        BackendSpec(
            backend_id="verl-train",
            kind=BackendKind.TRAINING,
            provider="verl",
            model_family="qwen",
        ),
        VerlTrainingBackend(
            backend_kwargs={"model_id": model_id},
        ),
    )
    return create_app(service)


def verl_ppo_app(
    storage_root: Path,
    *,
    model_path: str,
    trainer_config: dict,
):
    from verl_mint.backends.verl import VerlTrainingBackend

    service = MintService(storage=LocalStorageRepo(storage_root))
    service.backends.register(
        BackendSpec(
            backend_id="noop-infer",
            kind=BackendKind.INFERENCE,
            provider="noop",
            model_family="qwen",
        ),
        NoopInferenceBackend(),
    )
    service.backends.register(
        BackendSpec(
            backend_id="verl-train",
            kind=BackendKind.TRAINING,
            provider="verl",
            model_family="qwen",
        ),
        VerlTrainingBackend(
            model_path=model_path,
            backend_kwargs={"trainer": trainer_config},
        ),
    )
    return create_app(service)
