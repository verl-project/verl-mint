from __future__ import annotations

import argparse
import socket
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import mint
import uvicorn
from mint import types

from verl_mint import create_app
from verl_mint.backends.base import InferenceBackend, TrainingBackend
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
from verl_mint.service import MintService
from verl_mint.storage import LocalStorageRepo

SMOKE_CLIENT_TOKEN = "unused"


class FakeInferenceBackend(InferenceBackend):
    def generate(self, req: GenerateRequest) -> GenerateResult:
        return GenerateResult(text=req.prompt.upper(), raw={"provider": "fake-vllm"})


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


class ServerThread(threading.Thread):
    def __init__(self, *, host: str, port: int, storage_root: Path) -> None:
        super().__init__(daemon=True)
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
        app = create_app(service)
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


def wait_for_port(host: str, port: int, timeout_s: float = 10.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex((host, port)) == 0:
                return
        time.sleep(0.1)
    raise TimeoutError(f"server did not start on {host}:{port} within {timeout_s}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument(
        "--storage-root",
        default="/tmp/verl-mint-official-client-smoke",
    )
    args = parser.parse_args()

    port = args.port or find_free_port()
    server = ServerThread(host=args.host, port=port, storage_root=Path(args.storage_root))
    server.start()

    try:
        wait_for_port(args.host, port)
        svc = mint.ServiceClient(base_url=f"http://{args.host}:{port}", api_key=SMOKE_CLIENT_TOKEN)
        train = svc.create_lora_training_client(
            base_model="Qwen/Qwen3-0.6B",
            rank=16,
            user_metadata={"smoke": "official-client"},
        )
        info = train.get_info()
        print("model_id", info.model_id)
        print("model_name", info.model_name)

        datum = types.Datum(
            model_input=types.ModelInput(chunks=[types.EncodedTextChunk(tokens=[10, 11, 12])]),
            loss_fn_inputs={
                "target_tokens": types.TensorData(data=[11, 12, 13], dtype="int64", shape=[3]),
                "loss_mask": types.TensorData(data=[1.0, 1.0, 1.0], dtype="float32", shape=[3]),
            },
        )
        fb = train.forward_backward([datum], loss_fn="cross_entropy").result()
        print("fb_metrics", fb.metrics)

        opt = train.optim_step(types.AdamParams(learning_rate=3e-4)).result()
        print("opt_metrics", opt.metrics)

        saved = train.save_state("checkpoint-001").result()
        print("saved_path", saved.path)

        loaded = train.load_state(saved.path).result()
        print("loaded_path", loaded.path)

        resumed = svc.create_training_client_from_state(saved.path)
        print("resumed_model_id", resumed.get_info().model_id)

        resumed_opt = svc.create_training_client_from_state_with_optimizer(saved.path)
        print("resumed_opt_model_id", resumed_opt.get_info().model_id)

        sampler_saved = train.save_weights_for_sampler("sampler-001").result()
        print("sampler_saved_path", sampler_saved.path)

        sampling_client = train.save_weights_and_get_sampling_client()
        sample = sampling_client.sample(
            prompt=types.ModelInput(chunks=[types.EncodedTextChunk(tokens=[1, 2, 3])]),
            num_samples=1,
            sampling_params=types.SamplingParams(max_tokens=2),
        ).result()
        print("sample_sequences", len(sample.sequences))
    finally:
        server.stop()
        server.join(timeout=5)


if __name__ == "__main__":
    main()
