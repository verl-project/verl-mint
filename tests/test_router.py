from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from urllib.parse import urlsplit

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

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
from verl_mint.errors import BackendKindMismatchError, UnsupportedOperationError
from verl_mint.router import BackendRouter
from verl_mint import create_app
from verl_mint.schemas import CreateModelRequest, ForwardBackwardRequest, ForwardBackwardInput, ModelInput, EncodedTextChunk
from verl_mint.service import MintService
from verl_mint.storage import (
    LocalStorageRepo,
    default_storage_root,
    shared_storage_env_name,
    shared_storage_roots_env_name,
    storage_env_name,
)


QWEN30B_MODEL_ID = "Qwen/Qwen3-30B-A3B-Base"


class FakeInferenceBackend(InferenceBackend):
    def generate(self, req: GenerateRequest) -> GenerateResult:
        prompt_tokens = [int(tok) for tok in req.prompt.split() if tok.lstrip("-").isdigit()]
        token_ids = tuple(prompt_tokens or [len(req.prompt)])
        return GenerateResult(
            text=req.prompt.upper(),
            token_ids=token_ids,
            token_logprobs=tuple([-0.1 for _ in token_ids]),
            prompt_token_ids=tuple(prompt_tokens),
            prompt_logprobs=tuple([None] + [-2.0 for _ in prompt_tokens[1:]]),
            raw={"provider": "fake-vllm"},
        )


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
        assert handle.backend_session_id is not None
        self.sessions.pop(handle.backend_session_id, None)

    def capabilities(self) -> TrainingCapabilities:
        return TrainingCapabilities(
            supports_forward=True,
            supports_train_step=True,
            supports_reverse_kl=True,
            supports_tokenizer_info=True,
            supports_reset_expert_bias=True,
            extras={"loss_fns": ["sft", "ppo", "grpo"]},
        )

    def get_tokenizer_info(self, handle: SessionHandle) -> TokenizerInfo:
        return TokenizerInfo(metadata={"bos": 1, "eos": 2})

    def forward(self, handle: SessionHandle, req: TrainOpRequest) -> TrainOpResult:
        return TrainOpResult(state=self._state(handle), outputs={"mode": "forward"})

    def run_train_op(self, handle: SessionHandle, req: TrainOpRequest) -> TrainOpResult:
        assert handle.backend_session_id is not None
        session = self.sessions[handle.backend_session_id]
        if req.op in {TrainOp.FORWARD_BACKWARD, TrainOp.TRAIN_STEP, TrainOp.CUSTOM}:
            session.step += 1
        elif req.op is TrainOp.OPTIMIZER_STEP:
            session.step += 10
        return TrainOpResult(
            state=TrainState(step=session.step, extras=req.options),
            outputs={"op": req.op.value},
        )

    def forward_backward_ppo(
        self,
        handle: SessionHandle,
        req: TrainOpRequest,
    ) -> TrainOpResult:
        assert handle.backend_session_id is not None
        session = self.sessions[handle.backend_session_id]
        session.step += 1
        return TrainOpResult(
            state=TrainState(step=session.step, extras={"mode": "ppo"}),
            outputs={"op": req.op.value, "algorithm": "ppo"},
        )

    def forward_backward_reverse_kl(
        self,
        handle: SessionHandle,
        req: TrainOpRequest,
    ) -> TrainOpResult:
        assert handle.backend_session_id is not None
        session = self.sessions[handle.backend_session_id]
        session.step += 1
        return TrainOpResult(
            state=TrainState(step=session.step, extras={"mode": "reverse_kl"}),
            outputs={"op": req.op.value},
        )

    def reset_expert_bias(self, handle: SessionHandle) -> TrainOpResult:
        return TrainOpResult(state=self._state(handle), outputs={"reset": True})

    def save_checkpoint(self, handle: SessionHandle, req: CheckpointRequest) -> ArtifactRef:
        path = self._checkpoint_path(req.uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("checkpoint", encoding="utf-8")
        return ArtifactRef(uri=req.uri, format="pt", metadata={"optimizer": req.include_optimizer})

    def load_checkpoint(self, handle: SessionHandle, req: CheckpointRequest) -> TrainState:
        assert handle.backend_session_id is not None
        self.sessions[handle.backend_session_id].step = 7
        return TrainState(step=7)

    def _state(self, handle: SessionHandle) -> TrainState:
        assert handle.backend_session_id is not None
        session = self.sessions[handle.backend_session_id]
        return TrainState(step=session.step)

    def _checkpoint_path(self, uri: str) -> Path:
        parsed = urlsplit(uri)
        if not parsed.scheme or parsed.scheme == "file":
            return Path(parsed.path if parsed.scheme == "file" else uri)
        rel = Path(parsed.netloc, parsed.path.lstrip("/"))
        return self.checkpoint_root / parsed.scheme / rel


@pytest.fixture
def router() -> BackendRouter:
    router = BackendRouter()
    router.register_backend(
        BackendSpec(
            backend_id="vllm-main",
            kind=BackendKind.INFERENCE,
            provider="vllm",
            model_family="qwen",
        ),
        FakeInferenceBackend(),
    )
    router.register_backend(
        BackendSpec(
            backend_id="megatron-main",
            kind=BackendKind.TRAINING,
            provider="megatron",
            model_family="llama",
            config={"tp": 2, "nested": {"pp": 1}},
        ),
        FakeTrainingBackend(),
    )
    return router


def test_training_routes_match_mint_shape(router: BackendRouter) -> None:
    view = router.open_training_session(
        SessionSpec(session_id="m1", backend_id="megatron-main", batch_codec="torch")
    )
    assert view.session_id == "m1"
    assert view.batch_codec == "torch"

    caps = router.get_capabilities("m1")
    assert caps.supports_reverse_kl is True

    tok = router.get_tokenizer_info("m1")
    assert tok.metadata["eos"] == 2

    forward = router.forward(
        "m1",
        TrainOpRequest(op=TrainOp.FORWARD, batch_codec="torch", batch_payload={"x": 1}),
    )
    assert forward.outputs["mode"] == "forward"

    fb = router.run_train_op(
        "m1",
        TrainOpRequest(op=TrainOp.FORWARD_BACKWARD, batch_codec="torch", batch_payload={}),
    )
    assert fb.state.step == 1

    rev = router.forward_backward_reverse_kl(
        "m1",
        TrainOpRequest(
            op=TrainOp.FORWARD_BACKWARD_REVERSE_KL,
            batch_codec="torch",
            batch_payload={},
        ),
    )
    assert rev.state.step == 2

    opt = router.run_train_op(
        "m1",
        TrainOpRequest(op=TrainOp.OPTIMIZER_STEP, batch_codec="torch", batch_payload={}),
    )
    assert opt.state.step == 12

    reset = router.reset_expert_bias("m1")
    assert reset.outputs["reset"] is True

    ckpt = router.save_checkpoint("m1", CheckpointRequest(uri="ckpt://step12"))
    assert ckpt.format == "pt"

    state = router.load_checkpoint("m1", CheckpointRequest(uri="ckpt://step7"))
    assert state.step == 7


def test_fake_backend_does_not_materialize_uri_scheme_in_repo(tmp_path) -> None:
    backend = FakeTrainingBackend(checkpoint_root=tmp_path)
    handle = backend.open_session(
        SessionSpec(session_id="m1", backend_id="megatron-main", batch_codec="torch")
    )

    artifact = backend.save_checkpoint(handle, CheckpointRequest(uri="ckpt://step12"))

    assert artifact.uri == "ckpt://step12"
    assert not Path("ckpt:").exists()
    assert (tmp_path / "ckpt" / "step12").read_text(encoding="utf-8") == "checkpoint"


def test_rollout_train_op_maps_grpo_alias(router: BackendRouter) -> None:
    assert router._rollout_train_op({"train_op": "grpo"}) is TrainOp.FORWARD_BACKWARD_REVERSE_KL


def test_rollout_collect_defaults_to_custom_algorithm(tmp_path) -> None:
    service = MintService(storage=LocalStorageRepo(tmp_path))
    service.backends.register(
        BackendSpec(
            backend_id="vllm-main",
            kind=BackendKind.INFERENCE,
            provider="vllm",
            model_family="qwen",
        ),
        FakeInferenceBackend(),
    )
    service.backends.register(
        BackendSpec(
            backend_id="megatron-main",
            kind=BackendKind.TRAINING,
            provider="megatron",
            model_family="llama",
        ),
        FakeTrainingBackend(),
    )
    service.training.create_model("model-a", "megatron-main", "torch")
    service.rollouts.open_session("r-default", "vllm-main", "model-a")

    exp = service.rollouts.collect_experience(
        "r-default",
        prompt="1 2",
        batch_codec="torch",
        reward=1.0,
    )
    assert exp.batch_payload["algorithm"] == "custom"

    trained = service.rollouts.train_on_experience("r-default", batch_payload=exp.batch_payload)
    assert trained.outputs["op"] == "custom"


def test_grpo_rollout_session_snapshots_reference_checkpoint(tmp_path) -> None:
    service = MintService(storage=LocalStorageRepo(tmp_path))
    service.backends.register(
        BackendSpec(
            backend_id="vllm-main",
            kind=BackendKind.INFERENCE,
            provider="vllm",
            model_family="qwen",
        ),
        FakeInferenceBackend(),
    )
    service.backends.register(
        BackendSpec(
            backend_id="megatron-main",
            kind=BackendKind.TRAINING,
            provider="megatron",
            model_family="llama",
        ),
        FakeTrainingBackend(),
    )
    service.training.create_model("model-a", "megatron-main", "torch")
    service.rollouts.open_session(
        "r-grpo",
        "vllm-main",
        "model-a",
        metadata={"algo": "grpo"},
    )

    exp = service.rollouts.collect_experience(
        "r-grpo",
        prompt="1 2",
        batch_codec="torch",
        reward=1.0,
    )
    assert exp.batch_payload["algorithm"] == "grpo"
    assert exp.batch_payload["reference_model_path"] == "mint://rollouts/model-a/r-grpo/reference.pt"
    assert (tmp_path / "rollouts" / "model-a" / "r-grpo" / "reference.pt").exists()

    trained = service.rollouts.train_on_experience("r-grpo", batch_payload=exp.batch_payload)
    assert trained.outputs["op"] == "forward_backward_reverse_kl"


def test_verl_reverse_kl_preserves_portable_reference_uri(tmp_path, monkeypatch) -> None:
    class RecordingTrainingBackend(FakeTrainingBackend):
        def __init__(self) -> None:
            super().__init__()
            self.last_reverse_kl_req: TrainOpRequest | None = None

        def forward_backward_reverse_kl(
            self,
            handle: SessionHandle,
            req: TrainOpRequest,
        ) -> TrainOpResult:
            self.last_reverse_kl_req = req
            return super().forward_backward_reverse_kl(handle, req)

    monkeypatch.setenv(shared_storage_env_name(), "1")
    backend = RecordingTrainingBackend()
    shared_root = tmp_path / "shared"
    service = MintService(storage=LocalStorageRepo(shared_root))
    service.backends.register(
        BackendSpec(
            backend_id="train-main",
            kind=BackendKind.TRAINING,
            provider="verl",
            model_family="llama",
            config={
                "requires_shared_checkpoint_io": True,
                "requires_portable_reference_uri": True,
            },
        ),
        backend,
    )
    service.training.create_model("model-ray", "train-main", "torch")

    service.training.forward_backward_reverse_kl(
        "model-ray",
        batch_payload={
            "reference_model_path": "mint://refs/ref.pt",
            "data": [{"student_input": {}, "reference_input": {}, "target_tokens": {"data": [1]}, "weights": {"data": [1.0]}}],
        },
    )
    assert backend.last_reverse_kl_req is not None
    assert backend.last_reverse_kl_req.batch_payload["reference_model_path"] == "mint://refs/ref.pt"


def test_verl_reverse_kl_rejects_nonshared_absolute_reference_path(tmp_path) -> None:
    service = MintService(storage=LocalStorageRepo(tmp_path))
    service.backends.register(
        BackendSpec(
            backend_id="train-main",
            kind=BackendKind.TRAINING,
            provider="verl",
            model_family="llama",
            config={
                "requires_shared_checkpoint_io": True,
                "requires_portable_reference_uri": True,
            },
        ),
        FakeTrainingBackend(),
    )
    service.training.create_model("model-ray", "train-main", "torch")

    with pytest.raises(ValueError, match="shared checkpoint path"):
        service.training.forward_backward_reverse_kl(
            "model-ray",
            batch_payload={
                "reference_model_path": str(tmp_path / "refs" / "ref.pt"),
                "data": [{"student_input": {}, "reference_input": {}, "target_tokens": {"data": [1]}, "weights": {"data": [1.0]}}],
            },
        )


def test_verl_rollout_reference_path_rejects_nonshared_absolute_override(tmp_path, monkeypatch) -> None:
    shared_root = tmp_path / "shared"
    monkeypatch.setenv(shared_storage_roots_env_name(), str(shared_root))
    service = MintService(storage=LocalStorageRepo(shared_root))
    service.backends.register(
        BackendSpec(
            backend_id="vllm-main",
            kind=BackendKind.INFERENCE,
            provider="vllm",
            model_family="qwen",
        ),
        FakeInferenceBackend(),
    )
    service.backends.register(
        BackendSpec(
            backend_id="train-main",
            kind=BackendKind.TRAINING,
            provider="verl",
            model_family="llama",
            config={
                "requires_shared_checkpoint_io": True,
                "requires_portable_reference_uri": True,
            },
        ),
        FakeTrainingBackend(),
    )
    service.training.create_model("model-ray", "train-main", "torch")
    service.rollouts.open_session("roll-ray", "vllm-main", "model-ray")

    with pytest.raises(ValueError, match="shared checkpoint path"):
        service.rollouts.collect_experience(
            "roll-ray",
            prompt="1 2",
            batch_codec="torch",
            metadata={"reference_model_path": str(tmp_path / "not-shared" / "ref.pt")},
        )


def test_mint_service_does_not_leave_shared_flag_sticky(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv(shared_storage_env_name(), raising=False)
    shared_root = tmp_path / "shared"
    monkeypatch.setenv(shared_storage_roots_env_name(), str(shared_root))
    MintService(storage=LocalStorageRepo(shared_root))
    monkeypatch.delenv(shared_storage_roots_env_name(), raising=False)

    local_root = tmp_path / "local"
    service = MintService(storage=LocalStorageRepo(local_root))
    service.backends.register(
        BackendSpec(
            backend_id="train-main",
            kind=BackendKind.TRAINING,
            provider="verl",
            model_family="llama",
            config={
                "requires_shared_checkpoint_io": True,
                "requires_portable_reference_uri": True,
            },
        ),
        FakeTrainingBackend(),
    )
    service.training.create_model("model-ray", "train-main", "torch")

    with pytest.raises(ValueError, match="shared VERL_MINT_STORAGE_ROOT"):
        service.training.save_weights("model-ray", "mint://ckpts/model-ray.pt")


def test_group_advantages_preserves_singleton_reward_signal() -> None:
    from verl_mint.backends.qwen_sft import _group_advantages

    assert _group_advantages([2.5], ["g1"]) == [2.5]
    assert _group_advantages([1.0, 1.0], ["g2", "g2"]) == [0.0, 0.0]


def test_qwen_ppo_allows_missing_old_logprobs() -> None:
    torch = pytest.importorskip("torch")

    from verl_mint.backends.qwen_sft import QwenSFTTrainingBackend, _PolicyTrace

    backend = object.__new__(QwenSFTTrainingBackend)
    backend.sessions = {}
    backend._reference_models = {}
    session = type("Session", (), {})()
    session.device = torch.device("cpu")
    session.step = 0
    session.last_loss = 0.0
    backend.sessions["ppo-session"] = session

    def fake_completion_trace(_self, _session, prompt_tokens, completion_tokens, *, temperature=1.0):
        n = len(completion_tokens)
        values = torch.zeros(n, dtype=torch.float32, requires_grad=True)
        return _PolicyTrace(
            logprobs=torch.full((n,), -0.2, dtype=torch.float32, requires_grad=True),
            entropies=torch.zeros(n, dtype=torch.float32),
            values=values,
        )

    backend._completion_trace = fake_completion_trace.__get__(backend, QwenSFTTrainingBackend)
    req = TrainOpRequest(
        op=TrainOp.FORWARD_BACKWARD_PPO,
        batch_codec="torch",
        batch_payload={
            "samples": [
                {
                    "prompt_tokens": [1],
                    "completion_tokens": [2, 3],
                    "weights": [1.0, 1.0],
                    "advantages": [1.0, 1.0],
                    "returns": [1.0, 1.0],
                    "reward": 1.0,
                    "group_id": "g1",
                }
            ]
        },
    )

    result = QwenSFTTrainingBackend.forward_backward_ppo(
        backend,
        SessionHandle(backend_session_id="ppo-session"),
        req,
    )
    assert result.state.step == 1
    assert "approx_kl" in result.outputs


def _qwen_dpo_payload(*, include_reference: bool = True) -> dict:
    def row(pair_id: str, role: str, completion: list[int], ref: list[float]) -> dict:
        loss_inputs = {
            "pair_id": pair_id,
            "role": role,
            "prompt_tokens": {"data": [1], "shape": [1], "dtype": "int64"},
            "completion_tokens": {"data": completion, "shape": [len(completion)], "dtype": "int64"},
            "target_tokens": {"data": completion, "shape": [len(completion)], "dtype": "int64"},
            "weights": {"data": [1.0 for _ in completion], "shape": [len(completion)], "dtype": "float32"},
        }
        if include_reference:
            loss_inputs["reference_logprobs"] = {"data": ref, "shape": [len(ref)], "dtype": "float32"}
        return {
            "model_input": {"chunks": [{"type": "encoded_text", "tokens": [1, *completion]}]},
            "loss_fn_inputs": loss_inputs,
        }

    return {
        "loss_fn": "dpo",
        "loss_fn_config": {"beta": 0.1},
        "data": [
            row("p0", "chosen", [2], [0.0]),
            row("p0", "rejected", [3], [0.0]),
        ],
    }


def _qwen_dpo_backend():
    torch = pytest.importorskip("torch")

    from verl_mint.backends.qwen_sft import QwenSFTTrainingBackend, _PolicyTrace

    backend = object.__new__(QwenSFTTrainingBackend)
    backend.sessions = {}
    backend._reference_models = {}
    session = type("Session", (), {})()
    session.device = torch.device("cpu")
    session.step = 0
    session.last_loss = 0.0
    session.pref = torch.nn.Parameter(torch.tensor(1.0))
    session.optimizer = torch.optim.SGD([session.pref], lr=0.1)
    backend.sessions["dpo-session"] = session

    def fake_completion_trace(_self, sess, prompt_tokens, completion_tokens, *, temperature=1.0):
        scale = torch.tensor([float(tok) for tok in completion_tokens], dtype=torch.float32)
        logprobs = sess.pref * scale
        return _PolicyTrace(
            logprobs=logprobs,
            entropies=torch.zeros(len(completion_tokens), dtype=torch.float32),
            values=torch.zeros(len(completion_tokens), dtype=torch.float32),
        )

    backend._completion_trace = fake_completion_trace.__get__(backend, QwenSFTTrainingBackend)
    return backend, session


def test_qwen_dpo_forward_returns_metrics() -> None:
    from verl_mint.backends.qwen_sft import QwenSFTTrainingBackend

    backend, _session = _qwen_dpo_backend()
    result = QwenSFTTrainingBackend.forward(
        backend,
        SessionHandle(backend_session_id="dpo-session"),
        TrainOpRequest(
            op=TrainOp.FORWARD,
            batch_codec="torch",
            batch_payload=_qwen_dpo_payload(),
        ),
    )

    assert result.outputs["algorithm"] == "dpo"
    assert result.outputs["num_pairs"] == 1
    assert "pi_margin" in result.outputs
    assert "ref_margin" in result.outputs
    assert "loss_fn_outputs" not in result.outputs


def test_qwen_dpo_train_step_updates_parameter() -> None:
    from verl_mint.backends.qwen_sft import QwenSFTTrainingBackend

    backend, session = _qwen_dpo_backend()
    before = float(session.pref.detach().item())
    result = QwenSFTTrainingBackend.run_train_op(
        backend,
        SessionHandle(backend_session_id="dpo-session"),
        TrainOpRequest(
            op=TrainOp.TRAIN_STEP,
            batch_codec="torch",
            batch_payload=_qwen_dpo_payload(),
        ),
    )

    assert result.state.step == 1
    assert result.outputs["op"] == "train_step"
    assert float(session.pref.detach().item()) != before


def test_qwen_dpo_requires_reference_source() -> None:
    from verl_mint.backends.qwen_sft import QwenSFTTrainingBackend

    backend, _session = _qwen_dpo_backend()
    with pytest.raises(ValueError, match="reference_logprobs or reference_model_path"):
        QwenSFTTrainingBackend.forward(
            backend,
            SessionHandle(backend_session_id="dpo-session"),
            TrainOpRequest(
                op=TrainOp.FORWARD,
                batch_codec="torch",
                batch_payload=_qwen_dpo_payload(include_reference=False),
            ),
        )


def test_qwen_dpo_rejects_incomplete_pair() -> None:
    from verl_mint.backends.qwen_sft import QwenSFTTrainingBackend

    backend, _session = _qwen_dpo_backend()
    payload = _qwen_dpo_payload()
    payload["data"] = payload["data"][:1]
    with pytest.raises(ValueError, match="exactly one chosen and one rejected"):
        QwenSFTTrainingBackend.forward(
            backend,
            SessionHandle(backend_session_id="dpo-session"),
            TrainOpRequest(op=TrainOp.FORWARD, batch_codec="torch", batch_payload=payload),
        )


def test_qwen_dpo_rejects_duplicate_role() -> None:
    from verl_mint.backends.qwen_sft import QwenSFTTrainingBackend

    backend, _session = _qwen_dpo_backend()
    payload = _qwen_dpo_payload()
    payload["data"][1]["loss_fn_inputs"]["role"] = "chosen"
    with pytest.raises(ValueError, match="duplicate chosen"):
        QwenSFTTrainingBackend.forward(
            backend,
            SessionHandle(backend_session_id="dpo-session"),
            TrainOpRequest(op=TrainOp.FORWARD, batch_codec="torch", batch_payload=payload),
        )


def test_qwen_close_session_clears_reference_model_cache() -> None:
    from verl_mint.backends.qwen_sft import QwenSFTTrainingBackend

    backend = object.__new__(QwenSFTTrainingBackend)
    backend.sessions = {"sess": object()}
    backend._reference_models = {"mint://rollouts/sess/ref.pt": object()}

    QwenSFTTrainingBackend.close_session(backend, SessionHandle(backend_session_id="sess"))
    assert backend._reference_models == {}


def test_mint_service_supports_sft_and_rl_entrypoints(tmp_path) -> None:
    service = MintService(storage=LocalStorageRepo(tmp_path))
    service.backends.register(
        BackendSpec(
            backend_id="vllm-main",
            kind=BackendKind.INFERENCE,
            provider="vllm",
            model_family="qwen",
        ),
        FakeInferenceBackend(),
    )
    service.backends.register(
        BackendSpec(
            backend_id="megatron-main",
            kind=BackendKind.TRAINING,
            provider="megatron",
            model_family="llama",
        ),
        FakeTrainingBackend(),
    )

    model = service.training.create_model("model-a", "megatron-main", "torch")
    assert model.status == "active"

    info = service.training.get_info("model-a")
    assert info["capabilities"].supports_forward is True
    assert info["tokenizer"].metadata["bos"] == 1

    sft = service.training.train_step("model-a", batch_payload={"loss_mask": [1]})
    assert sft.state.step == 1

    rl = service.training.forward_backward_reverse_kl("model-a", batch_payload={"pair": 1})
    assert rl.state.step == 2

    ppo = service.training.forward_backward_ppo("model-a", batch_payload={"pair": 2})
    assert ppo.state.step == 3

    rollout = service.rollouts.open_session("r1", "vllm-main", "model-a", metadata={"algo": "ppo"})
    assert rollout.training_session_id == "model-a"

    exp = service.rollouts.collect_experience(
        "r1",
        prompt="1 2",
        batch_codec="torch",
        reward=1.0,
        metadata={"source": "unit", "group_id": "g1"},
        num_samples=2,
    )
    assert exp.batch_payload["response"] == "1 2"
    assert len(exp.batch_payload["samples"]) == 2
    assert exp.batch_payload["samples"][0]["completion_tokens"] == (1, 2)
    assert exp.batch_payload["samples"][0]["group_id"] == "g1"
    assert exp.metadata["source"] == "unit"

    trained = service.rollouts.train_on_experience(
        "r1",
        batch_payload=exp.batch_payload,
        options={"train_op": "forward_backward_ppo", "algo": "ppo"},
    )
    assert trained.state.step == 4
    assert trained.outputs["op"] == "forward_backward_ppo"
    assert exp.batch_payload["algorithm"] == "ppo"
    assert exp.batch_payload["samples"][0]["old_values"] == (0.0, 0.0)

    grpo_trained = service.rollouts.train_on_experience(
        "r1",
        batch_payload=exp.batch_payload,
        options={"train_op": "grpo"},
    )
    assert grpo_trained.state.step == 5
    assert grpo_trained.outputs["op"] == "forward_backward_reverse_kl"


def test_router_rejects_kind_mismatch(router: BackendRouter) -> None:
    with pytest.raises(BackendKindMismatchError):
        router.open_training_session(
            SessionSpec(session_id="s2", backend_id="vllm-main", batch_codec="torch")
        )

    with pytest.raises(BackendKindMismatchError):
        router.generate("megatron-main", GenerateRequest(prompt="hello"))


def test_router_rejects_duplicate_ids_and_codec_mismatch(router: BackendRouter) -> None:
    with pytest.raises(ValueError, match="duplicate backend_id"):
        router.register_backend(
            BackendSpec(
                backend_id="vllm-main",
                kind="inference",
                provider="vllm",
                model_family="qwen",
            ),
            FakeInferenceBackend(),
        )

    router.open_training_session(
        SessionSpec(session_id="dup", backend_id="megatron-main", batch_codec="torch")
    )

    with pytest.raises(ValueError, match="duplicate session_id"):
        router.open_training_session(
            SessionSpec(session_id="dup", backend_id="megatron-main", batch_codec="torch")
        )

    with pytest.raises(ValueError, match="batch_codec mismatch"):
        router.run_train_op(
            "dup",
            TrainOpRequest(op=TrainOp.FORWARD_BACKWARD, batch_codec="numpy", batch_payload={}),
        )


def test_specs_and_payloads_are_frozen(router: BackendRouter) -> None:
    specs = router.list_backends()
    backend = next(item for item in specs if item.backend_id == "megatron-main")
    assert isinstance(backend.config, MappingProxyType)
    with pytest.raises(TypeError):
        backend.config["tp"] = 8

    payload = {"items": ["x"]}
    req = TrainOpRequest(op=TrainOp.TRAIN_STEP, batch_codec="torch", batch_payload=payload)
    payload["items"].append("y")
    assert req.batch_payload["items"] == ("x",)


def test_training_service_uses_local_storage_repo(tmp_path) -> None:
    storage = LocalStorageRepo(tmp_path)
    service = MintService(storage=storage)
    service.backends.register(
        BackendSpec(
            backend_id="megatron-store",
            kind=BackendKind.TRAINING,
            provider="megatron",
            model_family="llama",
        ),
        FakeTrainingBackend(),
    )
    service.training.create_model("model-store", "megatron-store", "torch")

    saved = service.training.save_weights("model-store", "mint://ckpts/model-store.pt")
    assert saved.uri == "mint://ckpts/model-store.pt"
    assert (tmp_path / "ckpts" / "model-store.pt").exists()

    loaded = service.training.load_weights("model-store", "mint://ckpts/model-store.pt")
    assert loaded.step == 7


def test_schema_skeleton_matches_mint_shape() -> None:
    req = CreateModelRequest(
        session_id="sess",
        model_seq_id=1,
        base_model="llama",
        backend_id="megatron-main",
    )
    assert req.type == "create_model"

    fb = ForwardBackwardRequest(
        model_id="sess:1",
        forward_backward_input=ForwardBackwardInput(
            data=[
                {
                    "model_input": {"chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]},
                    "loss_fn_inputs": {"labels": [1, 2, 3]},
                }
            ],
            loss_fn="sft",
        ),
    )
    assert isinstance(fb.forward_backward_input.data[0].model_input.chunks[0], EncodedTextChunk)
    assert isinstance(fb.forward_backward_input.data[0].model_input, ModelInput)


def test_create_app_uses_storage_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(storage_env_name(), str(tmp_path))
    app = create_app()
    assert app.title == "verl-mint"
    assert app.state.mint_service.storage.root == tmp_path.resolve()


def test_create_app_defaults_storage_root(monkeypatch) -> None:
    monkeypatch.delenv(storage_env_name(), raising=False)
    app = create_app()
    assert app.state.mint_service.storage.root == LocalStorageRepo.from_env({}).root
    assert str(app.state.mint_service.storage.root) == default_storage_root()


def test_api_v1_and_legacy_routes_are_paired(tmp_path) -> None:
    service = MintService(storage=LocalStorageRepo(tmp_path))
    app = create_app(service)

    pairs: set[tuple[str, str]] = set()
    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        path = route.path
        if path.startswith("/_artifacts/") or path.startswith("/vla/"):
            continue
        if path in {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}:
            continue
        for method in route.methods:
            if method in {"GET", "POST", "DELETE"}:
                pairs.add((method, path))

    for method, path in sorted(pairs):
        if path.startswith("/api/v1/"):
            legacy_path = "/" + path[len("/api/v1/") :]
            assert (method, legacy_path) in pairs, f"missing legacy alias for {method} {path}"
        else:
            v1_path = f"/api/v1{path}"
            assert (method, v1_path) in pairs, f"missing v1 alias for {method} {path}"


def test_fastapi_returns_mint_like_response_skeletons(tmp_path) -> None:
    service = MintService(storage=LocalStorageRepo(tmp_path))
    service.backends.register(
        BackendSpec(
            backend_id="megatron-http",
            kind=BackendKind.TRAINING,
            provider="megatron",
            model_family="llama",
        ),
        FakeTrainingBackend(),
    )
    service.backends.register(
        BackendSpec(
            backend_id="vllm-http",
            kind=BackendKind.INFERENCE,
            provider="vllm",
            model_family="llama",
        ),
        FakeInferenceBackend(),
    )
    app = create_app(service)
    client = TestClient(app)

    healthz_v1 = client.get("/api/v1/healthz")
    assert healthz_v1.status_code == 200
    assert healthz_v1.json()["status"] == "ok"

    healthz_legacy = client.get("/healthz")
    assert healthz_legacy.status_code == 200
    assert healthz_legacy.json()["status"] == "ok"

    caps_v1 = client.get("/api/v1/get_server_capabilities")
    assert caps_v1.status_code == 200
    assert caps_v1.json()["status"] == "ready"
    assert any(item["model_name"] == "llama" for item in caps_v1.json()["supported_models"])

    caps_legacy = client.get("/get_server_capabilities")
    assert caps_legacy.status_code == 200
    assert caps_legacy.json()["status"] == "ready"
    assert any(item["model_name"] == "llama" for item in caps_legacy.json()["supported_models"])

    server_info_v1 = client.get("/api/v1/server_info")
    assert server_info_v1.status_code == 200
    assert server_info_v1.json()["server"] == "verl-mint"
    assert "storage_root" not in server_info_v1.json()

    server_info_legacy = client.get("/server_info")
    assert server_info_legacy.status_code == 200
    assert server_info_legacy.json()["server"] == "verl-mint"
    assert "storage_root" not in server_info_legacy.json()

    create_resp = client.post(
        "/create_model",
        json={
            "session_id": "sess",
            "model_seq_id": 1,
            "base_model": "llama",
            "backend_id": "megatron-http",
            "batch_codec": "torch",
            "lora_config": {"rank": 8},
        },
    )
    assert create_resp.status_code == 200
    create_future = create_resp.json()
    assert "request_id" in create_future
    create_payload = client.post("/retrieve_future", json=create_future).json()
    assert create_payload["type"] == "create_model"
    assert create_payload["model_id"] == "sess:1"

    forward_resp = client.post(
        "/forward",
        json={
            "model_id": "sess:1",
            "forward_input": {
                "data": [
                    {
                        "model_input": {"chunks": [{"type": "encoded_text", "tokens": [1, 2]}]},
                        "loss_fn_inputs": {
                            "target_tokens": {"data": [2, 3], "shape": [2], "dtype": "int64"},
                            "loss_mask": {"data": [1.0, 1.0], "shape": [2], "dtype": "float32"},
                        },
                    }
                ],
                "loss_fn": "sft",
            },
        },
    )
    assert forward_resp.status_code == 200
    forward_payload = client.post("/retrieve_future", json=forward_resp.json()).json()
    assert forward_payload["type"] == "forward"
    assert forward_payload["output"]["loss_fn_outputs"][0]["logprobs"]["dtype"] == "float32"
    assert forward_payload["output"]["loss_fn_outputs"][0]["logprobs"]["shape"] == [2]

    fb_resp = client.post(
        "/forward_backward",
        json={
            "model_id": "sess:1",
            "forward_backward_input": {
                "data": [
                    {
                        "model_input": {"chunks": [{"type": "encoded_text", "tokens": [1, 2]}]},
                        "loss_fn_inputs": {"labels": [1, 2]},
                    }
                ],
                "loss_fn": "sft",
            },
        },
    )
    assert fb_resp.status_code == 200
    body = client.post("/retrieve_future", json=fb_resp.json()).json()
    assert body["type"] == "forward_backward"
    assert body["output"]["loss_fn_output_type"] == "sft_loss"
    assert body["output"]["metrics"]["num_samples:sum"] == 1.0
    assert body["output"]["loss_fn_outputs"][0]["loss"]["dtype"] == "float32"

    forward_v1_resp = client.post(
        "/api/v1/forward",
        json={
            "model_id": "sess:1",
            "forward_input": {
                "data": [
                    {
                        "model_input": {"chunks": [{"type": "encoded_text", "tokens": [1, 2]}]},
                        "loss_fn_inputs": {
                            "target_tokens": {"data": [2, 3], "shape": [2], "dtype": "int64"},
                            "loss_mask": {"data": [1.0, 1.0], "shape": [2], "dtype": "float32"},
                        },
                    }
                ],
                "loss_fn": "sft",
            },
        },
    )
    assert forward_v1_resp.status_code == 200
    forward_v1_payload = client.post("/api/v1/retrieve_future", json=forward_v1_resp.json()).json()
    assert forward_v1_payload["loss_fn_output_type"] == "sft_loss"
    assert forward_v1_payload["metrics"]["num_tokens:sum"] == 2.0

    fb_v1_resp = client.post(
        "/api/v1/forward_backward",
        json={
            "model_id": "sess:1",
            "forward_backward_input": {
                "data": [
                    {
                        "model_input": {"chunks": [{"type": "encoded_text", "tokens": [1, 2]}]},
                        "loss_fn_inputs": {"labels": [1, 2]},
                    }
                ],
                "loss_fn": "sft",
            },
        },
    )
    assert fb_v1_resp.status_code == 200
    fb_v1_payload = client.post("/api/v1/retrieve_future", json=fb_v1_resp.json()).json()
    assert fb_v1_payload["loss_fn_output_type"] == "sft_loss"
    assert fb_v1_payload["metrics"]["num_samples:sum"] == 1.0

    optim_v1_resp = client.post(
        "/api/v1/optim_step",
        json={"model_id": "sess:1", "adam_params": {"learning_rate": 0.0003}},
    )
    assert optim_v1_resp.status_code == 200
    optim_v1_payload = client.post("/api/v1/retrieve_future", json=optim_v1_resp.json()).json()
    assert optim_v1_payload["type"] == "optim_step"
    assert optim_v1_payload["metrics"]["step"] >= 0.0

    train_step_resp = client.post(
        "/train_step",
        json={
            "model_id": "sess:1",
            "forward_backward_input": {
                "data": [
                    {
                        "model_input": {"chunks": [{"type": "encoded_text", "tokens": [1, 2]}]},
                        "loss_fn_inputs": {"labels": [1, 2]},
                    }
                ],
                "loss_fn": "sft",
            },
            "adam_params": {"learning_rate": 0.0003},
        },
    )
    assert train_step_resp.status_code == 200
    train_body = client.post("/retrieve_future", json=train_step_resp.json()).json()
    assert train_body["type"] == "train_step"
    assert train_body["output"]["loss_fn_output_type"] == "sft_loss"
    assert train_body["metrics"]["learning_rate"] == 0.0003

    train_step_v1_resp = client.post(
        "/api/v1/train_step",
        json={
            "model_id": "sess:1",
            "forward_backward_input": {
                "data": [
                    {
                        "model_input": {"chunks": [{"type": "encoded_text", "tokens": [1, 2]}]},
                        "loss_fn_inputs": {"labels": [1, 2]},
                    }
                ],
                "loss_fn": "sft",
            },
            "adam_params": {"learning_rate": 0.0003},
        },
    )
    assert train_step_v1_resp.status_code == 200
    train_v1_body = client.post("/api/v1/retrieve_future", json=train_step_v1_resp.json()).json()
    assert train_v1_body["type"] == "train_step"
    assert train_v1_body["metrics"]["learning_rate"] == 0.0003

    ppo_v1_resp = client.post(
        "/api/v1/forward_backward_ppo",
        json={
            "model_id": "sess:1",
            "data": [
                {
                    "student_input": {"chunks": [{"type": "encoded_text", "tokens": [1, 2]}]},
                    "target_tokens": {"data": [1, 2], "shape": [2], "dtype": "int64"},
                    "weights": {"data": [1.0, 1.0], "shape": [2], "dtype": "float32"},
                    "old_logprobs": {"data": [-0.1, -0.1], "shape": [2], "dtype": "float32"},
                    "old_values": {"data": [0.0, 0.0], "shape": [2], "dtype": "float32"},
                    "advantages": {"data": [1.0, 1.0], "shape": [2], "dtype": "float32"},
                    "returns": {"data": [1.0, 1.0], "shape": [2], "dtype": "float32"},
                    "prompt_tokens": [1],
                    "completion_tokens": [2, 3],
                }
            ],
            "clip_coef": 0.2,
            "value_coef": 0.5,
            "entropy_coef": 0.01,
            "value_clip": 0.2,
        },
    )
    assert ppo_v1_resp.status_code == 200
    ppo_v1_payload = client.post(
        "/api/v1/retrieve_future", json=ppo_v1_resp.json()
    ).json()
    assert ppo_v1_payload["type"] == "mint_forward_backward_ppo"

    (tmp_path / "checkpoints").mkdir(parents=True, exist_ok=True)
    (tmp_path / "checkpoints" / "ref.pt").write_text("ref", encoding="utf-8")

    reverse_kl_v1_resp = client.post(
        "/api/v1/forward_backward_reverse_kl",
        json={
            "model_id": "sess:1",
            "reference_model_path": "mint://checkpoints/ref.pt",
            "data": [
                {
                    "student_input": {"chunks": [{"type": "encoded_text", "tokens": [1, 2]}]},
                    "reference_input": {"chunks": [{"type": "encoded_text", "tokens": [1, 2]}]},
                    "target_tokens": {"data": [1, 2], "shape": [2], "dtype": "int64"},
                    "weights": {"data": [1.0, 1.0], "shape": [2], "dtype": "float32"},
                }
            ],
            "temperature": 1.0,
        },
    )
    assert reverse_kl_v1_resp.status_code == 200
    reverse_kl_v1_payload = client.post(
        "/api/v1/retrieve_future", json=reverse_kl_v1_resp.json()
    ).json()
    assert reverse_kl_v1_payload["type"] == "mint_forward_backward_reverse_kl"

    info_resp = client.post("/get_info", json={"model_id": "sess:1"})
    assert info_resp.status_code == 200
    assert info_resp.json()["type"] == "get_info"
    assert info_resp.json()["model_data"]["model_name"] == "llama"
    assert info_resp.json()["is_lora"] is True
    assert info_resp.json()["lora_rank"] == 8

    info_v1_resp = client.post("/api/v1/get_info", json={"model_id": "sess:1"})
    assert info_v1_resp.status_code == 200
    info_v1_payload = info_v1_resp.json()
    assert info_v1_payload["type"] == "get_info"
    assert info_v1_payload["model_data"]["model_name"] == "llama"

    save_resp = client.post(
        "/save_weights",
        json={"model_id": "sess:1", "path": "mint://checkpoints/sess-1.pt"},
    )
    assert save_resp.status_code == 200
    save_payload = client.post("/retrieve_future", json=save_resp.json()).json()
    assert save_payload["type"] == "save_weights"
    assert save_payload["path"] == "mint://checkpoints/sess-1.pt"

    weights_info_resp = client.post("/api/v1/weights_info", json={"mint_path": save_payload["path"]})
    assert weights_info_resp.status_code == 200
    weights_info_payload = weights_info_resp.json()
    assert weights_info_payload["base_model"] == "llama"
    assert weights_info_payload["is_lora"] is True
    assert weights_info_payload["lora_rank"] == 8

    weights_info_legacy = client.post("/weights_info", json={"mint_path": save_payload["path"]})
    assert weights_info_legacy.status_code == 200

    create_from_state_resp = client.post(
        "/create_model_from_state",
        json={
            "session_id": "sess",
            "model_seq_id": 2,
            "base_model": "llama",
            "backend_id": "megatron-http",
            "batch_codec": "torch",
            "state_path": "mint://checkpoints/sess-1.pt",
            "load_optimizer": True,
        },
    )
    assert create_from_state_resp.status_code == 200
    create_from_state_payload = client.post(
        "/retrieve_future", json=create_from_state_resp.json()
    ).json()
    assert create_from_state_payload["type"] == "create_model_from_state"
    assert create_from_state_payload["model_id"] == "sess:2"

    create_from_state_v1_resp = client.post(
        "/api/v1/create_model_from_state",
        json={
            "session_id": "sess",
            "model_seq_id": 4,
            "base_model": "llama",
            "backend_id": "megatron-http",
            "batch_codec": "torch",
            "state_path": "mint://checkpoints/sess-1.pt",
            "load_optimizer": True,
        },
    )
    assert create_from_state_v1_resp.status_code == 200
    create_from_state_v1_payload = client.post(
        "/api/v1/retrieve_future", json=create_from_state_v1_resp.json()
    ).json()
    assert create_from_state_v1_payload["type"] == "create_model_from_state"
    assert create_from_state_v1_payload["model_id"] == "sess:4"

    load_resp = client.post(
        "/load_weights",
        json={"model_id": "sess:1", "path": "mint://checkpoints/sess-1.pt", "optimizer": True},
    )
    assert load_resp.status_code == 200
    load_payload = client.post("/retrieve_future", json=load_resp.json()).json()
    assert load_payload["type"] == "load_weights"
    assert load_payload["path"] == "mint://checkpoints/sess-1.pt"

    session_resp = client.post("/create_session", json={})
    assert session_resp.status_code == 200
    session_id = session_resp.json()["session_id"]

    sampling_session_resp = client.post(
        "/create_sampling_session",
        json={"session_id": session_id, "sampling_session_seq_id": 1, "base_model": "llama"},
    )
    assert sampling_session_resp.status_code == 200
    sampler_id = sampling_session_resp.json()["sampling_session_id"]

    session_detail = client.get(f"/sessions/{session_id}")
    assert session_detail.status_code == 200
    assert sampler_id in session_detail.json()["sampler_ids"]

    session_detail_v1 = client.get(f"/api/v1/sessions/{session_id}")
    assert session_detail_v1.status_code == 200
    assert sampler_id in session_detail_v1.json()["sampler_ids"]

    sessions_v1 = client.get("/api/v1/sessions")
    assert sessions_v1.status_code == 200
    assert session_id in sessions_v1.json()["sessions"]

    sampler_detail = client.get(f"/samplers/{sampler_id}")
    assert sampler_detail.status_code == 200
    assert sampler_detail.json()["base_model"] == "llama"

    sampler_detail_v1 = client.get(f"/api/v1/samplers/{sampler_id}")
    assert sampler_detail_v1.status_code == 200
    assert sampler_detail_v1.json()["base_model"] == "llama"

    heartbeat = client.post("/session_heartbeat", json={"session_id": session_id})
    assert heartbeat.status_code == 200
    assert heartbeat.json()["type"] == "session_heartbeat"

    telemetry = client.post("/telemetry", json={})
    assert telemetry.status_code == 200
    assert telemetry.json()["status"] == "accepted"

    telemetry_v1 = client.post("/api/v1/telemetry", json={})
    assert telemetry_v1.status_code == 200
    assert telemetry_v1.json()["status"] == "accepted"

    action_session_resp = client.post(
        "/api/v1/action_sessions",
        json={"session_id": session_id, "action_session_seq_id": 1, "base_model": "llama"},
    )
    assert action_session_resp.status_code == 200
    action_session_id = action_session_resp.json()["action_session_id"]

    act_resp = client.post(
        f"/api/v1/action_sessions/{action_session_id}/act",
        json={
            "action_session_id": action_session_id,
            "observation": {"chunks": [{"type": "encoded_text", "tokens": [1, 2]}]},
        },
    )
    assert act_resp.status_code == 200
    act_payload = client.post("/api/v1/retrieve_future", json=act_resp.json()).json()
    assert act_payload["type"] == "act"

    delete_action_resp = client.delete(f"/api/v1/action_sessions/{action_session_id}")
    assert delete_action_resp.status_code == 200

    legacy_action_session_resp = client.post(
        "/action_sessions",
        json={"session_id": session_id, "action_session_seq_id": 2, "base_model": "llama"},
    )
    assert legacy_action_session_resp.status_code == 200
    legacy_action_session_id = legacy_action_session_resp.json()["action_session_id"]

    legacy_act_resp = client.post(
        f"/action_sessions/{legacy_action_session_id}/act",
        json={
            "action_session_id": legacy_action_session_id,
            "observation": {"chunks": [{"type": "encoded_text", "tokens": [3, 4]}]},
        },
    )
    assert legacy_act_resp.status_code == 200
    legacy_act_payload = client.post("/retrieve_future", json=legacy_act_resp.json()).json()
    assert legacy_act_payload["type"] == "act"

    legacy_delete_action_resp = client.delete(f"/action_sessions/{legacy_action_session_id}")
    assert legacy_delete_action_resp.status_code == 200

    sample_resp = client.post(
        "/asample",
        json={
            "sampling_session_id": sampler_id,
            "num_samples": 1,
            "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1, 2]}]},
            "sampling_params": {"max_tokens": 4},
        },
    )
    assert sample_resp.status_code == 200
    sample_payload = client.post("/retrieve_future", json=sample_resp.json()).json()
    assert sample_payload["type"] == "sample"
    assert sample_payload["sequences"][0]["tokens"] == [1, 2]

    logprob_resp = client.post(
        "/compute_logprobs",
        json={
            "sampling_session_id": sampler_id,
            "seq_id": 1,
            "sequence": {"chunks": [{"type": "encoded_text", "tokens": [1, 2]}]},
        },
    )
    assert logprob_resp.status_code == 200
    logprob_payload = client.post("/retrieve_future", json=logprob_resp.json()).json()
    assert logprob_payload["type"] == "compute_logprobs"
    assert logprob_payload["logprobs"] == [None, -2.0]

    logprob_v1_resp = client.post(
        "/api/v1/compute_logprobs",
        json={
            "sampling_session_id": sampler_id,
            "seq_id": 2,
            "sequence": {"chunks": [{"type": "encoded_text", "tokens": [1, 2]}]},
        },
    )
    assert logprob_v1_resp.status_code == 200
    logprob_v1_payload = client.post("/api/v1/retrieve_future", json=logprob_v1_resp.json()).json()
    assert logprob_v1_payload["type"] == "compute_logprobs"
    assert logprob_v1_payload["logprobs"] == [None, -2.0]

    sampler_save_resp = client.post(
        "/save_weights_for_sampler",
        json={"model_id": "sess:1", "path": "mint://sampler/sess-1.pt"},
    )
    assert sampler_save_resp.status_code == 200
    sampler_save_payload = client.post("/retrieve_future", json=sampler_save_resp.json()).json()
    assert sampler_save_payload["type"] == "save_weights_for_sampler"
    sampler_from_save = client.get(f"/samplers/{sampler_save_payload['sampling_session_id']}")
    assert sampler_from_save.status_code == 200
    sampler_from_save_v1 = client.get(f"/api/v1/samplers/{sampler_save_payload['sampling_session_id']}")
    assert sampler_from_save_v1.status_code == 200

    sampler_save_v1_resp = client.post(
        "/api/v1/save_weights_for_sampler",
        json={"model_id": "sess:1", "path": "sampler-v1-sess-1.pt"},
    )
    assert sampler_save_v1_resp.status_code == 200
    sampler_save_v1_payload = client.post("/api/v1/retrieve_future", json=sampler_save_v1_resp.json()).json()
    assert sampler_save_v1_payload["type"] == "save_weights_for_sampler"
    assert sampler_save_v1_payload["path"] == "mint://sess:1/sampler_weights/sampler-v1-sess-1.pt"
    assert "sampling_session_id" not in sampler_save_v1_payload or sampler_save_v1_payload["sampling_session_id"] is None

    sampler_checkpoint_id = sampler_save_v1_payload["path"].split("/")[-1]
    sampler_archive = client.get(
        f"/api/v1/training_runs/sess:1/checkpoints/{sampler_checkpoint_id}/archive",
        follow_redirects=False,
    )
    assert sampler_archive.status_code == 302
    sampler_archive_file = client.get(sampler_archive.headers["location"])
    assert sampler_archive_file.status_code == 200

    sampler_handoff_resp = client.post(
        "/api/v1/save_weights_for_sampler",
        json={"model_id": "sess:1", "sampling_session_seq_id": 9},
    )
    assert sampler_handoff_resp.status_code == 200
    sampler_handoff_payload = client.post("/api/v1/retrieve_future", json=sampler_handoff_resp.json()).json()
    assert sampler_handoff_payload["type"] == "save_weights_for_sampler"
    assert sampler_handoff_payload["path"] is None
    sampler_handoff_id = sampler_handoff_payload["sampling_session_id"]
    assert sampler_handoff_id == "sess:sampler:9"
    sampler_handoff_detail = client.get(f"/samplers/{sampler_handoff_id}")
    assert sampler_handoff_detail.status_code == 200
    handoff_model_path = sampler_handoff_detail.json()["model_path"]
    assert handoff_model_path == "mint://sess:1/sampler_weights/sampler-9.pt"
    handoff_checkpoint_id = handoff_model_path.split("/")[-1]
    handoff_archive = client.get(
        f"/api/v1/training_runs/sess:1/checkpoints/{handoff_checkpoint_id}/archive",
        follow_redirects=False,
    )
    assert handoff_archive.status_code == 302
    handoff_archive_file = client.get(handoff_archive.headers["location"])
    assert handoff_archive_file.status_code == 200

    runs_resp = client.get("/training_runs")
    assert runs_resp.status_code == 200
    assert runs_resp.json()["training_runs"][0]["base_model"] == "llama"

    checkpoints_resp = client.get("/training_runs/sess:1/checkpoints")
    assert checkpoints_resp.status_code == 200
    assert len(checkpoints_resp.json()["checkpoints"]) >= 2

    checkpoint_id = checkpoints_resp.json()["checkpoints"][0]["checkpoint_id"]
    checkpoint_detail = client.get(f"/training_runs/sess:1/checkpoints/{checkpoint_id}")
    assert checkpoint_detail.status_code == 200

    archive_resp = client.get(f"/training_runs/sess:1/checkpoints/{checkpoint_id}/archive")
    assert archive_resp.status_code == 200

    upload_resp = client.post("/checkpoints/upload")
    assert upload_resp.status_code == 200
    assert upload_resp.json()["checkpoint_id"] == "uploaded-checkpoint"

    upload_v1_resp = client.post("/api/v1/checkpoints/upload")
    assert upload_v1_resp.status_code == 200
    assert upload_v1_resp.json()["checkpoint_id"] == "uploaded-checkpoint"

    interpolate_resp = client.post(
        "/checkpoints/interpolate",
        json={"source_paths": ["mint://a", "mint://b"], "coefficients": [0.5, 0.5]},
    )
    assert interpolate_resp.status_code == 200
    interpolate_payload = client.post("/retrieve_future", json=interpolate_resp.json()).json()
    assert interpolate_payload["type"] == "mint_interpolate_checkpoints"

    interpolate_v1_resp = client.post(
        "/api/v1/checkpoints/interpolate",
        json={"source_paths": ["mint://a", "mint://b"], "coefficients": [0.5, 0.5]},
    )
    assert interpolate_v1_resp.status_code == 200
    interpolate_v1_payload = client.post("/api/v1/retrieve_future", json=interpolate_v1_resp.json()).json()
    assert interpolate_v1_payload["type"] == "mint_interpolate_checkpoints"

    models_resp = client.get("/models")
    assert models_resp.status_code == 200
    assert models_resp.json()["models"][0]["base_model"] == "llama"

    models_v1_resp = client.get("/api/v1/models")
    assert models_v1_resp.status_code == 200

    model_v1_resp = client.get("/api/v1/models/sess:1")
    assert model_v1_resp.status_code == 200

    caps_v1_resp = client.get("/api/v1/models/sess:1/capabilities")
    assert caps_v1_resp.status_code == 200

    tokenizer_v1_resp = client.get("/api/v1/models/sess:1/tokenizer")
    assert tokenizer_v1_resp.status_code == 200

    reset_v1_resp = client.post("/api/v1/reset_expert_bias", json={"model_id": "sess:1"})
    assert reset_v1_resp.status_code == 200
    assert reset_v1_resp.json()["model_id"] == "sess:1"

    rollout_v1_open = client.post(
        "/api/v1/rollout_sessions",
        json={
            "rollout_session_id": "roll-1",
            "inference_backend_id": "vllm-http",
            "training_session_id": "sess:1",
        },
    )
    assert rollout_v1_open.status_code == 200

    rollout_v1_collect = client.post(
        "/api/v1/rollout_sessions/roll-1/collect",
        json={"prompt": "1 2", "batch_codec": "torch", "num_samples": 2, "metadata": {"group_id": "roll-1:g"}},
    )
    assert rollout_v1_collect.status_code == 200
    rollout_v1_collect_payload = rollout_v1_collect.json()
    assert rollout_v1_collect_payload["rollout_session_id"] == "roll-1"
    assert rollout_v1_collect_payload["batch_payload"]["response"] == "1 2"
    assert len(rollout_v1_collect_payload["batch_payload"]["samples"]) == 2
    assert rollout_v1_collect_payload["batch_payload"]["samples"][0]["old_logprobs"] == [-0.1, -0.1]

    rollout_legacy_collect = client.post(
        "/rollout_sessions/roll-1/collect",
        json={"prompt": "3 4", "batch_codec": "torch", "num_samples": 1},
    )
    assert rollout_legacy_collect.status_code == 200
    rollout_legacy_collect_payload = rollout_legacy_collect.json()
    assert rollout_legacy_collect_payload["batch_payload"]["response"] == "3 4"
    assert rollout_legacy_collect_payload["batch_payload"]["samples"][0]["completion_tokens"] == [3, 4]

    rollout_legacy_train = client.post(
        "/rollout_sessions/roll-1/train",
        json={"batch_payload": rollout_v1_collect_payload["batch_payload"], "extension_op": "rl", "options": {"train_op": "forward_backward_ppo", "algo": "ppo"}},
    )
    assert rollout_legacy_train.status_code == 200
    rollout_legacy_train_payload = rollout_legacy_train.json()
    assert "state" in rollout_legacy_train_payload
    assert "outputs" in rollout_legacy_train_payload

    rollout_v1_train = client.post(
        "/api/v1/rollout_sessions/roll-1/train",
        json={"batch_payload": rollout_v1_collect_payload["batch_payload"], "extension_op": "rl", "options": {"train_op": "forward_backward_ppo", "algo": "ppo"}},
    )
    assert rollout_v1_train.status_code == 200
    rollout_v1_train_payload = rollout_v1_train.json()
    assert "state" in rollout_v1_train_payload
    assert "outputs" in rollout_v1_train_payload

    rollout_v1_close = client.delete("/api/v1/rollout_sessions/roll-1")
    assert rollout_v1_close.status_code == 200

    create_v1_model_resp = client.post(
        "/api/v1/create_model",
        json={
            "session_id": "sess",
            "model_seq_id": 3,
            "base_model": "llama",
            "backend_id": "megatron-http",
            "batch_codec": "torch",
        },
    )
    assert create_v1_model_resp.status_code == 200
    create_v1_model_payload = client.post(
        "/api/v1/retrieve_future", json=create_v1_model_resp.json()
    ).json()
    assert create_v1_model_payload["type"] == "create_model"
    assert create_v1_model_payload["model_id"] == "sess:3"

    delete_v1_model_resp = client.delete("/api/v1/models/sess:3")
    assert delete_v1_model_resp.status_code == 200
    assert delete_v1_model_resp.json()["ok"] is True

    after_delete_models = client.get("/models").json()["models"]
    assert "sess:3" not in {item["model_id"] for item in after_delete_models}

    unload_v1_resp = client.post("/api/v1/unload_model", json={"model_id": "sess:1"})
    assert unload_v1_resp.status_code == 200
    unload_v1_payload = client.post("/api/v1/retrieve_future", json=unload_v1_resp.json()).json()
    assert unload_v1_payload["type"] == "unload_model"

    unload_legacy_resp = client.post("/unload_model", json={"model_id": "sess:2"})
    assert unload_legacy_resp.status_code == 200
    unload_legacy_payload = client.post("/retrieve_future", json=unload_legacy_resp.json()).json()
    assert unload_legacy_payload["type"] == "unload_model"

    remaining_models = client.get("/models").json()["models"]
    remaining_ids = {item["model_id"] for item in remaining_models}
    assert "sess:1" not in remaining_ids
    assert "sess:2" not in remaining_ids


def test_versioned_save_uses_mint_paths_and_weights_info_aliases(tmp_path) -> None:
    service = MintService(storage=LocalStorageRepo(tmp_path))
    service.backends.register(
        BackendSpec(
            backend_id="megatron-http",
            kind=BackendKind.TRAINING,
            provider="megatron",
            model_family="llama",
        ),
        FakeTrainingBackend(),
    )
    app = create_app(service)
    client = TestClient(app)

    client.post(
        "/api/v1/create_model",
        json={
            "session_id": "sess",
            "model_seq_id": 1,
            "base_model": "llama",
            "backend_id": "megatron-http",
            "batch_codec": "torch",
            "lora_config": {"rank": 8},
        },
    )
    save_resp = client.post("/api/v1/save_weights", json={"model_id": "sess:1", "path": "checkpoint-001"})
    save_payload = client.post("/api/v1/retrieve_future", json=save_resp.json()).json()
    assert save_payload["path"] == "mint://sess:1/weights/checkpoint-001"

    weights_info_mint = client.post("/api/v1/weights_info", json={"mint_path": "mint://sess:1/weights/checkpoint-001"})
    assert weights_info_mint.status_code == 200
    weights_info_legacy = client.post("/weights_info", json={"mint_path": "mint://sess:1/weights/checkpoint-001"})
    assert weights_info_legacy.status_code == 200

    training_runs_resp = client.get("/api/v1/training_runs")
    assert training_runs_resp.status_code == 200
    training_runs_payload = training_runs_resp.json()
    assert training_runs_payload["object"] == "list"
    assert training_runs_payload["data"][0]["id"] == "sess:1"
    assert training_runs_payload["data"][0]["model"] == "llama"

    training_run_resp = client.get("/api/v1/training_runs/sess:1")
    assert training_run_resp.status_code == 200
    training_run_payload = training_run_resp.json()
    assert training_run_payload["id"] == "sess:1"
    assert training_run_payload["model"] == "llama"
    assert training_run_payload["last_checkpoint"]["path"] == "mint://sess:1/weights/checkpoint-001"

    checkpoints_resp = client.get("/api/v1/training_runs/sess:1/checkpoints")
    assert checkpoints_resp.status_code == 200
    checkpoints_payload = checkpoints_resp.json()
    assert checkpoints_payload["object"] == "list"
    assert checkpoints_payload["data"][0]["path"] == "mint://sess:1/weights/checkpoint-001"

    all_checkpoints_resp = client.get("/api/v1/checkpoints")
    assert all_checkpoints_resp.status_code == 200
    assert all_checkpoints_resp.json()["data"][0]["path"] == "mint://sess:1/weights/checkpoint-001"

    all_checkpoints_legacy_resp = client.get("/checkpoints")
    assert all_checkpoints_legacy_resp.status_code == 200
    assert all_checkpoints_legacy_resp.json()["checkpoints"][0]["path"] == "mint://sess:1/weights/checkpoint-001"

    paged_v1 = client.get("/api/v1/checkpoints?offset=0&limit=1")
    assert paged_v1.status_code == 200
    paged_v1_payload = paged_v1.json()
    assert paged_v1_payload["cursor"]["limit"] == 1
    assert len(paged_v1_payload["checkpoints"]) <= 1

    paged_legacy = client.get("/checkpoints?offset=0&limit=1")
    assert paged_legacy.status_code == 200
    paged_legacy_payload = paged_legacy.json()
    assert paged_legacy_payload["cursor"]["limit"] == 1
    assert len(paged_legacy_payload["checkpoints"]) <= 1

    checkpoint_id = checkpoints_payload["checkpoints"][0]["checkpoint_id"]
    checkpoint_detail_resp = client.get(f"/api/v1/training_runs/sess:1/checkpoints/{checkpoint_id}")
    assert checkpoint_detail_resp.status_code == 200
    checkpoint_detail_payload = checkpoint_detail_resp.json()
    assert checkpoint_detail_payload["checkpoint_id"] == checkpoint_id
    assert checkpoint_detail_payload["path"] == "mint://sess:1/weights/checkpoint-001"

    archive_resp = client.get(
        f"/api/v1/training_runs/sess:1/checkpoints/{checkpoint_id}/archive",
        follow_redirects=False,
    )
    assert archive_resp.status_code == 302
    assert archive_resp.headers["location"].endswith(f"/_artifacts/sess:1/checkpoints/{checkpoint_id}")
    archive_file_resp = client.get(archive_resp.headers["location"])
    assert archive_file_resp.status_code == 200

    delete_resp = client.delete(f"/api/v1/training_runs/sess:1/checkpoints/{checkpoint_id}")
    assert delete_resp.status_code == 204
    deleted_weights_info = client.post("/api/v1/weights_info", json={"mint_path": save_payload["path"]})
    assert deleted_weights_info.status_code == 404


def test_retrieve_future_supports_pending_try_again_and_metadata_only(tmp_path) -> None:
    service = MintService(storage=LocalStorageRepo(tmp_path))
    app = create_app(service)
    client = TestClient(app)

    pending_id = app.state.future_store.create_resolved(
        {"type": "save_weights", "path": "mint://checkpoints/pending.pt"},
        pending_polls=1,
        queue_state_reason="warming",
    )
    pending_resp = client.post("/api/v1/retrieve_future", json={"request_id": pending_id})
    assert pending_resp.status_code == 408
    assert pending_resp.json()["queue_state"] == "active"
    assert pending_resp.json()["queue_state_reason"] == "warming"
    pending_done = client.post("/api/v1/retrieve_future", json={"request_id": pending_id}).json()
    assert pending_done["path"] == "mint://checkpoints/pending.pt"
    pending_gone = client.post("/api/v1/retrieve_future", json={"request_id": pending_id})
    assert pending_gone.status_code == 404

    retry_id = app.state.future_store.create_resolved(
        {"type": "save_weights", "path": "mint://checkpoints/retry.pt"},
        allow_try_again_once=True,
    )
    retry_first = client.post("/api/v1/retrieve_future", json={"request_id": retry_id}).json()
    assert retry_first["type"] == "try_again"
    retry_second = client.post("/api/v1/retrieve_future", json={"request_id": retry_id}).json()
    assert retry_second["path"] == "mint://checkpoints/retry.pt"
    retry_gone = client.post("/api/v1/retrieve_future", json={"request_id": retry_id})
    assert retry_gone.status_code == 404

    metadata_id = app.state.future_store.create_resolved(
        {"type": "save_weights", "path": "mint://checkpoints/meta.pt", "payload": "x" * 64},
        metadata_only_threshold=1,
    )
    metadata_first = client.post(
        "/api/v1/retrieve_future", json={"request_id": metadata_id, "allow_metadata_only": True}
    ).json()
    assert metadata_first["status"] == "complete_metadata"
    assert metadata_first["response_payload_size"] > 0
    metadata_second = client.post(
        "/api/v1/retrieve_future", json={"request_id": metadata_id, "allow_metadata_only": False}
    ).json()
    assert metadata_second["path"] == "mint://checkpoints/meta.pt"
    metadata_gone = client.post(
        "/api/v1/retrieve_future", json={"request_id": metadata_id, "allow_metadata_only": False}
    )
    assert metadata_gone.status_code == 404
    assert len(app.state.future_store._items) == 0


def test_save_weights_for_sampler_succeeds_without_inference_backend(tmp_path) -> None:
    service = MintService(storage=LocalStorageRepo(tmp_path))
    service.backends.register(
        BackendSpec(
            backend_id="train-main",
            kind=BackendKind.TRAINING,
            provider="megatron",
            model_family="llama",
        ),
        FakeTrainingBackend(),
    )
    app = create_app(service)
    client = TestClient(app)

    create_resp = client.post(
        "/create_model",
        json={
            "session_id": "sess",
            "model_seq_id": 1,
            "base_model": "llama",
            "backend_id": "train-main",
            "batch_codec": "torch",
        },
    )
    assert create_resp.status_code == 200

    save_resp = client.post(
        "/save_weights_for_sampler",
        json={"model_id": "sess:1", "path": "mint://sampler/sess-1.pt"},
    )
    assert save_resp.status_code == 200
    save_payload = client.post("/retrieve_future", json=save_resp.json()).json()
    assert save_payload["path"] == "mint://sampler/sess-1.pt"


def test_sampling_endpoints_require_existing_sampler(tmp_path) -> None:
    service = MintService(storage=LocalStorageRepo(tmp_path))
    service.backends.register(
        BackendSpec(
            backend_id="vllm-main",
            kind=BackendKind.INFERENCE,
            provider="vllm",
            model_family="qwen",
        ),
        FakeInferenceBackend(),
    )
    app = create_app(service)
    client = TestClient(app)

    sample_resp = client.post(
        "/asample",
        json={
            "sampling_session_id": "missing-sampler",
            "num_samples": 1,
            "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1, 2]}]},
            "sampling_params": {"max_tokens": 4},
        },
    )
    assert sample_resp.status_code == 404

    logprob_resp = client.post(
        "/compute_logprobs",
        json={
            "sampling_session_id": "missing-sampler",
            "seq_id": 1,
            "sequence": {"chunks": [{"type": "encoded_text", "tokens": [1, 2]}]},
        },
    )
    assert logprob_resp.status_code == 404


def test_sampling_endpoints_require_existing_sampler_artifact(tmp_path) -> None:
    service = MintService(storage=LocalStorageRepo(tmp_path))
    service.backends.register(
        BackendSpec(
            backend_id="vllm-main",
            kind=BackendKind.INFERENCE,
            provider="vllm",
            model_family="qwen",
        ),
        FakeInferenceBackend(),
    )
    app = create_app(service)
    app.state.sampling_sessions["sess:sampler:1"] = {
        "session_id": "sess",
        "base_model": "llama",
        "model_path": "mint://sess/weights/missing.pt",
        "lora_rank": 32,
    }
    client = TestClient(app)

    sample_resp = client.post(
        "/asample",
        json={
            "sampling_session_id": "sess:sampler:1",
            "num_samples": 1,
            "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1, 2]}]},
            "sampling_params": {"max_tokens": 4},
        },
    )
    assert sample_resp.status_code == 404

    logprob_resp = client.post(
        "/compute_logprobs",
        json={
            "sampling_session_id": "sess:sampler:1",
            "seq_id": 1,
            "sequence": {"chunks": [{"type": "encoded_text", "tokens": [1, 2]}]},
        },
    )
    assert logprob_resp.status_code == 404


def test_asample_fallback_uses_prompt_length_and_clamps_tokens(tmp_path) -> None:
    class SparseInferenceBackend(InferenceBackend):
        def generate(self, req: GenerateRequest) -> GenerateResult:
            return GenerateResult(
                text="9 8 7 6",
                token_ids=(9, 8, 7, 6),
                token_logprobs=(-0.2, -0.2, -0.2, -0.2),
                prompt_logprobs=(),
                raw={"provider": "sparse"},
            )

    service = MintService(storage=LocalStorageRepo(tmp_path))
    service.backends.register(
        BackendSpec(
            backend_id="sparse-inf",
            kind=BackendKind.INFERENCE,
            provider="vllm",
            model_family="llama",
        ),
        SparseInferenceBackend(),
    )
    service.backends.register(
        BackendSpec(
            backend_id="train-main",
            kind=BackendKind.TRAINING,
            provider="megatron",
            model_family="llama",
        ),
        FakeTrainingBackend(),
    )
    app = create_app(service)
    client = TestClient(app)

    client.post(
        "/create_model",
        json={
            "session_id": "sess",
            "model_seq_id": 1,
            "base_model": "llama",
            "backend_id": "train-main",
            "batch_codec": "torch",
        },
    )
    sampler_save = client.post(
        "/save_weights_for_sampler",
        json={"model_id": "sess:1", "path": "mint://sampler/sess-1.pt"},
    )
    sampler_payload = client.post("/retrieve_future", json=sampler_save.json()).json()

    sample_resp = client.post(
        "/asample",
        json={
            "sampling_session_id": sampler_payload["sampling_session_id"],
            "num_samples": 1,
            "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]},
            "sampling_params": {"max_tokens": 2},
            "prompt_logprobs": True,
        },
    )
    assert sample_resp.status_code == 200
    sample_payload = client.post("/retrieve_future", json=sample_resp.json()).json()
    assert sample_payload["sequences"][0]["tokens"] == [9, 8]
    assert sample_payload["prompt_logprobs"] == [None, -0.1, -0.1]


def test_verl_requires_shared_storage_for_checkpoint_io(tmp_path) -> None:
    service = MintService(storage=LocalStorageRepo(tmp_path))
    service.backends.register(
        BackendSpec(
            backend_id="train-main",
            kind=BackendKind.TRAINING,
            provider="verl",
            model_family="llama",
            config={
                "requires_shared_checkpoint_io": True,
                "requires_portable_reference_uri": True,
            },
        ),
        FakeTrainingBackend(),
    )
    service.training.create_model("model-ray", "train-main", "torch")

    with pytest.raises(ValueError, match="shared VERL_MINT_STORAGE_ROOT"):
        service.training.save_weights("model-ray", "mint://ckpts/model-ray.pt")


def test_sampler_routes_to_matching_inference_backend(tmp_path) -> None:
    class ConstantInferenceBackend(InferenceBackend):
        def __init__(self, token: int) -> None:
            self.token = token

        def generate(self, req: GenerateRequest) -> GenerateResult:
            return GenerateResult(
                text=str(self.token),
                token_ids=(self.token,),
                token_logprobs=(-0.5,),
                raw={"provider": "constant"},
            )

    service = MintService(storage=LocalStorageRepo(tmp_path))
    service.backends.register(
        BackendSpec(
            backend_id="llama-inf",
            kind=BackendKind.INFERENCE,
            provider="vllm",
            model_family="llama",
        ),
        ConstantInferenceBackend(11),
    )
    service.backends.register(
        BackendSpec(
            backend_id="qwen-inf",
            kind=BackendKind.INFERENCE,
            provider="vllm",
            model_family="qwen",
        ),
        ConstantInferenceBackend(22),
    )
    service.backends.register(
        BackendSpec(
            backend_id="train-main",
            kind=BackendKind.TRAINING,
            provider="megatron",
            model_family="qwen",
        ),
        FakeTrainingBackend(),
    )

    app = create_app(service)
    client = TestClient(app)
    client.post(
        "/create_model",
        json={
            "session_id": "sess",
            "model_seq_id": 1,
            "base_model": "Qwen/Qwen3-0.6B",
            "backend_id": "train-main",
            "batch_codec": "torch",
        },
    )

    sampler_save = client.post(
        "/save_weights_for_sampler",
        json={"model_id": "sess:1", "path": "mint://sampler/sess-1.pt"},
    )
    sampler_payload = client.post("/retrieve_future", json=sampler_save.json()).json()

    sample_resp = client.post(
        "/asample",
        json={
            "sampling_session_id": sampler_payload["sampling_session_id"],
            "num_samples": 1,
            "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1, 2]}]},
            "sampling_params": {"max_tokens": 4},
        },
    )
    sample_payload = client.post("/retrieve_future", json=sample_resp.json()).json()
    assert sample_payload["sequences"][0]["tokens"] == [22]


def test_create_model_from_state_rolls_back_on_load_failure(tmp_path) -> None:
    service = MintService(storage=LocalStorageRepo(tmp_path))
    service.backends.register(
        BackendSpec(
            backend_id="megatron-main",
            kind=BackendKind.TRAINING,
            provider="megatron",
            model_family="llama",
        ),
        FakeTrainingBackend(),
    )

    with pytest.raises(FileNotFoundError):
        service.training.create_model_from_state(
            model_id="sess:1",
            backend_id="megatron-main",
            batch_codec="torch",
            uri="mint://checkpoints/missing.pt",
            include_optimizer=True,
        )

    retry = service.training.create_model(
        model_id="sess:1",
        backend_id="megatron-main",
        batch_codec="torch",
    )
    assert retry.session_id == "sess:1"


def test_create_model_from_state_rolls_back_even_if_close_fails(tmp_path) -> None:
    class FailingRollbackBackend(FakeTrainingBackend):
        def __init__(self) -> None:
            super().__init__()
            self.fail_close_once = True

        def close_session(self, handle: SessionHandle) -> None:
            if self.fail_close_once:
                self.fail_close_once = False
                raise RuntimeError("close failed during rollback")
            super().close_session(handle)

    service = MintService(storage=LocalStorageRepo(tmp_path))
    service.backends.register(
        BackendSpec(
            backend_id="megatron-rollback",
            kind=BackendKind.TRAINING,
            provider="megatron",
            model_family="llama",
        ),
        FailingRollbackBackend(),
    )

    with pytest.raises(RuntimeError, match="rollback failed while closing session 'sess:9'"):
        service.training.create_model_from_state(
            model_id="sess:9",
            backend_id="megatron-rollback",
            batch_codec="torch",
            uri="mint://checkpoints/missing.pt",
            include_optimizer=True,
        )

    assert service.training.get_model("sess:9").status == "close_failed"
    service.training.delete_model("sess:9")
    retry = service.training.create_model(
        model_id="sess:9",
        backend_id="megatron-rollback",
        batch_codec="torch",
    )
    assert retry.session_id == "sess:9"


def test_terminal_close_blocks_future_ops(tmp_path) -> None:
    class ClosingErrorBackend(FakeTrainingBackend):
        def close_session(self, handle: SessionHandle) -> None:
            super().close_session(handle)
            raise RuntimeError("close failed")

    service = MintService(storage=LocalStorageRepo(tmp_path))
    service.backends.register(
        BackendSpec(
            backend_id="megatron-close",
            kind=BackendKind.TRAINING,
            provider="megatron",
            model_family="llama",
        ),
        ClosingErrorBackend(),
    )
    service.training.create_model("model-close", "megatron-close", "torch")

    with pytest.raises(RuntimeError, match="close failed"):
        service.training.delete_model("model-close")

    assert service.training.get_model("model-close").status == "close_failed"
    with pytest.raises(UnsupportedOperationError, match="not active"):
        service.training.train_step("model-close", batch_payload={})


def test_create_model_routes_qwen30b_moe_to_megatron_backend(tmp_path) -> None:
    model = QWEN30B_MODEL_ID
    service = MintService(storage=LocalStorageRepo(tmp_path))
    service.backends.register(
        BackendSpec(
            backend_id="qwen-local",
            kind=BackendKind.TRAINING,
            provider="transformers",
            model_family="qwen",
        ),
        FakeTrainingBackend(),
    )
    service.backends.register(
        BackendSpec(
            backend_id="qwen-megatron",
            kind=BackendKind.TRAINING,
            provider="megatron",
            model_family="qwen",
        ),
        FakeTrainingBackend(),
    )
    app = create_app(service)
    client = TestClient(app)

    create_resp = client.post(
        "/create_model",
        json={
            "session_id": "qwen30b",
            "model_seq_id": 1,
            "base_model": model,
            "batch_codec": "verl",
        },
    )
    assert create_resp.status_code == 200
    payload = client.post("/retrieve_future", json=create_resp.json()).json()
    assert payload["backend"] == "qwen-megatron"
    assert service.training.get_model("qwen30b:1").backend_id == "qwen-megatron"


def test_create_model_rejects_qwen30b_without_megatron_backend(tmp_path) -> None:
    model = QWEN30B_MODEL_ID
    service = MintService(storage=LocalStorageRepo(tmp_path))
    service.backends.register(
        BackendSpec(
            backend_id="qwen-local",
            kind=BackendKind.TRAINING,
            provider="transformers",
            model_family="qwen",
        ),
        FakeTrainingBackend(),
    )
    app = create_app(service)
    client = TestClient(app)

    create_resp = client.post(
        "/create_model",
        json={
            "session_id": "qwen30b",
            "model_seq_id": 1,
            "base_model": model,
            "batch_codec": "torch",
        },
    )
    assert create_resp.status_code == 400
    assert "requires a Megatron training backend" in create_resp.json()["detail"]
