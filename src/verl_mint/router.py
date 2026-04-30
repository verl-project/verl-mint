from __future__ import annotations

from dataclasses import dataclass, replace

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
    TrainOpRequest,
    TrainOpResult,
    TrainOp,
    TrainState,
    TrainingCapabilities,
    TrainingSessionView,
)
from verl_mint.errors import (
    BackendKindMismatchError,
    UnknownBackendError,
    UnknownSessionError,
    UnsupportedOperationError,
)


@dataclass(frozen=True)
class RegisteredBackend:
    spec: BackendSpec
    backend: Backend


@dataclass(frozen=True)
class TrainingSessionRecord:
    session_id: str
    backend_id: str
    batch_codec: str
    metadata: dict[str, object]
    backend: TrainingBackend
    handle: SessionHandle
    state: TrainState | None = None
    status: str = "active"
    last_error: str | None = None


@dataclass(frozen=True)
class RolloutSessionRecord:
    rollout_session_id: str
    inference_backend_id: str
    training_session_id: str
    metadata: dict[str, object]
    status: str = "active"
    last_error: str | None = None


class BackendRegistry:
    def __init__(self) -> None:
        self._items: dict[str, RegisteredBackend] = {}

    def register(self, spec: BackendSpec, backend: Backend) -> None:
        if spec.backend_id in self._items:
            raise ValueError(f"duplicate backend_id: {spec.backend_id}")
        if spec.kind == BackendKind.INFERENCE and not isinstance(backend, InferenceBackend):
            raise BackendKindMismatchError("inference spec requires InferenceBackend")
        if spec.kind == BackendKind.TRAINING and not isinstance(backend, TrainingBackend):
            raise BackendKindMismatchError("training spec requires TrainingBackend")
        self._items[spec.backend_id] = RegisteredBackend(spec=spec, backend=backend)

    def list_specs(self) -> list[BackendSpec]:
        return [item.spec for item in self._items.values()]

    def get(self, backend_id: str) -> RegisteredBackend:
        item = self._items.get(backend_id)
        if item is None:
            raise UnknownBackendError(f"unknown backend_id: {backend_id}")
        return item


class BackendRouter:
    def __init__(self, registry: BackendRegistry | None = None) -> None:
        self._registry = registry or BackendRegistry()
        self._sessions: dict[str, TrainingSessionRecord] = {}
        self._rollouts: dict[str, RolloutSessionRecord] = {}

    def register_backend(self, spec: BackendSpec, backend: Backend) -> None:
        self._registry.register(spec, backend)

    def list_backends(self) -> list[BackendSpec]:
        return self._registry.list_specs()

    def get_backend_spec(self, backend_id: str) -> BackendSpec:
        return self._registry.get(backend_id).spec

    def generate(self, backend_id: str, req: GenerateRequest) -> GenerateResult:
        item = self._registry.get(backend_id)
        if item.spec.kind != BackendKind.INFERENCE:
            raise BackendKindMismatchError(f"backend {backend_id} is not inference")
        backend = item.backend
        assert isinstance(backend, InferenceBackend)
        return backend.generate(req)

    def open_training_session(self, spec: SessionSpec) -> TrainingSessionView:
        if spec.session_id in self._sessions:
            raise ValueError(f"duplicate session_id: {spec.session_id}")

        item = self._registry.get(spec.backend_id)
        if item.spec.kind != BackendKind.TRAINING:
            raise BackendKindMismatchError(f"backend {spec.backend_id} is not training")
        backend = item.backend
        assert isinstance(backend, TrainingBackend)
        handle = backend.open_session(spec)
        self._sessions[spec.session_id] = TrainingSessionRecord(
            session_id=spec.session_id,
            backend_id=spec.backend_id,
            batch_codec=spec.batch_codec,
            metadata=dict(spec.metadata),
            backend=backend,
            handle=handle,
        )
        return self.get_session_view(spec.session_id)

    def close_training_session(self, session_id: str) -> None:
        record = self._get_session(session_id)
        try:
            record.backend.close_session(record.handle)
        except Exception as exc:
            self._sessions[session_id] = replace(
                record,
                status="close_failed",
                last_error=str(exc),
            )
            raise
        del self._sessions[session_id]

    def list_training_sessions(self) -> list[TrainingSessionView]:
        return [self.get_session_view(session_id) for session_id in self._sessions]

    def get_session_view(self, session_id: str) -> TrainingSessionView:
        record = self._get_session(session_id)
        return TrainingSessionView(
            session_id=record.session_id,
            backend_id=record.backend_id,
            batch_codec=record.batch_codec,
            metadata=record.metadata,
            state=record.state,
            status=record.status,
            last_error=record.last_error,
        )

    def get_session_state(self, session_id: str) -> TrainState | None:
        return self._get_session(session_id).state

    def get_capabilities(self, session_id: str) -> TrainingCapabilities:
        record = self._get_active_session(session_id)
        return record.backend.capabilities()

    def get_tokenizer_info(self, session_id: str) -> TokenizerInfo:
        record = self._get_active_session(session_id)
        return record.backend.get_tokenizer_info(record.handle)

    def forward(self, session_id: str, req: TrainOpRequest) -> TrainOpResult:
        record = self._get_active_session(session_id)
        self._validate_batch_codec(record, req.batch_codec)
        return record.backend.forward(record.handle, req)

    def run_train_op(self, session_id: str, req: TrainOpRequest) -> TrainOpResult:
        record = self._get_active_session(session_id)
        self._validate_batch_codec(record, req.batch_codec)
        result = record.backend.run_train_op(record.handle, req)
        self._sessions[session_id] = replace(record, state=result.state)
        return result

    def forward_backward_ppo(self, session_id: str, req: TrainOpRequest) -> TrainOpResult:
        record = self._get_active_session(session_id)
        self._validate_batch_codec(record, req.batch_codec)
        result = record.backend.forward_backward_ppo(record.handle, req)
        self._sessions[session_id] = replace(record, state=result.state)
        return result

    def forward_backward_reverse_kl(self, session_id: str, req: TrainOpRequest) -> TrainOpResult:
        record = self._get_active_session(session_id)
        self._validate_batch_codec(record, req.batch_codec)
        result = record.backend.forward_backward_reverse_kl(record.handle, req)
        self._sessions[session_id] = replace(record, state=result.state)
        return result

    def reset_expert_bias(self, session_id: str) -> TrainOpResult:
        record = self._get_active_session(session_id)
        result = record.backend.reset_expert_bias(record.handle)
        self._sessions[session_id] = replace(record, state=result.state)
        return result

    def save_checkpoint(self, session_id: str, req: CheckpointRequest) -> ArtifactRef:
        record = self._get_active_session(session_id)
        return record.backend.save_checkpoint(record.handle, req)

    def load_checkpoint(self, session_id: str, req: CheckpointRequest) -> TrainState:
        record = self._get_active_session(session_id)
        state = record.backend.load_checkpoint(record.handle, req)
        self._sessions[session_id] = replace(record, state=state)
        return state

    def open_rollout_session(self, spec: RolloutSessionSpec) -> RolloutSessionView:
        if spec.rollout_session_id in self._rollouts:
            raise ValueError(f"duplicate rollout_session_id: {spec.rollout_session_id}")
        item = self._registry.get(spec.inference_backend_id)
        if item.spec.kind != BackendKind.INFERENCE:
            raise BackendKindMismatchError(
                f"backend {spec.inference_backend_id} is not inference"
            )
        self._get_active_session(spec.training_session_id)
        self._rollouts[spec.rollout_session_id] = RolloutSessionRecord(
            rollout_session_id=spec.rollout_session_id,
            inference_backend_id=spec.inference_backend_id,
            training_session_id=spec.training_session_id,
            metadata=dict(spec.metadata),
        )
        return self.get_rollout_session_view(spec.rollout_session_id)

    def close_rollout_session(self, rollout_session_id: str) -> None:
        self._get_active_rollout(rollout_session_id)
        del self._rollouts[rollout_session_id]

    def get_rollout_session_view(self, rollout_session_id: str) -> RolloutSessionView:
        record = self._get_rollout(rollout_session_id)
        return RolloutSessionView(
            rollout_session_id=record.rollout_session_id,
            inference_backend_id=record.inference_backend_id,
            training_session_id=record.training_session_id,
            status=record.status,
            last_error=record.last_error,
        )

    def collect_experience(
        self,
        rollout_session_id: str,
        *,
        prompt: str,
        batch_codec: str,
        sampling: dict | None = None,
        reward: float | None = None,
        policy_version: str | None = None,
        metadata: dict | None = None,
        num_samples: int = 1,
    ) -> ExperienceBatch:
        rollout = self._get_active_rollout(rollout_session_id)
        sample_count = max(1, int(num_samples))
        sampling = dict(sampling or {})
        rollout_metadata = dict(rollout.metadata)
        request_metadata = dict(metadata or {})
        merged_metadata = {**rollout_metadata, **request_metadata}
        group_id = str(merged_metadata.get("group_id") or rollout_session_id)
        samples = []
        prompt_tokens: tuple[int, ...] = ()

        for sample_index in range(sample_count):
            generated = self.generate(
                rollout.inference_backend_id,
                GenerateRequest(prompt=prompt, sampling=sampling),
            )
            completion_tokens = tuple(generated.token_ids)
            completion_logprobs = tuple(generated.token_logprobs)
            if not completion_tokens:
                parsed = [int(tok) for tok in generated.text.split() if tok.lstrip("-").isdigit()]
                completion_tokens = tuple(parsed or [len(generated.text)])
            if not completion_logprobs:
                completion_logprobs = tuple([-0.1 for _ in completion_tokens])
            if not prompt_tokens:
                prompt_tokens = tuple(generated.prompt_token_ids)
                if not prompt_tokens:
                    prompt_tokens = tuple(int(tok) for tok in prompt.split() if tok.lstrip("-").isdigit())

            sample_reward = float(reward) if reward is not None else 0.0
            samples.append(
                {
                    "sample_id": str(sample_index),
                    "group_id": group_id,
                    "prompt_tokens": prompt_tokens,
                    "completion_text": generated.text,
                    "completion_tokens": completion_tokens,
                    "old_logprobs": completion_logprobs,
                    "reference_logprobs": (),
                    "old_values": tuple(0.0 for _ in completion_tokens),
                    "advantages": (),
                    "returns": (),
                    "weights": tuple(1.0 for _ in completion_tokens),
                    "reward": sample_reward,
                    "stop_reason": generated.stop_reason,
                    "raw": dict(generated.raw),
                }
            )

        first = samples[0]
        algorithm = str(merged_metadata.get("algo") or sampling.get("algo") or "custom").lower()
        payload = {
            "algorithm": algorithm,
            "prompt": prompt,
            "prompt_tokens": prompt_tokens,
            "response": first["completion_text"],
            "raw": first["raw"],
            "reward": first["reward"],
            "reference_model_path": merged_metadata.get("reference_model_path")
            or sampling.get("reference_model_path")
            or sampling.get("model_path"),
            "samples": samples,
        }
        return ExperienceBatch(
            rollout_session_id=rollout.rollout_session_id,
            training_session_id=rollout.training_session_id,
            batch_codec=batch_codec,
            batch_payload=payload,
            policy_backend_id=rollout.inference_backend_id,
            policy_version=policy_version,
            metadata=merged_metadata,
        )

    def train_on_experience(
        self,
        rollout_session_id: str,
        *,
        extension_op: str = "rl",
        batch_payload: object | None = None,
        options: dict | None = None,
    ) -> TrainOpResult:
        rollout = self._get_active_rollout(rollout_session_id)
        session = self._get_active_session(rollout.training_session_id)
        payload = batch_payload if batch_payload is not None else {}
        req = TrainOpRequest(
            op=self._rollout_train_op(options, payload),
            batch_codec=session.batch_codec,
            batch_payload=payload,
            extension_op=extension_op,
            options=options or {},
        )
        if req.op is TrainOp.FORWARD_BACKWARD_PPO:
            return self.forward_backward_ppo(session.session_id, req)
        if req.op is TrainOp.FORWARD_BACKWARD_REVERSE_KL:
            return self.forward_backward_reverse_kl(session.session_id, req)
        return self.run_train_op(session.session_id, req)

    def _get_session(self, session_id: str) -> TrainingSessionRecord:
        record = self._sessions.get(session_id)
        if record is None:
            raise UnknownSessionError(f"unknown session_id: {session_id}")
        return record

    def _get_active_session(self, session_id: str) -> TrainingSessionRecord:
        record = self._get_session(session_id)
        if record.status != "active":
            raise UnsupportedOperationError(
                f"session {session_id} is not active: {record.status}"
            )
        return record

    def _get_rollout(self, rollout_session_id: str) -> RolloutSessionRecord:
        record = self._rollouts.get(rollout_session_id)
        if record is None:
            raise UnknownSessionError(f"unknown rollout_session_id: {rollout_session_id}")
        return record

    def _get_active_rollout(self, rollout_session_id: str) -> RolloutSessionRecord:
        record = self._get_rollout(rollout_session_id)
        if record.status != "active":
            raise UnsupportedOperationError(
                f"rollout session {rollout_session_id} is not active: {record.status}"
            )
        return record

    @staticmethod
    def _validate_batch_codec(record: TrainingSessionRecord, batch_codec: str) -> None:
        if record.batch_codec != batch_codec:
            raise ValueError(
                f"batch_codec mismatch: expected={record.batch_codec} got={batch_codec}"
            )

    @staticmethod
    def _rollout_train_op(options: dict | None, batch_payload: object | None = None) -> TrainOp:
        payload = dict(batch_payload) if hasattr(batch_payload, "get") else {}
        if not options:
            algo = str(payload.get("algorithm") or "").lower()
            if algo == "ppo":
                return TrainOp.FORWARD_BACKWARD_PPO
            if algo == "grpo":
                return TrainOp.FORWARD_BACKWARD_REVERSE_KL
            return TrainOp.CUSTOM
        op = options.get("train_op")
        algo = str(options.get("algo") or payload.get("algorithm") or "").lower()
        if op is None:
            if algo == "ppo":
                return TrainOp.FORWARD_BACKWARD_PPO
            if algo == "grpo":
                return TrainOp.FORWARD_BACKWARD_REVERSE_KL
            return TrainOp.CUSTOM
        op_name = str(op).lower()
        if op_name in {"ppo", "forward_backward_ppo"}:
            return TrainOp.FORWARD_BACKWARD_PPO
        if op_name in {"grpo", "reverse_kl", "forward_backward_reverse_kl"}:
            return TrainOp.FORWARD_BACKWARD_REVERSE_KL
        return TrainOp(op)
