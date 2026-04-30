from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from verl_mint.backends.base import Backend
from verl_mint.contracts import (
    ArtifactRef,
    BackendSpec,
    CheckpointRequest,
    ExperienceBatch,
    GenerateRequest,
    GenerateResult,
    RolloutSessionSpec,
    RolloutSessionView,
    SessionSpec,
    TokenizerInfo,
    TrainOp,
    TrainOpRequest,
    TrainOpResult,
    TrainingCapabilities,
    TrainingSessionView,
    TrainState,
)
from verl_mint.router import BackendRouter
from verl_mint.storage import LocalStorageRepo, storage_env_name


class BackendRegistryService:
    def __init__(self, router: BackendRouter) -> None:
        self._router = router

    def register(self, spec: BackendSpec, backend: Backend) -> None:
        self._router.register_backend(spec, backend)

    def list(self) -> list[BackendSpec]:
        return self._router.list_backends()


class InferenceService:
    def __init__(self, router: BackendRouter) -> None:
        self._router = router

    def generate(
        self,
        backend_id: str,
        prompt: str,
        sampling: Mapping[str, Any] | None = None,
    ) -> GenerateResult:
        return self._router.generate(
            backend_id,
            GenerateRequest(prompt=prompt, sampling=sampling or {}),
        )


class TrainingService:
    def __init__(self, router: BackendRouter, storage: LocalStorageRepo) -> None:
        self._router = router
        self._storage = storage

    def _resolve_checkpoint_uri(self, uri: str) -> str:
        if uri.startswith(("mint://", "tinker://", "repo://")):
            return self._storage.resolve_for_read(uri)
        return uri

    def _prepare_reference_model_path(self, model_id: str, batch_payload: Any) -> Any:
        if not hasattr(batch_payload, "items"):
            return batch_payload
        payload = deepcopy(dict(batch_payload))
        raw_path = payload.get("reference_model_path")
        if not isinstance(raw_path, str) or not raw_path:
            return payload

        requires_portable_uri = self._training_requires(
            model_id,
            "requires_portable_reference_uri",
        )
        if raw_path.startswith("file://"):
            normalized = str(Path(raw_path[len("file://") :]).expanduser().resolve())
            payload["reference_model_path"] = normalized
            if requires_portable_uri:
                self._storage.require_shared_path(
                    Path(normalized),
                    reason="portable reverse KL reference checkpoint",
                )
            return payload

        if raw_path.startswith(("mint://", "tinker://", "repo://")):
            if requires_portable_uri:
                self._storage.require_shared_storage(
                    reason="portable reverse KL reference checkpoint"
                )
            return payload

        if not requires_portable_uri:
            return payload

        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            raise ValueError(
                "portable reverse KL reference_model_path must use mint://, tinker://, repo://, "
                "or an absolute shared filesystem path"
            )
        normalized = path.resolve()
        self._storage.require_shared_path(
            normalized,
            reason="portable reverse KL reference checkpoint",
        )
        payload["reference_model_path"] = str(normalized)
        return payload

    def _training_flags(self, model_id: str) -> dict[str, Any]:
        view = self._router.get_session_view(model_id)
        spec = self._router.get_backend_spec(view.backend_id)
        caps = self._router.get_capabilities(model_id)
        return {**dict(spec.config), **dict(caps.extras)}

    def _training_requires(self, model_id: str, key: str) -> bool:
        return bool(self._training_flags(model_id).get(key))

    def _require_shared_storage_for_model(self, model_id: str, *, reason: str) -> None:
        if self._training_requires(model_id, "requires_shared_checkpoint_io"):
            self._storage.require_shared_storage(reason=reason)

    def create_model(
        self,
        model_id: str,
        backend_id: str,
        batch_codec: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> TrainingSessionView:
        return self._router.open_training_session(
            SessionSpec(
                session_id=model_id,
                backend_id=backend_id,
                batch_codec=batch_codec,
                metadata=metadata or {},
            )
        )

    def create_model_from_state(
        self,
        model_id: str,
        backend_id: str,
        batch_codec: str,
        uri: str,
        include_optimizer: bool = True,
        metadata: Mapping[str, Any] | None = None,
    ) -> TrainingSessionView:
        view = self.create_model(model_id, backend_id, batch_codec, metadata)
        try:
            self.load_weights(
                model_id,
                uri=uri,
                include_optimizer=include_optimizer,
                metadata=metadata,
            )
        except Exception as load_exc:
            try:
                self.delete_model(model_id)
            except Exception as cleanup_exc:
                raise RuntimeError(
                    f"rollback failed while closing session '{model_id}': {cleanup_exc}"
                ) from load_exc
            raise
        return self._router.get_session_view(view.session_id)

    def delete_model(self, model_id: str) -> None:
        self._router.close_training_session(model_id)

    def list_models(self) -> list[TrainingSessionView]:
        return self._router.list_training_sessions()

    def get_model(self, model_id: str) -> TrainingSessionView:
        return self._router.get_session_view(model_id)

    def get_capabilities(self, model_id: str) -> TrainingCapabilities:
        return self._router.get_capabilities(model_id)

    def get_tokenizer_info(self, model_id: str) -> TokenizerInfo:
        return self._router.get_tokenizer_info(model_id)

    def get_info(self, model_id: str) -> dict[str, Any]:
        view = self._router.get_session_view(model_id)
        return {
            "model": view,
            "capabilities": self.get_capabilities(model_id),
            "tokenizer": self.get_tokenizer_info(model_id),
        }

    def forward(
        self,
        model_id: str,
        batch_payload: Any,
        options: Mapping[str, Any] | None = None,
    ) -> TrainOpResult:
        session = self._router.get_session_view(model_id)
        return self._router.forward(
            model_id,
            TrainOpRequest(
                op=TrainOp.FORWARD,
                batch_codec=session.batch_codec,
                batch_payload=batch_payload,
                options=options or {},
            ),
        )

    def forward_backward(
        self,
        model_id: str,
        batch_payload: Any,
        options: Mapping[str, Any] | None = None,
    ) -> TrainOpResult:
        return self.run_train_op(
            model_id,
            op=TrainOp.FORWARD_BACKWARD,
            batch_payload=batch_payload,
            options=options,
        )

    def optim_step(
        self,
        model_id: str,
        batch_payload: Any | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> TrainOpResult:
        return self.run_train_op(
            model_id,
            op=TrainOp.OPTIMIZER_STEP,
            batch_payload=batch_payload or {},
            options=options,
        )

    def train_step(
        self,
        model_id: str,
        batch_payload: Any,
        options: Mapping[str, Any] | None = None,
    ) -> TrainOpResult:
        return self.run_train_op(
            model_id,
            op=TrainOp.TRAIN_STEP,
            batch_payload=batch_payload,
            options=options,
        )

    def forward_backward_ppo(
        self,
        model_id: str,
        batch_payload: Any,
        options: Mapping[str, Any] | None = None,
    ) -> TrainOpResult:
        session = self._router.get_session_view(model_id)
        return self._router.forward_backward_ppo(
            model_id,
            TrainOpRequest(
                op=TrainOp.FORWARD_BACKWARD_PPO,
                batch_codec=session.batch_codec,
                batch_payload=batch_payload,
                options=options or {},
            ),
        )

    def forward_backward_reverse_kl(
        self,
        model_id: str,
        batch_payload: Any,
        options: Mapping[str, Any] | None = None,
    ) -> TrainOpResult:
        session = self._router.get_session_view(model_id)
        return self._router.forward_backward_reverse_kl(
            model_id,
            TrainOpRequest(
                op=TrainOp.FORWARD_BACKWARD_REVERSE_KL,
                batch_codec=session.batch_codec,
                batch_payload=self._prepare_reference_model_path(model_id, batch_payload),
                options=options or {},
            ),
        )

    def reset_expert_bias(self, model_id: str) -> TrainOpResult:
        return self._router.reset_expert_bias(model_id)

    def run_train_op(
        self,
        model_id: str,
        op: TrainOp,
        batch_payload: Any,
        extension_op: str | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> TrainOpResult:
        session = self._router.get_session_view(model_id)
        return self._router.run_train_op(
            model_id,
            TrainOpRequest(
                op=op,
                batch_codec=session.batch_codec,
                batch_payload=batch_payload,
                extension_op=extension_op,
                options=options or {},
            ),
        )

    def save_weights(
        self,
        model_id: str,
        uri: str,
        include_optimizer: bool = True,
        metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        self._require_shared_storage_for_model(
            model_id,
            reason="portable checkpoint save",
        )
        resolved_uri = self._storage.resolve_for_write(uri)
        artifact = self._router.save_checkpoint(
            model_id,
            CheckpointRequest(
                uri=resolved_uri,
                include_optimizer=include_optimizer,
                metadata=metadata or {},
            ),
        )
        return ArtifactRef(uri=uri, format=artifact.format, metadata=dict(artifact.metadata))

    def load_weights(
        self,
        model_id: str,
        uri: str,
        include_optimizer: bool = True,
        metadata: Mapping[str, Any] | None = None,
    ) -> TrainState:
        self._require_shared_storage_for_model(
            model_id,
            reason="portable checkpoint load",
        )
        resolved_uri = self._storage.resolve_for_read(uri)
        return self._router.load_checkpoint(
            model_id,
            CheckpointRequest(
                uri=resolved_uri,
                include_optimizer=include_optimizer,
                metadata=metadata or {},
            ),
        )

    def get_state(self, model_id: str) -> TrainState | None:
        return self._router.get_session_state(model_id)


class RolloutService:
    def __init__(self, router: BackendRouter, storage: LocalStorageRepo) -> None:
        self._router = router
        self._storage = storage

    def _training_flags(self, training_session_id: str) -> dict[str, Any]:
        view = self._router.get_session_view(training_session_id)
        spec = self._router.get_backend_spec(view.backend_id)
        caps = self._router.get_capabilities(training_session_id)
        return {**dict(spec.config), **dict(caps.extras)}

    def _training_requires(self, training_session_id: str, key: str) -> bool:
        return bool(self._training_flags(training_session_id).get(key))

    def _normalize_reference_model_path(
        self,
        training_session_id: str,
        metadata: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        resolved = dict(metadata or {})
        raw_path = resolved.get("reference_model_path")
        if not isinstance(raw_path, str) or not raw_path:
            return resolved

        requires_portable_uri = self._training_requires(
            training_session_id,
            "requires_portable_reference_uri",
        )
        if raw_path.startswith("file://"):
            normalized = str(Path(raw_path[len("file://") :]).expanduser().resolve())
            resolved["reference_model_path"] = normalized
            if requires_portable_uri:
                self._storage.require_shared_path(
                    Path(normalized),
                    reason="portable rollout reference checkpoint",
                )
            return resolved

        if raw_path.startswith(("mint://", "tinker://", "repo://")):
            if requires_portable_uri:
                self._storage.require_shared_storage(
                    reason="portable rollout reference checkpoint"
                )
            return resolved

        if not requires_portable_uri:
            return resolved

        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            raise ValueError(
                "portable rollout reference_model_path must use mint://, tinker://, repo://, "
                "or an absolute shared filesystem path"
            )
        normalized = path.resolve()
        self._storage.require_shared_path(
            normalized,
            reason="portable rollout reference checkpoint",
        )
        resolved["reference_model_path"] = str(normalized)
        return resolved

    def _ensure_grpo_reference_snapshot(
        self,
        rollout_session_id: str,
        training_session_id: str,
        metadata: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        resolved = self._normalize_reference_model_path(training_session_id, metadata)
        if resolved.get("reference_model_path"):
            return resolved
        if str(resolved.get("algo") or "").lower() != "grpo":
            return resolved
        if self._training_requires(training_session_id, "requires_shared_checkpoint_io"):
            self._storage.require_shared_storage(reason="portable rollout reference checkpoint")

        reference_uri = f"mint://rollouts/{training_session_id}/{rollout_session_id}/reference.pt"
        resolved_uri = self._storage.resolve_for_write(reference_uri)
        self._router.save_checkpoint(
            training_session_id,
            CheckpointRequest(
                uri=resolved_uri,
                include_optimizer=False,
                metadata={"type": "reference", "rollout_session_id": rollout_session_id},
            ),
        )
        resolved["reference_model_path"] = reference_uri
        return resolved

    def open_session(
        self,
        rollout_session_id: str,
        inference_backend_id: str,
        training_session_id: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> RolloutSessionView:
        return self._router.open_rollout_session(
            RolloutSessionSpec(
                rollout_session_id=rollout_session_id,
                inference_backend_id=inference_backend_id,
                training_session_id=training_session_id,
                metadata=self._ensure_grpo_reference_snapshot(
                    rollout_session_id,
                    training_session_id,
                    metadata,
                ),
            )
        )

    def close_session(self, rollout_session_id: str) -> None:
        self._router.close_rollout_session(rollout_session_id)

    def collect_experience(
        self,
        rollout_session_id: str,
        prompt: str,
        batch_codec: str,
        sampling: Mapping[str, Any] | None = None,
        reward: float | None = None,
        policy_version: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        num_samples: int = 1,
    ) -> ExperienceBatch:
        rollout = self._router.get_rollout_session_view(rollout_session_id)
        return self._router.collect_experience(
            rollout_session_id,
            prompt=prompt,
            batch_codec=batch_codec,
            sampling=dict(sampling or {}),
            reward=reward,
            policy_version=policy_version,
            metadata=self._normalize_reference_model_path(rollout.training_session_id, metadata),
            num_samples=num_samples,
        )

    def train_on_experience(
        self,
        rollout_session_id: str,
        batch_payload: Any | None = None,
        extension_op: str = "rl",
        options: Mapping[str, Any] | None = None,
    ) -> TrainOpResult:
        rollout = self._router.get_rollout_session_view(rollout_session_id)
        return self._router.train_on_experience(
            rollout_session_id,
            extension_op=extension_op,
            batch_payload=self._normalize_reference_model_path(rollout.training_session_id, batch_payload),
            options=dict(options or {}),
        )


class MintService:
    def __init__(
        self,
        router: BackendRouter | None = None,
        storage: LocalStorageRepo | None = None,
    ) -> None:
        router = router or BackendRouter()
        storage = storage or LocalStorageRepo.from_env()
        storage.ensure()
        os.environ[storage_env_name()] = str(storage.root)
        self.storage = storage
        self.backends = BackendRegistryService(router)
        self.inference = InferenceService(router)
        self.training = TrainingService(router, storage)
        self.rollouts = RolloutService(router, storage)
