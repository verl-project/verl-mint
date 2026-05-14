from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Mapping

from verl_mint.backends.base import InferenceBackend, TrainingBackend
from verl_mint.backends.mint_style import MintStyleGRPOTrainer
from verl_mint.contracts import (
    ArtifactRef,
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
from verl_mint.model_registry import ModelConfig, get_model_config


class VerlRuntimeError(RuntimeError):
    pass


def _import_attr(module_name: str, attr_name: str) -> Any:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - depends on optional veRL install
        raise VerlRuntimeError(f"veRL runtime module is unavailable: {module_name}") from exc
    try:
        return getattr(module, attr_name)
    except AttributeError as exc:  # pragma: no cover - depends on optional veRL version
        raise VerlRuntimeError(f"veRL runtime attribute is unavailable: {module_name}.{attr_name}") from exc


def _call_first(obj: Any, names: tuple[str, ...], *args: Any, **kwargs: Any) -> Any:
    for name in names:
        fn = getattr(obj, name, None)
        if fn is not None:
            return fn(*args, **kwargs)
    raise VerlRuntimeError(f"object {type(obj).__name__} has none of methods {names}")


_VERL_META_KEYS = {
    "algorithm",
    "config",
    "config_name",
    "config_path",
    "config_overrides",
    "hydra",
    "hydra_overrides",
    "ppo_config",
    "verl_config",
    "verl_config_overrides",
    "verl_overrides",
}


def _merge_mapping(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _merge_mapping(current, value)
        else:
            merged[key] = value
    return merged


def _nested_get(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if isinstance(current, Mapping):
            current = current.get(part)
        else:
            current = getattr(current, part, None)
        if current is None:
            return None
    return current


def _model_ref_from_spec(spec: Mapping[str, Any]) -> str | None:
    for key in ("model_path", "model_id", "base_model"):
        value = spec.get(key)
        if isinstance(value, str) and value:
            return value
    hydra = spec.get("hydra")
    if isinstance(hydra, Mapping):
        for item in hydra.get("overrides") or []:
            raw = str(item)
            key, sep, value = raw.partition("=")
            if sep and key.lstrip("+") == "actor_rollout_ref.model.path" and value:
                return value
    return None


def _override_key(item: str) -> str:
    return item.partition("=")[0].lstrip("+")


def _merge_hydra_overrides(base: list[str], extra: list[str], *, replace: bool = False) -> list[str]:
    merged = list(base)
    index = {_override_key(item): i for i, item in enumerate(merged) if "=" in item}
    for item in extra:
        key = _override_key(item)
        if replace and key in index:
            merged[index[key]] = item
        elif key not in index:
            index[key] = len(merged)
            merged.append(item)
    return merged


def _drop_hydra_overrides(base: list[str], prefixes: tuple[str, ...]) -> list[str]:
    return [item for item in base if not _override_key(item).startswith(prefixes)]


def _mint_hydra_overrides(model_ref: str, cfg: ModelConfig) -> list[str]:
    values = [
        f"actor_rollout_ref.model.path={model_ref}",
        f"actor_rollout_ref.model.enable_gradient_checkpointing={str(cfg.gradient_checkpointing)}",
        f"trainer.n_gpus_per_node={cfg.train_gpus}",
        "trainer.nnodes=1",
    ]
    if cfg.is_moe:
        etp = cfg.train_etp if cfg.train_etp is not None else cfg.train_tp
        values.extend(
            [
                f"actor_rollout_ref.actor.megatron.tensor_model_parallel_size={cfg.train_tp}",
                f"actor_rollout_ref.actor.megatron.pipeline_model_parallel_size={cfg.train_pp}",
                f"actor_rollout_ref.actor.megatron.expert_model_parallel_size={cfg.train_ep}",
                f"actor_rollout_ref.actor.megatron.context_parallel_size={cfg.train_cp}",
                f"actor_rollout_ref.actor.megatron.expert_tensor_parallel_size={etp}",
                "actor_rollout_ref.actor.megatron.param_offload=True",
                "actor_rollout_ref.actor.megatron.optimizer_offload=True",
                "actor_rollout_ref.actor.megatron.grad_offload=True",
                "actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True",
                "actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra]",
                "actor_rollout_ref.actor.checkpoint.load_contents=[model,optimizer,extra]",
                "actor_rollout_ref.actor.megatron.use_mbridge=True",
                "actor_rollout_ref.actor.megatron.vanilla_mbridge=False",
                "+actor_rollout_ref.actor.megatron.override_ddp_config.grad_reduce_in_fp32=True",
                "+actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform",
                "+actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full",
                "+actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1",
                "+actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32",
                f"actor_rollout_ref.ref.megatron.tensor_model_parallel_size={cfg.train_tp}",
                f"actor_rollout_ref.ref.megatron.pipeline_model_parallel_size={cfg.train_pp}",
                f"actor_rollout_ref.ref.megatron.expert_model_parallel_size={cfg.train_ep}",
                f"actor_rollout_ref.ref.megatron.context_parallel_size={cfg.train_cp}",
                f"actor_rollout_ref.ref.megatron.expert_tensor_parallel_size={etp}",
                "actor_rollout_ref.ref.megatron.param_offload=True",
                f"critic.megatron.tensor_model_parallel_size={cfg.train_tp}",
                f"critic.megatron.pipeline_model_parallel_size={cfg.train_pp}",
                f"critic.megatron.expert_model_parallel_size={cfg.train_ep}",
                f"critic.megatron.context_parallel_size={cfg.train_cp}",
                f"critic.megatron.expert_tensor_parallel_size={etp}",
                "critic.megatron.param_offload=True",
                "critic.megatron.optimizer_offload=True",
                "critic.megatron.grad_offload=True",
                "critic.optim.override_optimizer_config.use_precision_aware_optimizer=True",
                "critic.checkpoint.save_contents=[model,optimizer,extra]",
                "critic.checkpoint.load_contents=[model,optimizer,extra]",
                f"actor_rollout_ref.rollout.tensor_model_parallel_size={cfg.inference_tp}",
                f"actor_rollout_ref.rollout.data_parallel_size={cfg.inference_dp}",
                f"actor_rollout_ref.rollout.max_model_len={cfg.max_model_len}",
            ]
        )
        if cfg.train_lora_rank is not None:
            values.extend(
                [
                    f"+actor_rollout_ref.model.lora.rank={cfg.train_lora_rank}",
                    f"+actor_rollout_ref.model.lora.alpha={cfg.train_lora_alpha or cfg.train_lora_rank * 2}",
                    "+actor_rollout_ref.model.lora.dtype=float16",
                    "+actor_rollout_ref.model.lora.type=lora",
                    "+actor_rollout_ref.model.lora.target_modules=[linear_qkv,linear_proj,linear_fc1,linear_fc2]",
                    "+actor_rollout_ref.model.lora.exclude_modules=[]",
                ]
            )
        if cfg.gpu_memory_utilization is not None:
            values.append(f"actor_rollout_ref.rollout.gpu_memory_utilization={cfg.gpu_memory_utilization}")
        if cfg.max_num_seqs is not None:
            values.append(f"actor_rollout_ref.rollout.max_num_seqs={cfg.max_num_seqs}")
        if cfg.max_num_batched_tokens is not None:
            values.append(f"actor_rollout_ref.rollout.max_num_batched_tokens={cfg.max_num_batched_tokens}")
        if cfg.max_lora_rank is not None:
            values.append(f"actor_rollout_ref.model.lora_rank={cfg.max_lora_rank}")
    return values


@dataclass(frozen=True)
class VerlRuntimeConfig:
    model_id: str | None = None
    model_path: str | None = None
    trainer: Mapping[str, Any] | None = None
    rollout: Mapping[str, Any] | None = None
    resources: Mapping[str, Any] | None = None
    model_config: ModelConfig | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "VerlRuntimeConfig":
        model_id = value.get("model_id")
        model_path = value.get("model_path")
        trainer = dict(value.get("trainer") or {})
        rollout = dict(value.get("rollout") or {})
        resources = dict(value.get("resources") or {})
        model_config = value.get("model_config")
        if model_config is not None and not isinstance(model_config, ModelConfig):
            raise TypeError("model_config must be a ModelConfig")
        if model_config is None:
            model_ref = model_path or model_id
            if isinstance(model_ref, str) and model_ref:
                try:
                    model_config = get_model_config(model_ref)
                except ValueError:
                    model_config = None
        if isinstance(model_config, ModelConfig):
            resources.setdefault("train_gpus", model_config.train_gpus)
            resources.setdefault("inference_gpus", model_config.inference_gpus)
        return cls(
            model_id=model_id,
            model_path=model_path,
            trainer=VerlPPOJobRunner._with_mint_model_overrides(trainer, fallback_model_ref=model_path or model_id),
            rollout=rollout,
            resources=resources,
            model_config=model_config,
        )

    def merged(self, value: Mapping[str, Any]) -> "VerlRuntimeConfig":
        merged = {
            "model_id": self.model_id,
            "model_path": self.model_path,
            "trainer": dict(self.trainer or {}),
            "rollout": dict(self.rollout or {}),
            "resources": dict(self.resources or {}),
            "model_config": self.model_config,
        }
        for k, v in value.items():
            if k in {"trainer", "rollout", "resources"} and isinstance(v, Mapping):
                merged[k] = _merge_mapping(dict(merged[k] or {}), v)
            elif k == "metadata" and isinstance(v, Mapping):
                continue
            else:
                merged[k] = v
        if isinstance(value.get("base_model"), str):
            merged["model_id"] = value["base_model"]
            merged["model_path"] = value["base_model"]
        return VerlRuntimeConfig.from_mapping(merged)

    @property
    def model_ref(self) -> str | None:
        return self.model_path or self.model_id


class VerlBatchAdapter:
    def __init__(self, data_proto_cls: type[Any] | None = None) -> None:
        self.data_proto_cls = data_proto_cls or _import_attr("verl.protocol", "DataProto")

    def to_data_proto(self, payload: Any) -> Any:
        if isinstance(payload, self.data_proto_cls):
            return payload
        if not isinstance(payload, Mapping):
            raise TypeError("veRL backend batch_payload must be a mapping or DataProto")

        tensors = dict(payload.get("tensors") or payload.get("batch") or {})
        non_tensor_batch = dict(
            payload.get("non_tensor_batch")
            or payload.get("non_tensors")
            or payload.get("data")
            or {}
        )
        meta_info = dict(payload.get("meta_info") or payload.get("metadata") or {})
        if "loss_fn" in payload:
            meta_info["loss_fn"] = payload["loss_fn"]
        if "loss_fn_config" in payload:
            meta_info["loss_fn_config"] = payload["loss_fn_config"]
        for key in _VERL_META_KEYS:
            if key in payload:
                meta_info[key] = payload[key]

        if hasattr(self.data_proto_cls, "from_dict"):
            try:
                return self.data_proto_cls.from_dict(
                    tensors=tensors,
                    non_tensors=non_tensor_batch,
                    meta_info=meta_info,
                )
            except TypeError:
                return self.data_proto_cls.from_dict(
                    tensors=tensors,
                    non_tensor_batch=non_tensor_batch,
                    meta_info=meta_info,
                )

        for kwargs in (
            {"batch": tensors, "non_tensor_batch": non_tensor_batch, "meta_info": meta_info},
            {"tensors": tensors, "non_tensor_batch": non_tensor_batch, "meta_info": meta_info},
        ):
            try:
                return self.data_proto_cls(**kwargs)
            except TypeError:
                continue
        raise VerlRuntimeError("unsupported DataProto constructor shape")


class VerlPPOJobRunner:
    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        run_ppo: Callable[[Any], Any] | None = None,
    ) -> None:
        self.config = dict(config or {})
        self._apply_mint_model_config()
        self.run_ppo = None if self.config.get("remote_run_ppo") else run_ppo or self._resolve_run_ppo(self.config)
        self.initialized = False
        self.closed = False
        self.last_batch: Any | None = None
        self.last_config: Any | None = None

    def init_workers(self) -> None:
        self.initialized = True

    def shutdown(self) -> None:
        self.closed = True

    def ppo_step(self, batch: Any) -> Mapping[str, Any]:
        self.last_batch = batch
        config = self._config_from_batch(batch)
        self.last_config = config
        out = self._run_ppo(config)
        metrics = dict(out or {}) if isinstance(out, Mapping) else {}
        step = metrics.pop("step", None) or self._expected_training_steps(config)
        if step is None:
            step = 0
        return {
            "step": int(step),
            "algorithm": "ppo",
            "execution_framework": "verl",
            "runner": "verl.trainer.main_ppo.run_ppo",
            **metrics,
        }

    def _run_ppo(self, config: Any) -> Any:
        if not self.config.get("remote_run_ppo"):
            if self.run_ppo is None:
                raise VerlRuntimeError("local veRL PPO runner is not configured")
            return self.run_ppo(config)

        import ray
        from omegaconf import OmegaConf

        if not ray.is_initialized():
            ray_init = self._ray_init_config(config)
            if isinstance(ray_init, Mapping):
                ray_init_kwargs = dict(ray_init)
            else:
                ray_init_kwargs = OmegaConf.to_container(ray_init, resolve=True)
            ray.init(**ray_init_kwargs)

        run_ppo_path = str(self.config.get("run_ppo") or "verl.trainer.main_ppo:run_ppo")

        @ray.remote(num_cpus=1)
        def _remote_run_ppo(spec: Any, path: str) -> Any:
            import importlib
            from collections.abc import Mapping as RemoteMapping

            if isinstance(spec, RemoteMapping):
                hydra_spec = spec.get("hydra")
                if isinstance(hydra_spec, RemoteMapping):
                    import hydra

                    config_dir = hydra_spec.get("config_dir") or spec.get("config_path")
                    config_name = str(hydra_spec.get("config_name") or spec.get("config_name") or "ppo_trainer")
                    overrides = list(hydra_spec.get("overrides") or spec.get("hydra_overrides") or [])
                    with hydra.initialize_config_dir(
                        config_dir=str(config_dir),
                        version_base=None,
                        job_name="verl_mint_remote_ppo",
                    ):
                        spec = hydra.compose(config_name=config_name, overrides=overrides)
            module_name, attr_name = path.split(":", 1)
            return getattr(importlib.import_module(module_name), attr_name)(spec)

        return ray.get(_remote_run_ppo.remote(config, run_ppo_path))

    @staticmethod
    def _resolve_run_ppo(config: Mapping[str, Any]) -> Callable[[Any], Any]:
        path = str(config.get("run_ppo") or "verl.trainer.main_ppo:run_ppo")
        module_name, attr_name = path.split(":", 1)
        return _import_attr(module_name, attr_name)

    def _config_from_batch(self, batch: Any) -> Any:
        meta_info = getattr(batch, "meta_info", {}) or {}
        if not isinstance(meta_info, Mapping):
            raise TypeError("veRL PPO DataProto meta_info must be a mapping")

        spec = self._with_mint_model_overrides(dict(self.config))
        for key in ("ppo_config", "verl_config", "config"):
            value = meta_info.get(key)
            if isinstance(value, Mapping):
                spec = _merge_mapping(spec, value)
        for key in _VERL_META_KEYS:
            if key in meta_info and key not in {"ppo_config", "verl_config", "config"}:
                spec[key] = meta_info[key]
        if self.config.get("remote_run_ppo"):
            return spec
        return self._compose_config(spec)

    def _apply_mint_model_config(self) -> None:
        self.config = self._with_mint_model_overrides(self.config)

    @staticmethod
    def _with_mint_model_overrides(spec: Mapping[str, Any], fallback_model_ref: str | None = None) -> dict[str, Any]:
        merged = dict(spec)
        model_ref = _model_ref_from_spec(merged) or fallback_model_ref
        if not model_ref:
            return merged
        try:
            cfg = get_model_config(model_ref)
        except ValueError:
            return merged
        hydra = dict(merged.get("hydra") or {}) if isinstance(merged.get("hydra"), Mapping) else {}
        overrides = [str(x) for x in hydra.get("overrides") or merged.get("hydra_overrides") or []]
        if cfg.is_moe:
            overrides = _drop_hydra_overrides(
                overrides,
                (
                    "actor_rollout_ref.actor.fsdp_config",
                    "actor_rollout_ref.ref.fsdp_config",
                    "critic.fsdp_config",
                ),
            )
            hydra.setdefault("config_name", "ppo_megatron_trainer")
        overrides = _merge_hydra_overrides(
            overrides,
            _mint_hydra_overrides(model_ref, cfg),
            replace=cfg.is_moe,
        )
        if overrides:
            hydra["overrides"] = overrides
            merged["hydra"] = hydra
        resources = dict(merged.get("resources") or {}) if isinstance(merged.get("resources"), Mapping) else {}
        resources.setdefault("train_gpus", cfg.train_gpus)
        resources.setdefault("inference_gpus", cfg.inference_gpus)
        merged["resources"] = resources
        merged["model_config"] = cfg
        return merged

    @staticmethod
    def _compose_config(spec: Mapping[str, Any]) -> Any:
        try:
            from omegaconf import OmegaConf
        except Exception:  # pragma: no cover - only used when veRL deps are absent
            OmegaConf = None

        hydra_spec = spec.get("hydra")
        if isinstance(hydra_spec, Mapping):
            config_dir = hydra_spec.get("config_dir") or spec.get("config_path")
            if config_dir is None:
                raise VerlRuntimeError("veRL PPO hydra config requires config_dir or config_path")
            try:
                import hydra
            except Exception as exc:  # pragma: no cover - depends on optional veRL install
                raise VerlRuntimeError("Hydra is required to compose veRL PPO config") from exc
            overrides = list(hydra_spec.get("overrides") or spec.get("hydra_overrides") or [])
            config_name = str(hydra_spec.get("config_name") or spec.get("config_name") or "ppo_trainer")
            with hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None, job_name="verl_mint_ppo"):
                return hydra.compose(config_name=config_name, overrides=overrides)

        config = spec.get("config") if isinstance(spec.get("config"), Mapping) else spec
        overrides = spec.get("config_overrides") or spec.get("verl_config_overrides") or spec.get("verl_overrides")
        if isinstance(overrides, Mapping):
            config = _merge_mapping(config, overrides)
        if OmegaConf is None:
            return dict(config)
        return OmegaConf.create(config)

    @staticmethod
    def _ray_init_config(config: Any) -> Any:
        ray_init = _nested_get(config, "ray_kwargs.ray_init") or {}
        if ray_init:
            return ray_init
        if not isinstance(config, Mapping):
            return {}
        hydra_spec = config.get("hydra")
        if not isinstance(hydra_spec, Mapping):
            return {}
        values: dict[str, Any] = {}
        for item in hydra_spec.get("overrides") or []:
            raw = str(item)
            key, sep, value = raw.partition("=")
            if not sep:
                continue
            key = key.lstrip("+")
            prefix = "ray_kwargs.ray_init."
            if key.startswith(prefix):
                target = values
                parts = key.removeprefix(prefix).split(".")
                for part in parts[:-1]:
                    target = target.setdefault(part, {})
                target[parts[-1]] = value
        return values

    @staticmethod
    def _expected_training_steps(config: Any) -> int | None:
        step = _nested_get(config, "trainer.total_training_steps")
        if step is not None:
            return int(step)
        if not isinstance(config, Mapping):
            return None
        hydra_spec = config.get("hydra")
        if not isinstance(hydra_spec, Mapping):
            return None
        for item in hydra_spec.get("overrides") or []:
            raw = str(item)
            key, sep, value = raw.partition("=")
            if sep and key.lstrip("+") == "trainer.total_training_steps":
                return int(value)
        return None


class VerlTrainingBackend(TrainingBackend):
    def __init__(
        self,
        *,
        model_id: str | None = None,
        model_path: str | None = None,
        config: Mapping[str, Any] | None = None,
        backend_kwargs: Mapping[str, Any] | None = None,
        batch_adapter: VerlBatchAdapter | None = None,
    ) -> None:
        runtime_config = VerlRuntimeConfig.from_mapping({**dict(config or {}), **dict(backend_kwargs or {})})
        self.config = runtime_config.merged({"model_id": model_id, "model_path": model_path})
        self.batch_adapter = batch_adapter or VerlBatchAdapter()
        self._trainers: dict[str, Any] = {}
        self._steps: dict[str, int] = {}

    def open_session(self, spec: SessionSpec) -> SessionHandle:
        trainer = self._build_trainer(self.config.merged(spec.metadata))
        self._init_trainer(trainer)
        self._trainers[spec.session_id] = trainer
        self._steps[spec.session_id] = 0
        return SessionHandle(
            backend_session_id=spec.session_id,
            metadata={
                "execution_framework": "verl",
                "placement_model": "verl_resource_pool",
            },
        )

    def close_session(self, handle: SessionHandle) -> None:
        sid = self._sid(handle)
        trainer = self._trainers.pop(sid)
        self._steps.pop(sid, None)
        shutdown = getattr(trainer, "shutdown", None) or getattr(trainer, "close", None)
        if shutdown is not None:
            shutdown()

    def capabilities(self) -> TrainingCapabilities:
        return TrainingCapabilities(
            supports_forward=True,
            supports_train_step=True,
            supports_reverse_kl=True,
            supports_checkpoint_load=True,
            supports_checkpoint_save=True,
            extras={
                "requires_shared_checkpoint_io": True,
                "requires_portable_reference_uri": True,
                "supports_remote_execution": True,
                "execution_framework": "verl",
                "placement_model": "verl_resource_pool",
                "batch_protocol": "verl.protocol.DataProto",
            },
        )

    def get_tokenizer_info(self, handle: SessionHandle) -> TokenizerInfo:
        trainer = self._trainer(handle)
        info = getattr(trainer, "get_tokenizer_info", None)
        if info is None:
            return TokenizerInfo(metadata={"backend": "verl"})
        result = info()
        if isinstance(result, TokenizerInfo):
            return result
        return TokenizerInfo(metadata=dict(result or {}))

    def forward(self, handle: SessionHandle, req: TrainOpRequest) -> TrainOpResult:
        data = self._data_for_request(req)
        out = _call_first(self._trainer(handle), ("forward", "eval_step", "compute_log_probs"), data)
        return self._result(handle, out)

    def run_train_op(self, handle: SessionHandle, req: TrainOpRequest) -> TrainOpResult:
        if req.op is TrainOp.FORWARD:
            return self.forward(handle, req)
        if req.op is TrainOp.FORWARD_BACKWARD_REVERSE_KL:
            return self.forward_backward_reverse_kl(handle, req)
        if self._is_dpo_request(req):
            return self._run_dpo_loss(handle, req)
        if req.op is TrainOp.FORWARD_BACKWARD_PPO:
            return self.forward_backward_ppo(handle, req)
        data = self._data_for_request(req)
        out = _call_first(self._trainer(handle), ("train_step", "fit_batch", "step", "update_actor"), data)
        return self._result(handle, out, increment=True)

    def forward_backward_ppo(self, handle: SessionHandle, req: TrainOpRequest) -> TrainOpResult:
        data = self._data_for_request(req)
        out = _call_first(self._trainer(handle), ("ppo_step", "train_step", "fit_batch", "step", "update_actor"), data)
        return self._result(handle, out, increment=True)

    def forward_backward_reverse_kl(self, handle: SessionHandle, req: TrainOpRequest) -> TrainOpResult:
        if self._uses_mint_style_grpo(req):
            out = MintStyleGRPOTrainer(
                self._trainer(handle),
                adapter=self.batch_adapter,
                max_token_len_per_gpu=int(req.options.get("max_token_len_per_gpu", 10240)),
            ).step(req)
            return self._result(handle, out, increment=True)
        data = self._data_for_request(req)
        out = _call_first(self._trainer(handle), ("grpo_step", "train_step", "fit_batch", "step", "update_actor"), data)
        return self._result(handle, out, increment=True)

    def _uses_mint_style_grpo(self, req: TrainOpRequest) -> bool:
        route = str(req.options.get("route") or req.options.get("execution_framework") or "").lower()
        if route in {"mint_style", "mint-style", "mint"}:
            return True
        if isinstance(req.batch_payload, Mapping):
            algo = str(req.batch_payload.get("algorithm") or req.options.get("algo") or "").lower()
            return algo == "grpo" and bool(req.batch_payload.get("samples"))
        return False

    def _run_dpo_loss(self, handle: SessionHandle, req: TrainOpRequest) -> TrainOpResult:
        data = self._data_for_request(req)
        out = _call_first(self._trainer(handle), ("dpo_step", "train_dpo"), data)
        return self._result(handle, out, increment=True)

    def _data_for_request(self, req: TrainOpRequest) -> Any:
        data = self.batch_adapter.to_data_proto(req.batch_payload)
        meta_info = getattr(data, "meta_info", None)
        if isinstance(meta_info, dict) and req.options:
            options = dict(req.options)
            meta_info["mint_options"] = options
            for key in _VERL_META_KEYS:
                if key in options:
                    meta_info[key] = options[key]
        return data

    def _is_dpo_request(self, req: TrainOpRequest) -> bool:
        if req.extension_op == "dpo":
            return True
        if isinstance(req.batch_payload, Mapping):
            loss_fn = str(req.batch_payload.get("loss_fn") or "").lower()
            if loss_fn in {"dpo", "direct_preference_optimization"}:
                return True
            loss_cfg = req.batch_payload.get("loss_fn_config")
            if isinstance(loss_cfg, Mapping):
                algo = str(loss_cfg.get("algorithm") or loss_cfg.get("type") or "").lower()
                return algo in {"dpo", "direct_preference_optimization"}
        return False

    def reset_expert_bias(self, handle: SessionHandle) -> TrainOpResult:
        out = _call_first(self._trainer(handle), ("reset_expert_bias",))
        return self._result(handle, out)

    def save_checkpoint(self, handle: SessionHandle, req: CheckpointRequest) -> ArtifactRef:
        out = _call_first(
            self._trainer(handle),
            ("save_checkpoint", "save"),
            req.uri,
            include_optimizer=req.include_optimizer,
            **dict(req.metadata),
        )
        metadata = dict(out or {}) if isinstance(out, Mapping) else {}
        return ArtifactRef(uri=req.uri, format="verl-checkpoint", metadata=metadata)

    def load_checkpoint(self, handle: SessionHandle, req: CheckpointRequest) -> TrainState:
        out = _call_first(
            self._trainer(handle),
            ("load_checkpoint", "load"),
            req.uri,
            include_optimizer=req.include_optimizer,
            **dict(req.metadata),
        )
        if isinstance(out, TrainState):
            return out
        if isinstance(out, Mapping) and "step" in out:
            step = int(out["step"])
            self._steps[self._sid(handle)] = step
            return TrainState(step=step, extras=dict(out))
        return TrainState(step=self._steps[self._sid(handle)])

    def _build_trainer(self, config: VerlRuntimeConfig) -> Any:
        trainer_cfg = dict(config.trainer or {})
        trainer_cls = trainer_cfg.pop("class", None)
        if trainer_cls is not None:
            module_name, attr_name = str(trainer_cls).rsplit(":", 1)
            return _import_attr(module_name, attr_name)({**trainer_cfg, "runtime": config})
        return VerlPPOJobRunner(trainer_cfg)

    def _init_trainer(self, trainer: Any) -> None:
        init = getattr(trainer, "init_workers", None) or getattr(trainer, "initialize", None) or getattr(trainer, "setup", None)
        if init is not None:
            init()

    def _sid(self, handle: SessionHandle) -> str:
        if handle.backend_session_id is None or handle.backend_session_id not in self._trainers:
            raise KeyError(f"unknown veRL session: {handle.backend_session_id}")
        return handle.backend_session_id

    def _trainer(self, handle: SessionHandle) -> Any:
        return self._trainers[self._sid(handle)]

    def _result(self, handle: SessionHandle, out: Any, *, increment: bool = False) -> TrainOpResult:
        sid = self._sid(handle)
        if increment:
            self._steps[sid] += 1
        if isinstance(out, TrainOpResult):
            self._steps[sid] = out.state.step
            return out
        if isinstance(out, Mapping):
            if "step" in out:
                self._steps[sid] = int(out["step"])
            return TrainOpResult(
                state=TrainState(step=self._steps[sid]),
                outputs={k: v for k, v in out.items() if k != "step"},
            )
        return TrainOpResult(state=TrainState(step=self._steps[sid]), outputs={"raw": out})


class VerlInferenceBackend(InferenceBackend):
    def __init__(
        self,
        *,
        model_id: str | None = None,
        model_path: str | None = None,
        config: Mapping[str, Any] | None = None,
        backend_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        runtime_config = VerlRuntimeConfig.from_mapping({**dict(config or {}), **dict(backend_kwargs or {})})
        self.config = runtime_config.merged({"model_id": model_id, "model_path": model_path})
        self.rollout = self._build_rollout(self.config)

    def generate(self, req: GenerateRequest) -> GenerateResult:
        out = _call_first(self.rollout, ("generate", "generate_sequences"), req.prompt, **dict(req.sampling))
        if isinstance(out, GenerateResult):
            return out
        if isinstance(out, Mapping):
            return GenerateResult(
                text=str(out.get("text", "")),
                token_ids=tuple(out.get("token_ids", ())),
                token_logprobs=tuple(out.get("token_logprobs", ())),
                prompt_token_ids=tuple(out.get("prompt_token_ids", ())),
                prompt_logprobs=tuple(out.get("prompt_logprobs", ())),
                stop_reason=str(out.get("stop_reason", "length")),
                raw=out,
            )
        return GenerateResult(text=str(out), raw={"backend": "verl"})

    def _build_rollout(self, config: VerlRuntimeConfig) -> Any:
        rollout_cfg = dict(config.rollout or {})
        rollout_cls = rollout_cfg.pop("class", None)
        if rollout_cls is not None:
            module_name, attr_name = str(rollout_cls).rsplit(":", 1)
            return _import_attr(module_name, attr_name)({**rollout_cfg, "runtime": config})
        raise VerlRuntimeError(
            "built-in veRL rollout requires config={'rollout': {'class': '<module>:<class>'}} "
            "until a project-default veRL rollout class is selected."
        )


def default_verl_training_backend(**kwargs: Any) -> VerlTrainingBackend:
    return VerlTrainingBackend(**kwargs)


def default_verl_inference_backend(**kwargs: Any) -> VerlInferenceBackend:
    return VerlInferenceBackend(**kwargs)
