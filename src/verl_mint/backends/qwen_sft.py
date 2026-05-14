from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verl_mint.backends.base import InferenceBackend, TrainingBackend
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
from verl_mint.model_registry import get_model_config
from verl_mint.errors import UnsupportedOperationError
from verl_mint.milestone1 import MILESTONE1_BASE_MODEL_ID, MILESTONE1_MODEL_PATH_ENV
from verl_mint.storage import LocalStorageRepo

try:
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception as _RUNTIME_IMPORT_ERROR:  # pragma: no cover - runtime-only dependency gate
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    LoraConfig = None
    PeftModel = None
    get_peft_model = None
else:  # pragma: no cover - import path only
    _RUNTIME_IMPORT_ERROR = None


def _require_runtime() -> None:
    if _RUNTIME_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Qwen SFT backend requires optional deps: "
            "pip install 'verl-mint[qwen-sft]'"
        ) from _RUNTIME_IMPORT_ERROR


def _resolve_model_ref(base_model: str | None) -> str:
    return os.environ.get(MILESTONE1_MODEL_PATH_ENV, base_model or MILESTONE1_BASE_MODEL_ID)


def _resolve_hidden_size(model: Any) -> int:
    candidates = [
        getattr(getattr(model, "config", None), "hidden_size", None),
        getattr(getattr(getattr(model, "base_model", None), "config", None), "hidden_size", None),
        getattr(
            getattr(getattr(getattr(model, "base_model", None), "model", None), "config", None),
            "hidden_size",
            None,
        ),
    ]
    for value in candidates:
        if value is not None:
            return int(value)
    raise RuntimeError("could not infer hidden_size from qwen model config")


def _build_lora_config(rank: int) -> Any:
    return LoraConfig(
        r=rank,
        lora_alpha=max(16, rank * 2),
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )


def _maybe_wrap_with_lora(model: Any, rank: int) -> Any:
    if get_peft_model is None:
        return model
    wrapped = get_peft_model(model, _build_lora_config(rank))
    wrapped.train()
    return wrapped


def _has_lora_state(state: dict[str, Any]) -> bool:
    return any("lora_" in key or "modules_to_save" in key for key in state)


def _to_int_tokens(prompt: str) -> list[int]:
    return [int(tok) for tok in prompt.split() if tok.lstrip("-").isdigit()]


def _chunk_tokens(model_input: Any) -> list[int]:
    chunks = model_input.get("chunks", []) if hasattr(model_input, "get") else []
    tokens: list[int] = []
    for chunk in chunks:
        if hasattr(chunk, "get") and chunk.get("type") == "encoded_text":
            tokens.extend(int(tok) for tok in chunk.get("tokens", []))
    return tokens


def _tensor_values(value: Any) -> list[float]:
    if hasattr(value, "get"):
        data = value.get("data", [])
        if isinstance(data, (list, tuple)):
            return [float(x) for x in data]
        return [float(data)]
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    if value is None:
        return []
    return [float(value)]


def _expand_token_values(
    values: tuple[float, ...],
    length: int,
    *,
    default: float,
    name: str,
) -> tuple[float, ...]:
    if length <= 0:
        return ()
    data = [float(x) for x in values]
    if not data:
        return tuple(default for _ in range(length))
    if len(data) == 1 and length > 1:
        data = data * length
    if len(data) != length:
        raise ValueError(f"{name} length {len(data)} does not match completion length {length}")
    return tuple(data)


def _weighted_mean(values: Any, weights: Any) -> Any:
    denom = weights.sum().clamp_min(1e-8)
    return (values * weights).sum() / denom


def _batch_loss_fn(batch_payload: Any) -> str:
    batch = dict(batch_payload) if hasattr(batch_payload, "items") else {}
    return str(batch.get("loss_fn") or "").lower()


def _tensor_ints(value: Any) -> list[int]:
    return [int(x) for x in _tensor_values(value)]


def _group_advantages(rewards: list[float], group_ids: list[str]) -> list[float]:
    groups: dict[str, list[int]] = {}
    for i, group_id in enumerate(group_ids):
        groups.setdefault(group_id, []).append(i)

    advantages = [0.0] * len(rewards)
    for indices in groups.values():
        vals = [rewards[i] for i in indices]
        if len(indices) == 1:
            advantages[indices[0]] = float(vals[0])
            continue
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = var ** 0.5
        if std < 1e-6:
            for idx in indices:
                advantages[idx] = 0.0
            continue
        for idx, value in zip(indices, vals):
            advantages[idx] = (value - mean) / std
    return advantages


def _clear_model_cache(cache: dict[Any, Any]) -> None:
    cache.clear()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


@dataclass
class _TrainSession:
    model: Any
    tokenizer: Any
    optimizer: Any
    value_head: Any
    device: Any
    step: int = 0
    last_loss: float = 0.0
    base_model: str = MILESTONE1_BASE_MODEL_ID
    lora_rank: int = 16


@dataclass(frozen=True)
class _RLSample:
    prompt_tokens: tuple[int, ...]
    reference_prompt_tokens: tuple[int, ...]
    completion_tokens: tuple[int, ...]
    old_logprobs: tuple[float, ...]
    reference_logprobs: tuple[float, ...]
    old_values: tuple[float, ...]
    advantages: tuple[float, ...]
    returns: tuple[float, ...]
    weights: tuple[float, ...]
    reward: float
    group_id: str
    sample_id: str


@dataclass(frozen=True)
class _PolicyTrace:
    logprobs: Any
    entropies: Any
    values: Any


class QwenSFTTrainingBackend(TrainingBackend):
    def __init__(
        self,
        *,
        model_id: str = MILESTONE1_BASE_MODEL_ID,
        learning_rate: float = 3e-4,
    ) -> None:
        _require_runtime()
        self.model_id = model_id
        self.learning_rate = learning_rate
        self.sessions: dict[str, _TrainSession] = {}
        self._reference_models: dict[str, Any] = {}

    def open_session(self, spec: SessionSpec) -> SessionHandle:
        metadata = dict(spec.metadata)
        base_model = str(metadata.get("base_model") or self.model_id)
        try:
            cfg = get_model_config(base_model)
        except ValueError:
            cfg = None
        if cfg is not None and cfg.is_moe:
            raise RuntimeError(
                f"{base_model} is a MoE model and requires the TP={cfg.train_tp} Megatron training backend; "
                "QwenSFTTrainingBackend is single-process and must not host 30B MoE training"
            )
        model_ref = _resolve_model_ref(base_model)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_ref,
            dtype=dtype,
            trust_remote_code=True,
        )
        model.to(device)
        model.train()

        lora_cfg = metadata.get("lora_config") or {}
        rank = int(lora_cfg.get("rank", 16)) if hasattr(lora_cfg, "get") else 16
        if rank > 0 and get_peft_model is not None:
            model = _maybe_wrap_with_lora(model, rank)

        hidden_size = _resolve_hidden_size(model)
        value_head = torch.nn.Linear(hidden_size, 1, bias=True)
        value_head.to(device=device, dtype=dtype)
        value_head.train()

        params = [p for p in model.parameters() if p.requires_grad]
        params.extend(value_head.parameters())
        optimizer = torch.optim.AdamW(params, lr=self.learning_rate)
        self.sessions[spec.session_id] = _TrainSession(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            value_head=value_head,
            device=device,
            base_model=base_model,
            lora_rank=rank,
        )
        return SessionHandle(backend_session_id=spec.session_id)

    def close_session(self, handle: SessionHandle) -> None:
        if handle.backend_session_id is None:
            return
        self.sessions.pop(handle.backend_session_id, None)
        _clear_model_cache(self._reference_models)

    def capabilities(self) -> TrainingCapabilities:
        return TrainingCapabilities(
            supports_forward=True,
            supports_train_step=True,
            supports_reverse_kl=True,
            supports_tokenizer_info=True,
            supports_reset_expert_bias=False,
            supports_checkpoint_load=True,
            supports_checkpoint_save=True,
            extras={"loss_fns": ["cross_entropy", "sft", "dpo", "ppo", "grpo", "reverse_kl"]},
        )

    def get_tokenizer_info(self, handle: SessionHandle) -> TokenizerInfo:
        session = self._session(handle)
        return TokenizerInfo(
            metadata={
                "bos": int(session.tokenizer.bos_token_id or -1),
                "eos": int(session.tokenizer.eos_token_id or -1),
                "pad": int(session.tokenizer.pad_token_id or -1),
                "vocab_size": int(session.tokenizer.vocab_size),
                "base_model": session.base_model,
            }
        )

    def forward(self, handle: SessionHandle, req: TrainOpRequest) -> TrainOpResult:
        session = self._session(handle)
        if _batch_loss_fn(req.batch_payload) == "dpo":
            return self._compute_dpo(session, req.batch_payload, backward=False)
        loss, token_count = self._compute_loss(session, req.batch_payload, backward=False)
        session.last_loss = loss
        return TrainOpResult(
            state=TrainState(step=session.step, extras={"loss": loss}),
            outputs={"loss": loss, "num_tokens": token_count, "mode": "forward"},
        )

    def run_train_op(self, handle: SessionHandle, req: TrainOpRequest) -> TrainOpResult:
        session = self._session(handle)
        if req.op is TrainOp.FORWARD_BACKWARD_PPO:
            return self.forward_backward_ppo(handle, req)
        if req.op is TrainOp.FORWARD_BACKWARD_REVERSE_KL:
            return self.forward_backward_reverse_kl(handle, req)
        if req.op in {TrainOp.FORWARD_BACKWARD, TrainOp.TRAIN_STEP, TrainOp.CUSTOM}:
            if _batch_loss_fn(req.batch_payload) == "dpo":
                result = self._compute_dpo(session, req.batch_payload, backward=True)
                session.step += 1
                if req.op in {TrainOp.TRAIN_STEP, TrainOp.CUSTOM}:
                    session.optimizer.step()
                    session.optimizer.zero_grad(set_to_none=True)
                session.last_loss = float(result.outputs["loss"])
                outputs = {**dict(result.outputs), "op": req.op.value}
                return TrainOpResult(
                    state=TrainState(step=session.step, extras={"loss": session.last_loss}),
                    outputs=outputs,
                )
            loss, token_count = self._compute_loss(session, req.batch_payload, backward=True)
            session.step += 1
            if req.op in {TrainOp.TRAIN_STEP, TrainOp.CUSTOM}:
                session.optimizer.step()
                session.optimizer.zero_grad(set_to_none=True)
            session.last_loss = loss
            return TrainOpResult(
                state=TrainState(step=session.step, extras={"loss": loss}),
                outputs={"loss": loss, "num_tokens": token_count, "op": req.op.value},
            )

        if req.op is TrainOp.OPTIMIZER_STEP:
            session.optimizer.step()
            session.optimizer.zero_grad(set_to_none=True)
            session.step += 1
            return TrainOpResult(
                state=TrainState(step=session.step, extras={"loss": session.last_loss}),
                outputs={"loss": session.last_loss, "op": req.op.value},
            )

        raise UnsupportedOperationError(f"unsupported train op for qwen-sft backend: {req.op.value}")

    def forward_backward_ppo(self, handle: SessionHandle, req: TrainOpRequest) -> TrainOpResult:
        session = self._session(handle)
        batch = dict(req.batch_payload) if hasattr(req.batch_payload, "items") else {}
        items = batch.get("data") or batch.get("samples") or []
        if not items:
            raise ValueError("PPO training requires non-empty RL batch data")

        options = dict(req.options)
        clip_coef = float(batch.get("clip_coef", options.get("clip_coef", options.get("cliprange", 0.2))))
        value_coef = float(batch.get("value_coef", options.get("value_coef", options.get("vf_coef", 0.5))))
        entropy_coef = float(batch.get("entropy_coef", options.get("entropy_coef", 0.0)))
        value_clip = float(batch.get("value_clip", options.get("value_clip", options.get("cliprange_value", 0.2))))

        samples = [self._rl_sample(item, i) for i, item in enumerate(items)]
        rewards = [sample.reward for sample in samples]
        default_advantages = _group_advantages(rewards, [sample.group_id for sample in samples])

        sample_losses = []
        policy_losses = []
        value_losses = []
        entropy_terms = []
        approx_kls = []
        clipfracs = []
        return_means = []
        value_means = []
        token_count = 0

        for sample, default_advantage in zip(samples, default_advantages):
            trace = self._completion_trace(session, sample.prompt_tokens, sample.completion_tokens)
            n = int(trace.logprobs.numel())
            if sample.old_logprobs:
                old_logprobs = torch.tensor(
                    _expand_token_values(sample.old_logprobs, n, default=0.0, name="old_logprobs"),
                    dtype=trace.logprobs.dtype,
                    device=session.device,
                )
            else:
                old_logprobs = trace.logprobs.detach()

            weights = torch.tensor(
                _expand_token_values(sample.weights, n, default=1.0, name="weights"),
                dtype=trace.logprobs.dtype,
                device=session.device,
            )
            advantages = torch.tensor(
                _expand_token_values(sample.advantages, n, default=float(default_advantage), name="advantages"),
                dtype=trace.logprobs.dtype,
                device=session.device,
            )
            if sample.returns:
                returns_values = _expand_token_values(sample.returns, n, default=sample.reward, name="returns")
            else:
                returns_values = tuple(sample.reward for _ in range(n))
            returns = torch.tensor(returns_values, dtype=trace.values.dtype, device=session.device)

            has_old_values = bool(sample.old_values)
            old_values = torch.tensor(
                _expand_token_values(sample.old_values, n, default=0.0, name="old_values"),
                dtype=trace.values.dtype,
                device=session.device,
            )

            logratio = trace.logprobs - old_logprobs
            ratio = torch.exp(logratio)
            clipped_ratio = ratio.clamp(1.0 - clip_coef, 1.0 + clip_coef)
            policy_terms = -torch.minimum(ratio * advantages, clipped_ratio * advantages)
            policy_loss = _weighted_mean(policy_terms, weights)

            value_error = (trace.values - returns) ** 2
            if has_old_values and value_clip > 0.0:
                clipped_values = old_values + (trace.values - old_values).clamp(-value_clip, value_clip)
                clipped_error = (clipped_values - returns) ** 2
                value_terms = 0.5 * torch.maximum(value_error, clipped_error)
            else:
                value_terms = 0.5 * value_error
            value_loss = _weighted_mean(value_terms, weights)

            entropy_bonus = _weighted_mean(trace.entropies, weights)
            total = policy_loss + value_coef * value_loss - entropy_coef * entropy_bonus

            clip_mask = (torch.abs(ratio - 1.0) > clip_coef).to(trace.logprobs.dtype)
            sample_losses.append(total)
            policy_losses.append(policy_loss.detach())
            value_losses.append(value_loss.detach())
            entropy_terms.append(entropy_bonus.detach())
            approx_kls.append((0.5 * _weighted_mean(logratio * logratio, weights)).detach())
            clipfracs.append(_weighted_mean(clip_mask, weights).detach())
            return_means.append(_weighted_mean(returns, weights).detach())
            value_means.append(_weighted_mean(trace.values.detach(), weights).detach())
            token_count += n

        total_loss = torch.stack(sample_losses).mean()
        total_loss.backward()
        session.step += 1
        session.last_loss = float(total_loss.detach().item())

        reward_mean = sum(rewards) / len(rewards)
        adv_mean = sum(float(x) for x in default_advantages) / len(default_advantages)
        per_sample_loss = [float(loss.detach().item()) for loss in sample_losses]
        policy_mean = torch.stack(policy_losses).mean().item()
        value_mean = torch.stack(value_losses).mean().item()
        entropy_mean = torch.stack(entropy_terms).mean().item()
        approx_kl_mean = torch.stack(approx_kls).mean().item()
        clipfrac_mean = torch.stack(clipfracs).mean().item()
        return_mean = torch.stack(return_means).mean().item()
        value_pred_mean = torch.stack(value_means).mean().item()

        return TrainOpResult(
            state=TrainState(
                step=session.step,
                extras={
                    "loss": session.last_loss,
                    "policy_loss": float(policy_mean),
                    "value_loss": float(value_mean),
                    "entropy": float(entropy_mean),
                    "approx_kl": float(approx_kl_mean),
                    "clipfrac": float(clipfrac_mean),
                    "reward_mean": float(reward_mean),
                    "adv_mean": float(adv_mean),
                    "return_mean": float(return_mean),
                    "value_mean": float(value_pred_mean),
                },
            ),
            outputs={
                "loss": session.last_loss,
                "policy_loss": float(policy_mean),
                "value_loss": float(value_mean),
                "entropy": float(entropy_mean),
                "approx_kl": float(approx_kl_mean),
                "clipfrac": float(clipfrac_mean),
                "reward_mean": float(reward_mean),
                "adv_mean": float(adv_mean),
                "return_mean": float(return_mean),
                "value_mean": float(value_pred_mean),
                "num_tokens": token_count,
                "algorithm": str(batch.get("algorithm") or options.get("algo") or "ppo"),
                "sequence_losses": tuple(per_sample_loss),
            },
        )

    def forward_backward_reverse_kl(self, handle: SessionHandle, req: TrainOpRequest) -> TrainOpResult:
        session = self._session(handle)
        batch = dict(req.batch_payload) if hasattr(req.batch_payload, "items") else {}
        items = batch.get("data") or batch.get("samples") or []
        if not items:
            raise ValueError("reverse KL training requires non-empty RL batch data")

        options = dict(req.options)
        kl_coef = float(batch.get("kl_coef", options.get("kl_coef", options.get("beta", 0.05))))
        temperature = float(batch.get("temperature", options.get("temperature", 1.0)))
        reference_model_path = batch.get("reference_model_path") or options.get("reference_model_path")
        samples = [self._rl_sample(item, i) for i, item in enumerate(items)]
        rewards = [sample.reward for sample in samples]
        advantages = _group_advantages(rewards, [sample.group_id for sample in samples])

        sample_losses = []
        policy_losses = []
        kl_losses = []
        token_count = 0

        for sample, advantage in zip(samples, advantages):
            trace = self._completion_trace(
                session,
                sample.prompt_tokens,
                sample.completion_tokens,
                temperature=temperature,
            )
            n = int(trace.logprobs.numel())
            weights = torch.tensor(
                _expand_token_values(sample.weights, n, default=1.0, name="weights"),
                dtype=trace.logprobs.dtype,
                device=session.device,
            )
            if sample.old_logprobs:
                old = torch.tensor(
                    _expand_token_values(sample.old_logprobs, n, default=0.0, name="old_logprobs"),
                    dtype=trace.logprobs.dtype,
                    device=session.device,
                )
            else:
                old = trace.logprobs.detach()
            if sample.reference_logprobs:
                ref = torch.tensor(
                    _expand_token_values(sample.reference_logprobs, n, default=0.0, name="reference_logprobs"),
                    dtype=trace.logprobs.dtype,
                    device=session.device,
                )
            else:
                if not reference_model_path:
                    raise ValueError("reverse KL requires reference_logprobs or reference_model_path")
                ref = self._reference_completion_logprobs(
                    session,
                    str(reference_model_path),
                    sample.reference_prompt_tokens,
                    sample.completion_tokens,
                    temperature=temperature,
                )
                if int(ref.numel()) != n:
                    raise ValueError("reference logprob length does not match completion length")

            ratio = torch.exp(trace.logprobs - old)
            adv = torch.tensor(float(advantage), dtype=trace.logprobs.dtype, device=session.device)
            policy_loss = -_weighted_mean(ratio * adv, weights)

            delta = ref - trace.logprobs
            kl = torch.exp(delta) - delta - 1.0
            kl_loss = _weighted_mean(kl, weights)
            total = policy_loss + kl_coef * kl_loss

            sample_losses.append(total)
            policy_losses.append(policy_loss.detach())
            kl_losses.append(kl_loss.detach())
            token_count += n

        total_loss = torch.stack(sample_losses).mean()
        total_loss.backward()
        session.step += 1
        session.last_loss = float(total_loss.detach().item())

        reward_mean = sum(rewards) / len(rewards)
        adv_mean = sum(float(x) for x in advantages) / len(advantages)
        per_sample_loss = [float(loss.detach().item()) for loss in sample_losses]
        policy_mean = torch.stack(policy_losses).mean().item()
        kl_mean = torch.stack(kl_losses).mean().item()

        return TrainOpResult(
            state=TrainState(
                step=session.step,
                extras={
                    "loss": session.last_loss,
                    "policy_loss": float(policy_mean),
                    "kl": float(kl_mean),
                    "reward_mean": float(reward_mean),
                    "adv_mean": float(adv_mean),
                },
            ),
            outputs={
                "loss": session.last_loss,
                "policy_loss": float(policy_mean),
                "kl": float(kl_mean),
                "reward_mean": float(reward_mean),
                "adv_mean": float(adv_mean),
                "num_tokens": token_count,
                "algorithm": str(batch.get("algorithm") or options.get("algo") or "grpo"),
                "sequence_losses": tuple(per_sample_loss),
            },
        )

    def reset_expert_bias(self, handle: SessionHandle) -> TrainOpResult:
        raise UnsupportedOperationError("reset_expert_bias is out of scope for qwen backend")

    def save_checkpoint(self, handle: SessionHandle, req: CheckpointRequest) -> ArtifactRef:
        session = self._session(handle)
        path = self._artifact_path_for_write(req.uri)

        model_state = session.model.state_dict()
        if PeftModel is not None and isinstance(session.model, PeftModel):
            model_state = {
                k: v.detach().cpu()
                for k, v in model_state.items()
                if "lora_" in k or "modules_to_save" in k
            }
        else:
            model_state = {k: v.detach().cpu() for k, v in model_state.items() if v.requires_grad}

        payload: dict[str, Any] = {
            "step": session.step,
            "last_loss": session.last_loss,
            "base_model": session.base_model,
            "lora_rank": session.lora_rank,
            "model_state": model_state,
            "value_head_state": {k: v.detach().cpu() for k, v in session.value_head.state_dict().items()},
        }
        if req.include_optimizer:
            payload["optimizer_state"] = session.optimizer.state_dict()

        torch.save(payload, path)
        return ArtifactRef(uri=req.uri, format="pt", metadata={"step": session.step})

    def load_checkpoint(self, handle: SessionHandle, req: CheckpointRequest) -> TrainState:
        session = self._session(handle)
        path = self._artifact_path_for_read(req.uri)
        payload = torch.load(path, map_location=session.device)
        model_state = payload.get("model_state", {})
        if not model_state:
            raise ValueError("checkpoint is missing model_state")

        checkpoint_base_model = str(payload.get("base_model") or session.base_model)
        if checkpoint_base_model != session.base_model:
            raise ValueError(
                f"checkpoint base_model mismatch: expected={session.base_model} got={checkpoint_base_model}"
            )
        checkpoint_lora_rank = int(payload.get("lora_rank", session.lora_rank))
        if checkpoint_lora_rank != session.lora_rank:
            raise ValueError(
                f"checkpoint lora_rank mismatch: expected={session.lora_rank} got={checkpoint_lora_rank}"
            )

        has_lora_state = _has_lora_state(model_state)
        uses_lora_runtime = PeftModel is not None and isinstance(session.model, PeftModel)
        if has_lora_state != uses_lora_runtime:
            raise ValueError(
                f"checkpoint adapter topology mismatch: runtime_lora={uses_lora_runtime} checkpoint_lora={has_lora_state}"
            )

        incompatible = session.model.load_state_dict(model_state, strict=False)
        unexpected = list(getattr(incompatible, "unexpected_keys", []))
        if unexpected:
            raise ValueError(f"checkpoint contains unexpected model keys: {unexpected[:5]}")
        missing = list(getattr(incompatible, "missing_keys", []))
        required_missing = [
            key for key in missing if "lora_" in key or "modules_to_save" in key
        ]
        if required_missing:
            raise ValueError(f"checkpoint is missing adapter keys: {required_missing[:5]}")

        value_head_state = payload.get("value_head_state") or {}
        if not value_head_state:
            raise ValueError("checkpoint is missing value_head_state")
        session.value_head.load_state_dict(value_head_state, strict=True)

        if req.include_optimizer and "optimizer_state" in payload:
            session.optimizer.load_state_dict(payload["optimizer_state"])

        session.step = int(payload.get("step", session.step))
        session.last_loss = float(payload.get("last_loss", session.last_loss))
        return TrainState(step=session.step, extras={"loss": session.last_loss})

    def _session(self, handle: SessionHandle) -> _TrainSession:
        if handle.backend_session_id is None or handle.backend_session_id not in self.sessions:
            raise KeyError(f"unknown backend session: {handle.backend_session_id}")
        return self.sessions[handle.backend_session_id]

    def _artifact_path_for_write(self, uri: str) -> Path:
        if uri.startswith(("mint://", "repo://")):
            return Path(LocalStorageRepo.from_env().resolve_for_write(uri))
        path = Path(uri).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _artifact_path_for_read(self, uri: str) -> Path:
        if uri.startswith(("mint://", "repo://")):
            return Path(LocalStorageRepo.from_env().resolve_for_read(uri))
        return Path(uri).expanduser()

    def _compute_loss(self, session: _TrainSession, batch_payload: Any, *, backward: bool) -> tuple[float, int]:
        batch = dict(batch_payload) if hasattr(batch_payload, "items") else {}
        data = batch.get("data") or []
        if not data:
            raise ValueError("qwen-sft backend requires non-empty batch payload data")

        input_rows: list[list[int]] = []
        label_rows: list[list[int]] = []

        for item in data:
            model_input = item.get("model_input", {}) if hasattr(item, "get") else {}
            tokens = _chunk_tokens(model_input)
            if not tokens:
                continue

            loss_inputs = item.get("loss_fn_inputs", {}) if hasattr(item, "get") else {}
            target = loss_inputs.get("target_tokens") if hasattr(loss_inputs, "get") else None
            if hasattr(target, "get"):
                labels = [int(x) for x in target.get("data", [])]
            else:
                raw_labels = loss_inputs.get("labels", tokens) if hasattr(loss_inputs, "get") else tokens
                labels = [int(x) for x in raw_labels]

            if len(labels) < len(tokens):
                labels = labels + [-100] * (len(tokens) - len(labels))
            if len(labels) > len(tokens):
                labels = labels[: len(tokens)]

            loss_mask = loss_inputs.get("loss_mask") if hasattr(loss_inputs, "get") else None
            if hasattr(loss_mask, "get"):
                mask_vals = [float(v) for v in loss_mask.get("data", [])]
                for i, mask_value in enumerate(mask_vals[: len(labels)]):
                    if mask_value <= 0.0:
                        labels[i] = -100

            input_rows.append(tokens)
            label_rows.append(labels)

        if not input_rows:
            raise ValueError("qwen-sft backend could not extract tokenized rows from batch payload")

        pad_id = int(session.tokenizer.pad_token_id or session.tokenizer.eos_token_id or 0)
        max_len = max(len(row) for row in input_rows)
        input_ids = []
        labels = []
        attention_mask = []
        token_count = 0

        for tokens, labels_row in zip(input_rows, label_rows):
            pad_len = max_len - len(tokens)
            input_ids.append(tokens + [pad_id] * pad_len)
            labels.append(labels_row + [-100] * pad_len)
            attention_mask.append([1] * len(tokens) + [0] * pad_len)
            token_count += sum(1 for value in labels_row if value != -100)

        input_ids_t = torch.tensor(input_ids, dtype=torch.long, device=session.device)
        labels_t = torch.tensor(labels, dtype=torch.long, device=session.device)
        attention_mask_t = torch.tensor(attention_mask, dtype=torch.long, device=session.device)

        outputs = session.model(
            input_ids=input_ids_t,
            attention_mask=attention_mask_t,
            labels=labels_t,
        )
        loss = outputs.loss
        if backward:
            loss.backward()
        return float(loss.detach().item()), int(token_count)

    def _compute_dpo(self, session: _TrainSession, batch_payload: Any, *, backward: bool) -> TrainOpResult:
        batch = dict(batch_payload) if hasattr(batch_payload, "items") else {}
        data = batch.get("data") or []
        if not data:
            raise ValueError("DPO training requires non-empty batch payload data")

        cfg = batch.get("loss_fn_config") or {}
        if not hasattr(cfg, "get"):
            raise ValueError("DPO loss_fn_config must be a mapping")
        beta = float(cfg.get("beta", 0.1))
        reference_model_path = cfg.get("reference_model_path")
        temperature = float(cfg.get("temperature", 1.0))

        rows = [self._dpo_row(item, i) for i, item in enumerate(data)]
        pairs: dict[str, dict[str, dict[str, Any]]] = {}
        for row in rows:
            pair = pairs.setdefault(row["pair_id"], {})
            role = row["role"]
            if role in pair:
                raise ValueError(f"DPO pair {row['pair_id']} has duplicate {role} row")
            pair[role] = row

        losses = []
        pair_records = []
        total_tokens = 0

        for pair_id, pair in pairs.items():
            if set(pair) != {"chosen", "rejected"}:
                raise ValueError(f"DPO pair {pair_id} must contain exactly one chosen and one rejected row")
            chosen = pair["chosen"]
            rejected = pair["rejected"]

            chosen_trace = self._completion_trace(
                session,
                chosen["prompt_tokens"],
                chosen["completion_tokens"],
                temperature=temperature,
            )
            rejected_trace = self._completion_trace(
                session,
                rejected["prompt_tokens"],
                rejected["completion_tokens"],
                temperature=temperature,
            )
            chosen_weights = self._dpo_weights(session, chosen, int(chosen_trace.logprobs.numel()))
            rejected_weights = self._dpo_weights(session, rejected, int(rejected_trace.logprobs.numel()))

            chosen_pi = (chosen_trace.logprobs * chosen_weights).sum()
            rejected_pi = (rejected_trace.logprobs * rejected_weights).sum()
            chosen_ref = self._dpo_reference_sum(
                session,
                chosen,
                chosen_weights,
                reference_model_path,
                temperature=temperature,
            )
            rejected_ref = self._dpo_reference_sum(
                session,
                rejected,
                rejected_weights,
                reference_model_path,
                temperature=temperature,
            )

            pi_margin = chosen_pi - rejected_pi
            ref_margin = chosen_ref - rejected_ref
            reward_margin = beta * (pi_margin - ref_margin)
            pair_loss = -torch.nn.functional.logsigmoid(reward_margin)
            losses.append(pair_loss)
            total_tokens += int((chosen_weights > 0).sum().item() + (rejected_weights > 0).sum().item())

            record = {
                "pair_id": pair_id,
                "loss": float(pair_loss.detach().item()),
                "pi_margin": float(pi_margin.detach().item()),
                "ref_margin": float(ref_margin.detach().item()),
                "reward_margin": float(reward_margin.detach().item()),
                "chosen_logprob": float(chosen_pi.detach().item()),
                "rejected_logprob": float(rejected_pi.detach().item()),
                "chosen_ref_logprob": float(chosen_ref.detach().item()),
                "rejected_ref_logprob": float(rejected_ref.detach().item()),
                "accuracy": 1.0 if float(reward_margin.detach().item()) > 0.0 else 0.0,
            }
            pair_records.append(record)

        if not losses:
            raise ValueError("DPO training requires at least one complete preference pair")
        if total_tokens <= 0:
            raise ValueError("DPO training requires at least one nonmasked completion token")

        total_loss = torch.stack(losses).mean()
        if backward:
            total_loss.backward()
        loss_value = float(total_loss.detach().item())
        session.last_loss = loss_value

        def mean(key: str) -> float:
            return sum(float(record[key]) for record in pair_records) / len(pair_records)

        outputs = {
            "loss": loss_value,
            "dpo_loss": loss_value,
            "pi_margin": mean("pi_margin"),
            "ref_margin": mean("ref_margin"),
            "reward_margin": mean("reward_margin"),
            "accuracy": mean("accuracy"),
            "chosen_logprob": mean("chosen_logprob"),
            "rejected_logprob": mean("rejected_logprob"),
            "chosen_ref_logprob": mean("chosen_ref_logprob"),
            "rejected_ref_logprob": mean("rejected_ref_logprob"),
            "beta": beta,
            "num_pairs": len(pair_records),
            "num_tokens": total_tokens,
            "algorithm": "dpo",
        }
        return TrainOpResult(
            state=TrainState(step=session.step, extras={"loss": loss_value}),
            outputs=outputs,
        )

    def _dpo_row(self, item: Any, index: int) -> dict[str, Any]:
        if not hasattr(item, "get"):
            raise ValueError("DPO datum must be a mapping")
        loss_inputs = item.get("loss_fn_inputs", {})
        if not hasattr(loss_inputs, "get"):
            raise ValueError("DPO loss_fn_inputs must be a mapping")

        pair_id = str(loss_inputs.get("pair_id") or "")
        if not pair_id:
            raise ValueError("DPO datum is missing pair_id")
        role = str(loss_inputs.get("role") or "").lower()
        if role not in {"chosen", "rejected"}:
            raise ValueError("DPO datum role must be chosen or rejected")

        prompt_tokens = _tensor_ints(loss_inputs.get("prompt_tokens"))
        completion_tokens = _tensor_ints(loss_inputs.get("completion_tokens")) or _tensor_ints(loss_inputs.get("target_tokens"))
        if not prompt_tokens:
            raise ValueError("DPO datum is missing prompt_tokens")
        if not completion_tokens:
            raise ValueError("DPO datum is missing completion_tokens")

        weights_source = loss_inputs.get("weights") or loss_inputs.get("loss_mask") or loss_inputs.get("mask")
        weights = tuple(_tensor_values(weights_source))
        reference_logprobs = tuple(_tensor_values(loss_inputs.get("reference_logprobs")))
        return {
            "index": index,
            "pair_id": pair_id,
            "role": role,
            "prompt_tokens": tuple(prompt_tokens),
            "completion_tokens": tuple(completion_tokens),
            "weights": weights,
            "reference_logprobs": reference_logprobs,
        }

    def _dpo_weights(self, session: _TrainSession, row: dict[str, Any], length: int) -> Any:
        weights = _expand_token_values(row["weights"], length, default=1.0, name="weights")
        if not any(float(x) > 0.0 for x in weights):
            raise ValueError("DPO row has zero nonmasked completion tokens")
        return torch.tensor(weights, dtype=torch.float32, device=session.device)

    def _dpo_reference_sum(
        self,
        session: _TrainSession,
        row: dict[str, Any],
        weights: Any,
        reference_model_path: Any,
        *,
        temperature: float,
    ) -> Any:
        if row["reference_logprobs"]:
            values = _expand_token_values(
                row["reference_logprobs"],
                int(weights.numel()),
                default=0.0,
                name="reference_logprobs",
            )
            ref = torch.tensor(values, dtype=torch.float32, device=session.device)
        else:
            if not reference_model_path:
                raise ValueError("DPO requires reference_logprobs or reference_model_path")
            ref = self._reference_completion_logprobs(
                session,
                str(reference_model_path),
                row["prompt_tokens"],
                row["completion_tokens"],
                temperature=temperature,
            )
            if int(ref.numel()) != int(weights.numel()):
                raise ValueError("reference logprob length does not match completion length")
        return (ref * weights).sum()

    def _rl_sample(self, item: Any, sample_index: int) -> _RLSample:
        if not hasattr(item, "get"):
            raise ValueError("RL datum must be a mapping")

        prompt_tokens = [int(x) for x in item.get("prompt_tokens", [])]
        if not prompt_tokens:
            prompt_tokens = _chunk_tokens(item.get("student_input", {}))
        if not prompt_tokens:
            raise ValueError("RL datum is missing prompt tokens")

        reference_prompt_tokens = [int(x) for x in item.get("reference_prompt_tokens", [])]
        if not reference_prompt_tokens:
            reference_prompt_tokens = _chunk_tokens(item.get("reference_input", {}))
        if not reference_prompt_tokens:
            reference_prompt_tokens = list(prompt_tokens)

        completion_tokens = [int(x) for x in item.get("completion_tokens", [])]
        if not completion_tokens:
            target = item.get("target_tokens")
            completion_tokens = [int(x) for x in _tensor_values(target)]
        if not completion_tokens:
            raise ValueError("RL datum is missing completion tokens")

        old_logprobs = tuple(_tensor_values(item.get("old_logprobs")))
        reference_logprobs = tuple(_tensor_values(item.get("reference_logprobs")))

        old_values = tuple(_tensor_values(item.get("old_values") or item.get("values")))
        advantages = tuple(_tensor_values(item.get("advantages") or item.get("advantage")))
        returns = tuple(_tensor_values(item.get("returns") or item.get("value_targets")))
        weights = tuple(_tensor_values(item.get("weights")))
        reward = float(item.get("reward", 0.0))
        group_id = str(item.get("group_id") or "default")
        sample_id = str(item.get("sample_id") or sample_index)

        return _RLSample(
            prompt_tokens=tuple(prompt_tokens),
            reference_prompt_tokens=tuple(reference_prompt_tokens),
            completion_tokens=tuple(completion_tokens),
            old_logprobs=tuple(float(x) for x in old_logprobs),
            reference_logprobs=tuple(float(x) for x in reference_logprobs),
            old_values=tuple(float(x) for x in old_values),
            advantages=tuple(float(x) for x in advantages),
            returns=tuple(float(x) for x in returns),
            weights=tuple(float(x) for x in weights),
            reward=reward,
            group_id=group_id,
            sample_id=sample_id,
        )

    def _completion_trace(
        self,
        session: _TrainSession,
        prompt_tokens: tuple[int, ...],
        completion_tokens: tuple[int, ...],
        *,
        temperature: float = 1.0,
    ) -> _PolicyTrace:
        full_tokens = list(prompt_tokens) + list(completion_tokens)
        if len(full_tokens) < 2:
            raise ValueError("RL sample must contain at least two tokens")

        input_ids = torch.tensor([full_tokens[:-1]], dtype=torch.long, device=session.device)
        attention_mask = torch.ones_like(input_ids)
        target_ids = torch.tensor(full_tokens[1:], dtype=torch.long, device=session.device)

        outputs = session.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        scale = max(1e-5, float(temperature))
        logits = (outputs.logits[0].float()) / scale
        logprobs = torch.log_softmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        gathered = logprobs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        entropies = -(probs * logprobs).sum(dim=-1)

        hidden = outputs.hidden_states[-1][0]
        values = session.value_head(hidden).squeeze(-1).float()

        start = max(0, len(prompt_tokens) - 1)
        end = start + len(completion_tokens)
        return _PolicyTrace(
            logprobs=gathered[start:end],
            entropies=entropies[start:end],
            values=values[start:end],
        )

    def _reference_completion_logprobs(
        self,
        session: _TrainSession,
        reference_model_path: str,
        prompt_tokens: tuple[int, ...],
        completion_tokens: tuple[int, ...],
        *,
        temperature: float = 1.0,
    ) -> Any:
        model = self._reference_model(session, reference_model_path)
        full_tokens = list(prompt_tokens) + list(completion_tokens)
        if len(full_tokens) < 2:
            raise ValueError("reference sample must contain at least two tokens")
        input_ids = torch.tensor([full_tokens[:-1]], dtype=torch.long, device=session.device)
        attention_mask = torch.ones_like(input_ids)
        target_ids = torch.tensor(full_tokens[1:], dtype=torch.long, device=session.device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[0].float()
        logits = logits / max(1e-5, float(temperature))
        logprobs = torch.log_softmax(logits, dim=-1)
        gathered = logprobs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        start = max(0, len(prompt_tokens) - 1)
        end = start + len(completion_tokens)
        return gathered[start:end]

    def _reference_model(self, session: _TrainSession, reference_model_path: str) -> Any:
        cached = self._reference_models.get(reference_model_path)
        if cached is not None:
            return cached
        if self._reference_models:
            _clear_model_cache(self._reference_models)

        model, _ = _load_qwen_model_from_reference(
            reference_model_path,
            default_base_model=session.base_model,
            default_lora_rank=session.lora_rank,
            device=session.device,
            error_prefix="reference checkpoint",
        )
        model.eval()
        self._reference_models[reference_model_path] = model
        return model


def _resolve_artifact_reference(uri: str) -> str:
    if uri.startswith(("mint://", "repo://")):
        return LocalStorageRepo.from_env().resolve_for_read(uri)
    if uri.startswith("file://"):
        return str(Path(uri[len("file://") :]).expanduser().resolve())
    return uri


def _load_qwen_model_from_reference(
    reference: str,
    *,
    default_base_model: str,
    default_lora_rank: int,
    device: Any,
    error_prefix: str,
) -> tuple[Any, str]:
    resolved = _resolve_artifact_reference(reference)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    if Path(resolved).is_file():
        payload = torch.load(resolved, map_location=device)
        base_model = str(payload.get("base_model") or default_base_model)
        rank = int(payload.get("lora_rank", default_lora_rank))
        model = AutoModelForCausalLM.from_pretrained(
            _resolve_model_ref(base_model),
            dtype=dtype,
            trust_remote_code=True,
        )
        model.to(device)
        model_state = payload.get("model_state", {})
        if _has_lora_state(model_state):
            model = _maybe_wrap_with_lora(model, rank)
        incompatible = model.load_state_dict(model_state, strict=False)
        unexpected = list(getattr(incompatible, "unexpected_keys", []))
        if unexpected:
            raise ValueError(f"{error_prefix} has unexpected model keys: {unexpected[:5]}")
        return model, base_model

    base_model = str(resolved)
    model = AutoModelForCausalLM.from_pretrained(
        _resolve_model_ref(base_model),
        dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device)
    return model, base_model


class QwenTextInferenceBackend(InferenceBackend):
    def __init__(self, *, model_id: str = MILESTONE1_BASE_MODEL_ID) -> None:
        _require_runtime()
        self.model_id = model_id
        self.model_ref = _resolve_model_ref(model_id)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_ref, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_ref,
            dtype=dtype,
            trust_remote_code=True,
        )
        self.model.to(self.device)
        self.model.eval()
        self._tokenizers: dict[str, Any] = {self.model_ref: self.tokenizer}
        self._sampling_models: dict[str, tuple[Any, Any]] = {}

    def generate(self, req: GenerateRequest) -> GenerateResult:
        model, tokenizer = self._sampling_assets(req.sampling.get("model_path"))
        input_ids, attention_mask, prompt_tokens = self._prompt_inputs(req.prompt, tokenizer)
        mode = str(req.sampling.get("mode", "generate"))
        prompt_logprobs = tuple(self._prompt_logprobs(model, input_ids, attention_mask))

        if mode == "logprobs":
            return GenerateResult(
                text=req.prompt,
                prompt_token_ids=tuple(prompt_tokens),
                prompt_logprobs=prompt_logprobs,
                raw={"provider": "qwen-transformers", "mode": mode},
            )

        max_new_tokens = int(req.sampling.get("max_tokens", req.sampling.get("max_new_tokens", 8)))
        max_new_tokens = max(1, max_new_tokens)
        temperature = float(req.sampling.get("temperature", 1.0))
        top_p = float(req.sampling.get("top_p", 1.0))
        top_k = int(req.sampling.get("top_k", 0))
        top_k = max(0, top_k)
        do_sample = bool(req.sampling.get("do_sample", temperature > 0.0))

        generation_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        if do_sample:
            generation_kwargs["temperature"] = max(1e-5, temperature)
            generation_kwargs["top_p"] = min(max(top_p, 0.0), 1.0)
            if top_k > 0:
                generation_kwargs["top_k"] = top_k

        with torch.no_grad():
            generated = model.generate(**generation_kwargs)

        new_tokens = generated.sequences[0, input_ids.shape[1] :].tolist()
        token_logprobs: list[float] = []
        for score_t, token in zip(generated.scores, new_tokens):
            logprob = torch.log_softmax(score_t[0].float(), dim=-1)[int(token)]
            token_logprobs.append(float(logprob.detach().item()))

        if not new_tokens:
            eos_token = int(tokenizer.eos_token_id or 0)
            new_tokens = [eos_token]
            token_logprobs = [0.0]

        stop_reason = "length"
        if new_tokens and int(tokenizer.eos_token_id or -1) == int(new_tokens[-1]):
            stop_reason = "eos"

        text = " ".join(str(token) for token in new_tokens)
        return GenerateResult(
            text=text,
            token_ids=tuple(new_tokens),
            token_logprobs=tuple(token_logprobs),
            prompt_token_ids=tuple(prompt_tokens),
            prompt_logprobs=prompt_logprobs,
            stop_reason=stop_reason,
            raw={"provider": "qwen-transformers", "mode": mode},
        )

    def _sampling_assets(self, model_path: Any) -> tuple[Any, Any]:
        if not isinstance(model_path, str) or not model_path:
            return self.model, self.tokenizer
        cached = self._sampling_models.get(model_path)
        if cached is not None:
            return cached
        if self._sampling_models:
            _clear_model_cache(self._sampling_models)

        model, base_model = _load_qwen_model_from_reference(
            model_path,
            default_base_model=self.model_id,
            default_lora_rank=16,
            device=self.device,
            error_prefix="sampler checkpoint",
        )
        model.eval()
        assets = (model, self._tokenizer_for_model_ref(base_model))
        self._sampling_models[model_path] = assets
        return assets

    def _tokenizer_for_model_ref(self, base_model: str) -> Any:
        model_ref = _resolve_model_ref(base_model)
        cached = self._tokenizers.get(model_ref)
        if cached is not None:
            return cached
        tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        self._tokenizers[model_ref] = tokenizer
        return tokenizer

    def _prompt_inputs(self, prompt: str, tokenizer: Any) -> tuple[Any, Any, list[int]]:
        int_tokens = _to_int_tokens(prompt)
        if int_tokens:
            input_ids = torch.tensor([int_tokens], dtype=torch.long, device=self.device)
            attention_mask = torch.ones_like(input_ids)
            return input_ids, attention_mask, list(int_tokens)

        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        return input_ids, attention_mask, input_ids[0].tolist()

    def _prompt_logprobs(self, model: Any, input_ids: Any, attention_mask: Any) -> list[float | None]:
        if int(input_ids.shape[1]) <= 0:
            return []
        if int(input_ids.shape[1]) == 1:
            return [None]

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[0].float()
        logprobs = torch.log_softmax(logits, dim=-1)

        values: list[float | None] = [None]
        tokens = input_ids[0].tolist()
        for i in range(1, len(tokens)):
            values.append(float(logprobs[i - 1, int(tokens[i])].detach().item()))
        return values
