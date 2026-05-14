from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Mapping

from verl_mint.contracts import TrainOpRequest


class MintStyleBatchError(ValueError):
    pass


def _as_list(value: Any, *, field: str) -> list[Any]:
    if isinstance(value, Mapping) and "data" in value:
        value = value["data"]
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return value
    raise MintStyleBatchError(f"{field} must be a list, tuple, or {{'data': ...}}")


def _as_float_list(value: Any, *, field: str, n: int) -> list[float]:
    xs = [float(x) for x in _as_list(value, field=field)]
    if len(xs) != n:
        raise MintStyleBatchError(f"{field} length {len(xs)} != completion length {n}")
    return xs


def _as_int_list(value: Any, *, field: str) -> list[int]:
    return [int(x) for x in _as_list(value, field=field)]


def _extract_samples(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    samples = payload.get("samples")
    if isinstance(samples, (list, tuple)):
        if not samples:
            raise MintStyleBatchError("samples must not be empty")
        return [dict(s) for s in samples]
    if "prompt_tokens" not in payload or "completion_tokens" not in payload:
        raise MintStyleBatchError("payload requires samples or prompt_tokens/completion_tokens")
    return [dict(payload)]


def _group_centered_advantages(samples: list[dict[str, Any]]) -> dict[int, list[float]]:
    rewards_by_group: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for i, sample in enumerate(samples):
        group = str(sample.get("group_id") or "default")
        rewards_by_group[group].append((i, float(sample.get("reward", 0.0))))

    out: dict[int, list[float]] = {}
    for rows in rewards_by_group.values():
        mean = sum(r for _, r in rows) / len(rows)
        variance = sum((r - mean) ** 2 for _, r in rows) / len(rows)
        std = variance**0.5
        for i, reward in rows:
            n = len(_as_list(samples[i]["completion_tokens"], field="completion_tokens"))
            adv = reward - mean
            if std > 0:
                adv /= std
            out[i] = [float(adv) for _ in range(n)]
    return out


def build_mint_style_datum(sample: Mapping[str, Any], advantages: list[float] | None = None) -> dict[str, Any]:
    prompt = _as_int_list(sample.get("prompt_tokens", ()), field="prompt_tokens")
    completion = _as_int_list(sample.get("completion_tokens", ()), field="completion_tokens")
    if not completion:
        raise MintStyleBatchError("completion_tokens must not be empty")
    tokens = prompt + completion
    n = len(completion)

    old_logprobs = _as_float_list(sample.get("old_logprobs", [-0.0 for _ in completion]), field="old_logprobs", n=n)
    weights = _as_float_list(sample.get("weights", [1.0 for _ in completion]), field="weights", n=n)
    if advantages is None:
        raw_adv = sample.get("advantages")
        advantages = _as_float_list(raw_adv, field="advantages", n=n) if raw_adv else [float(sample.get("reward", 0.0))] * n
    elif len(advantages) != n:
        raise MintStyleBatchError(f"advantages length {len(advantages)} != completion length {n}")

    return {
        "model_input": {"chunks": [{"tokens": tokens}]},
        "loss_fn_inputs": {
            "target_tokens": {"data": completion},
            "weights": {"data": weights},
            "logprobs": {"data": old_logprobs},
            "advantages": {"data": [float(x) for x in advantages]},
        },
        "metadata": {
            "sample_id": str(sample.get("sample_id", "")),
            "group_id": str(sample.get("group_id", "default")),
            "prompt_len": len(prompt),
            "response_len": n,
            "reward": float(sample.get("reward", 0.0)),
        },
    }


def build_mint_style_grpo_datums(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    samples = _extract_samples(payload)
    computed_advantages = _group_centered_advantages(samples)
    return [build_mint_style_datum(sample, computed_advantages.get(i)) for i, sample in enumerate(samples)]


@dataclass
class MintStyleGRPOTrainer:
    trainer: Any
    adapter: Any | None = None
    max_token_len_per_gpu: int = 10240
    optimizer_methods: tuple[str, ...] = ("optimizer_step", "optim_step", "step_optimizer")
    export_methods: tuple[str, ...] = ("export_lora_adapter", "save_lora_adapter", "save_checkpoint")
    history: list[dict[str, Any]] = field(default_factory=list)

    def step(self, req: TrainOpRequest) -> Mapping[str, Any]:
        if not isinstance(req.batch_payload, Mapping):
            raise MintStyleBatchError("Mint-style GRPO batch_payload must be a mapping")
        payload = dict(req.batch_payload)
        datums = build_mint_style_grpo_datums(payload)
        train_batch = self._to_train_batch(datums)
        out = self._forward_backward(train_batch, req)
        opt = self._optimizer_step(req)
        adapter = self._export_adapter(req)
        record = {"datums": datums, "forward_backward": out, "optimizer": opt, "adapter": adapter}
        self.history.append(record)
        return {
            "algorithm": "grpo",
            "execution_framework": "mint_style",
            "num_samples": len(datums),
            "num_tokens": sum(len(d["model_input"]["chunks"][0]["tokens"]) for d in datums),
            "forward_backward": out,
            "optimizer": opt,
            "adapter": adapter,
        }

    def _to_train_batch(self, datums: list[dict[str, Any]]) -> Any:
        if self.adapter is None:
            return datums
        fn = getattr(self.adapter, "to_train_batch", None)
        if fn is not None:
            return fn(datums, max_token_len_per_gpu=self.max_token_len_per_gpu)
        fn = getattr(self.adapter, "to_data_proto", None)
        if fn is not None:
            return fn({"tensors": {"mint_datums": datums}, "meta_info": {"algorithm": "grpo", "route": "mint_style"}})
        return datums

    def _forward_backward(self, train_batch: Any, req: TrainOpRequest) -> Any:
        for name in ("forward_backward_ppo", "forward_backward", "train_step", "fit_batch", "step", "update_actor"):
            fn = getattr(self.trainer, name, None)
            if fn is not None:
                return fn(train_batch)
        raise MintStyleBatchError(f"trainer {type(self.trainer).__name__} has no forward/backward method")

    def _optimizer_step(self, req: TrainOpRequest) -> Any:
        if req.options.get("skip_optimizer_step"):
            return None
        for name in self.optimizer_methods:
            fn = getattr(self.trainer, name, None)
            if fn is not None:
                return fn()
        return None

    def _export_adapter(self, req: TrainOpRequest) -> Any:
        adapter_uri = req.options.get("adapter_uri") or req.options.get("export_adapter_uri")
        if not adapter_uri:
            return None
        for name in self.export_methods:
            fn = getattr(self.trainer, name, None)
            if fn is not None:
                return fn(str(adapter_uri))
        return None
