from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Mapping

from verl_mint.contracts import TrainOpRequest


class MinTTrainingBatchError(ValueError):
    pass


def _as_list(value: Any, *, field: str) -> list[Any]:
    if isinstance(value, Mapping) and "data" in value:
        value = value["data"]
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return value
    raise MinTTrainingBatchError(f"{field} must be a list, tuple, or {{'data': ...}}")


def _as_float_list(value: Any, *, field: str, n: int) -> list[float]:
    xs = [float(x) for x in _as_list(value, field=field)]
    if len(xs) != n:
        raise MinTTrainingBatchError(f"{field} length {len(xs)} != completion length {n}")
    return xs


def _as_int_list(value: Any, *, field: str) -> list[int]:
    return [int(x) for x in _as_list(value, field=field)]


def _extract_samples(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    samples = payload.get("samples")
    if isinstance(samples, (list, tuple)):
        if not samples:
            raise MinTTrainingBatchError("samples must not be empty")
        return [dict(s) for s in samples]
    if "prompt_tokens" not in payload or "completion_tokens" not in payload:
        raise MinTTrainingBatchError("payload requires samples or prompt_tokens/completion_tokens")
    return [dict(payload)]


def is_mint_reverse_kl_payload(payload: Mapping[str, Any]) -> bool:
    data = payload.get("data")
    if not isinstance(data, (list, tuple)) or not data:
        return False
    first = data[0]
    return isinstance(first, Mapping) and "student_input" in first and "target_tokens" in first


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


def build_mint_grpo_datum(sample: Mapping[str, Any], advantages: list[float] | None = None) -> dict[str, Any]:
    prompt = _as_int_list(sample.get("prompt_tokens", ()), field="prompt_tokens")
    completion = _as_int_list(sample.get("completion_tokens", ()), field="completion_tokens")
    if not completion:
        raise MinTTrainingBatchError("completion_tokens must not be empty")
    tokens = prompt + completion
    n = len(completion)

    old_logprobs = _as_float_list(sample.get("old_logprobs", [-0.0 for _ in completion]), field="old_logprobs", n=n)
    weights = _as_float_list(sample.get("weights", [1.0 for _ in completion]), field="weights", n=n)
    if advantages is None:
        raw_adv = sample.get("advantages")
        advantages = _as_float_list(raw_adv, field="advantages", n=n) if raw_adv else [float(sample.get("reward", 0.0))] * n
    elif len(advantages) != n:
        raise MinTTrainingBatchError(f"advantages length {len(advantages)} != completion length {n}")

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


def build_mint_grpo_datums(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    samples = _extract_samples(payload)
    computed_advantages = _group_centered_advantages(samples)
    return [build_mint_grpo_datum(sample, computed_advantages.get(i)) for i, sample in enumerate(samples)]


def build_mint_reverse_kl_datums(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    data = payload.get("data")
    if not isinstance(data, (list, tuple)) or not data:
        raise MinTTrainingBatchError("reverse-KL payload requires non-empty data")

    datums: list[dict[str, Any]] = []
    for i, raw in enumerate(data):
        if not isinstance(raw, Mapping):
            raise MinTTrainingBatchError(f"reverse-KL data[{i}] must be a mapping")
        item = dict(raw)
        student_input = item.get("student_input")
        target_tokens = item.get("target_tokens")
        weights = item.get("weights")
        if not isinstance(student_input, Mapping):
            raise MinTTrainingBatchError(f"reverse-KL data[{i}].student_input must be a mapping")
        if target_tokens is None:
            raise MinTTrainingBatchError(f"reverse-KL data[{i}].target_tokens is required")
        if weights is None:
            raise MinTTrainingBatchError(f"reverse-KL data[{i}].weights is required")

        loss_fn_inputs = {
            "target_tokens": target_tokens,
            "weights": weights,
            "reference_input": item.get("reference_input"),
            "reference_logprobs": item.get("reference_logprobs"),
            "temperature": payload.get("temperature"),
            "reference_model_path": payload.get("reference_model_path"),
        }
        for key in ("old_logprobs", "old_values", "advantages", "returns"):
            if item.get(key) is not None:
                loss_fn_inputs[key] = item[key]

        datums.append(
            {
                "model_input": dict(student_input),
                "loss_fn_inputs": {k: v for k, v in loss_fn_inputs.items() if v is not None},
                "metadata": {
                    "algorithm": "reverse_kl",
                    "sample_id": str(item.get("sample_id", "")),
                    "group_id": str(item.get("group_id", "default")),
                    "reference_model_path": str(payload.get("reference_model_path", "")),
                },
            }
        )
    return datums


@dataclass
class MinTTrainingAdapter:
    trainer: Any
    adapter: Any | None = None
    max_token_len_per_gpu: int = 10240
    optimizer_methods: tuple[str, ...] = ("optimizer_step", "optim_step", "step_optimizer")
    export_methods: tuple[str, ...] = ("export_lora_adapter", "save_lora_adapter", "save_checkpoint")
    history: list[dict[str, Any]] = field(default_factory=list)

    def step(self, req: TrainOpRequest) -> Mapping[str, Any]:
        if not isinstance(req.batch_payload, Mapping):
            raise MinTTrainingBatchError("MinT training GRPO batch_payload must be a mapping")
        payload = dict(req.batch_payload)
        if is_mint_reverse_kl_payload(payload):
            algorithm = "reverse_kl"
            payload = {**payload, "temperature": req.options.get("temperature", payload.get("temperature"))}
            datums = build_mint_reverse_kl_datums(payload)
        else:
            algorithm = "grpo"
            datums = build_mint_grpo_datums(payload)
        train_batch = self._to_train_batch(datums, algorithm=algorithm)
        out = self._forward_backward(train_batch, req)
        opt = self._optimizer_step(req)
        adapter = self._export_adapter(req)
        record = {"datums": datums, "forward_backward": out, "optimizer": opt, "adapter": adapter}
        self.history.append(record)
        return {
            "algorithm": algorithm,
            "execution_framework": "mint_training",
            "num_samples": len(datums),
            "num_tokens": sum(len(d["model_input"]["chunks"][0]["tokens"]) for d in datums),
            "forward_backward": out,
            "optimizer": opt,
            "adapter": adapter,
        }

    def _to_train_batch(self, datums: list[dict[str, Any]], *, algorithm: str = "grpo") -> Any:
        if self.adapter is None:
            return datums
        fn = getattr(self.adapter, "to_train_batch", None)
        if fn is not None:
            return fn(datums, max_token_len_per_gpu=self.max_token_len_per_gpu)
        return datums

    def _forward_backward(self, train_batch: Any, req: TrainOpRequest) -> Any:
        for name in ("forward_backward_ppo", "forward_backward", "train_step", "fit_batch", "step", "update_actor"):
            fn = getattr(self.trainer, name, None)
            if fn is not None:
                return fn(train_batch)
        raise MinTTrainingBatchError(f"trainer {type(self.trainer).__name__} has no forward/backward method")

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
