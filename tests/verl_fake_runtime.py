from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


RUN_PPO_CALLS: list[Any] = []


def run_ppo(config: Any) -> Mapping[str, Any]:
    RUN_PPO_CALLS.append(config)
    trainer = config.get("trainer", {}) if isinstance(config, Mapping) else getattr(config, "trainer", {})
    step = trainer.get("total_training_steps", 0) if isinstance(trainer, Mapping) else getattr(trainer, "total_training_steps", 0)
    return {"step": step, "actor/pg_loss": 0.1, "critic/vf_loss": 0.2}


def remote_run_ppo(config: Any) -> Mapping[str, Any]:
    return run_ppo(config)


class FakeTrainer:
    def __init__(self, config) -> None:
        self.config = config
        self.initialized = False
        self.closed = False
        self.last_batch = None
        self.step = 0

    def init_workers(self) -> None:
        self.initialized = True

    def train_step(self, batch) -> Mapping[str, Any]:
        self.last_batch = batch
        self.step += 1
        return {"loss": 0.5, "step": self.step}

    def ppo_step(self, batch) -> Mapping[str, Any]:
        self.last_batch = batch
        self.step += 1
        return {"loss": 0.25, "op": "ppo", "step": self.step}

    def grpo_step(self, batch) -> Mapping[str, Any]:
        self.last_batch = batch
        self.step += 1
        return {"loss": 0.125, "op": "grpo", "step": self.step}

    def forward_backward_ppo(self, batch) -> Mapping[str, Any]:
        self.last_batch = batch
        return {"loss": 0.03125, "op": "mint_style_forward_backward"}

    def optimizer_step(self) -> Mapping[str, Any]:
        self.step += 1
        return {"step": self.step, "op": "optimizer_step"}

    def export_lora_adapter(self, uri: str) -> Mapping[str, Any]:
        Path(uri).write_text("adapter", encoding="utf-8")
        return {"uri": uri, "format": "lora"}

    def dpo_step(self, batch) -> Mapping[str, Any]:
        self.last_batch = batch
        self.step += 1
        return {"loss": 0.0625, "op": "dpo", "step": self.step}

    def forward(self, batch) -> Mapping[str, Any]:
        self.last_batch = batch
        return {"logprobs": [-1.0]}

    def save_checkpoint(self, uri: str, *, include_optimizer: bool = True, **metadata: Any) -> Mapping[str, Any]:
        Path(uri).write_text(str(self.step), encoding="utf-8")
        return {"include_optimizer": include_optimizer, **metadata}

    def load_checkpoint(self, uri: str, *, include_optimizer: bool = True, **metadata: Any) -> Mapping[str, Any]:
        self.step = int(Path(uri).read_text(encoding="utf-8"))
        return {"step": self.step, "include_optimizer": include_optimizer, **metadata}

    def shutdown(self) -> None:
        self.closed = True


class FakeRollout:
    def __init__(self, config) -> None:
        self.config = config

    def generate(self, prompt: str, **sampling: Any) -> Mapping[str, Any]:
        return {"text": f"{prompt} done", "stop_reason": "stop", "sampling": sampling}
