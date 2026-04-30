from __future__ import annotations

from typing import Any

from verl_mint.backends.verl import VerlBatchAdapter, VerlInferenceBackend, VerlTrainingBackend
from verl_mint.contracts import (
    CheckpointRequest,
    GenerateRequest,
    SessionSpec,
    TrainOp,
    TrainOpRequest,
)


class _FakeDataProto:
    def __init__(self, *, batch=None, non_tensor_batch=None, meta_info=None) -> None:
        self.batch = dict(batch or {})
        self.non_tensor_batch = dict(non_tensor_batch or {})
        self.meta_info = dict(meta_info or {})

    @classmethod
    def from_dict(cls, *, tensors, non_tensors=None, non_tensor_batch=None, meta_info=None):
        return cls(
            batch=tensors,
            non_tensor_batch=non_tensor_batch if non_tensor_batch is not None else non_tensors,
            meta_info=meta_info,
        )


def test_verl_batch_adapter_builds_dataproto() -> None:
    adapter = VerlBatchAdapter(_FakeDataProto)
    batch = adapter.to_data_proto(
        {
            "tensors": {"input_ids": [[1, 2]]},
            "non_tensor_batch": {"uids": ["a"]},
            "meta_info": {"temperature": 1.0},
        }
    )

    assert isinstance(batch, _FakeDataProto)
    assert batch.batch == {"input_ids": [[1, 2]]}
    assert batch.non_tensor_batch == {"uids": ["a"]}
    assert batch.meta_info == {"temperature": 1.0}


def test_verl_training_backend_runs_ppo_through_verl_job_runner() -> None:
    import verl_fake_runtime

    verl_fake_runtime.RUN_PPO_CALLS.clear()
    backend = VerlTrainingBackend(
        config={
            "trainer": {
                "run_ppo": "verl_fake_runtime:run_ppo",
                "config": {"trainer": {"total_training_steps": 4}, "algorithm": {"adv_estimator": "gae"}},
            }
        },
        batch_adapter=VerlBatchAdapter(_FakeDataProto),
    )
    handle = backend.open_session(SessionSpec(session_id="ppo-job", backend_id="train", batch_codec="verl"))

    result = backend.forward_backward_ppo(
        handle,
        TrainOpRequest(
            op=TrainOp.FORWARD_BACKWARD_PPO,
            batch_codec="verl",
            batch_payload={
                "tensors": {},
                "algorithm": "ppo",
                "verl_config_overrides": {"trainer": {"total_training_steps": 20}},
            },
            options={"hydra_overrides": ["trainer.logger=['console']"]},
        ),
    )

    assert result.state.step == 20
    assert result.outputs["execution_framework"] == "verl"
    assert result.outputs["runner"] == "verl.trainer.main_ppo.run_ppo"
    assert result.outputs["actor/pg_loss"] == 0.1
    assert result.outputs["critic/vf_loss"] == 0.2
    trainer = backend._trainer(handle)
    assert trainer.last_batch.meta_info["algorithm"] == "ppo"
    assert trainer.last_batch.meta_info["mint_options"] == {"hydra_overrides": ("trainer.logger=['console']",)}
    assert len(verl_fake_runtime.RUN_PPO_CALLS) == 1


def test_verl_training_backend_uses_dataproto_and_trainer(tmp_path: Path) -> None:
    backend = VerlTrainingBackend(
        config={"trainer": {"class": "verl_fake_runtime:FakeTrainer", "lr": 1e-5}},
        batch_adapter=VerlBatchAdapter(_FakeDataProto),
    )
    handle = backend.open_session(SessionSpec(session_id="sess-1", backend_id="train", batch_codec="verl"))
    trainer = backend._trainer(handle)
    assert handle.metadata["execution_framework"] == "verl"
    assert trainer.initialized is True
    assert trainer.config["lr"] == 1e-5

    result = backend.run_train_op(
        handle,
        TrainOpRequest(
            op=TrainOp.FORWARD_BACKWARD,
            batch_codec="verl",
            batch_payload={"tensors": {"input_ids": [[1]]}, "meta_info": {"algo": "sft"}},
        ),
    )
    assert result.state.step == 1
    assert result.outputs["loss"] == 0.5
    assert trainer.last_batch is not None
    assert trainer.last_batch.meta_info == {"algo": "sft"}

    ppo = backend.forward_backward_ppo(
        handle,
        TrainOpRequest(op=TrainOp.FORWARD_BACKWARD_PPO, batch_codec="verl", batch_payload={"tensors": {}}),
    )
    assert ppo.outputs["op"] == "ppo"

    grpo = backend.forward_backward_reverse_kl(
        handle,
        TrainOpRequest(
            op=TrainOp.FORWARD_BACKWARD_REVERSE_KL,
            batch_codec="verl",
            batch_payload={"tensors": {}},
        ),
    )
    assert grpo.outputs["op"] == "grpo"

    dpo = backend.run_train_op(
        handle,
        TrainOpRequest(
            op=TrainOp.FORWARD_BACKWARD,
            batch_codec="verl",
            batch_payload={"tensors": {}, "meta_info": {"algo": "dpo"}, "loss_fn": "dpo"},
        ),
    )
    assert dpo.outputs["op"] == "dpo"
    assert trainer.last_batch.meta_info == {"algo": "dpo", "loss_fn": "dpo"}

    ckpt = tmp_path / "ckpt.txt"
    artifact = backend.save_checkpoint(handle, CheckpointRequest(uri=str(ckpt), metadata={"tag": "x"}))
    assert artifact.format == "verl-checkpoint"
    assert artifact.metadata["tag"] == "x"
    loaded = backend.load_checkpoint(handle, CheckpointRequest(uri=str(ckpt)))
    assert loaded.step == 4

    caps = backend.capabilities()
    assert caps.extras["batch_protocol"] == "verl.protocol.DataProto"
    assert caps.extras["placement_model"] == "verl_resource_pool"

    backend.close_session(handle)
    assert trainer.closed is True


def test_verl_inference_backend_generate() -> None:
    backend = VerlInferenceBackend(config={"rollout": {"class": "verl_fake_runtime:FakeRollout", "temperature": 1.0}})
    result = backend.generate(GenerateRequest(prompt="1 2", sampling={"max_tokens": 2}))
    assert result.text == "1 2 done"
    assert result.stop_reason == "stop"
    assert backend.rollout.config["temperature"] == 1.0
