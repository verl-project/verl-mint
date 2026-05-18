from __future__ import annotations

import os
import sys
from pathlib import Path

from verl_mint.backends.verl import (
    VerlBatchAdapter,
    VerlInferenceBackend,
    VerlPPOJobRunner,
    VerlTrainingBackend,
    _runtime_pythonpath_entries,
)
from verl_mint.model_registry import get_model_config, get_training_parallelism, normalize_model_name
from verl_mint.contracts import (
    CheckpointRequest,
    GenerateRequest,
    SessionSpec,
    TrainOp,
    TrainOpRequest,
)


QWEN30B_MODEL_ID = "Qwen/Qwen3-30B-A3B-Base"


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


def test_qwen30b_model_registry_matches_mint_tp4() -> None:
    model = QWEN30B_MODEL_ID

    assert normalize_model_name(model) == "Qwen/Qwen3-30B-A3B-Base"
    cfg = get_model_config(model)
    assert cfg.is_moe is True
    assert cfg.inference_tp == 4
    assert cfg.inference_dp == 1
    assert cfg.train_tp == 4
    assert cfg.train_ep == 1
    assert cfg.train_lora_rank == 16
    assert cfg.train_lora_alpha == 32
    assert cfg.train_gpus == 4
    assert cfg.inference_gpus == 4
    assert get_training_parallelism(model) == (4, 1, 1, 1, None)


def test_verl_ppo_job_runner_injects_mint_qwen30b_tp4_overrides() -> None:
    import verl_fake_runtime

    verl_fake_runtime.RUN_PPO_CALLS.clear()
    model = QWEN30B_MODEL_ID
    backend = VerlTrainingBackend(
        config={
            "model_path": model,
            "trainer": {
                "run_ppo": "verl_fake_runtime:run_ppo",
                "remote_run_ppo": True,
                "hydra": {"config_dir": "/path/to/verl/trainer/config", "overrides": []},
            },
        },
        batch_adapter=VerlBatchAdapter(_FakeDataProto),
    )
    handle = backend.open_session(SessionSpec(session_id="qwen30b-tp4", backend_id="train", batch_codec="verl"))

    trainer = backend._trainer(handle)
    spec = trainer._config_from_batch(_FakeDataProto())
    overrides = spec["hydra"]["overrides"]
    assert f"actor_rollout_ref.model.path={model}" in overrides
    assert "trainer.n_gpus_per_node=4" in overrides
    assert spec["hydra"]["config_name"] == "ppo_megatron_trainer"
    assert "actor_rollout_ref.rollout.tensor_model_parallel_size=4" in overrides
    assert "actor_rollout_ref.rollout.data_parallel_size=1" in overrides
    assert "actor_rollout_ref.rollout.max_model_len=32768" in overrides
    assert "actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4" in overrides
    assert "actor_rollout_ref.actor.megatron.expert_model_parallel_size=1" in overrides
    assert "actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=4" in overrides
    assert "actor_rollout_ref.actor.megatron.param_offload=True" in overrides
    assert "actor_rollout_ref.actor.megatron.optimizer_offload=True" in overrides
    assert "actor_rollout_ref.actor.megatron.grad_offload=True" in overrides
    assert "actor_rollout_ref.actor.megatron.vanilla_mbridge=False" in overrides
    assert "+actor_rollout_ref.actor.megatron.override_ddp_config.grad_reduce_in_fp32=True" in overrides
    assert "+actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True" in overrides
    assert "actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra]" in overrides
    assert "actor_rollout_ref.actor.checkpoint.load_contents=[model,optimizer,extra]" in overrides
    assert "actor_rollout_ref.model.lora.rank=16" in overrides
    assert "actor_rollout_ref.model.lora.alpha=32" in overrides
    assert "actor_rollout_ref.model.lora.dtype=float16" in overrides
    assert "actor_rollout_ref.model.lora.type=lora" in overrides
    assert "actor_rollout_ref.ref.megatron.tensor_model_parallel_size=4" in overrides
    assert "critic.megatron.tensor_model_parallel_size=4" in overrides
    assert "+critic.optim.override_optimizer_config.use_precision_aware_optimizer=True" in overrides
    assert "critic.checkpoint.save_contents=[model,optimizer,extra]" in overrides
    assert "critic.checkpoint.load_contents=[model,optimizer,extra]" in overrides
    assert not any("fsdp_config" in item for item in overrides)
    assert backend.config.resources["train_gpus"] == 4
    assert backend.config.resources["inference_gpus"] == 4


def test_verl_ppo_job_runner_switches_qwen30b_default_hydra_config() -> None:
    spec = VerlTrainingBackend(
        config={
            "model_path": QWEN30B_MODEL_ID,
            "trainer": {
                "run_ppo": "verl_fake_runtime:run_ppo",
                "hydra": {
                    "config_dir": "/path/to/verl/trainer/config",
                    "config_name": "ppo_trainer",
                    "overrides": [
                        "actor_rollout_ref.actor.fsdp_config.param_offload=True",
                        "actor_rollout_ref.model.lora_rank=16",
                        "actor_rollout_ref.model.lora_alpha=16",
                        "actor_rollout_ref.model.target_modules=all-linear",
                        "critic.model.enable_gradient_checkpointing=True",
                    ],
                },
            },
        },
        batch_adapter=VerlBatchAdapter(_FakeDataProto),
    ).config.trainer

    assert spec["hydra"]["config_name"] == "ppo_megatron_trainer"
    overrides = spec["hydra"]["overrides"]
    assert not any("fsdp_config" in item for item in overrides)
    assert not any(item.startswith("actor_rollout_ref.model.lora_rank=") for item in overrides)
    assert not any(item.startswith("actor_rollout_ref.model.lora_alpha=") for item in overrides)
    assert not any(item.startswith("actor_rollout_ref.model.target_modules=") for item in overrides)
    assert not any(item.startswith("critic.model.enable_gradient_checkpointing=") for item in overrides)


def test_verl_runtime_env_uses_pfs_manifest_without_host_only_sources(tmp_path: Path) -> None:
    root = tmp_path / "runtime"
    (root / "site-packages").mkdir(parents=True)
    (root / "src" / "verl").mkdir(parents=True)
    (root / "src" / "vllm").mkdir(parents=True)
    (root / "manifest.json").write_text(
        """
        {
          "runtime_env": {"site_packages_dir": "site-packages", "source_dir": "src"},
          "sources": [
            {"name": "verl", "pythonpath": ["."]},
            {"name": "vllm", "pythonpath": ["."], "host_only": true}
          ]
        }
        """,
        encoding="utf-8",
    )

    entries = _runtime_pythonpath_entries(root)

    assert entries == [str(root / "site-packages"), str(root / "src" / "verl")]


def test_verl_ppo_job_runner_builds_ray_runtime_env_from_runtime_root(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "runtime"
    (root / "site-packages").mkdir(parents=True)
    (root / "src" / "verl").mkdir(parents=True)
    (root / "manifest.json").write_text(
        """
        {
          "runtime_env": {"site_packages_dir": "site-packages", "source_dir": "src"},
          "sources": [{"name": "verl", "pythonpath": ["."]}]
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setenv("VERL_MINT_HF_MODULES_PATH", str(tmp_path / "hf_modules"))

    runtime_env = VerlPPOJobRunner._runtime_env_config(
        {
            "runtime_env_root": str(root),
            "ray_runtime_env": {
                "working_dir": "/tmp/work",
                "env_vars": {"PYTHONPATH": "/custom", "EXTRA": "1"},
            },
        }
    )

    env_vars = runtime_env["env_vars"]
    assert runtime_env["working_dir"] == "/tmp/work"
    assert env_vars["VERL_MINT_RUNTIME_ENV_ROOT"] == str(root)
    assert env_vars["VERL_MINT_HF_MODULES_PATH"] == str(tmp_path / "hf_modules")
    assert env_vars["EXTRA"] == "1"
    assert env_vars["PYTHONPATH"].split(":")[:3] == [
        str(root / "site-packages"),
        str(root / "src" / "verl"),
        str(tmp_path / "hf_modules"),
    ]
    assert env_vars["PYTHONPATH"].split(":")[-1] == "/custom"


def test_verl_ppo_job_runner_resolves_run_ppo_from_runtime_root(tmp_path: Path) -> None:
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    (site_packages / "runtime_only_runner.py").write_text(
        "def run_ppo(config):\n"
        "    return {'step': config['trainer']['total_training_steps'], 'runtime': True}\n",
        encoding="utf-8",
    )
    old_sys_path = list(sys.path)
    old_env = {
        name: os.environ.get(name)
        for name in ("PYTHONPATH", "VERL_MINT_RUNTIME_ENV_ROOT")
    }

    try:
        runner = VerlPPOJobRunner(
            config={
                "runtime_env_root": str(tmp_path),
                "run_ppo": "runtime_only_runner:run_ppo",
            }
        )
        out = runner.run_ppo({"trainer": {"total_training_steps": 7}})
    finally:
        sys.path[:] = old_sys_path
        for name, value in old_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    assert out == {"step": 7, "runtime": True}


def test_mint_training_grpo_datums_group_center_advantages() -> None:
    from verl_mint.backends.mint_training import build_mint_grpo_datums

    payload = {
        "algorithm": "grpo",
        "samples": [
            {
                "sample_id": "a",
                "group_id": "g",
                "prompt_tokens": [1, 2],
                "completion_tokens": [3, 4],
                "old_logprobs": [-0.1, -0.2],
                "weights": [1.0, 1.0],
                "reward": 2.0,
            },
            {
                "sample_id": "b",
                "group_id": "g",
                "prompt_tokens": [1, 2],
                "completion_tokens": [5],
                "old_logprobs": [-0.3],
                "weights": [1.0],
                "reward": 0.0,
            },
        ],
    }

    datums = build_mint_grpo_datums(payload)

    assert datums[0]["model_input"]["chunks"][0]["tokens"] == [1, 2, 3, 4]
    assert datums[0]["loss_fn_inputs"]["target_tokens"]["data"] == [3, 4]
    assert datums[0]["loss_fn_inputs"]["logprobs"]["data"] == [-0.1, -0.2]
    assert datums[0]["loss_fn_inputs"]["advantages"]["data"] == [1.0, 1.0]
    assert datums[1]["loss_fn_inputs"]["advantages"]["data"] == [-1.0]


def test_mint_training_reverse_kl_datums_preserve_official_sdk_shape() -> None:
    from verl_mint.backends.mint_training import build_mint_reverse_kl_datums

    payload = {
        "algorithm": "grpo",
        "reference_model_path": "mint://checkpoints/ref.pt",
        "temperature": 0.7,
        "data": [
            {
                "student_input": {"chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]},
                "reference_input": {"chunks": [{"type": "encoded_text", "tokens": [1, 2, 4]}]},
                "target_tokens": {"data": [2, 3], "shape": [2], "dtype": "int64"},
                "weights": {"data": [1.0, 1.0], "shape": [2], "dtype": "float32"},
            }
        ],
    }

    datums = build_mint_reverse_kl_datums(payload)

    assert datums[0]["model_input"]["chunks"][0]["tokens"] == [1, 2, 3]
    assert datums[0]["loss_fn_inputs"]["reference_input"]["chunks"][0]["tokens"] == [1, 2, 4]
    assert datums[0]["loss_fn_inputs"]["target_tokens"]["data"] == [2, 3]
    assert datums[0]["loss_fn_inputs"]["reference_model_path"] == "mint://checkpoints/ref.pt"
    assert datums[0]["loss_fn_inputs"]["temperature"] == 0.7


def test_verl_training_backend_routes_grpo_samples_through_mint_training(tmp_path: Path) -> None:
    backend = VerlTrainingBackend(
        config={"trainer": {"class": "verl_fake_runtime:FakeTrainer"}},
        batch_adapter=VerlBatchAdapter(_FakeDataProto),
    )
    handle = backend.open_session(SessionSpec(session_id="mint-training-grpo", backend_id="train", batch_codec="verl"))
    adapter_uri = tmp_path / "adapter.bin"

    result = backend.forward_backward_reverse_kl(
        handle,
        TrainOpRequest(
            op=TrainOp.FORWARD_BACKWARD_REVERSE_KL,
            batch_codec="verl",
            batch_payload={
                "algorithm": "grpo",
                "samples": [
                    {
                        "sample_id": "0",
                        "group_id": "g",
                        "prompt_tokens": [1],
                        "completion_tokens": [2, 3],
                        "old_logprobs": [-0.1, -0.2],
                        "weights": [1.0, 1.0],
                        "reward": 1.0,
                    },
                    {
                        "sample_id": "1",
                        "group_id": "g",
                        "prompt_tokens": [1],
                        "completion_tokens": [4],
                        "old_logprobs": [-0.3],
                        "weights": [1.0],
                        "reward": 0.0,
                    },
                ],
            },
            options={"route": "mint_training", "adapter_uri": str(adapter_uri)},
        ),
    )

    trainer = backend._trainer(handle)
    assert result.state.step == 1
    assert result.outputs["execution_framework"] == "mint_training"
    assert result.outputs["num_samples"] == 2
    assert result.outputs["forward_backward"]["op"] == "mint_training_forward_backward"
    assert result.outputs["optimizer"]["op"] == "optimizer_step"
    assert result.outputs["adapter"]["uri"] == str(adapter_uri)
    assert adapter_uri.read_text(encoding="utf-8") == "adapter"
    datums = trainer.last_batch
    assert datums[0]["loss_fn_inputs"]["advantages"]["data"] == [1.0, 1.0]
    assert datums[1]["loss_fn_inputs"]["advantages"]["data"] == [-1.0]


def test_verl_training_backend_routes_official_reverse_kl_payload_through_mint_training() -> None:
    backend = VerlTrainingBackend(
        config={"trainer": {"class": "verl_fake_runtime:FakeTrainer"}},
        batch_adapter=VerlBatchAdapter(_FakeDataProto),
    )
    handle = backend.open_session(SessionSpec(session_id="sdk-reverse-kl", backend_id="train", batch_codec="verl"))

    result = backend.forward_backward_reverse_kl(
        handle,
        TrainOpRequest(
            op=TrainOp.FORWARD_BACKWARD_REVERSE_KL,
            batch_codec="verl",
            batch_payload={
                "algorithm": "grpo",
                "reference_model_path": "mint://checkpoints/ref.pt",
                "data": [
                    {
                        "student_input": {"chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]},
                        "reference_input": {"chunks": [{"type": "encoded_text", "tokens": [1, 2, 4]}]},
                        "target_tokens": {"data": [2, 3], "shape": [2], "dtype": "int64"},
                        "weights": {"data": [1.0, 1.0], "shape": [2], "dtype": "float32"},
                    }
                ],
            },
            options={"temperature": 1.0},
        ),
    )

    trainer = backend._trainer(handle)
    assert result.state.step == 1
    assert result.outputs["algorithm"] == "reverse_kl"
    assert result.outputs["execution_framework"] == "mint_training"
    datums = trainer.last_batch
    assert datums[0]["loss_fn_inputs"]["reference_model_path"] == "mint://checkpoints/ref.pt"
    assert datums[0]["loss_fn_inputs"]["temperature"] == 1.0
