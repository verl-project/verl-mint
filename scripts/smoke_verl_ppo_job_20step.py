from __future__ import annotations

import argparse
import os
from pathlib import Path

from server_smoke import ServerThread, find_free_port, verl_ppo_app, wait_for_port
from client_scenarios import run_verl_ppo_job_smoke
from verl_mint.defaults import DEFAULT_BASE_MODEL_ID
from verl_mint.storage import default_storage_root, storage_env_name


def _overrides(args: argparse.Namespace) -> list[str]:
    mini_batch = args.ppo_mini_batch_size or args.train_batch_size
    values = [
        "algorithm.adv_estimator=gae",
        "critic.enable=True",
        "trainer.val_before_train=False",
        f"data.train_files={args.train_files}",
        f"data.val_files={args.val_files}",
        f"data.train_batch_size={args.train_batch_size}",
        f"data.val_batch_size={args.val_batch_size}",
        f"data.max_prompt_length={args.max_prompt_length}",
        f"data.max_response_length={args.max_response_length}",
        "data.shuffle=False",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        f"actor_rollout_ref.model.path={args.model_path}",
        f"critic.model.path={args.model_path}",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        f"actor_rollout_ref.model.lora_rank={args.lora_rank}",
        f"actor_rollout_ref.model.lora_alpha={args.lora_alpha}",
        "actor_rollout_ref.model.target_modules=all-linear",
        f"actor_rollout_ref.actor.optim.lr={args.actor_lr}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={mini_batch}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={args.ppo_micro_batch_size_per_gpu}",
        "actor_rollout_ref.actor.use_kl_loss=False",
        "actor_rollout_ref.actor.use_dynamic_bsz=False",
        "actor_rollout_ref.actor.fsdp_config.param_offload=True",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
        "actor_rollout_ref.actor.ulysses_sequence_parallel_size=1",
        f"actor_rollout_ref.rollout.name={args.rollout_backend}",
        f"actor_rollout_ref.rollout.n={args.rollout_n}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={args.gpu_memory_utilization}",
        f"actor_rollout_ref.rollout.max_model_len={args.max_model_len}",
        f"actor_rollout_ref.rollout.max_num_batched_tokens={args.max_num_batched_tokens}",
        f"actor_rollout_ref.rollout.max_num_seqs={args.max_num_seqs}",
        "actor_rollout_ref.rollout.enable_chunked_prefill=False",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={args.ppo_micro_batch_size_per_gpu}",
        f"critic.optim.lr={args.critic_lr}",
        f"critic.ppo_micro_batch_size_per_gpu={args.ppo_micro_batch_size_per_gpu}",
        "critic.use_dynamic_bsz=False",
        "critic.model.enable_gradient_checkpointing=True",
        "trainer.critic_warmup=0",
        "trainer.logger=['console']",
        f"trainer.project_name={args.project_name}",
        f"trainer.experiment_name={args.experiment_name}",
        f"trainer.n_gpus_per_node={args.n_gpus_per_node}",
        "trainer.nnodes=1",
        "trainer.save_freq=-1",
        "trainer.test_freq=-1",
        "trainer.total_epochs=1",
        f"trainer.total_training_steps={args.total_steps}",
    ]
    values.extend(args.override or [])
    return values


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--storage-root", default=os.environ.get(storage_env_name(), default_storage_root()))
    parser.add_argument("--config-dir", default=os.environ.get("VERL_CONFIG_DIR"))
    parser.add_argument("--config-name", default="ppo_trainer")
    parser.add_argument("--run-ppo", default="verl.trainer.main_ppo:run_ppo")
    parser.add_argument("--remote-run-ppo", action="store_true")
    parser.add_argument(
        "--runtime-env-root",
        default=os.environ.get("VERL_MINT_RUNTIME_ENV_ROOT"),
    )
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH", DEFAULT_BASE_MODEL_ID))
    parser.add_argument("--train-files", required=True)
    parser.add_argument("--val-files", required=True)
    parser.add_argument("--total-steps", type=_positive_int, default=20)
    parser.add_argument("--train-batch-size", type=_positive_int, default=2)
    parser.add_argument("--val-batch-size", type=_positive_int, default=2)
    parser.add_argument("--ppo-mini-batch-size", type=int, default=0)
    parser.add_argument("--ppo-micro-batch-size-per-gpu", type=_positive_int, default=1)
    parser.add_argument("--max-prompt-length", type=_positive_int, default=128)
    parser.add_argument("--max-response-length", type=_positive_int, default=128)
    parser.add_argument("--max-model-len", type=_positive_int, default=256)
    parser.add_argument("--max-num-batched-tokens", type=_positive_int, default=256)
    parser.add_argument("--max-num-seqs", type=_positive_int, default=4)
    parser.add_argument("--rollout-backend", default="vllm")
    parser.add_argument("--rollout-n", type=_positive_int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.05)
    parser.add_argument("--n-gpus-per-node", type=_positive_int, default=1)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--actor-lr", type=float, default=3e-5)
    parser.add_argument("--critic-lr", type=float, default=1e-5)
    parser.add_argument("--project-name", default="verl_mint_ppo")
    parser.add_argument("--experiment-name", default="qwen3_06b_lora_20step")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    if args.config_dir is None:
        raise SystemExit("--config-dir or VERL_CONFIG_DIR is required")

    trainer_config = {
        "run_ppo": args.run_ppo,
        "remote_run_ppo": args.remote_run_ppo,
        "hydra": {
            "config_dir": args.config_dir,
            "config_name": args.config_name,
            "overrides": _overrides(args),
        },
    }
    if args.runtime_env_root:
        trainer_config["runtime_env_root"] = args.runtime_env_root

    port = args.port or find_free_port()
    server = ServerThread(
        app=verl_ppo_app(
            Path(args.storage_root),
            model_path=args.model_path,
            trainer_config=trainer_config,
        ),
        host=args.host,
        port=port,
    )
    server.start()

    try:
        wait_for_port(args.host, port, timeout_s=120)
        run_verl_ppo_job_smoke(
            f"http://{args.host}:{port}",
            base_model=args.model_path,
            learning_rate=args.actor_lr,
        )
    finally:
        server.stop()
        server.join(timeout=10)


if __name__ == "__main__":
    main()
