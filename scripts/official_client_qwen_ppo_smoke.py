from __future__ import annotations

import argparse
from pathlib import Path

from _grpo_smoke_common import (
    ServerThread,
    assign_ppo_rewards,
    checkpoint_delta_l2,
    find_free_port,
    post_json,
    retrieve_future,
    state_delta_l2,
    wait_for_port,
)
from verl_mint import create_app
from verl_mint.backends.qwen_sft import QwenSFTTrainingBackend, QwenTextInferenceBackend
from verl_mint.contracts import BackendKind, BackendSpec
from verl_mint.milestone1 import MILESTONE1_BASE_MODEL_ID
from verl_mint.service import MintService
from verl_mint.storage import LocalStorageRepo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--storage-root", default="/tmp/verl-mint-qwen-ppo-smoke")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    args = parser.parse_args()

    storage = LocalStorageRepo(Path(args.storage_root))
    storage.ensure()
    service = MintService(storage=storage)
    service.backends.register(
        BackendSpec(
            backend_id="qwen-infer-local",
            kind=BackendKind.INFERENCE,
            provider="transformers",
            model_family="qwen",
        ),
        QwenTextInferenceBackend(model_id=MILESTONE1_BASE_MODEL_ID),
    )
    service.backends.register(
        BackendSpec(
            backend_id="qwen-train-local",
            kind=BackendKind.TRAINING,
            provider="transformers-peft",
            model_family="qwen",
        ),
        QwenSFTTrainingBackend(model_id=MILESTONE1_BASE_MODEL_ID),
    )

    port = args.port or find_free_port()
    server = ServerThread(app=create_app(service), host=args.host, port=port)
    server.start()
    base_url = f"http://{args.host}:{port}"

    try:
        wait_for_port(args.host, port)
        create_future = post_json(
            base_url,
            "/create_model",
            {
                "session_id": "ppo-smoke",
                "model_seq_id": 1,
                "base_model": MILESTONE1_BASE_MODEL_ID,
                "backend_id": "qwen-train-local",
                "batch_codec": "torch",
                "lora_config": {"rank": 16},
                "user_metadata": {"smoke": "official-client-qwen-ppo"},
            },
        )
        create_payload = retrieve_future(base_url, create_future)
        model_id = create_payload["model_id"]
        print("model_id", model_id)

        info = post_json(base_url, "/get_info", {"model_id": model_id})
        print("model_name", info["model_name"])

        before_future = post_json(
            base_url,
            "/save_state",
            {"model_id": model_id, "path": "mint://ppo-smoke/checkpoint-before.pt"},
        )
        before_payload = retrieve_future(base_url, before_future)
        before_path = storage.resolve_for_read(before_payload["path"])
        print("before_path", before_payload["path"])

        rollout = post_json(
            base_url,
            "/rollout_sessions",
            {
                "rollout_session_id": "ppo-rollout-1",
                "inference_backend_id": "qwen-infer-local",
                "training_session_id": model_id,
                "metadata": {"algo": "ppo"},
            },
        )
        print("rollout_session_id", rollout["rollout_session_id"])

        collect_payload = post_json(
            base_url,
            "/rollout_sessions/ppo-rollout-1/collect",
            {
                "prompt": {"chunks": [{"type": "encoded_text", "tokens": [101, 102, 103]}]},
                "batch_codec": "torch",
                "num_samples": args.num_samples,
                "sampling": {
                    "max_tokens": args.max_tokens,
                    "temperature": 1.0,
                    "top_k": 20,
                    "top_p": 0.95,
                },
                "metadata": {"group_id": "prompt-1", "algo": "ppo"},
            },
        )
        samples = collect_payload["batch_payload"]["samples"]
        print("collected_samples", len(samples))
        print("first_completion_tokens", samples[0]["completion_tokens"])

        train_batch = assign_ppo_rewards(collect_payload["batch_payload"])
        train_payload = post_json(
            base_url,
            "/rollout_sessions/ppo-rollout-1/train",
            {
                "batch_payload": train_batch,
                "extension_op": "rl",
                "options": {
                    "train_op": "forward_backward_ppo",
                    "algo": "ppo",
                    "clip_coef": 0.2,
                    "value_coef": 0.5,
                    "entropy_coef": 0.01,
                    "value_clip": 0.2,
                },
            },
        )
        train_outputs = train_payload["outputs"]
        print("ppo_loss", train_outputs["loss"])
        print("ppo_policy_loss", train_outputs["policy_loss"])
        print("ppo_value_loss", train_outputs["value_loss"])
        print("ppo_entropy", train_outputs["entropy"])
        print("ppo_approx_kl", train_outputs["approx_kl"])
        print("ppo_clipfrac", train_outputs["clipfrac"])
        print("ppo_reward_mean", train_outputs["reward_mean"])
        print("ppo_return_mean", train_outputs["return_mean"])
        print("ppo_value_mean", train_outputs["value_mean"])
        print("ppo_num_tokens", train_outputs["num_tokens"])

        optim_future = post_json(
            base_url,
            "/optim_step",
            {
                "model_id": model_id,
                "adam_params": {"learning_rate": args.learning_rate},
            },
        )
        optim_payload = retrieve_future(base_url, optim_future)
        print("optim_metrics", optim_payload["metrics"])

        after_future = post_json(
            base_url,
            "/save_state",
            {"model_id": model_id, "path": "mint://ppo-smoke/checkpoint-after.pt"},
        )
        after_payload = retrieve_future(base_url, after_future)
        after_path = storage.resolve_for_read(after_payload["path"])
        print("after_path", after_payload["path"])

        actor_delta = checkpoint_delta_l2(before_path, after_path)
        value_delta = state_delta_l2(before_path, after_path, "value_head_state")
        print("actor_delta_l2", actor_delta)
        print("value_head_delta_l2", value_delta)
        if actor_delta <= 0.0:
            raise RuntimeError("actor weight delta is zero after PPO train and optimizer step")
        if value_delta <= 0.0:
            raise RuntimeError("value-head delta is zero after PPO train and optimizer step")

        load_future = post_json(
            base_url,
            "/load_state",
            {"model_id": model_id, "path": after_payload["path"], "optimizer": True},
        )
        load_payload = retrieve_future(base_url, load_future)
        print("loaded_path", load_payload["path"])

    finally:
        server.stop()
        server.join(timeout=10)


if __name__ == "__main__":
    main()
