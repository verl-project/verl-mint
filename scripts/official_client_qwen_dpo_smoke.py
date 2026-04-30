from __future__ import annotations

import argparse
from pathlib import Path

from _grpo_smoke_common import (
    ServerThread,
    checkpoint_delta_l2,
    find_free_port,
    post_json,
    retrieve_future,
    wait_for_port,
)
from verl_mint import create_app
from verl_mint.backends.qwen_sft import QwenSFTTrainingBackend
from verl_mint.contracts import BackendKind, BackendSpec
from verl_mint.milestone1 import MILESTONE1_BASE_MODEL_ID
from verl_mint.service import MintService
from verl_mint.storage import LocalStorageRepo


def _tensor(data: list[int] | list[float], dtype: str) -> dict:
    return {"data": data, "shape": [len(data)], "dtype": dtype}


def _dpo_row(pair_id: str, role: str, prompt: list[int], completion: list[int]) -> dict:
    return {
        "model_input": {"chunks": [{"type": "encoded_text", "tokens": [*prompt, *completion]}]},
        "loss_fn_inputs": {
            "pair_id": pair_id,
            "role": role,
            "prompt_tokens": _tensor(prompt, "int64"),
            "completion_tokens": _tensor(completion, "int64"),
            "target_tokens": _tensor(completion, "int64"),
            "weights": _tensor([1.0 for _ in completion], "float32"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--storage-root", default="/tmp/verl-mint-qwen-dpo-smoke")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--beta", type=float, default=0.1)
    args = parser.parse_args()

    storage = LocalStorageRepo(Path(args.storage_root))
    storage.ensure()
    service = MintService(storage=storage)
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
                "session_id": "dpo-smoke",
                "model_seq_id": 1,
                "base_model": MILESTONE1_BASE_MODEL_ID,
                "backend_id": "qwen-train-local",
                "batch_codec": "torch",
                "lora_config": {"rank": 16},
                "user_metadata": {"smoke": "official-client-qwen-dpo"},
            },
        )
        create_payload = retrieve_future(base_url, create_future)
        model_id = create_payload["model_id"]
        print("model_id", model_id)

        before_future = post_json(
            base_url,
            "/save_state",
            {"model_id": model_id, "path": "mint://dpo-smoke/checkpoint-before.pt"},
        )
        before_payload = retrieve_future(base_url, before_future)
        before_path = storage.resolve_for_read(before_payload["path"])
        print("before_path", before_payload["path"])

        prompt = [101, 102, 103]
        chosen = [104, 105]
        rejected = [106, 107]
        train_future = post_json(
            base_url,
            "/train_step",
            {
                "model_id": model_id,
                "forward_backward_input": {
                    "loss_fn": "dpo",
                    "loss_fn_config": {
                        "beta": args.beta,
                        "reference_model_path": before_payload["path"],
                    },
                    "data": [
                        _dpo_row("pair-0", "chosen", prompt, chosen),
                        _dpo_row("pair-0", "rejected", prompt, rejected),
                    ],
                },
                "adam_params": {"learning_rate": args.learning_rate},
            },
        )
        train_payload = retrieve_future(base_url, train_future)
        metrics = train_payload["output"]["metrics"]
        print("dpo_loss", metrics["dpo_loss:last"])
        print("dpo_accuracy", metrics["accuracy:last"])
        print("dpo_pi_margin", metrics["pi_margin:last"])
        print("dpo_ref_margin", metrics["ref_margin:last"])
        after_future = post_json(
            base_url,
            "/save_state",
            {"model_id": model_id, "path": "mint://dpo-smoke/checkpoint-after.pt"},
        )
        after_payload = retrieve_future(base_url, after_future)
        after_path = storage.resolve_for_read(after_payload["path"])
        print("after_path", after_payload["path"])

        delta = checkpoint_delta_l2(before_path, after_path)
        print("weight_delta_l2", delta)
        if delta <= 0.0:
            raise RuntimeError("weight delta is zero after DPO train_step")

    finally:
        server.stop()
        server.join(timeout=10)


if __name__ == "__main__":
    main()
