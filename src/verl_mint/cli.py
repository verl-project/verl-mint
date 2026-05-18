from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import uvicorn

from verl_mint import create_app
from verl_mint.contracts import BackendKind, BackendSpec, GenerateRequest, GenerateResult
from verl_mint.backends.base import InferenceBackend
from verl_mint.service import MintService
from verl_mint.storage import LocalStorageRepo, default_storage_root, storage_env_name


class _NoopInferenceBackend(InferenceBackend):
    def generate(self, req: GenerateRequest) -> GenerateResult:
        return GenerateResult(text=req.prompt, stop_reason="noop", raw={"provider": "noop"})


def _trainer_config(args: argparse.Namespace) -> dict[str, Any]:
    trainer: dict[str, Any] = {
        "run_ppo": args.run_ppo,
        "remote_run_ppo": args.remote_run_ppo,
    }
    if args.runtime_env_root:
        trainer["runtime_env_root"] = args.runtime_env_root
    if args.verl_config_dir:
        trainer["hydra"] = {
            "config_dir": args.verl_config_dir,
            "config_name": args.verl_config_name,
            "overrides": list(args.override or []),
        }
    return trainer


def _serve(args: argparse.Namespace) -> None:
    service = MintService(storage=LocalStorageRepo(Path(args.storage_root)))
    service.backends.register(
        BackendSpec(
            backend_id=args.inference_backend_id,
            kind=BackendKind.INFERENCE,
            provider="noop",
            model_family="qwen",
        ),
        _NoopInferenceBackend(),
    )

    if args.backend == "verl":
        from verl_mint.backends.verl import VerlTrainingBackend

        backend = VerlTrainingBackend(
            model_path=args.model_path,
            backend_kwargs={"trainer": _trainer_config(args)},
        )
        provider = "verl"
    else:
        from verl_mint.backends.qwen_sft import QwenSFTTrainingBackend

        backend = QwenSFTTrainingBackend(model_id=args.model_path)
        provider = "transformers-peft"

    service.backends.register(
        BackendSpec(
            backend_id=args.training_backend_id,
            kind=BackendKind.TRAINING,
            provider=provider,
            model_family="qwen",
        ),
        backend,
    )
    uvicorn.run(create_app(service), host=args.host, port=args.port, log_level=args.log_level)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="verl-mint")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="start a simple verl-mint HTTP service")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--storage-root", default=os.environ.get(storage_env_name(), default_storage_root()))
    serve.add_argument("--backend", choices=["verl", "qwen"], default="verl")
    serve.add_argument("--model-path", default=os.environ.get("VERL_MINT_MODEL_PATH", "Qwen/Qwen3-0.6B"))
    serve.add_argument("--training-backend-id", default="verl-train")
    serve.add_argument("--inference-backend-id", default="noop-infer")
    serve.add_argument("--verl-config-dir", default=os.environ.get("VERL_CONFIG_DIR"))
    serve.add_argument("--verl-config-name", default=os.environ.get("VERL_CONFIG_NAME", "ppo_trainer"))
    serve.add_argument("--run-ppo", default=os.environ.get("VERL_MINT_RUN_PPO", "verl.trainer.main_ppo:run_ppo"))
    serve.add_argument("--remote-run-ppo", action="store_true", default=os.environ.get("VERL_MINT_REMOTE_RUN_PPO") == "1")
    serve.add_argument(
        "--runtime-env-root",
        default=os.environ.get("VERL_MINT_RUNTIME_ENV_ROOT"),
    )
    serve.add_argument("--override", action="append", default=[])
    serve.add_argument("--log-level", default="info")
    serve.set_defaults(func=_serve)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
