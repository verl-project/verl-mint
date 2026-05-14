from __future__ import annotations

import argparse
import os
import socket
import threading
import time
from pathlib import Path

import mint
import uvicorn
from mint import types

from verl_mint import create_app
from verl_mint.backends.verl import VerlInferenceBackend, VerlTrainingBackend
from verl_mint.contracts import BackendKind, BackendSpec
from verl_mint.milestone1 import MILESTONE1_BASE_MODEL_ID
from verl_mint.service import MintService
from verl_mint.storage import LocalStorageRepo, default_storage_root, storage_env_name

SMOKE_CLIENT_TOKEN = "unused"


class ServerThread(threading.Thread):
    def __init__(
        self,
        *,
        host: str,
        port: int,
        storage_root: Path,
    ) -> None:
        super().__init__(daemon=True)
        service = MintService(storage=LocalStorageRepo(storage_root))

        service.backends.register(
            BackendSpec(
                backend_id="verl-infer",
                kind=BackendKind.INFERENCE,
                provider="verl",
                model_family="qwen",
            ),
            VerlInferenceBackend(
                backend_kwargs={"model_id": MILESTONE1_BASE_MODEL_ID},
            ),
        )

        service.backends.register(
            BackendSpec(
                backend_id="verl-train",
                kind=BackendKind.TRAINING,
                provider="verl",
                model_family="qwen",
            ),
            VerlTrainingBackend(
                backend_kwargs={"model_id": MILESTONE1_BASE_MODEL_ID},
            ),
        )

        app = create_app(service)
        config = uvicorn.Config(app, host=host, port=port, log_level="warning")
        self.server = uvicorn.Server(config)

    def run(self) -> None:
        self.server.run()

    def stop(self) -> None:
        self.server.should_exit = True


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_port(host: str, port: int, timeout_s: float = 30.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex((host, port)) == 0:
                return
        time.sleep(0.1)
    raise TimeoutError(f"server did not start on {host}:{port} within {timeout_s}s")


def tiny_datum() -> types.Datum:
    return types.Datum(
        model_input=types.ModelInput(chunks=[types.EncodedTextChunk(tokens=[10, 11, 12])]),
        loss_fn_inputs={
            "target_tokens": types.TensorData(data=[11, 12, 13], dtype="int64", shape=[3]),
            "loss_mask": types.TensorData(data=[1.0, 1.0, 1.0], dtype="float32", shape=[3]),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument(
        "--storage-root",
        default=os.environ.get(storage_env_name(), default_storage_root()),
        help="Shared filesystem root for checkpoint/sampler artifacts. `/tmp` only works for single-node local runs.",
    )
    args = parser.parse_args()

    port = args.port or find_free_port()
    server = ServerThread(
        host=args.host,
        port=port,
        storage_root=Path(args.storage_root),
    )
    server.start()

    try:
        wait_for_port(args.host, port)
        svc = mint.ServiceClient(base_url=f"http://{args.host}:{port}", api_key=SMOKE_CLIENT_TOKEN)
        train = svc.create_lora_training_client(
            base_model=MILESTONE1_BASE_MODEL_ID,
            rank=16,
            user_metadata={"smoke": "official-client-verl"},
        )

        info = train.get_info()
        print("model_id", info.model_id)
        print("model_name", info.model_name)

        fb = train.forward_backward([tiny_datum()], loss_fn="cross_entropy").result()
        print("fb_metrics", fb.metrics)

        opt = train.optim_step(types.AdamParams(learning_rate=3e-4)).result()
        print("opt_metrics", opt.metrics)

        saved = train.save_state("checkpoint-001").result()
        print("saved_path", saved.path)

        loaded = train.load_state(saved.path).result()
        print("loaded_path", loaded.path)

        loaded_with_opt = train.load_state_with_optimizer(saved.path).result()
        print("loaded_with_optimizer_path", loaded_with_opt.path)

        resumed = svc.create_training_client_from_state(saved.path)
        print("resumed_model_id", resumed.get_info().model_id)

        resumed_opt = svc.create_training_client_from_state_with_optimizer(saved.path)
        print("resumed_opt_model_id", resumed_opt.get_info().model_id)

        sampler_saved = train.save_weights_for_sampler("sampler-001").result()
        print("sampler_saved_path", sampler_saved.path)

        named_sampling_client = train.create_sampling_client(sampler_saved.path)
        named_sample = named_sampling_client.sample(
            prompt=types.ModelInput(chunks=[types.EncodedTextChunk(tokens=[1, 2, 3])]),
            num_samples=1,
            sampling_params=types.SamplingParams(max_tokens=2),
        ).result()
        print("named_sampler_sequences", len(named_sample.sequences))

        handoff_sampling_client = train.save_weights_and_get_sampling_client()
        handoff_sample = handoff_sampling_client.sample(
            prompt=types.ModelInput(chunks=[types.EncodedTextChunk(tokens=[1, 2, 3])]),
            num_samples=1,
            sampling_params=types.SamplingParams(max_tokens=2),
        ).result()
        print("handoff_sampler_sequences", len(handoff_sample.sequences))

    finally:
        server.stop()
        server.join(timeout=10)


if __name__ == "__main__":
    main()
