from __future__ import annotations

import argparse
from pathlib import Path

from server_smoke import ServerThread, find_free_port, qwen_app, wait_for_port
from client_scenarios import run_qwen_sft_smoke
from verl_mint.defaults import DEFAULT_BASE_MODEL_ID


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument(
        "--storage-root",
        default="/tmp/verl-mint-qwen-sft-smoke",
    )
    args = parser.parse_args()

    port = args.port or find_free_port()
    server = ServerThread(
        app=qwen_app(Path(args.storage_root), inference=True),
        host=args.host,
        port=port,
    )
    server.start()

    try:
        wait_for_port(args.host, port)
        run_qwen_sft_smoke(f"http://{args.host}:{port}", base_model=DEFAULT_BASE_MODEL_ID)
    finally:
        server.stop()
        server.join(timeout=5)


if __name__ == "__main__":
    main()
