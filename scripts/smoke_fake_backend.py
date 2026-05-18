from __future__ import annotations

import argparse
from pathlib import Path

from server_smoke import ServerThread, fake_app, find_free_port, wait_for_port
from client_scenarios import run_fake_contract_smoke


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument(
        "--storage-root",
        default="/tmp/verl-mint-sdk-client-smoke",
    )
    args = parser.parse_args()

    port = args.port or find_free_port()
    server = ServerThread(
        app=fake_app(Path(args.storage_root)),
        host=args.host,
        port=port,
    )
    server.start()

    try:
        wait_for_port(args.host, port)
        run_fake_contract_smoke(f"http://{args.host}:{port}")
    finally:
        server.stop()
        server.join(timeout=5)


if __name__ == "__main__":
    main()
