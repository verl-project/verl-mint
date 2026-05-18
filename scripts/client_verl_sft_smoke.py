from __future__ import annotations

import argparse

from client_scenarios import run_qwen_sft_smoke


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    args = parser.parse_args()

    run_qwen_sft_smoke(
        args.base_url,
        base_model=args.base_model,
        smoke_label="sdk-client-verl",
    )


if __name__ == "__main__":
    main()
