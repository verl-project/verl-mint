from __future__ import annotations

import argparse

from client_scenarios import run_qwen_ppo_smoke


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    args = parser.parse_args()

    run_qwen_ppo_smoke(
        args.base_url,
        base_model=args.base_model,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()
