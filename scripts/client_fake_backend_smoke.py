from __future__ import annotations

import argparse

from client_scenarios import run_fake_contract_smoke


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    args = parser.parse_args()

    run_fake_contract_smoke(args.base_url)


if __name__ == "__main__":
    main()
