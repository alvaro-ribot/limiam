"""Convenience wrapper for the full replication bundle."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run_command(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run tiny smoke-test settings.")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel jobs for the synthetic simulations.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    quick_flag = ["--quick"] if args.quick else []
    run_command([sys.executable, str(ROOT / "figure2c.py")])
    run_command([sys.executable, str(ROOT / "run_simulations.py"), "--jobs", str(args.jobs), *quick_flag])
    run_command([sys.executable, str(ROOT / "oil_market.py"), *quick_flag])


if __name__ == "__main__":
    main()
