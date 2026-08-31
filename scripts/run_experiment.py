from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_apfrl.algorithms import run_experiment
from jit_apfrl.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = run_experiment(load_config(args.config), args.output, device=args.device)
    print(json.dumps({"summary": result["metrics"]["summary"], "runtime": result["runtime"]}, indent=2))


if __name__ == "__main__":
    main()
