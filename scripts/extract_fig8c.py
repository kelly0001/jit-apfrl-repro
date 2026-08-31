from __future__ import annotations
import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np

SETTINGS = [
    "alpha_0p50x",
    "base",
    "alpha_1p50x",
    "beta_0p50x",
    "beta_1p50x",
    "chi_0p50x",
    "chi_1p50x",
]
SEEDS = [3407, 4518, 5629]
COMPONENTS = ["weighted_jit", "weighted_staleness", "weighted_concentration"]


def setting_meta(setting):
    if setting == "base":
        return "alpha", 1.0
    coefficient, multiplier = setting.split("_")
    return coefficient, float(multiplier.replace("p", ".").replace("x", ""))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    rows = []
    for setting in SETTINGS:
        coefficient, multiplier = setting_meta(setting)
        for seed in SEEDS:
            path = a.runs_root / setting / f"seed_{seed}" / "metrics.json"
            data = json.loads(path.read_text())
            events = data["async_events"]
            denominator = [float(event["quality_denominator"]) for event in events]
            for component in COMPONENTS:
                values = [float(event[component]) for event in events]
                rows.append(
                    {
                        "setting_id": setting,
                        "coefficient": coefficient,
                        "multiplier": multiplier,
                        "seed": seed,
                        "component": component,
                        "component_mean": statistics.mean(values),
                        "component_sd": statistics.stdev(values),
                        "component_median": statistics.median(values),
                        "component_p95": float(np.percentile(values, 95)),
                        "denominator_mean": statistics.mean(denominator),
                    }
                )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", newline="") as f:
        fields = [
            "setting_id",
            "coefficient",
            "multiplier",
            "seed",
            "component",
            "component_mean",
            "component_sd",
            "component_median",
            "component_p95",
            "denominator_mean",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
