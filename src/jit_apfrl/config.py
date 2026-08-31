from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml


SUPPORTED_METHODS = {
    "FedPPO",
    "DP-FedRL",
    "Async-FedDRL",
    "JIT-APFRL",
    "JIT-FedRL",
    "JIT-APFRL-no-adaptive-privacy",
    "JIT-APFRL-no-MAD",
    "JIT-APFRL-no-MAD-pure",
    "JIT-APFRL-local-only",
    "JIT-APFRL-no-Q",
    "JIT-APFRL-no-async",
}


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    config = copy.deepcopy(config)
    config["_config_path"] = str(config_path.resolve())
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required = {"run", "environment", "privacy", "model", "training", "aggregation", "data"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"missing configuration sections: {sorted(missing)}")

    run = config["run"]
    method = run.get("method")
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"unsupported method: {method!r}")
    if int(run.get("seed", -1)) < 0:
        raise ValueError("run.seed must be non-negative")

    env = config["environment"]
    if int(env.get("state_dim", -1)) != 14:
        raise ValueError("environment.state_dim must be 14")
    if int(env.get("num_nodes", 0)) < 1 or int(env.get("horizon", 0)) < 1:
        raise ValueError("num_nodes and horizon must be positive")
    levels = env.get("action_levels")
    if not isinstance(levels, list) or len(levels) < 2:
        raise ValueError("environment.action_levels must contain at least two choices")
    if any(not 0.0 < float(level) <= 1.0 for level in levels):
        raise ValueError("every action level must be in (0, 1]")

    privacy = config["privacy"]
    epsilon_min = float(privacy["epsilon_min"])
    epsilon_max = float(privacy["epsilon_max"])
    total = float(privacy["episode_budget"])
    horizon = int(env["horizon"])
    if not 0.0 < epsilon_min <= total / horizon <= epsilon_max:
        raise ValueError("privacy pacing requires 0 < epsilon_min <= E_total/T <= epsilon_max")
    if privacy.get("mode") not in {
        "off",
        "literal_original",
        "manuscript_literal",
        "revised_coordinate",
        "revised_coordinate_postclip",
    }:
        raise ValueError("unknown privacy mode")

    training = config["training"]
    if int(training.get("federated_rounds", 0)) < 1:
        raise ValueError("training.federated_rounds must be positive")
    if int(training.get("ppo_epochs", 0)) < 1:
        raise ValueError("training.ppo_epochs must be positive")


def write_resolved_config(config: dict[str, Any], path: str | Path) -> None:
    output = copy.deepcopy(config)
    output.pop("_config_path", None)
    Path(path).write_text(yaml.safe_dump(output, sort_keys=True), encoding="utf-8")


def canonical_config_json(config: dict[str, Any]) -> str:
    output = copy.deepcopy(config)
    output.pop("_config_path", None)
    return json.dumps(output, sort_keys=True, separators=(",", ":"))
