from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MethodSpec:
    name: str
    privacy_mode: str
    dynamic_privacy: bool
    mad_regularization: bool
    asynchronous: bool
    safety_quality: bool
    dual_constraint: bool


def method_spec(name: str, config: dict[str, Any]) -> MethodSpec:
    if name == "FedPPO":
        return MethodSpec(name, "off", False, False, False, False, False)
    if name == "DP-FedRL":
        return MethodSpec(name, str(config["privacy"]["mode"]), False, False, False, False, False)
    if name == "Async-FedDRL":
        return MethodSpec(name, "off", False, False, True, False, False)
    if name in {"JIT-APFRL", "JIT-FedRL"}:
        return MethodSpec(name, str(config["privacy"]["mode"]), True, True, True, True, True)
    if name == "JIT-APFRL-no-adaptive-privacy":
        return MethodSpec(name, str(config["privacy"]["mode"]), False, True, True, True, True)
    if name == "JIT-APFRL-no-MAD":
        return MethodSpec(name, str(config["privacy"]["mode"]), True, False, True, True, True)
    if name == "JIT-APFRL-no-MAD-pure":
        return MethodSpec(
            name=name,
            privacy_mode=str(config["privacy"]["mode"]),
            dynamic_privacy=True,
            mad_regularization=False,
            asynchronous=True,
            safety_quality=True,
            dual_constraint=True,
        )
    if name == "JIT-APFRL-local-only":
        return MethodSpec(
            name=name,
            privacy_mode=str(config["privacy"]["mode"]),
            dynamic_privacy=True,
            mad_regularization=False,
            asynchronous=True,
            safety_quality=True,
            dual_constraint=True,
        )
    if name == "JIT-APFRL-no-Q":
        return MethodSpec(name, str(config["privacy"]["mode"]), True, True, True, False, True)
    if name == "JIT-APFRL-no-async":
        return MethodSpec(name, str(config["privacy"]["mode"]), True, True, False, True, True)
    raise ValueError(f"unknown method: {name}")


def uses_nr_gat(method: MethodSpec) -> bool:
    return method.name in {
        "JIT-APFRL",
        "JIT-FedRL",
        "JIT-APFRL-no-adaptive-privacy",
        "JIT-APFRL-no-Q",
        "JIT-APFRL-no-async",
        "JIT-APFRL-no-MAD-pure",
    }


def is_local_only(method: MethodSpec) -> bool:
    return method.name == "JIT-APFRL-local-only"
