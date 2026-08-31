from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


@dataclass(frozen=True)
class UpdateQuality:
    value: float
    normalized_jit: float
    normalized_staleness: float
    concentration: float
    weighted_jit: float
    weighted_staleness: float
    weighted_concentration: float
    denominator: float


@dataclass(frozen=True)
class UploadEvent:
    arrival_time: float
    client_id: int
    pull_round: int
    pull_time: float
    local_train_time: float
    network_time: float

    @property
    def total_delay(self) -> float:
        return float(self.local_train_time + self.network_time)


@dataclass
class PendingUpload:
    arrival_time: float
    client_id: int
    pull_round: int
    pull_time: float
    local_train_time: float
    network_time: float
    client_state: dict[str, torch.Tensor]
    l_sum_ms: float
    concentration: float

    @property
    def total_delay(self) -> float:
        return float(self.local_train_time + self.network_time)


def compute_update_quality(
    l_sum_ms: float,
    staleness: int,
    concentration: float,
    *,
    alpha: float,
    beta: float,
    chi: float,
    l_max_ms: float,
    staleness_scale: float,
    normalized: bool = True,
) -> UpdateQuality:
    if l_sum_ms < 0.0 or staleness < 0 or concentration < 0.0:
        raise ValueError("quality inputs must be non-negative")
    if min(alpha, beta, chi) < 0.0:
        raise ValueError("quality coefficients must be non-negative")
    if normalized:
        if l_max_ms <= 0.0 or staleness_scale <= 0.0:
            raise ValueError("normalization scales must be positive")
        jit_term = l_sum_ms / l_max_ms
        stale_term = staleness / staleness_scale
    else:
        jit_term = l_sum_ms
        stale_term = float(staleness)
    weighted_jit = alpha * jit_term
    weighted_staleness = beta * stale_term
    weighted_concentration = chi * concentration
    denominator = 1.0 + weighted_jit + weighted_staleness + weighted_concentration
    value = 1.0 / denominator
    return UpdateQuality(
        value=float(value),
        normalized_jit=float(jit_term),
        normalized_staleness=float(stale_term),
        concentration=float(concentration),
        weighted_jit=float(weighted_jit),
        weighted_staleness=float(weighted_staleness),
        weighted_concentration=float(weighted_concentration),
        denominator=float(denominator),
    )


def synchronous_average(
    updates: list[Mapping[str, torch.Tensor]], weights: list[float] | None = None
) -> dict[str, torch.Tensor]:
    if not updates:
        raise ValueError("at least one update is required")
    keys = set(updates[0])
    if any(set(update) != keys for update in updates):
        raise ValueError("all model updates must have identical parameter keys")
    if weights is None:
        weights = [1.0 / len(updates)] * len(updates)
    if len(weights) != len(updates) or any(weight < 0.0 for weight in weights):
        raise ValueError("invalid aggregation weights")
    total = float(sum(weights))
    if total <= 0.0:
        raise ValueError("aggregation weights must have positive mass")
    normalized = [float(weight) / total for weight in weights]
    averaged: dict[str, torch.Tensor] = {}
    for key in sorted(keys):
        first = updates[0][key].detach()
        if not torch.is_floating_point(first):
            averaged[key] = first.clone()
            continue
        value = torch.zeros_like(first)
        for weight, update in zip(normalized, updates, strict=True):
            value.add_(update[key].detach(), alpha=weight)
        if not torch.isfinite(value).all():
            raise FloatingPointError("non-finite synchronous aggregate")
        averaged[key] = value
    return averaged


class AsyncFederatedServer:
    def __init__(self, initial_state: Mapping[str, torch.Tensor], eta_global: float) -> None:
        if not 0.0 < eta_global <= 1.0:
            raise ValueError("eta_global must be in (0, 1]")
        self.state = {key: value.detach().clone() for key, value in initial_state.items()}
        self.eta_global = float(eta_global)
        self.global_round = 0

    def apply(self, client_state: Mapping[str, torch.Tensor], quality: float) -> float:
        if set(client_state) != set(self.state):
            raise ValueError("client and server parameter keys differ")
        if not 0.0 < quality <= 1.0:
            raise ValueError("quality must be in (0, 1]")
        mixing = self.eta_global * float(quality)
        for key, current in self.state.items():
            candidate = client_state[key].detach().to(device=current.device)
            if torch.is_floating_point(current):
                updated = (1.0 - mixing) * current + mixing * candidate
                if not torch.isfinite(updated).all():
                    raise FloatingPointError("non-finite asynchronous aggregate")
                self.state[key] = updated
            else:
                self.state[key] = candidate.clone()
        self.global_round += 1
        return mixing


def make_upload_events(
    *,
    pull_round: int,
    pull_time: float,
    workload_pressure: list[float],
    seed: int,
    base_train_time: float,
    train_time_jitter: float,
    base_network_time: float,
    network_time_jitter: float,
    jit_delay_ms: list[float] | None = None,
) -> list[UploadEvent]:







    if min(base_train_time, base_network_time) < 0.0:
        raise ValueError("base event delays must be non-negative")
    rng = torch.Generator().manual_seed(int(seed))
    events: list[UploadEvent] = []
    for client_id, pressure in enumerate(workload_pressure):
        train_noise = float(torch.rand((), generator=rng).item()) * max(0.0, train_time_jitter)
        net_noise = float(torch.rand((), generator=rng).item()) * max(0.0, network_time_jitter)
        local_train_time = (
            base_train_time
            + max(0.0, float(pressure)) * train_time_jitter
            + train_noise
        )
        network_time = base_network_time + max(0.0, float(pressure)) * network_time_jitter + net_noise
        arrival = float(pull_time + local_train_time + network_time)
        events.append(
            UploadEvent(
                arrival_time=arrival,
                client_id=int(client_id),
                pull_round=int(pull_round),
                pull_time=float(pull_time),
                local_train_time=float(local_train_time),
                network_time=float(network_time),
            )
        )
    return sorted(events, key=lambda event: (event.arrival_time, event.client_id))
