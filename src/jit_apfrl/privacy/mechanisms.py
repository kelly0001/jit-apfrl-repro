from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def laplace_scale(sensitivity: float, epsilon: float) -> float:
    if sensitivity <= 0.0:
        raise ValueError("sensitivity must be positive")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    return float(sensitivity / epsilon)


@dataclass
class BudgetPacer:


    horizon: int
    episode_budget: float
    epsilon_min: float
    epsilon_max: float
    rho_per_ms: float

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        if not 0.0 < self.epsilon_min <= self.episode_budget / self.horizon <= self.epsilon_max:
            raise ValueError("invalid pacing parameters")
        if self.rho_per_ms < 0.0:
            raise ValueError("rho must be non-negative")
        self.remaining = float(self.episode_budget)
        self.step_index = 0
        self.spent: list[float] = []

    def next_epsilon(self, tolerance_ms: float) -> float:
        if self.step_index >= self.horizon:
            raise RuntimeError("privacy budget sequence is complete")
        raw = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * np.exp(
            -self.rho_per_ms * max(0.0, float(tolerance_ms))
        )
        future_steps = self.horizon - 1 - self.step_index
        upper = self.remaining - future_steps * self.epsilon_min
        epsilon = float(min(raw, upper))
        if epsilon <= 0.0:
            raise RuntimeError("budget pacing generated a non-positive release")
        self.remaining = max(0.0, self.remaining - epsilon)
        self.spent.append(epsilon)
        self.step_index += 1
        return epsilon

    @property
    def concentration(self) -> float:
        if not self.spent:
            return 0.0
        uniform = self.episode_budget / self.horizon
        return float(sum(max(0.0, value - uniform) for value in self.spent) / self.episode_budget)


class LocalPrivacyMechanism:









    def __init__(self, mode: str, dimension: int, seed: int) -> None:
        if mode not in {
            "off",
            "literal_original",
            "manuscript_literal",
            "revised_coordinate",
            "revised_coordinate_postclip",
        }:
            raise ValueError("unsupported privacy mode")
        if dimension < 1:
            raise ValueError("dimension must be positive")
        self.mode = mode
        self.dimension = int(dimension)
        self.rng = np.random.default_rng(seed)

    @property
    def sensitivity(self) -> float:
        if self.mode in {"literal_original", "manuscript_literal"}:
            return float(self.dimension)
        if self.mode in {"revised_coordinate", "revised_coordinate_postclip"}:
            return 1.0
        return 0.0

    def scale(self, epsilon: float) -> float:
        if self.mode == "off":
            return 0.0
        return laplace_scale(self.sensitivity, epsilon)

    def release(self, normalized_state: np.ndarray, epsilon: float) -> tuple[np.ndarray, dict[str, float]]:
        state = np.asarray(normalized_state, dtype=np.float64)
        if state.shape[-1] != self.dimension:
            raise ValueError("released state has the wrong dimension")
        if not np.all(np.isfinite(state)):
            raise ValueError("released state must be finite")
        clipped = np.clip(state, 0.0, 1.0)
        if self.mode == "off":
            return clipped.copy(), {"epsilon": float("inf"), "sensitivity": 0.0, "scale": 0.0}
        scale = self.scale(epsilon)
        noise = self.rng.laplace(0.0, scale, size=clipped.shape)
        released = clipped + noise
        if self.mode == "revised_coordinate_postclip":
            released = np.clip(released, 0.0, 1.0)
        return released, {
            "epsilon": float(epsilon),
            "sensitivity": self.sensitivity,
            "scale": scale,
            "mean_abs_noise": float(np.mean(np.abs(noise))),
            "signal_rms": float(np.sqrt(np.mean(clipped**2))),
            "noise_rms": float(np.sqrt(np.mean(noise**2))),
            "_diagnostic_signal": clipped,
            "_diagnostic_raw_noise": noise,
            "_diagnostic_released": released,
        }


def episode_privacy_loss(epsilons: list[float] | np.ndarray) -> float:
    values = np.asarray(epsilons, dtype=np.float64)
    if np.any(values <= 0.0) or not np.all(np.isfinite(values)):
        raise ValueError("every composed epsilon must be positive and finite")
    return float(np.sum(values))
