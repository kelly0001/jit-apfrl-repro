from __future__ import annotations

import numpy as np


def summarize_episode(
    rewards: np.ndarray,
    violations_ms: np.ndarray,
    delays_ms: np.ndarray,
    deadlines_ms: np.ndarray,
    backlog: np.ndarray,
) -> dict[str, float]:
    arrays = [rewards, violations_ms, delays_ms, deadlines_ms, backlog]
    if any(not np.all(np.isfinite(value)) for value in arrays):
        raise ValueError("non-finite value in episode metrics")
    violations = np.asarray(violations_ms, dtype=np.float64)
    return {
        "mean_reward": float(np.mean(rewards)),
        "sum_reward": float(np.sum(rewards)),
        "mean_delay_ms": float(np.mean(delays_ms)),
        "p95_delay_ms": float(np.percentile(delays_ms, 95)),
        "mean_deadline_ms": float(np.mean(deadlines_ms)),
        "mean_jit_violation_ms": float(np.mean(violations)),
        "sum_jit_violation_ms": float(np.sum(violations)),
        "p95_jit_violation_ms": float(np.percentile(violations, 95)),
        "deadline_miss_rate": float(np.mean(violations > 0.0)),
        "mean_backlog": float(np.mean(backlog)),
    }
