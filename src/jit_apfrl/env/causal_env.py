from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def jit_violation(delay_ms: np.ndarray | float, deadline_ms: np.ndarray | float) -> np.ndarray:

    return np.maximum(0.0, np.asarray(delay_ms) - np.asarray(deadline_ms))


@dataclass(frozen=True)
class TransitionOutcome:
    allocated_resource: np.ndarray
    completed_work: np.ndarray
    processing_delay_ms: np.ndarray
    deadline_ms: np.ndarray
    jit_violation_ms: np.ndarray
    reward: np.ndarray
    next_backlog: np.ndarray
    next_availability: np.ndarray


class CausalSchedulingEnv:







    STATE_NAMES = (
        "sensor_2",
        "sensor_3",
        "sensor_4",
        "sensor_7",
        "sensor_8",
        "sensor_9",
        "sensor_11",
        "sensor_12",
        "task_cpu",
        "task_memory",
        "task_gpu",
        "background_pressure",
        "backlog",
        "availability",
    )

    def __init__(
        self,
        equipment: np.ndarray,
        health: np.ndarray,
        workloads: np.ndarray,
        config: dict[str, Any],
        equipment_node_pools: np.ndarray | None = None,
        workload_node_pools: np.ndarray | None = None,
    ) -> None:
        self.config = config
        self.num_nodes = int(config["num_nodes"])
        self.horizon = int(config["horizon"])
        self.action_levels = np.asarray(config["action_levels"], dtype=np.float64)
        self.state_dim = int(config["state_dim"])
        if self.state_dim != len(self.STATE_NAMES):
            raise ValueError("state dimension does not match the explicit state definition")
        self.equipment = np.asarray(equipment, dtype=np.float64)
        self.health = np.asarray(health, dtype=np.float64)
        self.workloads = np.asarray(workloads, dtype=np.float64)
        if self.equipment.ndim != 2 or self.equipment.shape[1] != 8:
            raise ValueError("equipment data must have shape [samples, 8]")
        if self.health.shape != (self.equipment.shape[0],):
            raise ValueError("health data must align with equipment rows")
        if self.workloads.ndim != 2 or self.workloads.shape[1] != 4:
            raise ValueError("workloads must have columns cpu, memory, gpu, pressure")
        if min(len(self.equipment), len(self.workloads)) < self.num_nodes:
            raise ValueError("dataset is too small for the configured node count")
        self._equipment_node_pools = self._validate_node_pools(
            equipment_node_pools,
            length=len(self.equipment),
            label="equipment",
        )
        self._workload_node_pools = self._validate_node_pools(
            workload_node_pools,
            length=len(self.workloads),
            label="workload",
        )

        self._rng = np.random.default_rng(0)
        self._seed = 0
        self._step = 0
        self._equipment_indices = np.zeros((self.horizon + 1, self.num_nodes), dtype=np.int64)
        self._workload_indices = np.zeros_like(self._equipment_indices)
        self._uncertainty = np.zeros((self.horizon, self.num_nodes), dtype=np.float64)
        self._capacity_shock = np.zeros_like(self._uncertainty)
        self._mes_timetable_ms = np.zeros((self.horizon, self.num_nodes), dtype=np.float64)
        self.backlog = np.zeros(self.num_nodes, dtype=np.float64)
        self.availability = np.ones(self.num_nodes, dtype=np.float64)
        self.state = np.zeros((self.num_nodes, self.state_dim), dtype=np.float64)

    def _validate_node_pools(
        self,
        pools: np.ndarray | None,
        *,
        length: int,
        label: str,
    ) -> np.ndarray | None:
        if pools is None:
            return None
        pools = np.asarray(pools, dtype=np.int64)
        if pools.ndim != 2 or pools.shape[0] != self.num_nodes or pools.shape[1] < 1:
            raise ValueError(f"{label} node pools must have shape [num_nodes, pool_size]")
        if np.any(pools < 0) or np.any(pools >= length):
            raise ValueError(f"{label} node pools contain out-of-range indices")
        return pools

    @classmethod
    def from_npz(
        cls,
        dataset_path: str | Path,
        config: dict[str, Any],
        split: str = "train",
    ) -> "CausalSchedulingEnv":
        if split not in {"train", "eval"}:
            raise ValueError("split must be train or eval")
        with np.load(dataset_path, allow_pickle=False) as data:
            equipment_pool_key = f"{split}_equipment_node_pools"
            workload_pool_key = f"{split}_workload_node_pools"
            return cls(
                equipment=data[f"{split}_equipment"],
                health=data[f"{split}_health"],
                workloads=data[f"{split}_workloads"],
                config=config,
                equipment_node_pools=data[equipment_pool_key] if equipment_pool_key in data.files else None,
                workload_node_pools=data[workload_pool_key] if workload_pool_key in data.files else None,
            )

    @property
    def action_count(self) -> int:
        return int(len(self.action_levels))

    @property
    def current_step(self) -> int:
        return self._step

    def reset(self, seed: int) -> np.ndarray:
        self._seed = int(seed)
        self._rng = np.random.default_rng(self._seed)
        self._step = 0

        offsets = np.arange(self.horizon + 1, dtype=np.int64)[:, None]
        node_stride = (np.arange(self.num_nodes, dtype=np.int64) * 17)[None, :]
        if self._equipment_node_pools is None:
            equipment_starts = self._rng.integers(0, len(self.equipment), size=self.num_nodes)
            self._equipment_indices = (equipment_starts[None, :] + offsets + node_stride) % len(self.equipment)
        else:
            pool_width = self._equipment_node_pools.shape[1]
            equipment_starts = self._rng.integers(0, pool_width, size=self.num_nodes)
            positions = (equipment_starts[None, :] + offsets + node_stride) % pool_width
            self._equipment_indices = self._equipment_node_pools[np.arange(self.num_nodes)[None, :], positions]
        if self._workload_node_pools is None:
            workload_starts = self._rng.integers(0, len(self.workloads), size=self.num_nodes)
            self._workload_indices = (workload_starts[None, :] + 3 * offsets + node_stride) % len(self.workloads)
        else:
            pool_width = self._workload_node_pools.shape[1]
            workload_starts = self._rng.integers(0, pool_width, size=self.num_nodes)
            positions = (workload_starts[None, :] + 3 * offsets + node_stride) % pool_width
            self._workload_indices = self._workload_node_pools[np.arange(self.num_nodes)[None, :], positions]
        self._uncertainty = self._rng.normal(
            loc=0.0,
            scale=float(self.config["delay_uncertainty_ms"]),
            size=(self.horizon, self.num_nodes),
        )
        self._capacity_shock = self._rng.normal(
            loc=0.0,
            scale=float(self.config["capacity_uncertainty"]),
            size=(self.horizon, self.num_nodes),
        )
        deadline_min = float(self.config["deadline_min_ms"])
        deadline_max = float(self.config["deadline_max_ms"])
        phases = (
            np.arange(self.horizon, dtype=np.float64)[:, None]
            + np.arange(self.num_nodes, dtype=np.float64)[None, :] * 3.0
        ) / max(1, self.horizon)
        timetable_wave = 0.5 + 0.5 * np.sin(2.0 * np.pi * phases)
        self._mes_timetable_ms = deadline_max - (deadline_max - deadline_min) * timetable_wave

        initial_workload = self.workloads[self._workload_indices[0]]
        initial_health = self.health[self._equipment_indices[0]]
        self.backlog = np.asarray(
            float(self.config["initial_backlog"]) + 0.1 * initial_workload[:, 3], dtype=np.float64
        )
        self.availability = np.clip(
            float(self.config["initial_availability"]) * (0.75 + 0.25 * initial_health),
            float(self.config["minimum_availability"]),
            1.0,
        )
        self.state = self._compose_state(0)
        return self.state.copy()

    def _compose_state(self, time_index: int) -> np.ndarray:
        equipment = self.equipment[self._equipment_indices[time_index]]
        workload = self.workloads[self._workload_indices[time_index]]
        state = np.concatenate(
            [
                equipment,
                workload[:, :4],
                np.clip(self.backlog / float(self.config["backlog_normalizer"]), 0.0, 1.0)[:, None],
                self.availability[:, None],
            ],
            axis=1,
        )
        return np.clip(state, 0.0, 1.0)

    def _validate_actions(self, actions: np.ndarray | list[int]) -> np.ndarray:
        actions_array = np.asarray(actions)
        if actions_array.shape != (self.num_nodes,):
            raise ValueError(f"actions must have shape ({self.num_nodes},)")
        if not np.issubdtype(actions_array.dtype, np.integer):
            if not np.all(actions_array == np.floor(actions_array)):
                raise ValueError("actions must be integer indices")
            actions_array = actions_array.astype(np.int64)
        actions_array = actions_array.astype(np.int64, copy=False)
        if np.any(actions_array < 0) or np.any(actions_array >= self.action_count):
            raise ValueError("action index outside the discrete action space")
        return actions_array

    def current_deadline_ms(self) -> np.ndarray:

        if self._step >= self.horizon:
            raise RuntimeError("episode has terminated; call reset")
        return self._mes_timetable_ms[self._step].copy()

    def step(self, actions: np.ndarray | list[int]) -> tuple[np.ndarray, np.ndarray, bool, dict[str, np.ndarray]]:
        if self._step >= self.horizon:
            raise RuntimeError("episode has terminated; call reset")
        action_indices = self._validate_actions(actions)
        allocation = self.action_levels[action_indices]
        workload = self.workloads[self._workload_indices[self._step]]
        health = np.clip(self.health[self._equipment_indices[self._step]], 0.0, 1.0)
        background = np.clip(workload[:, 3], 0.0, 1.0)

        shared_demand = float(np.mean(allocation * workload[:, 2]))
        congestion = np.clip(
            1.0 - float(self.config["interference_strength"]) * shared_demand,
            float(self.config["minimum_congestion_factor"]),
            1.0,
        )
        health_factor = 0.6 + 0.4 * health
        allocated_resource = np.clip(
            allocation * self.availability * health_factor * congestion * (1.0 - 0.35 * background),
            float(self.config["minimum_effective_capacity"]),
            1.0,
        )

        incoming_work = (
            float(self.config["minimum_task_work"])
            + 0.45 * workload[:, 0]
            + 0.25 * workload[:, 1]
            + 0.45 * workload[:, 2]
        )
        work_before_service = self.backlog + incoming_work
        service_capacity = float(self.config["service_rate"]) * allocated_resource
        completed_work = np.minimum(work_before_service, service_capacity)
        next_backlog = np.maximum(0.0, work_before_service - completed_work)
        shared_overflow = max(0.0, float(np.mean(incoming_work - service_capacity)))
        next_backlog += float(self.config["shared_queue_coupling"]) * shared_overflow
        next_backlog = np.clip(next_backlog, 0.0, float(self.config["maximum_backlog"]))

        processing_delay_ms = (
            float(self.config["dispatch_overhead_ms"])
            + float(self.config["base_service_ms"])
            * work_before_service
            / np.maximum(service_capacity, float(self.config["minimum_effective_capacity"]))
            * (1.0 + 0.5 * background)
            + self._uncertainty[self._step]
        )
        processing_delay_ms = np.maximum(float(self.config["minimum_delay_ms"]), processing_delay_ms)

        deadline_ms = self.current_deadline_ms()
        violation_ms = jit_violation(processing_delay_ms, deadline_ms)

        reward = (
            float(self.config["throughput_reward"]) * completed_work
            - float(self.config["allocation_cost"]) * allocation**2
            - float(self.config["backlog_cost"]) * next_backlog
            - float(self.config["physical_tardiness_cost"]) * violation_ms
        )

        trace_availability = 0.55 + 0.45 * health * (1.0 - 0.4 * background)
        next_availability = np.clip(
            float(self.config["availability_persistence"]) * self.availability
            + (1.0 - float(self.config["availability_persistence"])) * trace_availability
            - float(self.config["allocation_fatigue"]) * allocation
            + float(self.config["idle_recovery"]) * (1.0 - allocation)
            + self._capacity_shock[self._step],
            float(self.config["minimum_availability"]),
            1.0,
        )

        self.backlog = next_backlog
        self.availability = next_availability
        self._step += 1
        self.state = self._compose_state(self._step)
        done = self._step >= self.horizon
        outcome = TransitionOutcome(
            allocated_resource=allocated_resource,
            completed_work=completed_work,
            processing_delay_ms=processing_delay_ms,
            deadline_ms=deadline_ms,
            jit_violation_ms=violation_ms,
            reward=reward,
            next_backlog=next_backlog,
            next_availability=next_availability,
        )
        info = {field: np.asarray(getattr(outcome, field)).copy() for field in outcome.__dataclass_fields__}
        return self.state.copy(), reward.copy(), done, info
