from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from jit_apfrl.aggregation.async_server import (
    AsyncFederatedServer,
    PendingUpload,
    compute_update_quality,
    make_upload_events,
    synchronous_average,
)
from jit_apfrl.algorithms.methods import is_local_only, method_spec, uses_nr_gat
from jit_apfrl.algorithms.ppo import GraphPolicyBatch, Trajectory, graph_attention_policy_update, ppo_update
from jit_apfrl.config import write_resolved_config
from jit_apfrl.env import CausalSchedulingEnv
from jit_apfrl.metrics import summarize_episode
from jit_apfrl.models import ActorCritic, RobustGraphAttention
from jit_apfrl.privacy import BudgetPacer, LocalPrivacyMechanism, laplace_scale


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ring_adjacency(nodes: int) -> torch.Tensor:
    adjacency = torch.zeros((nodes, nodes), dtype=torch.bool)
    if nodes == 1:
        adjacency[0, 0] = True
    else:
        for node in range(nodes):
            adjacency[node, (node - 1) % nodes] = True
            adjacency[node, (node + 1) % nodes] = True
    return adjacency


def _set_seeds(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _state_l2_distance(first: dict[str, torch.Tensor], second: dict[str, torch.Tensor]) -> float:
    total = 0.0
    for key in first:
        if torch.is_floating_point(first[key]):
            diff = (first[key].detach().cpu() - second[key].detach().cpu()).reshape(-1)
            total += float(torch.sum(diff * diff))
    return float(np.sqrt(total))


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


class BoundedHistogramStats:
    def __init__(self, edges: np.ndarray) -> None:
        edges = np.asarray(edges, dtype=np.float64)
        if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0.0):
            raise ValueError("histogram edges must be strictly increasing")
        self.edges = edges
        self.histogram = np.zeros(len(edges) - 1, dtype=np.int64)
        self.count = 0
        self.total = 0.0
        self.total_sq = 0.0
        self.minimum = float("inf")
        self.maximum = float("-inf")

    def update(self, values: np.ndarray | float) -> None:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if array.size == 0:
            return
        if not np.all(np.isfinite(array)):
            raise ValueError("diagnostic values must be finite")
        self.count += int(array.size)
        self.total += float(np.sum(array))
        self.total_sq += float(np.sum(array * array))
        self.minimum = min(self.minimum, float(np.min(array)))
        self.maximum = max(self.maximum, float(np.max(array)))
        clipped = np.clip(array, self.edges[0], self.edges[-1])
        counts, _ = np.histogram(clipped, bins=self.edges)
        self.histogram += counts.astype(np.int64)

    def _quantile(self, probability: float) -> float:
        if self.count == 0:
            return 0.0
        target = probability * max(self.count - 1, 0)
        cumulative = np.cumsum(self.histogram)
        index = int(np.searchsorted(cumulative, target + 1.0, side="left"))
        index = min(max(index, 0), len(self.histogram) - 1)
        return float(0.5 * (self.edges[index] + self.edges[index + 1]))

    def summary(self) -> dict[str, Any]:
        if self.count == 0:
            return {}
        mean = self.total / self.count
        variance = max(0.0, self.total_sq / self.count - mean * mean)
        return {
            "count": int(self.count),
            "mean": float(mean),
            "std": float(np.sqrt(variance)),
            "p05": self._quantile(0.05),
            "p50": self._quantile(0.50),
            "median": self._quantile(0.50),
            "p90": self._quantile(0.90),
            "p95": self._quantile(0.95),
            "p99": self._quantile(0.99),
            "min": float(self.minimum),
            "max": float(self.maximum),
            "histogram_edges": self.edges,
            "histogram_counts": self.histogram,
        }


class PrivacyDiagnosticsAccumulator:
    def __init__(self, feature_dim: int, epsilon_max: float, scale_max: float) -> None:
        self.feature_dim = int(feature_dim)
        self.coordinate_count = 0
        self.release_count = 0
        self.epsilon_stats = BoundedHistogramStats(np.linspace(0.0, max(float(epsilon_max), 1e-6), 257))
        self.scale_stats = BoundedHistogramStats(np.linspace(0.0, max(float(scale_max), 1e-6), 257))
        perturbation_edges = np.concatenate(([0.0], np.geomspace(1e-9, 1e4, 512)))
        self.raw_abs_stats = BoundedHistogramStats(perturbation_edges)
        self.effective_abs_stats = BoundedHistogramStats(perturbation_edges)
        self.raw_stats = BoundedHistogramStats(np.linspace(-1e4, 1e4, 513))
        self.effective_stats = BoundedHistogramStats(np.linspace(-1e4, 1e4, 513))
        self.signal_power = np.zeros(self.feature_dim, dtype=np.float64)
        self.raw_noise_power = np.zeros(self.feature_dim, dtype=np.float64)
        self.effective_noise_power = np.zeros(self.feature_dim, dtype=np.float64)
        self.clipped_coordinates = 0

    def update(self, diagnostic: dict[str, Any]) -> None:
        signal = np.asarray(diagnostic["_diagnostic_signal"], dtype=np.float64)
        raw_noise = np.asarray(diagnostic["_diagnostic_raw_noise"], dtype=np.float64)
        released = np.asarray(diagnostic["_diagnostic_released"], dtype=np.float64)
        effective_noise = released - signal
        self.release_count += 1
        self.coordinate_count += int(signal.size)
        self.epsilon_stats.update(float(diagnostic["epsilon"]))
        self.scale_stats.update(float(diagnostic["scale"]))
        self.raw_abs_stats.update(np.abs(raw_noise))
        self.effective_abs_stats.update(np.abs(effective_noise))
        self.raw_stats.update(raw_noise)
        self.effective_stats.update(effective_noise)
        self.signal_power += signal**2
        self.raw_noise_power += raw_noise**2
        self.effective_noise_power += effective_noise**2
        self.clipped_coordinates += int(np.count_nonzero(np.abs(effective_noise - raw_noise) > 1e-12))

    def summary(self) -> dict[str, Any]:
        signal_total = float(np.sum(self.signal_power))
        raw_noise_total = float(np.sum(self.raw_noise_power))
        effective_noise_total = float(np.sum(self.effective_noise_power))
        release_count = max(self.release_count, 1)
        per_feature_signal = self.signal_power / release_count
        per_feature_raw = self.raw_noise_power / release_count
        per_feature_effective = self.effective_noise_power / release_count
        return {
            "release_count": int(self.release_count),
            "coordinate_count": int(self.coordinate_count),
            "epsilon": self.epsilon_stats.summary(),
            "laplace_scale": self.scale_stats.summary(),
            "raw_perturbation": self.raw_stats.summary(),
            "raw_perturbation_abs": self.raw_abs_stats.summary(),
            "effective_perturbation": self.effective_stats.summary(),
            "effective_perturbation_abs": self.effective_abs_stats.summary(),
            "per_feature_signal_power": per_feature_signal,
            "per_feature_raw_noise_power": per_feature_raw,
            "per_feature_effective_noise_power": per_feature_effective,
            "per_feature_raw_snr": per_feature_signal / np.maximum(per_feature_raw, 1e-12),
            "per_feature_effective_snr": per_feature_signal / np.maximum(per_feature_effective, 1e-12),
            "snr_raw_db": float(10.0 * np.log10(signal_total / max(raw_noise_total, 1e-12))),
            "snr_effective_db": float(10.0 * np.log10(signal_total / max(effective_noise_total, 1e-12))),
            "clipping_fraction": float(self.clipped_coordinates / max(self.coordinate_count, 1)),
        }


def _assert_finite_json(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite_json(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite_json(item, f"{path}[{index}]")
    elif isinstance(value, float) and not np.isfinite(value):
        raise ValueError(f"non-finite output at {path}")


def _write_json(path: Path, value: Any) -> None:
    ready = _json_ready(value)
    _assert_finite_json(ready)
    path.write_text(json.dumps(ready, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_training_checkpoint(
    path: Path,
    *,
    completed_rounds: int,
    global_model: ActorCritic,
    graph_encoder: RobustGraphAttention,
    multipliers: np.ndarray,
    server_round: int,
    simulated_global_time: float,
    config: dict[str, Any],
    pending_uploads: list[PendingUpload] | None = None,
    async_event_records: list[dict[str, float]] | None = None,
    current_launch_index: int | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "completed_rounds": int(completed_rounds),
            "global_model_state": global_model.state_dict(),
            "graph_encoder_state": graph_encoder.state_dict(),
            "dual_multipliers": np.asarray(multipliers, dtype=np.float64),
            "server_round": int(server_round),
            "simulated_global_time": float(simulated_global_time),
            "current_launch_index": int(current_launch_index if current_launch_index is not None else completed_rounds),
            "pending_uploads": pending_uploads or [],
            "async_event_records": async_event_records or [],
            "config": _json_ready(config),
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        path,
    )


def _pending_upload_metadata(pending_uploads: list[PendingUpload]) -> list[dict[str, float]]:
    return [
        {
            "arrival_time": float(upload.arrival_time),
            "client_id": float(upload.client_id),
            "pull_round": float(upload.pull_round),
            "pull_time": float(upload.pull_time),
            "local_train_time": float(upload.local_train_time),
            "network_time": float(upload.network_time),
            "l_sum_ms": float(upload.l_sum_ms),
            "concentration": float(upload.concentration),
        }
        for upload in sorted(pending_uploads, key=lambda item: (item.arrival_time, item.client_id))
    ]


def _apply_due_async_uploads(
    *,
    server: AsyncFederatedServer,
    pending_uploads: list[PendingUpload],
    cutoff_time: float,
    method_safety_quality: bool,
    alpha: float,
    beta: float,
    chi: float,
    l_max_ms: float,
    staleness_scale: float,
    normalized: bool,
    eta_global: float,
    round_index: int,
) -> tuple[list[dict[str, float]], int]:
    applied: list[dict[str, float]] = []
    pending_uploads.sort(key=lambda item: (item.arrival_time, item.client_id))
    accepted = 0
    while pending_uploads and pending_uploads[0].arrival_time <= cutoff_time:
        upload = pending_uploads.pop(0)
        staleness = int(server.global_round - upload.pull_round)
        if staleness < 0:
            raise RuntimeError("negative asynchronous staleness")
        server_round_at_arrival = int(server.global_round)
        if method_safety_quality:
            quality = compute_update_quality(
                float(upload.l_sum_ms),
                staleness,
                float(upload.concentration),
                alpha=alpha,
                beta=beta,
                chi=chi,
                l_max_ms=l_max_ms,
                staleness_scale=staleness_scale,
                normalized=normalized,
            )
        else:
            quality = compute_update_quality(
                0.0,
                staleness,
                0.0,
                alpha=0.0,
                beta=beta,
                chi=0.0,
                l_max_ms=1.0,
                staleness_scale=staleness_scale,
                normalized=True,
            )
        mixing = server.apply(upload.client_state, quality.value)
        applied.append(
            {
                "round": float(round_index),
                "node": float(upload.client_id),
                "arrival_time": float(upload.arrival_time),
                "pull_time": float(upload.pull_time),
                "pull_round": float(upload.pull_round),
                "server_round_at_arrival": float(server_round_at_arrival),
                "local_train_time": float(upload.local_train_time),
                "network_time": float(upload.network_time),
                "staleness": float(staleness),
                "quality": float(quality.value),
                "normalized_jit": float(quality.normalized_jit),
                "normalized_staleness": float(quality.normalized_staleness),
                "concentration": float(quality.concentration),
                "weighted_jit": float(quality.weighted_jit),
                "weighted_staleness": float(quality.weighted_staleness),
                "weighted_concentration": float(quality.weighted_concentration),
                "quality_denominator": float(quality.denominator),
                "ema_mixing": float(mixing),
                "eta_global": float(eta_global),
            }
        )
        accepted += 1
    return applied, accepted


def _latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    checkpoints = sorted(checkpoint_dir.glob("round_*.pt"))
    return checkpoints[-1] if checkpoints else None


def _device_metadata(device: torch.device) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
    }
    if device.type == "cuda":
        metadata["gpu_name"] = torch.cuda.get_device_name(device)
        properties = torch.cuda.get_device_properties(device)
        metadata["gpu_total_memory_bytes"] = int(properties.total_memory)
    else:
        metadata["gpu_name"] = None
    return metadata


def run_experiment(
    config: dict[str, Any],
    output_dir: str | Path,
    *,
    device: str | torch.device = "cpu",
    checkpoint_callback: Callable[[int, Path], None] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    seed = int(config["run"]["seed"])
    _set_seeds(seed)
    torch_device = torch.device(device)
    device_info = _device_metadata(torch_device)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(config["data"]["dataset_path"])
    if not dataset_path.is_absolute():
        config_path = Path(config.get("_config_path", Path.cwd()))
        project_root = config_path.parent.parent if config_path.is_file() else Path.cwd()
        dataset_path = (project_root / dataset_path).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    method = method_spec(str(config["run"]["method"]), config)
    env = CausalSchedulingEnv.from_npz(
        dataset_path,
        config=config["environment"],
        split=str(config["data"].get("split", "train")),
    )
    model_config = config["model"]
    nr_gat_enabled = uses_nr_gat(method)
    local_only = is_local_only(method)
    eta_noise_effective = float(model_config["eta_noise"]) if method.mad_regularization else 0.0
    global_model = ActorCritic(
        input_dim=env.state_dim if local_only else 2 * env.state_dim,
        action_count=env.action_count,
        hidden_dim=int(model_config["hidden_dim"]),
    ).to(torch_device)
    graph_encoder = RobustGraphAttention(
        state_dim=env.state_dim,
        eta_noise=eta_noise_effective,
    ).to(torch_device)
    adjacency = _ring_adjacency(env.num_nodes).to(torch_device)
    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in global_model.parameters())
    graph_parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in graph_encoder.parameters())

    training = config["training"]
    privacy_config = config["privacy"]
    aggregation_config = config["aggregation"]
    federated_rounds = int(training["federated_rounds"])
    checkpoint_interval = int(training.get("checkpoint_interval", 0) or 0)
    quality_jit_reference_ms = float(
        aggregation_config.get(
            "jit_reference_ms_for_quality",
            float(aggregation_config["l_max_ms"]) * env.horizon,
        )
    )
    dual_jit_budget_ms = float(
        aggregation_config.get(
            "jit_budget_ms_per_episode",
            float(aggregation_config["l_max_ms"]) * env.horizon,
        )
    )
    normalize_dual_terms = bool(aggregation_config.get("normalize_dual_terms", True))
    dual_penalty_normalizer_ms = dual_jit_budget_ms if normalize_dual_terms else 1.0
    multipliers = np.zeros(env.num_nodes, dtype=np.float64)
    server_round = 0
    simulated_global_time = 0.0
    async_launch_interval = float(aggregation_config.get("async_launch_interval", 1.0))
    if async_launch_interval <= 0.0:
        raise ValueError("aggregation.async_launch_interval must be positive")
    async_server = AsyncFederatedServer(global_model.state_dict(), float(aggregation_config["eta_global"]))
    pending_uploads: list[PendingUpload] = []
    async_event_records: list[dict[str, float]] = []
    round_records: list[dict[str, Any]] = []
    privacy_records: list[dict[str, float]] = []
    diagnostics_config = config.get("diagnostics", {})
    detailed_privacy = bool(diagnostics_config.get("privacy_detail", False))
    privacy_accumulator = None
    if detailed_privacy:
        privacy_accumulator = PrivacyDiagnosticsAccumulator(
            int(config["environment"]["state_dim"]),
            epsilon_max=float(privacy_config["epsilon_max"]),
            scale_max=laplace_scale(1.0, float(privacy_config["epsilon_min"])),
        )
    total_communication_bytes = 0
    total_upload_events = 0
    start_round = 0
    if bool(training.get("resume_from_checkpoint", False)):
        latest = _latest_checkpoint(output_path / "checkpoints")
        progress_path = output_path / "checkpoint_progress.json"
        if latest is not None and progress_path.exists():
            checkpoint = torch.load(latest, map_location=torch_device, weights_only=False)
            global_model.load_state_dict(checkpoint["global_model_state"])
            graph_encoder.load_state_dict(checkpoint["graph_encoder_state"])
            multipliers = np.asarray(checkpoint["dual_multipliers"], dtype=np.float64)
            server_round = int(checkpoint["server_round"])
            simulated_global_time = float(checkpoint["simulated_global_time"])
            pending_uploads = list(checkpoint.get("pending_uploads", []))
            async_event_records = list(checkpoint.get("async_event_records", []))
            async_server = AsyncFederatedServer(global_model.state_dict(), float(aggregation_config["eta_global"]))
            async_server.global_round = server_round
            start_round = int(checkpoint["completed_rounds"])
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            round_records = list(progress.get("round_records", []))
            privacy_records = list(progress.get("privacy_records", []))
            total_communication_bytes = int(progress.get("total_communication_bytes", 0))
            total_upload_events = int(progress.get("total_upload_events", 0))
            random.setstate(checkpoint["python_random_state"])
            np.random.set_state(checkpoint["numpy_random_state"])
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
            if torch.cuda.is_available() and checkpoint.get("torch_cuda_rng_state_all") is not None:
                torch.cuda.set_rng_state_all(
                    [state.cpu() for state in checkpoint["torch_cuda_rng_state_all"]]
                )

    for round_index in range(start_round, federated_rounds):
        round_seed = seed + 1009 * round_index
        launch_time = float(round_index) * async_launch_interval
        qualities: list[dict[str, float]] = []
        if method.asynchronous:
            due, accepted = _apply_due_async_uploads(
                server=async_server,
                pending_uploads=pending_uploads,
                cutoff_time=launch_time,
                method_safety_quality=method.safety_quality,
                alpha=float(aggregation_config["alpha"]),
                beta=float(aggregation_config["beta"]),
                chi=float(aggregation_config["chi"]),
                l_max_ms=quality_jit_reference_ms,
                staleness_scale=float(aggregation_config["staleness_scale"]),
                normalized=bool(aggregation_config["normalize_quality_terms"]),
                eta_global=float(aggregation_config["eta_global"]),
                round_index=round_index,
            )
            qualities.extend(due)
            async_event_records.extend(due)
            server_round = async_server.global_round
            simulated_global_time = launch_time
            global_model.load_state_dict(async_server.state)
        pull_round = server_round
        pull_time = launch_time if method.asynchronous else simulated_global_time
        state = env.reset(round_seed)
        local_models = [copy.deepcopy(global_model).to(torch_device) for _ in range(env.num_nodes)]
        trajectories = [Trajectory() for _ in range(env.num_nodes)]
        graph_batch = GraphPolicyBatch()
        pacers = [
            BudgetPacer(
                horizon=env.horizon,
                episode_budget=float(privacy_config["episode_budget"]),
                epsilon_min=float(privacy_config["epsilon_min"]),
                epsilon_max=float(privacy_config["epsilon_max"]),
                rho_per_ms=float(privacy_config["rho_per_ms"]),
            )
            for _ in range(env.num_nodes)
        ]
        mechanisms = [
            LocalPrivacyMechanism(method.privacy_mode, env.state_dim, round_seed + 37 * node)
            for node in range(env.num_nodes)
        ]
        release_history: list[torch.Tensor] = []
        reward_steps: list[np.ndarray] = []
        violation_steps: list[np.ndarray] = []
        delay_steps: list[np.ndarray] = []
        deadline_steps: list[np.ndarray] = []
        backlog_steps: list[np.ndarray] = []
        allocation_steps: list[np.ndarray] = []
        attention_steps: list[np.ndarray] = []
        workload_pressure_steps: list[np.ndarray] = []

        for _ in range(env.horizon):
            deadlines = env.current_deadline_ms()
            workload_pressure_steps.append(state[:, 11].copy())
            releases: list[np.ndarray] = []
            for node in range(env.num_nodes):
                if method.privacy_mode == "off":
                    epsilon = 1.0
                elif method.dynamic_privacy:
                    epsilon = pacers[node].next_epsilon(float(deadlines[node]))
                else:
                    epsilon = float(privacy_config["episode_budget"]) / env.horizon
                released, diagnostic = mechanisms[node].release(state[node], epsilon)
                releases.append(released)
                if privacy_accumulator is not None and method.privacy_mode != "off":
                    privacy_accumulator.update(diagnostic)
                if method.privacy_mode != "off":
                    scalar_diagnostic = {key: value for key, value in diagnostic.items() if not key.startswith("_")}
                    privacy_records.append(
                        {
                            "round": float(round_index),
                            "node": float(node),
                            "step": float(env.current_step),
                            **scalar_diagnostic,
                        }
                    )

            state_tensor = torch.as_tensor(state, device=torch_device, dtype=torch.float32)
            released_tensor = torch.as_tensor(np.asarray(releases), device=torch_device, dtype=torch.float32)
            release_history.append(released_tensor.detach())
            window = int(model_config["mad_window"])
            history_tensor = torch.stack(release_history[-window:])
            with torch.no_grad():
                if local_only:
                    aggregate = torch.zeros_like(state_tensor)
                    attention = torch.zeros_like(adjacency, dtype=torch.float32)
                    features = state_tensor
                elif nr_gat_enabled:
                    aggregate, attention = graph_encoder(
                        state_tensor,
                        released_tensor,
                        history_tensor,
                        adjacency,
                    )
                elif method.privacy_mode != "off":
                    aggregate = released_tensor
                    attention = torch.eye(env.num_nodes, device=torch_device, dtype=torch.float32)
                else:
                    aggregate = torch.zeros_like(state_tensor)
                    attention = adjacency.to(dtype=torch.float32)
                    attention = attention / attention.sum(dim=1, keepdim=True)
                if not local_only:
                    features = torch.cat([state_tensor, aggregate], dim=-1)
                actions: list[int] = []
                log_probabilities: list[float] = []
                values: list[float] = []
                for node, local_model in enumerate(local_models):
                    logits, value = local_model(features[node])
                    distribution = torch.distributions.Categorical(logits=logits)
                    action = distribution.sample()
                    actions.append(int(action.item()))
                    log_probabilities.append(float(distribution.log_prob(action).cpu()))
                    values.append(float(value.cpu()))

            next_state, physical_rewards, done, info = env.step(np.asarray(actions, dtype=np.int64))
            dual_violation_signal = info["jit_violation_ms"] / dual_penalty_normalizer_ms
            augmented_rewards = physical_rewards - multipliers * dual_violation_signal
            if nr_gat_enabled:
                graph_batch.append(
                    state_tensor,
                    released_tensor,
                    history_tensor,
                    torch.as_tensor(actions, device=torch_device, dtype=torch.int64),
                    torch.as_tensor(log_probabilities, device=torch_device, dtype=torch.float32),
                    torch.as_tensor(augmented_rewards, device=torch_device, dtype=torch.float32),
                    torch.as_tensor(values, device=torch_device, dtype=torch.float32),
                )
            for node in range(env.num_nodes):
                trajectories[node].append(
                    features[node],
                    actions[node],
                    log_probabilities[node],
                    float(augmented_rewards[node]),
                    values[node],
                )
            reward_steps.append(physical_rewards)
            violation_steps.append(info["jit_violation_ms"])
            delay_steps.append(info["processing_delay_ms"])
            deadline_steps.append(info["deadline_ms"])
            backlog_steps.append(info["next_backlog"])
            allocation_steps.append(info["allocated_resource"])
            attention_steps.append(attention.detach().cpu().numpy())
            state = next_state
            if done:
                break

        graph_loss = (
            graph_attention_policy_update(
                global_model,
                graph_encoder,
                graph_batch,
                adjacency,
                training,
                torch_device,
            )
            if nr_gat_enabled
            else {"graph_policy_loss": 0.0, "graph_value_loss": 0.0, "graph_grad_norm": 0.0}
        )
        losses = [
            ppo_update(local_models[node], trajectories[node], training, torch_device)
            for node in range(env.num_nodes)
        ]
        client_states = [local_model.state_dict() for local_model in local_models]
        pre_aggregation_state = {key: value.detach().clone() for key, value in global_model.state_dict().items()}
        client_update_norms = [
            _state_l2_distance(pre_aggregation_state, client_state) for client_state in client_states
        ]
        violations = np.stack(violation_steps)
        l_sum_per_node = violations.sum(axis=0)

        if method.asynchronous:
            workload_pressure = np.mean(np.stack(workload_pressure_steps), axis=0)
            events = make_upload_events(
                pull_round=pull_round,
                pull_time=pull_time,
                workload_pressure=[float(value) for value in workload_pressure],
                seed=round_seed + 7919,
                base_train_time=float(aggregation_config["base_train_time"]),
                train_time_jitter=float(aggregation_config["train_time_jitter"]),
                base_network_time=float(aggregation_config["base_network_time"]),
                network_time_jitter=float(aggregation_config["network_time_jitter"]),
            )
            for event in events:
                node = event.client_id
                concentration = pacers[int(node)].concentration if method.dynamic_privacy else 0.0
                pending_uploads.append(
                    PendingUpload(
                        arrival_time=float(event.arrival_time),
                        client_id=int(node),
                        pull_round=int(event.pull_round),
                        pull_time=float(event.pull_time),
                        local_train_time=float(event.local_train_time),
                        network_time=float(event.network_time),
                        client_state={key: value.detach().clone() for key, value in client_states[int(node)].items()},
                        l_sum_ms=float(l_sum_per_node[node]),
                        concentration=float(concentration),
                    )
                )
            pending_uploads.sort(key=lambda item: (item.arrival_time, item.client_id))
            total_upload_events += env.num_nodes
            server_round = async_server.global_round
            global_model.load_state_dict(async_server.state)
            simulated_global_time = max(simulated_global_time, launch_time)
        else:
            global_model.load_state_dict(synchronous_average(client_states))
            server_round += 1
            simulated_global_time += 1.0
            total_upload_events += env.num_nodes

        global_update_norm = _state_l2_distance(pre_aggregation_state, global_model.state_dict())
        penalty_steps = np.asarray(
            [multipliers * (step_violation / dual_penalty_normalizer_ms) for step_violation in violation_steps],
            dtype=np.float64,
        )
        augmented_reward_steps = np.asarray(
            [
                reward - multipliers * (violation / dual_penalty_normalizer_ms)
                for reward, violation in zip(reward_steps, violation_steps, strict=True)
            ],
            dtype=np.float64,
        )

        if method.dual_constraint:
            step_size = float(training["dual_step_size_initial"]) / (
                (1.0 + pull_round) ** float(training["dual_clock_exponent"])
            )
            multipliers = np.maximum(
                0.0,
                multipliers
                + step_size
                * (
                    (l_sum_per_node - dual_jit_budget_ms) / dual_jit_budget_ms
                    if normalize_dual_terms
                    else (l_sum_per_node - dual_jit_budget_ms)
                ),
            )

        rewards = np.stack(reward_steps)
        delays = np.stack(delay_steps)
        deadlines_array = np.stack(deadline_steps)
        backlog = np.stack(backlog_steps)
        episode_metrics = summarize_episode(rewards, violations, delays, deadlines_array, backlog)
        episode_metrics.update(
            {
                "round": float(round_index),
                "mean_allocated_resource": float(np.mean(np.stack(allocation_steps))),
                "mean_attention_entropy": float(
                    np.mean(
                        -np.sum(
                            np.stack(attention_steps)
                            * np.log(np.clip(np.stack(attention_steps), 1e-12, 1.0)),
                            axis=-1,
                        )
                    )
                ),
                "mean_dual_multiplier": float(np.mean(multipliers)),
                "server_round": float(server_round),
                "pending_upload_count": float(len(pending_uploads)),
                "async_launch_time": float(launch_time if method.asynchronous else simulated_global_time),
                "mean_policy_loss": float(np.mean([loss["policy_loss"] for loss in losses])),
                "mean_value_loss": float(np.mean([loss["value_loss"] for loss in losses])),
                "mean_entropy": float(np.mean([loss["entropy"] for loss in losses])),
                "mean_approx_kl": float(np.mean([loss["approx_kl"] for loss in losses])),
                "mean_clip_fraction": float(np.mean([loss["clip_fraction"] for loss in losses])),
                "mean_advantage_mean": float(np.mean([loss["advantage_mean"] for loss in losses])),
                "mean_advantage_std": float(np.mean([loss["advantage_std"] for loss in losses])),
                "mean_advantage_abs_mean": float(np.mean([loss["advantage_abs_mean"] for loss in losses])),
                "graph_grad_norm": float(graph_loss["graph_grad_norm"]),
                "mean_augmented_reward": float(np.mean(augmented_reward_steps)),
                "mean_dual_penalty": float(np.mean(penalty_steps)),
                "dual_penalty_to_raw_reward_abs_ratio": float(
                    np.mean(np.abs(penalty_steps)) / max(np.mean(np.abs(rewards)), 1e-12)
                ),
                "mean_client_update_norm": float(np.mean(client_update_norms)),
                "max_client_update_norm": float(np.max(client_update_norms)),
                "global_update_norm": float(global_update_norm),
                "global_to_client_update_norm_ratio": float(
                    global_update_norm / max(float(np.mean(client_update_norms)), 1e-12)
                ),
                "mean_q_u": float(np.mean([quality["quality"] for quality in qualities])) if qualities else 1.0,
                "p05_q_u": float(np.quantile([quality["quality"] for quality in qualities], 0.05)) if qualities else 1.0,
                "p95_q_u": float(np.quantile([quality["quality"] for quality in qualities], 0.95)) if qualities else 1.0,
                "mean_ema_mixing": float(np.mean([quality["ema_mixing"] for quality in qualities])) if qualities else 1.0,
                "mean_normalized_jit_for_q": float(np.mean([quality["normalized_jit"] for quality in qualities]))
                if qualities
                else 0.0,
                "attention_weight_std": float(np.std(np.stack(attention_steps))),
                "attention_weight_min": float(np.min(np.stack(attention_steps))),
                "attention_weight_max": float(np.max(np.stack(attention_steps))),
            }
        )
        round_records.append({"metrics": episode_metrics, "aggregation": qualities})
        neighbor_messages = (
            int(adjacency.sum().item()) * env.state_dim * 4 * env.horizon
            if nr_gat_enabled
            else 0
        )
        total_communication_bytes += env.num_nodes * parameter_bytes + neighbor_messages + (
            graph_parameter_bytes if nr_gat_enabled else 0
        )
        completed_rounds = round_index + 1
        if checkpoint_interval > 0 and (
            completed_rounds % checkpoint_interval == 0 or completed_rounds == federated_rounds
        ):
            _write_training_checkpoint(
                output_path / "checkpoints" / f"round_{completed_rounds:04d}.pt",
                completed_rounds=completed_rounds,
                global_model=global_model,
                graph_encoder=graph_encoder,
                multipliers=multipliers,
                server_round=server_round,
                simulated_global_time=simulated_global_time,
                config=config,
                pending_uploads=pending_uploads,
                async_event_records=async_event_records,
                current_launch_index=completed_rounds,
            )
            _write_json(
                output_path / "checkpoint_progress.json",
                {
                    "completed_rounds": completed_rounds,
                    "round_records": round_records,
                    "privacy_records": privacy_records,
                    "total_communication_bytes": total_communication_bytes,
                    "total_upload_events": total_upload_events,
                    "pending_upload_count": len(pending_uploads),
                    "pending_uploads": _pending_upload_metadata(pending_uploads),
                    "async_event_records": async_event_records,
                    "latest_checkpoint": f"checkpoints/round_{completed_rounds:04d}.pt",
                },
            )
            if checkpoint_callback is not None:
                checkpoint_callback(completed_rounds, output_path)

    if method.asynchronous and pending_uploads:
        final_cutoff = max(float(upload.arrival_time) for upload in pending_uploads)
        final_due, _ = _apply_due_async_uploads(
            server=async_server,
            pending_uploads=pending_uploads,
            cutoff_time=final_cutoff,
            method_safety_quality=method.safety_quality,
            alpha=float(aggregation_config["alpha"]),
            beta=float(aggregation_config["beta"]),
            chi=float(aggregation_config["chi"]),
            l_max_ms=quality_jit_reference_ms,
            staleness_scale=float(aggregation_config["staleness_scale"]),
            normalized=bool(aggregation_config["normalize_quality_terms"]),
            eta_global=float(aggregation_config["eta_global"]),
            round_index=federated_rounds,
        )
        async_event_records.extend(final_due)
        server_round = async_server.global_round
        simulated_global_time = max(simulated_global_time, final_cutoff)
        global_model.load_state_dict(async_server.state)

    metric_rows = [record["metrics"] for record in round_records]
    summary_keys = [key for key in metric_rows[0] if key not in {"round", "server_round"}]
    summary = {key: float(np.mean([row[key] for row in metric_rows])) for key in summary_keys}
    summary["final_round_mean_reward"] = float(metric_rows[-1]["mean_reward"])
    summary["final_round_mean_jit_violation_ms"] = float(metric_rows[-1]["mean_jit_violation_ms"])
    summary["communication_bytes"] = float(total_communication_bytes)
    summary["upload_event_count"] = float(total_upload_events)
    summary["accepted_async_upload_count"] = float(len(async_event_records))
    summary["simulated_global_time"] = float(simulated_global_time)

    privacy_summary: dict[str, Any]
    if privacy_records:
        scales = np.asarray([record["scale"] for record in privacy_records], dtype=np.float64)
        noise_rms = np.asarray([record["noise_rms"] for record in privacy_records], dtype=np.float64)
        signal_rms = np.asarray([record["signal_rms"] for record in privacy_records], dtype=np.float64)
        privacy_summary = {
            "mode": method.privacy_mode,
            "release_count": len(privacy_records),
            "scale_min": float(scales.min()),
            "scale_max": float(scales.max()),
            "scale_mean": float(scales.mean()),
            "noise_to_signal_rms": float(np.mean(noise_rms / np.maximum(signal_rms, 1e-12))),
            "episode_budget": float(privacy_config["episode_budget"]),
            "max_episode_privacy_loss": float(privacy_config["episode_budget"]),
        }
        if privacy_accumulator is not None:
            privacy_summary["detailed"] = privacy_accumulator.summary()
    else:
        privacy_summary = {"mode": "off", "release_count": 0}

    elapsed = time.perf_counter() - started
    runtime = {
        "wall_seconds": float(elapsed),
        "started_unix_seconds": float(time.time() - elapsed),
        "completed_unix_seconds": float(time.time()),
        **device_info,
    }
    metadata = {
        "method": method.name,
        "seed": seed,
        "dataset_path": str(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy_version": np.__version__,
    }
    metrics_payload = {"summary": summary, "rounds": round_records, "async_events": async_event_records}
    _write_json(output_path / "metrics.json", metrics_payload)
    _write_json(output_path / "runtime.json", runtime)
    _write_json(output_path / "privacy_diagnostics.json", {"summary": privacy_summary, "releases": privacy_records})
    _write_json(output_path / "run_metadata.json", metadata)
    write_resolved_config(config, output_path / "resolved_config.yaml")
    return {
        "metrics": metrics_payload,
        "runtime": runtime,
        "privacy": privacy_summary,
        "metadata": metadata,
        "output_dir": str(output_path.resolve()),
    }
