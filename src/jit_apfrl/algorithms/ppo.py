from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from jit_apfrl.models import ActorCritic
from jit_apfrl.models import RobustGraphAttention


@dataclass
class Trajectory:
    features: list[torch.Tensor] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    log_probabilities: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)

    def append(
        self,
        features: torch.Tensor,
        action: int,
        log_probability: float,
        reward: float,
        value: float,
    ) -> None:
        self.features.append(features.detach().cpu())
        self.actions.append(int(action))
        self.log_probabilities.append(float(log_probability))
        self.rewards.append(float(reward))
        self.values.append(float(value))


def generalized_advantages(
    rewards: np.ndarray,
    values: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    if rewards.shape != values.shape or rewards.ndim != 1:
        raise ValueError("rewards and values must be aligned vectors")
    advantages = np.zeros_like(rewards, dtype=np.float64)
    last_advantage = 0.0
    next_value = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        delta = rewards[index] + gamma * next_value - values[index]
        last_advantage = delta + gamma * gae_lambda * last_advantage
        advantages[index] = last_advantage
        next_value = values[index]
    returns = advantages + values
    return advantages, returns


def ppo_update(
    model: ActorCritic,
    trajectory: Trajectory,
    config: dict[str, float | int],
    device: torch.device,
) -> dict[str, float]:
    if not trajectory.features:
        raise ValueError("cannot update PPO from an empty trajectory")
    features = torch.stack(trajectory.features).to(device=device, dtype=torch.float32)
    actions = torch.tensor(trajectory.actions, device=device, dtype=torch.int64)
    old_log_probabilities = torch.tensor(
        trajectory.log_probabilities, device=device, dtype=torch.float32
    )
    reward_array = np.asarray(trajectory.rewards, dtype=np.float64)
    value_array = np.asarray(trajectory.values, dtype=np.float64)
    advantages_np, returns_np = generalized_advantages(
        reward_array,
        value_array,
        gamma=float(config["gamma"]),
        gae_lambda=float(config["gae_lambda"]),
    )
    advantages = torch.tensor(advantages_np, device=device, dtype=torch.float32)
    returns = torch.tensor(returns_np, device=device, dtype=torch.float32)
    raw_advantages = advantages.detach().clone()
    if len(advantages) > 1 and float(advantages.std(unbiased=False)) > 1e-8:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    last = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "grad_norm": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
        "advantage_mean": float(raw_advantages.mean().detach().cpu()),
        "advantage_std": float(raw_advantages.std(unbiased=False).detach().cpu()),
        "advantage_abs_mean": float(raw_advantages.abs().mean().detach().cpu()),
    }
    for _ in range(int(config["ppo_epochs"])):
        logits, values = model(features)
        distribution = torch.distributions.Categorical(logits=logits)
        log_probabilities = distribution.log_prob(actions)
        ratio = torch.exp(log_probabilities - old_log_probabilities)
        log_ratio = log_probabilities - old_log_probabilities
        clipped = torch.clamp(
            ratio,
            1.0 - float(config["clip_ratio"]),
            1.0 + float(config["clip_ratio"]),
        )
        policy_loss = -torch.minimum(ratio * advantages, clipped * advantages).mean()
        value_loss = torch.mean((returns - values) ** 2)
        entropy = distribution.entropy().mean()
        loss = (
            policy_loss
            + float(config["value_coefficient"]) * value_loss
            - float(config["entropy_coefficient"]) * entropy
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), float(config["max_grad_norm"]))
        optimizer.step()
        if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
            raise FloatingPointError("PPO produced non-finite parameters")
        clip_fraction = torch.mean(
            ((ratio < 1.0 - float(config["clip_ratio"])) | (ratio > 1.0 + float(config["clip_ratio"]))).to(
                dtype=torch.float32
            )
        )
        last = {
            "policy_loss": float(policy_loss.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()),
            "entropy": float(entropy.detach().cpu()),
            "grad_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
            "approx_kl": float((old_log_probabilities - log_probabilities).mean().detach().cpu()),
            "clip_fraction": float(clip_fraction.detach().cpu()),
            "advantage_mean": float(raw_advantages.mean().detach().cpu()),
            "advantage_std": float(raw_advantages.std(unbiased=False).detach().cpu()),
            "advantage_abs_mean": float(raw_advantages.abs().mean().detach().cpu()),
        }
    return last


@dataclass
class GraphPolicyBatch:
    states: list[torch.Tensor] = field(default_factory=list)
    releases: list[torch.Tensor] = field(default_factory=list)
    histories: list[torch.Tensor] = field(default_factory=list)
    actions: list[torch.Tensor] = field(default_factory=list)
    old_log_probabilities: list[torch.Tensor] = field(default_factory=list)
    rewards: list[torch.Tensor] = field(default_factory=list)
    values: list[torch.Tensor] = field(default_factory=list)

    def append(
        self,
        states: torch.Tensor,
        releases: torch.Tensor,
        history: torch.Tensor,
        actions: torch.Tensor,
        old_log_probabilities: torch.Tensor,
        rewards: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        self.states.append(states.detach().cpu())
        self.releases.append(releases.detach().cpu())
        self.histories.append(history.detach().cpu())
        self.actions.append(actions.detach().cpu())
        self.old_log_probabilities.append(old_log_probabilities.detach().cpu())
        self.rewards.append(rewards.detach().cpu())
        self.values.append(values.detach().cpu())


def graph_attention_policy_update(
    policy: ActorCritic,
    graph: RobustGraphAttention,
    batch: GraphPolicyBatch,
    adjacency: torch.Tensor,
    config: dict[str, float | int],
    device: torch.device,
) -> dict[str, float]:
    if not batch.states:
        return {"graph_policy_loss": 0.0, "graph_value_loss": 0.0, "graph_grad_norm": 0.0}
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)

    rewards_np = torch.stack(batch.rewards).numpy().reshape(-1)
    values_np = torch.stack(batch.values).numpy().reshape(-1)
    advantages_np, returns_np = generalized_advantages(
        rewards_np,
        values_np,
        gamma=float(config["gamma"]),
        gae_lambda=float(config["gae_lambda"]),
    )
    advantages = torch.tensor(advantages_np, device=device, dtype=torch.float32).reshape(
        len(batch.states), -1
    )
    returns = torch.tensor(returns_np, device=device, dtype=torch.float32).reshape(len(batch.states), -1)
    if advantages.numel() > 1 and float(advantages.std(unbiased=False)) > 1e-8:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    optimizer = torch.optim.Adam(graph.parameters(), lr=float(config["learning_rate"]))
    last = {"graph_policy_loss": 0.0, "graph_value_loss": 0.0, "graph_grad_norm": 0.0}
    for _ in range(int(config["ppo_epochs"])):
        policy_losses = []
        value_losses = []
        for index, states_cpu in enumerate(batch.states):
            states = states_cpu.to(device=device, dtype=torch.float32)
            releases = batch.releases[index].to(device=device, dtype=torch.float32)
            history = batch.histories[index].to(device=device, dtype=torch.float32)
            actions = batch.actions[index].to(device=device, dtype=torch.int64)
            old_logp = batch.old_log_probabilities[index].to(device=device, dtype=torch.float32)
            aggregate, _ = graph(states, releases, history, adjacency)
            features = torch.cat([states, aggregate], dim=-1)
            logits, values = policy(features)
            distribution = torch.distributions.Categorical(logits=logits)
            logp = distribution.log_prob(actions)
            ratio = torch.exp(logp - old_logp)
            clipped = torch.clamp(
                ratio,
                1.0 - float(config["clip_ratio"]),
                1.0 + float(config["clip_ratio"]),
            )
            policy_losses.append(
                -torch.minimum(ratio * advantages[index], clipped * advantages[index]).mean()
            )
            value_losses.append(torch.mean((returns[index] - values) ** 2))
        policy_loss = torch.stack(policy_losses).mean()
        value_loss = torch.stack(value_losses).mean()
        loss = policy_loss + float(config["value_coefficient"]) * value_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(graph.parameters(), float(config["max_grad_norm"]))
        optimizer.step()
        if not all(torch.isfinite(parameter).all() for parameter in graph.parameters()):
            raise FloatingPointError("NR-GAT update produced non-finite parameters")
        last = {
            "graph_policy_loss": float(policy_loss.detach().cpu()),
            "graph_value_loss": float(value_loss.detach().cpu()),
            "graph_grad_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
        }

    for parameter in policy.parameters():
        parameter.requires_grad_(True)
    policy.train()
    return last
