from __future__ import annotations

import torch
from torch import nn


class RobustGraphAttention(nn.Module):








    def __init__(self, state_dim: int, eta_noise: float = 1.0) -> None:
        super().__init__()
        if state_dim < 1 or eta_noise < 0.0:
            raise ValueError("invalid NR-GAT parameters")
        self.state_dim = int(state_dim)
        self.eta_noise = float(eta_noise)
        self.projection = nn.Linear(state_dim, state_dim, bias=False)
        self.attention = nn.Sequential(
            nn.Linear(2 * state_dim, state_dim),
            nn.Tanh(),
            nn.Linear(state_dim, 1, bias=False),
        )
        with torch.no_grad():
            self.projection.weight.copy_(torch.eye(state_dim))

    @staticmethod
    def neighbor_mad(history: torch.Tensor) -> torch.Tensor:

        if history.ndim != 3 or history.shape[0] < 1:
            raise ValueError("history must have shape [window, nodes, dim]")
        center = history.median(dim=0).values
        absolute_l1 = torch.abs(history - center.unsqueeze(0)).sum(dim=-1)
        return absolute_l1.median(dim=0).values

    def forward(
        self,
        local_states: torch.Tensor,
        released_states: torch.Tensor,
        history: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if local_states.shape != released_states.shape:
            raise ValueError("local and released states must have the same shape")
        if local_states.ndim != 2 or local_states.shape[1] != self.state_dim:
            raise ValueError("states must have shape [nodes, state_dim]")
        nodes = local_states.shape[0]
        if adjacency.shape != (nodes, nodes):
            raise ValueError("adjacency must have shape [nodes, nodes]")
        if history.shape[1:] != (nodes, self.state_dim):
            raise ValueError("history shape is incompatible with states")

        finite_release = torch.nan_to_num(released_states, nan=0.0, posinf=1e6, neginf=-1e6)
        finite_history = torch.nan_to_num(history, nan=0.0, posinf=1e6, neginf=-1e6)
        mad = self.neighbor_mad(finite_history)
        pair_features = torch.cat(
            [
                local_states[:, None, :].expand(nodes, nodes, self.state_dim),
                finite_release[None, :, :].expand(nodes, nodes, self.state_dim),
            ],
            dim=-1,
        )
        logits = self.attention(pair_features).squeeze(-1) - self.eta_noise * mad[None, :]
        mask = adjacency.to(dtype=torch.bool, device=logits.device)
        if torch.any(mask.sum(dim=1) == 0):
            raise ValueError("every node must have at least one visible neighbor")
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        attention = torch.softmax(logits, dim=1)
        projected = self.projection(finite_release)
        aggregate = torch.tanh(attention @ projected)
        if not torch.isfinite(aggregate).all() or not torch.isfinite(attention).all():
            raise FloatingPointError("non-finite NR-GAT output")
        return aggregate, attention


class ActorCritic(nn.Module):
    def __init__(self, input_dim: int, action_count: int, hidden_dim: int) -> None:
        super().__init__()
        if min(input_dim, action_count, hidden_dim) < 1:
            raise ValueError("network dimensions must be positive")
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, action_count)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(features)
        return self.actor(hidden), self.critic(hidden).squeeze(-1)
