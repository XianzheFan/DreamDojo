from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import torch
import torch.nn as nn

AgentActionSlices = tuple[tuple[tuple[int, int], ...], ...]


def _is_slice_pair(value: Sequence) -> bool:
    return len(value) == 2 and all(isinstance(v, int) for v in value)


def _coerce_agent_action_dims(agent_action_dims: Optional[Sequence[Sequence]]) -> Optional[AgentActionSlices]:
    if agent_action_dims is None:
        return None

    dims = []
    for group in agent_action_dims:
        if _is_slice_pair(group):
            slices = ((int(group[0]), int(group[1])),)
        else:
            slices = tuple((int(start), int(end)) for start, end in group)
        for start, end in slices:
            if start < 0 or end <= start:
                raise ValueError(f"Invalid agent action slice {(start, end)}")
        dims.append(slices)
    return tuple(dims)


def _pad_last_dim(x: torch.Tensor, target_dim: int) -> torch.Tensor:
    if x.shape[-1] == target_dim:
        return x
    if x.shape[-1] > target_dim:
        raise ValueError(f"Action dim {x.shape[-1]} exceeds configured per-agent action_dim {target_dim}")
    pad_shape = (*x.shape[:-1], target_dim - x.shape[-1])
    return torch.cat([x, torch.zeros(pad_shape, dtype=x.dtype, device=x.device)], dim=-1)


def _split_fused_agent_action(action: torch.Tensor, num_agents: int, agent_dim: int) -> torch.Tensor:
    batch, steps = action.shape[:2]
    return action.reshape(batch, steps, num_agents, agent_dim).permute(0, 2, 1, 3).contiguous()


def normalize_multi_agent_action(
    action: torch.Tensor,
    *,
    action_dim: int,
    num_agents: int = 1,
    agent_action_dims: Optional[Sequence[Sequence]] = None,
) -> torch.Tensor:
    """Return action as ``[B, P, T, D_agent]``.

    DreamDojo's public datasets currently feed a fused ``[B, T, D]`` action
    vector. Multi-agent training can either pass explicit ``[B, P, T, D]``
    tensors or configure slices into that fused vector.
    """
    dims = _coerce_agent_action_dims(agent_action_dims)
    if dims is not None:
        num_agents = len(dims)

    if action.ndim == 4:
        if dims is not None:
            raise ValueError("agent_action_dims should only be used with fused [B, T, D] action tensors")
        if action.shape[1] != num_agents:
            raise ValueError(f"Expected {num_agents} agents, got action shape {tuple(action.shape)}")
        return _pad_last_dim(action, action_dim)

    if action.ndim != 3:
        raise ValueError(f"Expected action shape [B, T, D] or [B, P, T, D], got {tuple(action.shape)}")

    if dims is not None:
        parts = []
        for agent_slices in dims:
            agent_action = torch.zeros(*action.shape[:-1], action_dim, dtype=action.dtype, device=action.device)
            for start, end in agent_slices:
                if end > action.shape[-1]:
                    raise ValueError(
                        f"Agent action slice {(start, end)} exceeds fused action dim {action.shape[-1]}"
                    )
                if end > action_dim:
                    raise ValueError(f"Agent action slice {(start, end)} exceeds per-agent action_dim {action_dim}")
                agent_action[:, :, start:end] = action[:, :, start:end]
            parts.append(agent_action)
        return torch.stack(parts, dim=1)

    if num_agents > 1:
        fused_dim = action.shape[-1]
        expected_dim = num_agents * action_dim
        if fused_dim == expected_dim:
            return _split_fused_agent_action(action, num_agents, action_dim)
        if fused_dim % num_agents == 0:
            inferred_dim = fused_dim // num_agents
            split = _split_fused_agent_action(action, num_agents, inferred_dim)
            return _pad_last_dim(split, action_dim)
        raise ValueError(
            f"Cannot split fused action dim {fused_dim} into {num_agents} agents; "
            "configure agent_action_dims for non-contiguous or padded layouts."
        )

    return _pad_last_dim(action.unsqueeze(1), action_dim)


def merge_agent_embeddings(
    embedding: torch.Tensor,
    *,
    role_embedding: Optional[nn.Embedding],
    merge: str,
) -> torch.Tensor:
    """Merge ``[B, P, ..., D]`` per-agent embeddings into ``[B, ..., D]``."""
    if embedding.ndim < 3:
        raise ValueError(f"Expected at least [B, P, D], got {tuple(embedding.shape)}")
    if role_embedding is not None:
        num_agents = embedding.shape[1]
        if num_agents > role_embedding.num_embeddings:
            raise ValueError(
                f"Action has {num_agents} agents, but role embedding only supports {role_embedding.num_embeddings}"
            )
        role_ids = torch.arange(num_agents, device=embedding.device)
        role = role_embedding(role_ids).to(dtype=embedding.dtype)
        view_shape = (1, num_agents, *([1] * (embedding.ndim - 3)), role.shape[-1])
        embedding = embedding + role.view(view_shape)

    if merge == "mean":
        return embedding.mean(dim=1)
    if merge == "sum":
        return embedding.sum(dim=1)
    raise ValueError(f"Unsupported agent_action_merge={merge!r}; expected 'mean' or 'sum'")


def add_flattened_agent_role(
    embedding: torch.Tensor,
    *,
    role_embedding: Optional[nn.Embedding],
    agent_ids: Optional[torch.Tensor],
) -> torch.Tensor:
    """Add role embeddings to flattened per-agent batches."""
    if role_embedding is None or agent_ids is None:
        return embedding
    agent_ids = agent_ids.to(device=embedding.device, dtype=torch.long)
    if agent_ids.ndim != 1:
        agent_ids = agent_ids.reshape(-1)
    if agent_ids.shape[0] != embedding.shape[0]:
        raise ValueError(
            f"agent_ids length {agent_ids.shape[0]} does not match embedding batch {embedding.shape[0]}"
        )
    role = role_embedding(agent_ids).to(dtype=embedding.dtype)
    view_shape = (embedding.shape[0], *([1] * (embedding.ndim - 2)), role.shape[-1])
    return embedding + role.view(view_shape)


def zero_init_mlp_output(mlp: nn.Module) -> None:
    nn.init.zeros_(mlp.fc2.weight)
    if mlp.fc2.bias is not None:
        nn.init.zeros_(mlp.fc2.bias)
