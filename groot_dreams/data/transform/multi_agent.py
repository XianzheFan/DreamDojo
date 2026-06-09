from typing import Any

import numpy as np
import torch
from pydantic import Field

from groot_dreams.data.transform.base import InvertibleModalityTransform


class MultiAgentStackTransform(InvertibleModalityTransform):
    """Insert an explicit agent axis after concatenating modalities.

    The single-agent DreamDojo path keeps action/state/video as one fused
    stream. Gamma-World-style multi-agent training needs the agent dimension to
    remain explicit so each actor can keep a private stream while later model
    layers decide how to share context.

    Expected input shapes, after ``ConcatTransform``:
      video:  [T, V, H, W, C]
      state:  [T, D]
      action: [T, D]

    Output shapes:
      video:  [P, T, V_per_agent, H, W, C]
      state:  [P, T, D_state_per_agent]
      action: [P, T, D_action_per_agent]
    """

    apply_to: list[str] = Field(
        default_factory=list,
        description="Unused; this transform operates on concat output keys.",
    )
    agent_video_views: list[list[int]] | None = Field(
        default=None,
        description="Per-agent indices into the concat video view axis.",
    )
    agent_state_dims: list[tuple[int, int]] | None = Field(
        default=None,
        description="Per-agent (start, end) slices into the concat state dim.",
    )
    agent_action_dims: list[tuple[int, int]] | None = Field(
        default=None,
        description="Per-agent (start, end) slices into the concat action dim.",
    )

    @property
    def num_agents(self) -> int:
        for groups in (
            self.agent_video_views,
            self.agent_state_dims,
            self.agent_action_dims,
        ):
            if groups is not None:
                return len(groups)
        return 1

    def _validate(self) -> None:
        configured = [
            groups
            for groups in (
                self.agent_video_views,
                self.agent_state_dims,
                self.agent_action_dims,
            )
            if groups is not None
        ]
        if not configured:
            return

        num_agents = len(configured[0])
        for groups in configured:
            assert len(groups) == num_agents, (
                "All configured multi-agent groups must declare the same "
                f"number of agents; got {[len(x) for x in configured]}"
            )

        if self.agent_video_views is not None:
            widths = {len(indices) for indices in self.agent_video_views}
            assert len(widths) == 1, (
                "Each agent must have the same number of video views; got "
                f"{[len(indices) for indices in self.agent_video_views]}"
            )

        if self.agent_state_dims is not None:
            widths = {end - start for start, end in self.agent_state_dims}
            assert len(widths) == 1, (
                "Each agent must have the same state width; got "
                f"{[end - start for start, end in self.agent_state_dims]}"
            )

        if self.agent_action_dims is not None:
            widths = {end - start for start, end in self.agent_action_dims}
            assert len(widths) == 1, (
                "Each agent must have the same action width; got "
                f"{[end - start for start, end in self.agent_action_dims]}"
            )

    @staticmethod
    def _expand_axis0(x):
        if isinstance(x, torch.Tensor):
            return x.unsqueeze(0)
        return np.expand_dims(x, axis=0)

    @staticmethod
    def _stack_axis0(parts):
        if isinstance(parts[0], torch.Tensor):
            return torch.stack(parts, dim=0)
        return np.stack(parts, axis=0)

    @staticmethod
    def _concat(parts, axis: int):
        if isinstance(parts[0], torch.Tensor):
            return torch.cat(parts, dim=axis)
        return np.concatenate(parts, axis=axis)

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        self._validate()
        num_agents = self.num_agents

        if "video" in data:
            video = data["video"]
            assert video.ndim == 5, (
                "MultiAgentStackTransform expects concat video shape "
                f"[T, V, H, W, C], got {tuple(video.shape)}"
            )
            if self.agent_video_views is None:
                data["video"] = self._expand_axis0(video)
            else:
                data["video"] = self._stack_axis0(
                    [video[:, indices, :, :, :] for indices in self.agent_video_views]
                )

        if "state" in data:
            state = data["state"]
            assert state.ndim == 2, (
                "MultiAgentStackTransform expects concat state shape "
                f"[T, D], got {tuple(state.shape)}"
            )
            if self.agent_state_dims is None:
                data["state"] = self._expand_axis0(state)
            else:
                data["state"] = self._stack_axis0(
                    [state[:, start:end] for start, end in self.agent_state_dims]
                )

        if "action" in data:
            action = data["action"]
            assert action.ndim == 2, (
                "MultiAgentStackTransform expects concat action shape "
                f"[T, D], got {tuple(action.shape)}"
            )
            if self.agent_action_dims is None:
                data["action"] = self._expand_axis0(action)
            else:
                data["action"] = self._stack_axis0(
                    [action[:, start:end] for start, end in self.agent_action_dims]
                )

        data["num_agents"] = num_agents
        return data

    def unapply(self, data: dict[str, Any]) -> dict[str, Any]:
        data.pop("num_agents", None)

        if "video" in data:
            video = data["video"]
            if video.shape[0] == 1:
                data["video"] = video.squeeze(0) if isinstance(video, torch.Tensor) else np.squeeze(video, axis=0)
            else:
                data["video"] = self._concat([video[p] for p in range(video.shape[0])], axis=1)

        if "state" in data:
            state = data["state"]
            if state.shape[0] == 1:
                data["state"] = state.squeeze(0) if isinstance(state, torch.Tensor) else np.squeeze(state, axis=0)
            else:
                data["state"] = self._concat([state[p] for p in range(state.shape[0])], axis=-1)

        if "action" in data:
            action = data["action"]
            if action.shape[0] == 1:
                data["action"] = (
                    action.squeeze(0) if isinstance(action, torch.Tensor) else np.squeeze(action, axis=0)
                )
            else:
                data["action"] = self._concat([action[p] for p in range(action.shape[0])], axis=-1)

        return data
