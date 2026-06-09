from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import torch

from cosmos_predict2._src.predict2.action.models.action_utils import (
    normalize_multi_agent_action,
)

MULTI_AGENT_FLATTENED_KEY = "_multi_agent_flattened"
SHARED_VIDEO_LATENT_KEY = "shared_video_latent"
NUM_AGENTS_KEY = "num_agents"
AGENT_IDS_KEY = "agent_ids"


def _repeat_first_dim(value: Any, repeats: int) -> Any:
    if repeats == 1:
        return value
    if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] > 0:
        return value.repeat_interleave(repeats, dim=0)
    if isinstance(value, list):
        return [item for item in value for _ in range(repeats)]
    return value


def _num_agents_as_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 1
        return int(value.reshape(-1)[0].item())
    if isinstance(value, (list, tuple)):
        if not value:
            return 1
        return int(value[0])
    return int(value)


def prepare_multi_agent_batch_inplace(
    data_batch: MutableMapping[str, Any],
    *,
    input_video_key: str,
    net: Any,
) -> None:
    """Flatten explicit multi-agent streams for the existing shared DiT path.

    Multi-agent data enters as ``video=[B, P, C, T, H, W]`` and usually
    ``action=[B, P, T, D]``. The current DreamDojo VAE/DiT stack consumes
    ``[B, C, T, H, W]``, so each agent stream is flattened into the batch:
    ``[B*P, C, T, H, W]``. ``agent_ids`` keeps the role identity available to
    the action-conditioned network.
    """
    if data_batch.get(MULTI_AGENT_FLATTENED_KEY):
        return
    if input_video_key not in data_batch:
        return

    video = data_batch[input_video_key]
    if not isinstance(video, torch.Tensor) or video.ndim != 6:
        return

    batch, num_agents, channels, frames, height, width = video.shape
    data_batch[input_video_key] = video.reshape(batch * num_agents, channels, frames, height, width)
    data_batch[NUM_AGENTS_KEY] = num_agents
    data_batch[AGENT_IDS_KEY] = torch.arange(num_agents, device=video.device).repeat(batch)

    action = data_batch.get("action")
    if isinstance(action, torch.Tensor):
        if action.ndim == 4:
            if action.shape[0] != batch or action.shape[1] != num_agents:
                raise ValueError(
                    f"Multi-agent action shape {tuple(action.shape)} does not match "
                    f"video batch/agents {batch}/{num_agents}"
                )
            data_batch["action"] = action.reshape(batch * num_agents, action.shape[2], action.shape[3])
        elif action.ndim == 3:
            per_agent_action = normalize_multi_agent_action(
                action,
                action_dim=getattr(net, "action_dim", action.shape[-1]),
                num_agents=num_agents,
                agent_action_dims=getattr(net, "agent_action_dims", None),
            )
            data_batch["action"] = per_agent_action.reshape(
                batch * num_agents,
                per_agent_action.shape[2],
                per_agent_action.shape[3],
            )
        else:
            raise ValueError(f"Expected action [B, T, D] or [B, P, T, D], got {tuple(action.shape)}")

    lam_video = data_batch.get("lam_video")
    if isinstance(lam_video, torch.Tensor):
        if lam_video.ndim == 6:
            # [B, P, T, H, W, C] -> [B*P, T, H, W, C]
            data_batch["lam_video"] = lam_video.reshape(
                batch * num_agents,
                lam_video.shape[2],
                lam_video.shape[3],
                lam_video.shape[4],
                lam_video.shape[5],
            )
        elif lam_video.ndim == 5 and lam_video.shape[0] == batch:
            data_batch["lam_video"] = lam_video.repeat_interleave(num_agents, dim=0)

    for key in (
        "fps",
        "padding_mask",
        "image_size",
        "t5_text_embeddings",
        "t5_text_mask",
        "neg_t5_text_embeddings",
        "neg_t5_text_mask",
    ):
        if key in data_batch:
            data_batch[key] = _repeat_first_dim(data_batch[key], num_agents)

    if "ai_caption" in data_batch:
        data_batch["ai_caption"] = _repeat_first_dim(data_batch["ai_caption"], num_agents)

    data_batch[MULTI_AGENT_FLATTENED_KEY] = True


def encode_shared_video_condition_inplace(
    model: Any,
    data_batch: MutableMapping[str, Any],
    *,
    shared_video_key: str = "shared_video",
    output_key: str = SHARED_VIDEO_LATENT_KEY,
) -> None:
    """Encode a shared visual stream once, then broadcast its latent to agents.

    ``shared_video`` stays at shape ``[B, C, T, H, W]`` after multi-agent
    flattening, while the agent video has become ``[B*P, C, T, H, W]``. This
    function uses the model tokenizer/encoder once for the shared stream and
    repeats the resulting latent over the agent axis so the DiT receives a
    per-agent condition without duplicating the main camera pixels through the
    VAE.
    """
    if output_key in data_batch or shared_video_key not in data_batch:
        return

    shared_video = data_batch[shared_video_key]
    if not isinstance(shared_video, torch.Tensor):
        return
    if shared_video.ndim == 6:
        batch, num_agents = shared_video.shape[:2]
        shared_video = shared_video.reshape(batch * num_agents, *shared_video.shape[2:])
        repeat_agents = 1
    elif shared_video.ndim == 5:
        num_agents = _num_agents_as_int(data_batch.get(NUM_AGENTS_KEY, 1))
        repeat_agents = num_agents
    else:
        raise ValueError(f"Expected shared_video [B,C,T,H,W] or [B,P,C,T,H,W], got {tuple(shared_video.shape)}")

    if shared_video.dtype == torch.uint8:
        shared_video = shared_video.to(**model.tensor_kwargs) / 127.5 - 1.0
    elif torch.is_floating_point(shared_video):
        shared_video = shared_video.to(**model.tensor_kwargs)
    else:
        raise ValueError(f"Unsupported shared_video dtype {shared_video.dtype}")

    shared_latent = model.encode(shared_video).contiguous().float()
    if repeat_agents > 1:
        shared_latent = shared_latent.repeat_interleave(repeat_agents, dim=0)
    data_batch[output_key] = shared_latent
