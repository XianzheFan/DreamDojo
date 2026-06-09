# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Tuple

import torch
import torch.amp as amp
import torch.nn as nn
from einops import rearrange

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.action.models.action_utils import (
    _coerce_agent_action_dims,
    add_flattened_agent_role,
    merge_agent_embeddings,
    normalize_multi_agent_action,
    zero_init_mlp_output,
)
from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.networks.minimal_v4_dit import MiniTrainDIT


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.activation = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ActionConditionedMinimalV1LVGDiT(MiniTrainDIT):
    def __init__(self, *args, timestep_scale: float = 1.0, **kwargs):
        assert "in_channels" in kwargs, "in_channels must be provided"
        latent_channels = kwargs["in_channels"]
        kwargs["in_channels"] += 1  # Add 1 for the condition mask

        action_dim = kwargs.get("action_dim", 10 * 8)
        if "action_dim" in kwargs:
            del kwargs["action_dim"]
        self.action_dim = action_dim

        self.agent_action_dims = _coerce_agent_action_dims(kwargs.pop("agent_action_dims", None))
        self.num_agents = int(kwargs.pop("num_agents", len(self.agent_action_dims) if self.agent_action_dims else 1))
        self.agent_action_merge = kwargs.pop("agent_action_merge", "mean")
        self.shared_video_conditioning = bool(kwargs.pop("shared_video_conditioning", False))

        num_action_per_chunk = kwargs.get("num_action_per_chunk", 12)
        if "num_action_per_chunk" in kwargs:
            del kwargs["num_action_per_chunk"]
        self.num_action_per_chunk = num_action_per_chunk

        # NOTE: this is not used in the original code, but we need it for the rectified flow model

        self.timestep_scale = timestep_scale
        log.info(f"timestep_scale: {timestep_scale}")

        super().__init__(*args, **kwargs)

        # add action embedding
        self.action_embedder_B_D = Mlp(
            in_features=action_dim * num_action_per_chunk,
            hidden_features=self.model_channels * 4,
            out_features=self.model_channels,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0,
        )
        self.action_embedder_B_3D = Mlp(
            in_features=action_dim * num_action_per_chunk,
            hidden_features=self.model_channels * 4,
            out_features=self.model_channels * 3,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0,
        )
        if self.num_agents > 1:
            self.agent_action_role_B_D = nn.Embedding(self.num_agents, self.model_channels)
            self.agent_action_role_B_3D = nn.Embedding(self.num_agents, self.model_channels * 3)
            nn.init.normal_(self.agent_action_role_B_D.weight, std=0.02)
            nn.init.normal_(self.agent_action_role_B_3D.weight, std=0.02)
        else:
            self.agent_action_role_B_D = None
            self.agent_action_role_B_3D = None
        if self.shared_video_conditioning:
            self.shared_video_embedder_B_D = Mlp(
                in_features=latent_channels,
                hidden_features=self.model_channels * 4,
                out_features=self.model_channels,
                act_layer=lambda: nn.GELU(approximate="tanh"),
                drop=0,
            )
            self.shared_video_embedder_B_3D = Mlp(
                in_features=latent_channels,
                hidden_features=self.model_channels * 4,
                out_features=self.model_channels * 3,
                act_layer=lambda: nn.GELU(approximate="tanh"),
                drop=0,
            )
            zero_init_mlp_output(self.shared_video_embedder_B_D)
            zero_init_mlp_output(self.shared_video_embedder_B_3D)
        else:
            self.shared_video_embedder_B_D = None
            self.shared_video_embedder_B_3D = None

    def _get_action_embeddings(
        self,
        action: torch.Tensor,
        agent_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action_is_flattened_agent = agent_ids is not None and action.ndim == 3
        action = normalize_multi_agent_action(
            action,
            action_dim=self.action_dim,
            num_agents=1 if action_is_flattened_agent else self.num_agents,
            agent_action_dims=None if action_is_flattened_agent else self.agent_action_dims,
        )
        action = rearrange(action, "b p t d -> b p (t d)")
        action_emb_B_P_D = self.action_embedder_B_D(action)
        action_emb_B_P_3D = self.action_embedder_B_3D(action)
        action_emb_B_D = merge_agent_embeddings(
            action_emb_B_P_D,
            role_embedding=self.agent_action_role_B_D,
            merge=self.agent_action_merge,
        ).unsqueeze(1)
        action_emb_B_3D = merge_agent_embeddings(
            action_emb_B_P_3D,
            role_embedding=self.agent_action_role_B_3D,
            merge=self.agent_action_merge,
        ).unsqueeze(1)
        action_emb_B_D = add_flattened_agent_role(
            action_emb_B_D,
            role_embedding=self.agent_action_role_B_D,
            agent_ids=agent_ids,
        )
        action_emb_B_3D = add_flattened_agent_role(
            action_emb_B_3D,
            role_embedding=self.agent_action_role_B_3D,
            agent_ids=agent_ids,
        )
        return action_emb_B_D, action_emb_B_3D

    def _get_shared_video_embeddings(self, shared_video_latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.shared_video_embedder_B_D is None or self.shared_video_embedder_B_3D is None:
            raise ValueError("shared_video_latent was provided but shared_video_conditioning is disabled")
        if shared_video_latent.ndim != 5:
            raise ValueError(
                f"Expected shared_video_latent [B,C,T,H,W], got {tuple(shared_video_latent.shape)}"
            )
        pooled_B_C = shared_video_latent.mean(dim=(2, 3, 4)).to(
            dtype=self.shared_video_embedder_B_D.fc1.weight.dtype
        )
        return (
            self.shared_video_embedder_B_D(pooled_B_C).unsqueeze(1),
            self.shared_video_embedder_B_3D(pooled_B_C).unsqueeze(1),
        )

    def forward(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        condition_video_input_mask_B_C_T_H_W: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        data_type: Optional[DataType] = DataType.VIDEO,
        img_context_emb: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        agent_ids: Optional[torch.Tensor] = None,
        shared_video_latent: Optional[torch.Tensor] = None,
        intermediate_feature_ids: Optional[List[int]] = None,
        **kwargs,
    ) -> torch.Tensor | List[torch.Tensor] | Tuple[torch.Tensor, List[torch.Tensor]]:
        del kwargs

        if data_type == DataType.VIDEO:
            x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, condition_video_input_mask_B_C_T_H_W.type_as(x_B_C_T_H_W)], dim=1)
        else:
            B, _, T, H, W = x_B_C_T_H_W.shape
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, torch.zeros((B, 1, T, H, W), dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device)], dim=1
            )

        # NOTE: we need to scale the timesteps, which is added for rectified flow model
        timesteps_B_T = timesteps_B_T * self.timestep_scale

        assert action is not None, "action must be provided"
        action_emb_B_D, action_emb_B_3D = self._get_action_embeddings(action, agent_ids=agent_ids)

        assert isinstance(data_type, DataType), (
            f"Expected DataType, got {type(data_type)}. We need discuss this flag later."
        )
        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = self.prepare_embedded_sequence(
            x_B_C_T_H_W,
            fps=fps,
            padding_mask=padding_mask,
        )

        if self.use_crossattn_projection:
            crossattn_emb = self.crossattn_proj(crossattn_emb)

        if img_context_emb is not None:
            assert self.extra_image_context_dim is not None, (
                "extra_image_context_dim must be set if img_context_emb is provided"
            )
            img_context_emb = self.img_context_proj(img_context_emb)
            context_input = (crossattn_emb, img_context_emb)
        else:
            context_input = crossattn_emb

        with amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
            if timesteps_B_T.ndim == 1:
                timesteps_B_T = timesteps_B_T.unsqueeze(1)
            t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)

            # add action embedding to the timestep embedding and adaln_lora
            t_embedding_B_T_D = t_embedding_B_T_D + action_emb_B_D
            adaln_lora_B_T_3D = adaln_lora_B_T_3D + action_emb_B_3D
            if shared_video_latent is not None:
                shared_emb_B_D, shared_emb_B_3D = self._get_shared_video_embeddings(shared_video_latent)
                t_embedding_B_T_D = t_embedding_B_T_D + shared_emb_B_D.to(dtype=t_embedding_B_T_D.dtype)
                adaln_lora_B_T_3D = adaln_lora_B_T_3D + shared_emb_B_3D.to(dtype=adaln_lora_B_T_3D.dtype)

            t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

        # for logging purpose
        affline_scale_log_info = {}
        affline_scale_log_info["t_embedding_B_T_D"] = t_embedding_B_T_D.detach()
        self.affline_scale_log_info = affline_scale_log_info
        self.affline_emb = t_embedding_B_T_D
        self.crossattn_emb = crossattn_emb

        if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
            assert x_B_T_H_W_D.shape == extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape, (
                f"{x_B_T_H_W_D.shape} != {extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape}"
            )

        B, T, H, W, D = x_B_T_H_W_D.shape

        intermediate_features_outputs = []
        for i, block in enumerate(self.blocks):
            x_B_T_H_W_D = block(
                x_B_T_H_W_D,
                t_embedding_B_T_D,
                context_input,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                adaln_lora_B_T_3D=adaln_lora_B_T_3D,
                extra_per_block_pos_emb=extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
            )
            if intermediate_feature_ids and i in intermediate_feature_ids:
                x_reshaped_for_disc = rearrange(x_B_T_H_W_D, "b tp hp wp d -> b (tp hp wp) d")
                intermediate_features_outputs.append(x_reshaped_for_disc)

        x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
        x_B_C_Tt_Hp_Wp = self.unpatchify(x_B_T_H_W_O)
        if intermediate_feature_ids:
            if len(intermediate_features_outputs) != len(intermediate_feature_ids):
                log.warning(
                    f"Collected {len(intermediate_features_outputs)} intermediate features, "
                    f"but expected {len(intermediate_feature_ids)}. "
                    f"Requested IDs: {intermediate_feature_ids}"
                )
            return x_B_C_Tt_Hp_Wp, intermediate_features_outputs

        return x_B_C_Tt_Hp_Wp


class ActionChunkConditionedMinimalV1LVGDiT(MiniTrainDIT):
    def __init__(self, *args, timestep_scale: float = 1.0, **kwargs):
        assert "in_channels" in kwargs, "in_channels must be provided"
        latent_channels = kwargs["in_channels"]
        kwargs["in_channels"] += 1  # Add 1 for the condition mask

        action_dim = kwargs.get("action_dim", 10 * 8)
        if "action_dim" in kwargs:
            del kwargs["action_dim"]
        self.action_dim = action_dim

        self.agent_action_dims = _coerce_agent_action_dims(kwargs.pop("agent_action_dims", None))
        self.num_agents = int(kwargs.pop("num_agents", len(self.agent_action_dims) if self.agent_action_dims else 1))
        self.agent_action_merge = kwargs.pop("agent_action_merge", "mean")
        self.shared_video_conditioning = bool(kwargs.pop("shared_video_conditioning", False))

        self._num_action_per_latent_frame = kwargs.get("temporal_compression_ratio", 4)
        if "temporal_compression_ratio" in kwargs:
            del kwargs["temporal_compression_ratio"]

        if "num_action_per_chunk" in kwargs:
            del kwargs["num_action_per_chunk"]

        self._hidden_dim_in_action_embedder = kwargs.get("hidden_dim_in_action_embedder", None)
        if "hidden_dim_in_action_embedder" in kwargs:
            del kwargs["hidden_dim_in_action_embedder"]

        # NOTE: this is not used in the original code, but we need it for the rectified flow model
        self.timestep_scale = timestep_scale

        super().__init__(*args, **kwargs)

        if self._hidden_dim_in_action_embedder is None:
            self._hidden_dim_in_action_embedder = self.model_channels * 4

        log.info(f"hidden_dim_in_action_embedder: {self._hidden_dim_in_action_embedder}")

        # add action embedding
        self.action_embedder_B_D = Mlp(
            in_features=action_dim * self._num_action_per_latent_frame,
            hidden_features=self._hidden_dim_in_action_embedder,
            out_features=self.model_channels,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0,
        )
        self.action_embedder_B_3D = Mlp(
            in_features=action_dim * self._num_action_per_latent_frame,
            hidden_features=self._hidden_dim_in_action_embedder,
            out_features=self.model_channels * 3,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0,
        )
        if self.num_agents > 1:
            self.agent_action_role_B_D = nn.Embedding(self.num_agents, self.model_channels)
            self.agent_action_role_B_3D = nn.Embedding(self.num_agents, self.model_channels * 3)
            nn.init.normal_(self.agent_action_role_B_D.weight, std=0.02)
            nn.init.normal_(self.agent_action_role_B_3D.weight, std=0.02)
        else:
            self.agent_action_role_B_D = None
            self.agent_action_role_B_3D = None
        if self.shared_video_conditioning:
            self.shared_video_embedder_B_D = Mlp(
                in_features=latent_channels,
                hidden_features=self._hidden_dim_in_action_embedder,
                out_features=self.model_channels,
                act_layer=lambda: nn.GELU(approximate="tanh"),
                drop=0,
            )
            self.shared_video_embedder_B_3D = Mlp(
                in_features=latent_channels,
                hidden_features=self._hidden_dim_in_action_embedder,
                out_features=self.model_channels * 3,
                act_layer=lambda: nn.GELU(approximate="tanh"),
                drop=0,
            )
            zero_init_mlp_output(self.shared_video_embedder_B_D)
            zero_init_mlp_output(self.shared_video_embedder_B_3D)
        else:
            self.shared_video_embedder_B_D = None
            self.shared_video_embedder_B_3D = None

    def _get_action_embeddings(
        self,
        action: torch.Tensor,
        agent_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action_is_flattened_agent = agent_ids is not None and action.ndim == 3
        action = normalize_multi_agent_action(
            action,
            action_dim=self.action_dim,
            num_agents=1 if action_is_flattened_agent else self.num_agents,
            agent_action_dims=None if action_is_flattened_agent else self.agent_action_dims,
        )
        num_actions = action.shape[2]
        if num_actions % self._num_action_per_latent_frame != 0:
            raise ValueError(
                f"Expected action frames ({num_actions}) to be divisible by "
                f"temporal_compression_ratio ({self._num_action_per_latent_frame})"
            )
        action = rearrange(
            action,
            "b p (t k) d -> b p t (k d)",
            k=self._num_action_per_latent_frame,
        )
        action_emb_B_P_T_D = self.action_embedder_B_D(action)
        action_emb_B_P_T_3D = self.action_embedder_B_3D(action)
        action_emb_B_T_D = merge_agent_embeddings(
            action_emb_B_P_T_D,
            role_embedding=self.agent_action_role_B_D,
            merge=self.agent_action_merge,
        )
        action_emb_B_T_3D = merge_agent_embeddings(
            action_emb_B_P_T_3D,
            role_embedding=self.agent_action_role_B_3D,
            merge=self.agent_action_merge,
        )

        zero_pad_action_emb_B_D = torch.zeros_like(action_emb_B_T_D[:, :1, :], device=action_emb_B_T_D.device)
        zero_pad_action_emb_B_3D = torch.zeros_like(action_emb_B_T_3D[:, :1, :], device=action_emb_B_T_3D.device)

        action_emb_B_T_D = torch.cat([zero_pad_action_emb_B_D, action_emb_B_T_D], dim=1)
        action_emb_B_T_3D = torch.cat([zero_pad_action_emb_B_3D, action_emb_B_T_3D], dim=1)
        action_emb_B_T_D = add_flattened_agent_role(
            action_emb_B_T_D,
            role_embedding=self.agent_action_role_B_D,
            agent_ids=agent_ids,
        )
        action_emb_B_T_3D = add_flattened_agent_role(
            action_emb_B_T_3D,
            role_embedding=self.agent_action_role_B_3D,
            agent_ids=agent_ids,
        )
        return action_emb_B_T_D, action_emb_B_T_3D

    def _get_shared_video_embeddings(self, shared_video_latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.shared_video_embedder_B_D is None or self.shared_video_embedder_B_3D is None:
            raise ValueError("shared_video_latent was provided but shared_video_conditioning is disabled")
        if shared_video_latent.ndim != 5:
            raise ValueError(
                f"Expected shared_video_latent [B,C,T,H,W], got {tuple(shared_video_latent.shape)}"
            )
        pooled_B_C = shared_video_latent.mean(dim=(2, 3, 4)).to(
            dtype=self.shared_video_embedder_B_D.fc1.weight.dtype
        )
        return (
            self.shared_video_embedder_B_D(pooled_B_C).unsqueeze(1),
            self.shared_video_embedder_B_3D(pooled_B_C).unsqueeze(1),
        )

    def forward(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        condition_video_input_mask_B_C_T_H_W: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        data_type: Optional[DataType] = DataType.VIDEO,
        img_context_emb: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        agent_ids: Optional[torch.Tensor] = None,
        shared_video_latent: Optional[torch.Tensor] = None,
        intermediate_feature_ids: Optional[List[int]] = None,
        **kwargs,
    ) -> torch.Tensor | List[torch.Tensor] | Tuple[torch.Tensor, List[torch.Tensor]]:
        del kwargs

        if data_type == DataType.VIDEO:
            x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, condition_video_input_mask_B_C_T_H_W.type_as(x_B_C_T_H_W)], dim=1)
        else:
            B, _, T, H, W = x_B_C_T_H_W.shape
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, torch.zeros((B, 1, T, H, W), dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device)], dim=1
            )

        timesteps_B_T = timesteps_B_T * self.timestep_scale

        # calculate action embedding
        assert action is not None, "action must be provided"
        action_emb_B_D, action_emb_B_3D = self._get_action_embeddings(action, agent_ids=agent_ids)

        # NOTE: adjust the action embedding according to the number of frames
        # if condition_video_input_mask_B_C_T_H_W is not None and data_type == DataType.VIDEO:
        #     condition_video_input_mask_B_T = (1 - condition_video_input_mask_B_C_T_H_W[:, 0, :, 0, 0]).unsqueeze(-1)
        #     action_emb_B_D = action_emb_B_D * condition_video_input_mask_B_T
        #     action_emb_B_3D = action_emb_B_3D * condition_video_input_mask_B_T
        # -------------------------------------------------------------

        assert isinstance(data_type, DataType), (
            f"Expected DataType, got {type(data_type)}. We need discuss this flag later."
        )
        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = self.prepare_embedded_sequence(
            x_B_C_T_H_W,
            fps=fps,
            padding_mask=padding_mask,
        )

        if self.use_crossattn_projection:
            crossattn_emb = self.crossattn_proj(crossattn_emb)

        if img_context_emb is not None:
            assert self.extra_image_context_dim is not None, (
                "extra_image_context_dim must be set if img_context_emb is provided"
            )
            img_context_emb = self.img_context_proj(img_context_emb)
            context_input = (crossattn_emb, img_context_emb)
        else:
            context_input = crossattn_emb

        with amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
            if timesteps_B_T.ndim == 1:
                timesteps_B_T = timesteps_B_T.unsqueeze(1)
            t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)

            t_embedding_B_T_D = t_embedding_B_T_D + action_emb_B_D
            adaln_lora_B_T_3D = adaln_lora_B_T_3D + action_emb_B_3D
            if shared_video_latent is not None:
                shared_emb_B_D, shared_emb_B_3D = self._get_shared_video_embeddings(shared_video_latent)
                t_embedding_B_T_D = t_embedding_B_T_D + shared_emb_B_D.to(dtype=t_embedding_B_T_D.dtype)
                adaln_lora_B_T_3D = adaln_lora_B_T_3D + shared_emb_B_3D.to(dtype=adaln_lora_B_T_3D.dtype)

            t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

        # for logging purpose
        affline_scale_log_info = {}
        affline_scale_log_info["t_embedding_B_T_D"] = t_embedding_B_T_D.detach()
        self.affline_scale_log_info = affline_scale_log_info
        self.affline_emb = t_embedding_B_T_D
        self.crossattn_emb = crossattn_emb

        if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
            assert x_B_T_H_W_D.shape == extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape, (
                f"{x_B_T_H_W_D.shape} != {extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape}"
            )

        B, T, H, W, D = x_B_T_H_W_D.shape

        intermediate_features_outputs = []
        for i, block in enumerate(self.blocks):
            x_B_T_H_W_D = block(
                x_B_T_H_W_D,
                t_embedding_B_T_D,
                context_input,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                adaln_lora_B_T_3D=adaln_lora_B_T_3D,
                extra_per_block_pos_emb=extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
            )
            if intermediate_feature_ids and i in intermediate_feature_ids:
                x_reshaped_for_disc = rearrange(x_B_T_H_W_D, "b tp hp wp d -> b (tp hp wp) d")
                intermediate_features_outputs.append(x_reshaped_for_disc)

        x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
        x_B_C_Tt_Hp_Wp = self.unpatchify(x_B_T_H_W_O)
        if intermediate_feature_ids:
            if len(intermediate_features_outputs) != len(intermediate_feature_ids):
                log.warning(
                    f"Collected {len(intermediate_features_outputs)} intermediate features, "
                    f"but expected {len(intermediate_feature_ids)}. "
                    f"Requested IDs: {intermediate_feature_ids}"
                )
            return x_B_C_Tt_Hp_Wp, intermediate_features_outputs

        return x_B_C_Tt_Hp_Wp
