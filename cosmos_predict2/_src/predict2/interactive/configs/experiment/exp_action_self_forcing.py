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

import math

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.predict2.callbacks.validation_draw_sample import ValidationDrawSample
from cosmos_predict2._src.predict2.models.video2world_model import HighSigmaStrategy
from cosmos_predict2._src.predict2.text_encoders.text_encoder import EmbeddingConcatStrategy
from cosmos_predict2._src.predict2.distill.utils.config_helper import build_no_s3_run, deep_update_config_dict


def make_experiment(
    name: str,
    data: str,
    model: str = "action_video2world_self_forcing_fsdp",
    net: str = "action_causal_cosmos_v1_2B",
    conditioner: str = "video_action_conditioner",
    tokenizer: str = "wan2pt1_tokenizer",
    overrides: dict | None = None,
) -> LazyDict:
    defaults = [
        {"override /data_train": data},
        {"override /data_val": data},
        {"override /model": model},
        {"override /net": net},
        {"override /net_teacher": "cosmos_v1_2B_action_chunk_conditioned"},
        {"override /net_fake_score": "cosmos_v1_2B_action_chunk_conditioned"},
        {"override /conditioner": conditioner},
        {"override /ckpt_type": "dcp_distill"},
        {"override /optimizer": "fusedadamw"},
        {"override /callbacks": ["basic", "wandb", "cluster_speed"]},
        {"override /checkpoint": "s3"},
        {"override /tokenizer": tokenizer},
        "_self_",
    ]
    node = dict(
        defaults=defaults,
        job=dict(
            group="self_forcing_action",
            name=name,
        ),
        model_parallel=dict(
            context_parallel_size=1,
        ),
        optimizer=dict(
            lr=1e-7,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            # `fusedadamw` defaults to master_weights=True (FP32 master copy), which is very memory-expensive
            # for 2B-scale nets and can trigger OOM once optimizer state is first materialized.
            master_weights=False,
        ),
        scheduler=dict(
            f_max=[1.0],
            f_min=[1.0],
            warm_up_steps=[0],
            cycle_lengths=[400_000],
        ),
        model=dict(
            config=dict(
                adjust_video_noise=True,
                # cpu_offload=True,
                conditional_frames_probs=None,
                conditioner=dict(
                    text=dict(
                        dropout_rate=0.0,
                        use_empty_string=False,
                    ),
                ),
                dmd=True,
                grad_clip=True,
                high_sigma_ratio=0.05,
                high_sigma_strategy=str(HighSigmaStrategy.NONE),
                init_student_with_teacher=True,
                intermediate_feature_ids=None,
                loss_scale=0.0,
                loss_scale_GAN_discriminator=1.0,
                loss_scale_GAN_generator=1.0,
                loss_scale_fake_score=1.0,
                loss_scale_sid=1.0,
                max_num_conditional_frames=0,
                max_simulation_steps=1,
                max_simulation_steps_fake=4,
                min_num_conditional_frames=0,
                net=dict(
                    action_dim=384,
                    temporal_compression_ratio=4,
                    rope_enable_fps_modulation=False,
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=1.0,
                    sac_config=dict(mode="mm_only"),
                    # sac_config=dict(mode="block_wise"),
                    use_crossattn_projection=True,
                    crossattn_proj_in_channels=100352,
                    crossattn_emb_channels=1024,
                    max_frames=256,
                ),
                net_fake_score=dict(
                    action_dim=384,
                    temporal_compression_ratio=4,
                    rope_enable_fps_modulation=False,
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=1.0,
                    sac_config=dict(mode="mm_only"),
                    # sac_config=dict(mode="block_wise"),
                    use_crossattn_projection=True,
                    crossattn_proj_in_channels=100352,
                    crossattn_emb_channels=1024,
                ),
                net_teacher=dict(
                    action_dim=384,
                    temporal_compression_ratio=4,
                    rope_enable_fps_modulation=False,
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=1.0,
                    sac_config=dict(mode="mm_only"),
                    # sac_config=dict(mode="block_wise"),
                    use_crossattn_projection=True,
                    crossattn_proj_in_channels=100352,
                    crossattn_emb_channels=1024,
                ),
                optimizer_discriminator_config=dict(
                    lr=1e-5,
                    weight_decay=0.01,
                    betas=(0.9, 0.999),
                    master_weights=False,
                ),
                optimizer_fake_score_config=dict(
                    lr=1e-5,
                    weight_decay=0.01,
                    betas=(0.9, 0.999),
                    # Avoid allocating FP32 master weights for the fake-score optimizer (big memory spike after first step).
                    master_weights=False,
                ),
                rectified_flow_loss_weight_uniform=False,
                resolution="720",
                resize_online=True,
                scaling="rectified_flow",
                sde=dict(
                    p_mean=-0.8,
                    p_std=1.6,
                    sigma_max=80,
                    sigma_min=0.0002,
                ),
                sde_D=dict(
                    p_mean=0.0,
                    p_std=1.6,
                    sigma_max=80,
                    sigma_min=0.0002,
                ),
                selected_sampling_time=[math.pi / 2, math.atan(15), math.atan(5), math.atan(5 / 3)],
                sigma_conditional=0.0001,
                sigma_data=1.0,
                state_t=1 + 12 // 4,
                student_update_freq=5,
                tangent_warmup=1,
                teacher_load_from=dict(
                    load_path="s3://bucket/cosmos_predict2_action_conditioned/action_conditional/cosmos_predict2p5_2B_action_conditioned_gr00t_gr1_customized_13frame_full_16nodes/checkpoints/iter_000014000/model",
                    credentials="credentials/s3_checkpoint.secret",
                ),
                teacher_guidance=0.0,
                text_encoder_class="reason1p1_7B",
                # Enable generating a decoded video during training so the interactive
                # W&B callback can log `train/backward_simulation_video`.
                vis_debug=False,
                vis_debug_every_n=100,
                text_encoder_config=dict(
                    ckpt_path="s3://bucket/cosmos_reasoning1/sft_exp700/sft_exp721-1_qwen7b_tl_721_5vs5_s3_balanced_n32_resume_16k/checkpoints/iter_000016000/model/",
                    embedding_concat_strategy=str(EmbeddingConcatStrategy.FULL_CONCAT),
                    compute_online=False,
                ),
                timestep_shift=5,
            ),
        ),
        checkpoint=dict(
            save_iter=1000,
            save_to_object_store=dict(
                enabled=True,
            ),
            load_from_object_store=dict(
                enabled=True,
            ),
            load_training_state=False,
            strict_resume=True,
        ),
        trainer=dict(
            max_iter=3000,
            logging_iter=50,
            callbacks=dict(
                iter_speed=dict(hit_thres=200),
                grad_clip=dict(
                    clip_norm=1.0,
                ),
                every_n_sample_reg=dict(
                    every_n=25000000000000000000,
                    is_image=False,
                    num_samples=5,
                ),
                every_n_sample_ema=dict(
                    every_n=25000000000000000000,
                    is_image=False,
                    num_samples=5,
                ),
            ),
        ),
        dataloader_train=dict(
            batch_size=1,
            pin_memory=False,
        ),
        upload_reproducible_setup=True,
    )
    if overrides:
        deep_update_config_dict(node, overrides)
    return LazyDict(node, flags={"allow_objects": True})


####################################
# Create and register experiments #
####################################

ACTION_GR00T_GR1_SELF_FORCING = make_experiment(
    name="gr1",
    data="gr00t_customized_gr1_long",
    overrides=dict(
        job=dict(
            project="cosmos_predict2_action_conditioned",
            group="interactive_self_forcing",
        ),
        checkpoint=dict(
            load_path="checkpoints/self_forcing/gr1/iter_000010000",
        ),
        model=dict(
            config=dict(
                teacher_load_from=dict(
                    load_path="checkpoints/warmup/gr1/iter_000050000/model",
                    credentials="credentials/s3_checkpoint.secret",
                ),
            ),
        ),
    ),
)

ACTION_GR00T_G1_SELF_FORCING = make_experiment(
    name="g1",
    data="gr00t_customized_g1_long",
    overrides=dict(
        job=dict(
            project="cosmos_predict2_action_conditioned",
            group="interactive_self_forcing",
        ),
        checkpoint=dict(
            load_path="checkpoints/self_forcing/g1/iter_000010000",
        ),
        model=dict(
            config=dict(
                teacher_load_from=dict(
                    load_path="checkpoints/warmup/g1/iter_000050000/model",
                    credentials="credentials/s3_checkpoint.secret",
                ),
            ),
        ),
    ),
)

ACTION_GR00T_AGIBOT_SELF_FORCING = make_experiment(
    name="agibot",
    data="gr00t_customized_agibot_long",
    overrides=dict(
        job=dict(
            project="cosmos_predict2_action_conditioned",
            group="interactive_self_forcing",
        ),
        checkpoint=dict(
            load_path="checkpoints/self_forcing/agibot/iter_000010000",
        ),
        model=dict(
            config=dict(
                teacher_load_from=dict(
                    load_path="checkpoints/warmup/agibot/iter_000050000/model",
                    credentials="credentials/s3_checkpoint.secret",
                ),
            ),
        ),
    ),
)

ACTION_GR00T_AGIBOT_FRUIT_SELF_FORCING = make_experiment(
    name="agibot_fruit",
    data="gr00t_customized_agibot_fruit_long",
    overrides=dict(
        job=dict(
            project="cosmos_predict2_action_conditioned",
            group="interactive_self_forcing",
        ),
        checkpoint=dict(
            load_path="checkpoints/self_forcing/agibot_fruit/iter_000010000",
        ),
        model=dict(
            config=dict(
                teacher_load_from=dict(
                    load_path="checkpoints/warmup/agibot_fruit/iter_000100000/model",
                    credentials="credentials/s3_checkpoint.secret",
                ),
            ),
        ),
    ),
)

ACTION_GR00T_YAM_SELF_FORCING = make_experiment(
    name="yam",
    data="gr00t_customized_yam_long",
    overrides=dict(
        job=dict(
            project="cosmos_predict2_action_conditioned",
            group="interactive_self_forcing",
        ),
        checkpoint=dict(
            load_path="checkpoints/self_forcing/yam/iter_000010000",
        ),
        model=dict(
            config=dict(
                teacher_load_from=dict(
                    load_path="checkpoints/warmup/yam/iter_000050000/model",
                    credentials="credentials/s3_checkpoint.secret",
                ),
            ),
        ),
    ),
)

ACTION_GR00T_PRETRAIN_SELF_FORCING = make_experiment(
    name="pretrain",
    data="gr00t_customized_pretrain_long",
    overrides=dict(
        job=dict(
            project="cosmos_predict2_action_conditioned",
            group="interactive_self_forcing",
        ),
        checkpoint=dict(
            load_path="checkpoints/self_forcing/pretrain/iter_000010000",
        ),
        model=dict(
            config=dict(
                teacher_load_from=dict(
                    load_path="checkpoints/warmup/pretrain/iter_000140000/model",
                    credentials="credentials/s3_checkpoint.secret",
                ),
            ),
        ),
    ),
)

ACTION_GR00T_FOLD_TOWEL_AGILEX_3VIEW_SELF_FORCING = make_experiment(
    name="fold_towel_agilex_3view",
    data="gr00t_customized_fold_towel_agilex_3view_long",
    overrides=dict(
        job=dict(
            project="cosmos_predict2_action_conditioned",
            group="interactive_self_forcing",
        ),
        checkpoint=dict(
            load_path="logs/cosmos_interactive/debug/fold_towel_agilex_3view_no_s3_2026-04-11_02-06-25/checkpoints/iter_000020000",
        ),
        model=dict(
            config=dict(
                teacher_load_from=dict(
                    load_path="outputs_0321_libero_ood/dreamdojo/exp1201/fold_towel_agilex_3view/checkpoints/iter_000008000/model",
                    credentials="credentials/s3_checkpoint.secret",
                ),
            ),
        ),
        trainer=dict(
            run_validation=True,
            validation_iter=1000,
            max_val_iter=50,
            callbacks=dict(
                validation_sample=L(ValidationDrawSample)(
                    n_samples=1,
                    guidance=[0],
                    fps=5,
                    is_ema=True,
                    do_x0_prediction=False,
                ),
            ),
        ),
    ),
)

cs = ConfigStore.instance()

cs.store(
    group="experiment",
    package="_global_",
    name="cosmos_predict2p5_2B_action_gr00t_gr1_self_forcing_no_s3",
    node=build_no_s3_run(ACTION_GR00T_GR1_SELF_FORCING),
)
cs.store(
    group="experiment",
    package="_global_",
    name="cosmos_predict2p5_2B_action_gr00t_g1_self_forcing_no_s3",
    node=build_no_s3_run(ACTION_GR00T_G1_SELF_FORCING),
)
cs.store(
    group="experiment",
    package="_global_",
    name="cosmos_predict2p5_2B_action_gr00t_agibot_self_forcing_no_s3",
    node=build_no_s3_run(ACTION_GR00T_AGIBOT_SELF_FORCING),
)
cs.store(
    group="experiment",
    package="_global_",
    name="cosmos_predict2p5_2B_action_gr00t_agibot_fruit_self_forcing_no_s3",
    node=build_no_s3_run(ACTION_GR00T_AGIBOT_FRUIT_SELF_FORCING),
)
cs.store(
    group="experiment",
    package="_global_",
    name="cosmos_predict2p5_2B_action_gr00t_yam_self_forcing_no_s3",
    node=build_no_s3_run(ACTION_GR00T_YAM_SELF_FORCING),
)
cs.store(
    group="experiment",
    package="_global_",
    name="cosmos_predict2p5_2B_action_gr00t_pretrain_self_forcing_no_s3",
    node=build_no_s3_run(ACTION_GR00T_PRETRAIN_SELF_FORCING),
)
cs.store(
    group="experiment",
    package="_global_",
    name="cosmos_predict2p5_2B_action_gr00t_fold_towel_agilex_3view_self_forcing_no_s3",
    node=build_no_s3_run(ACTION_GR00T_FOLD_TOWEL_AGILEX_3VIEW_SELF_FORCING),
)