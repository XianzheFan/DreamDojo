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

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.predict2.callbacks.validation_draw_sample import ValidationDrawSample
from cosmos_predict2._src.predict2.distill.utils.config_helper import build_no_s3_run, deep_update_config_dict


def make_experiment(
    name: str,
    data: str,
    model: str = "action_video2world_warmup_fsdp",
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
        {"override /conditioner": conditioner},
        {"override /tokenizer": tokenizer},
        {"override /ckpt_type": "dcp"},
        {"override /checkpoint": "s3"},
        {"override /callbacks": ["basic_warmup", "wandb_warmup", "cluster_speed"]},
        "_self_",
    ]
    node = dict(
        defaults=defaults,
        job=dict(
            project="cosmos_predict2_action_conditioned",
            group="interactive_warmup",
            name=name,
        ),
        checkpoint=dict(
            save_iter=1000,
            save_to_object_store=dict(enabled=True),
            load_from_object_store=dict(enabled=True),
            load_training_state=False,
            strict_resume=True,
        ),
        model_parallel=dict(
            context_parallel_size=1,
        ),
        optimizer=dict(
            lr=3e-5,
            weight_decay=0.1,
            betas=[0.9, 0.999],
            master_weights=False,
        ),
        scheduler=dict(
            warm_up_steps=[0],
            f_min=[1.0],
            f_max=[1.0],
        ),
        model=dict(
            config=dict(
                state_t=1 + 12 // 4,
                net=dict(
                    action_dim=384,
                    num_action_per_chunk=12,
                    timestep_scale=0.001,
                    # sac_config=dict(mode="block_wise"),
                ),
                tokenizer=dict(
                    temporal_window=16,
                ),
                resolution=720,
            ),
        ),
        trainer=dict(
            max_iter=20000,
            logging_iter=20,
            callbacks=dict(
                grad_clip=dict(
                    clip_norm=0.1,
                ),
                manual_gc=dict(
                    every_n=200,
                ),
                every_n_sample_reg=dict(
                    every_n=5000000000000000000,
                    do_x0_prediction=False,
                    guidance=[0],
                    fps=16,
                ),
                every_n_sample_ema=dict(
                    every_n=5000000000000000000,
                    do_x0_prediction=False,
                    guidance=[0],
                    fps=16,
                ),
            ),
        ),
        dataloader_train=dict(
            batch_size=4,
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

ACTION_GR00T_WARMUP_GR1 = make_experiment(
    name="gr1",
    data="gr00t_gr1_warmup",
    overrides=dict(
        checkpoint=dict(
            load_path="checkpoints/warmup/gr1/iter_000050000",
        ),
    ),
)

ACTION_GR00T_WARMUP_G1 = make_experiment(
    name="g1",
    data="gr00t_g1_warmup",
    overrides=dict(
        checkpoint=dict(
            load_path="checkpoints/warmup/g1/iter_000050000",
        ),
    ),
)

ACTION_GR00T_WARMUP_AGIBOT = make_experiment(
    name="agibot",
    data="gr00t_agibot_warmup",
    overrides=dict(
        checkpoint=dict(
            load_path="checkpoints/warmup/agibot/iter_000050000",
        ),
    ),
)

ACTION_GR00T_WARMUP_AGIBOT_FRUIT = make_experiment(
    name="agibot_fruit",
    data="gr00t_agibot_fruit_warmup",
    overrides=dict(
        checkpoint=dict(
            load_path="checkpoints/warmup/agibot_fruit/iter_000100000",
        ),
    ),
)

ACTION_GR00T_WARMUP_YAM = make_experiment(
    name="yam",
    data="gr00t_yam_warmup",
    overrides=dict(
        checkpoint=dict(
            load_path="checkpoints/warmup/yam/iter_000050000",
        ),
    ),
)

ACTION_GR00T_WARMUP_PRETRAIN = make_experiment(
    name="pretrain",
    data="gr00t_pretrain_warmup",
    overrides=dict(
        checkpoint=dict(
            load_path="checkpoints/warmup/pretrain/iter_000140000",
        ),
    ),
)

ACTION_GR00T_WARMUP_FOLD_TOWEL_AGILEX_3VIEW = make_experiment(
    name="fold_towel_agilex_3view",
    data="gr00t_fold_towel_agilex_3view_warmup",
    overrides=dict(
        checkpoint=dict(
            load_path="checkpoints/DreamDojo/2B_pretrain/iter_000140000",
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

"""
torchrun --nproc_per_node=1 --master_port=12341 -m scripts.train --config=cosmos_predict2/_src/predict2/interactive/configs/config_warmup.py -- experiment=cosmos_predict2p5_2B_action_gr00t_gr1_warmup
"""

cs = ConfigStore.instance()

# cs.store(
#     group="experiment",
#     package="_global_",
#     name="cosmos_predict2p5_2B_action_gr00t_gr1_warmup",
#     node=ACTION_GR00T_WARMUP_GR1,
# )
# cs.store(
#     group="experiment",
#     package="_global_",
#     name="cosmos_predict2p5_2B_action_gr00t_g1_warmup",
#     node=ACTION_GR00T_WARMUP_G1,
# )
cs.store(
    group="experiment",
    package="_global_",
    name="cosmos_predict2p5_2B_action_gr00t_gr1_warmup_no_s3",
    node=build_no_s3_run(ACTION_GR00T_WARMUP_GR1),
)
cs.store(
    group="experiment",
    package="_global_",
    name="cosmos_predict2p5_2B_action_gr00t_g1_warmup_no_s3",
    node=build_no_s3_run(ACTION_GR00T_WARMUP_G1),
)
cs.store(
    group="experiment",
    package="_global_",
    name="cosmos_predict2p5_2B_action_gr00t_agibot_warmup_no_s3",
    node=build_no_s3_run(ACTION_GR00T_WARMUP_AGIBOT),
)
cs.store(
    group="experiment",
    package="_global_",
    name="cosmos_predict2p5_2B_action_gr00t_agibot_fruit_warmup_no_s3",
    node=build_no_s3_run(ACTION_GR00T_WARMUP_AGIBOT_FRUIT),
)
cs.store(
    group="experiment",
    package="_global_",
    name="cosmos_predict2p5_2B_action_gr00t_yam_warmup_no_s3",
    node=build_no_s3_run(ACTION_GR00T_WARMUP_YAM),
)
cs.store(
    group="experiment",
    package="_global_",
    name="cosmos_predict2p5_2B_action_gr00t_pretrain_warmup_no_s3",
    node=build_no_s3_run(ACTION_GR00T_WARMUP_PRETRAIN),
)
cs.store(
    group="experiment",
    package="_global_",
    name="cosmos_predict2p5_2B_action_gr00t_fold_towel_agilex_3view_warmup_no_s3",
    node=build_no_s3_run(ACTION_GR00T_WARMUP_FOLD_TOWEL_AGILEX_3VIEW),
)