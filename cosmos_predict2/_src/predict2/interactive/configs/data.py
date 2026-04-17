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
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L


# ----------- Pre-generated Warmup Datasets -----------

from cosmos_predict2._src.predict2.interactive.datasets.dataset_action_warmup import ActionDatasetSFWarmup

dataset_gr00t_gr1_warmup = L(ActionDatasetSFWarmup)(
    data_path="datasets/gr1_warmup_regenerated_4step",
    cr1_embeddings_path="datasets/cr1_empty_string_text_embeddings.pt",
)

dataset_gr00t_g1_warmup = L(ActionDatasetSFWarmup)(
    data_path="datasets/g1_warmup_regenerated_4step",
    cr1_embeddings_path="datasets/cr1_empty_string_text_embeddings.pt",
)

dataset_gr00t_agibot_warmup = L(ActionDatasetSFWarmup)(
    data_path="datasets/agibot_warmup_regenerated_4step",
    cr1_embeddings_path="datasets/cr1_empty_string_text_embeddings.pt",
)

dataset_gr00t_agibot_fruit_warmup = L(ActionDatasetSFWarmup)(
    data_path="datasets/agibot_fruit_warmup_regenerated_4step",
    cr1_embeddings_path="datasets/cr1_empty_string_text_embeddings.pt",
)

dataset_gr00t_yam_warmup = L(ActionDatasetSFWarmup)(
    data_path="datasets/yam_warmup_regenerated_4step",
    cr1_embeddings_path="datasets/cr1_empty_string_text_embeddings.pt",
)

dataset_gr00t_pretrain_warmup = L(ActionDatasetSFWarmup)(
    data_path="datasets/pretrain_warmup_regenerated_4step",
    cr1_embeddings_path="datasets/cr1_empty_string_text_embeddings.pt",
)

dataset_gr00t_old_gr1_dreamdojo_warmup = L(ActionDatasetSFWarmup)(
    data_path="datasets/old_gr1_dreamdojo_warmup_regenerated_4step",
    cr1_embeddings_path="datasets/cr1_empty_string_text_embeddings.pt",
)

dataset_gr00t_old_gr1_cosmos_warmup = L(ActionDatasetSFWarmup)(
    data_path="datasets/old_gr1_cosmos_warmup_regenerated_4step",
    cr1_embeddings_path="datasets/cr1_empty_string_text_embeddings.pt",
)

dataset_gr00t_fold_towel_agilex_3view_warmup = L(ActionDatasetSFWarmup)(
    data_path="datasets/teacher_gen_output_fold_towel_agilex_3view",
    cr1_embeddings_path="datasets/cr1_empty_string_text_embeddings.pt",
)

# ----------- Standard GR00T Datasets -----------

from groot_dreams.dataloader import MultiVideoActionDataset, get_data_path

gr1_path, gr1_mixing_weights = get_data_path("gr1")
gr00t_customized_gr1_dataset = L(MultiVideoActionDataset)(
    num_frames=13,
    dataset_path=gr1_path,
    dataset_mixing_weights=gr1_mixing_weights,
    data_split="train",
    cr1_embeddings_path="datasets/cr1_empty_string_text_embeddings.pt"
)
gr00t_customized_gr1_dataset_long = L(MultiVideoActionDataset)(
    num_frames=49,
    dataset_path=gr1_path,
    dataset_mixing_weights=gr1_mixing_weights,
    data_split="train",
    cr1_embeddings_path="datasets/cr1_empty_string_text_embeddings.pt"
)

g1_path, g1_mixing_weights = get_data_path("g1")
gr00t_customized_g1_dataset = L(MultiVideoActionDataset)(
    num_frames=13,
    dataset_path=g1_path,
    dataset_mixing_weights=g1_mixing_weights,
    data_split="train",
    cr1_embeddings_path="datasets/cr1_empty_string_text_embeddings.pt"
)
gr00t_customized_g1_dataset_long = L(MultiVideoActionDataset)(
    num_frames=49,
    dataset_path=g1_path,
    dataset_mixing_weights=g1_mixing_weights,
    data_split="train",
    cr1_embeddings_path="datasets/cr1_empty_string_text_embeddings.pt"
)

agibot_path, agibot_mixing_weights = get_data_path("agibot")
gr00t_customized_agibot_dataset = L(MultiVideoActionDataset)(
    num_frames=13,
    dataset_path=agibot_path,
    dataset_mixing_weights=agibot_mixing_weights,
    data_split="train",
    cr1_embeddings_path="datasets/cr1_empty_string_text_embeddings.pt"
)
gr00t_customized_agibot_dataset_long = L(MultiVideoActionDataset)(
    num_frames=49,
    dataset_path=agibot_path,
    dataset_mixing_weights=agibot_mixing_weights,
    data_split="train",
    cr1_embeddings_path="datasets/cr1_empty_string_text_embeddings.pt"
)

agibot_fruit_path, agibot_fruit_mixing_weights = get_data_path("agibot_fruit")
gr00t_customized_agibot_fruit_dataset = L(MultiVideoActionDataset)(
    num_frames=13,
    dataset_path=agibot_fruit_path,
    dataset_mixing_weights=agibot_fruit_mixing_weights,
    data_split="train",
    cr1_embeddings_path="datasets/cr1_empty_string_text_embeddings.pt"
)
gr00t_customized_agibot_fruit_dataset_long = L(MultiVideoActionDataset)(
    num_frames=49,
    dataset_path=agibot_fruit_path,
    dataset_mixing_weights=agibot_fruit_mixing_weights,
    data_split="train",
    cr1_embeddings_path="datasets/cr1_empty_string_text_embeddings.pt"
)

yam_path, yam_mixing_weights = get_data_path("yam")
gr00t_customized_yam_dataset = L(MultiVideoActionDataset)(
    num_frames=13,
    dataset_path=yam_path,
    dataset_mixing_weights=yam_mixing_weights,
    data_split="train",
    cr1_embeddings_path="datasets/cr1_empty_string_text_embeddings.pt"
)
gr00t_customized_yam_dataset_long = L(MultiVideoActionDataset)(
    num_frames=49,
    dataset_path=yam_path,
    dataset_mixing_weights=yam_mixing_weights,
    data_split="train",
    cr1_embeddings_path="datasets/cr1_empty_string_text_embeddings.pt"
)

fold_towel_agilex_3view_path, fold_towel_agilex_3view_mixing_weights = get_data_path("fold_towel_agilex_3view")
gr00t_customized_fold_towel_agilex_3view_dataset = L(MultiVideoActionDataset)(
    num_frames=13,
    dataset_path=fold_towel_agilex_3view_path,
    dataset_mixing_weights=fold_towel_agilex_3view_mixing_weights,
    data_split="train",
    cr1_embeddings_path="datasets/cr1_empty_string_text_embeddings.pt",
    embodiment_override="agilex_3view",
)
gr00t_customized_fold_towel_agilex_3view_dataset_long = L(MultiVideoActionDataset)(
    num_frames=49,
    dataset_path=fold_towel_agilex_3view_path,
    dataset_mixing_weights=fold_towel_agilex_3view_mixing_weights,
    data_split="train",
    cr1_embeddings_path="datasets/cr1_empty_string_text_embeddings.pt",
    embodiment_override="agilex_3view",
)

pretrain_path, pretrain_mixing_weights = get_data_path("pretrain")
gr00t_customized_pretrain_dataset = L(MultiVideoActionDataset)(
    num_frames=13,
    dataset_path=pretrain_path,
    dataset_mixing_weights=pretrain_mixing_weights,
    data_split="train",
    cr1_embeddings_path="datasets/cr1_empty_string_text_embeddings.pt"
)
gr00t_customized_pretrain_dataset_long = L(MultiVideoActionDataset)(
    num_frames=49,
    dataset_path=pretrain_path,
    dataset_mixing_weights=pretrain_mixing_weights,
    data_split="train",
    cr1_embeddings_path="datasets/cr1_empty_string_text_embeddings.pt"
)

# ----------- Dataloaders -----------


def get_sampler(dataset) -> DistributedSampler:
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=1234,
    )


def make_dataloader(dataset, batch_size=1, drop_last=True, num_workers=4, pin_memory=True, **kwargs):
    sampler = get_sampler(dataset)
    return DataLoader(
        dataset=dataset,
        sampler=sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **kwargs,
    )


def register_interactive_data():
    cs = ConfigStore.instance()

    for split in ["train", "val"]:
        cs.store(
            group=f"data_{split}",
            package=f"dataloader_{split}",
            name="gr00t_gr1_warmup",
            node=L(make_dataloader)(dataset=dataset_gr00t_gr1_warmup),
        )
        cs.store(
            group=f"data_{split}",
            package=f"dataloader_{split}",
            name="gr00t_g1_warmup",
            node=L(make_dataloader)(dataset=dataset_gr00t_g1_warmup),
        )
        cs.store(
            group=f"data_{split}",
            package=f"dataloader_{split}",
            name="gr00t_agibot_warmup",
            node=L(make_dataloader)(dataset=dataset_gr00t_agibot_warmup),
        )
        cs.store(
            group=f"data_{split}",
            package=f"dataloader_{split}",
            name="gr00t_agibot_fruit_warmup",
            node=L(make_dataloader)(dataset=dataset_gr00t_agibot_fruit_warmup),
        )
        cs.store(
            group=f"data_{split}",
            package=f"dataloader_{split}",
            name="gr00t_yam_warmup",
            node=L(make_dataloader)(dataset=dataset_gr00t_yam_warmup),
        )
        cs.store(
            group=f"data_{split}",
            package=f"dataloader_{split}",
            name="gr00t_pretrain_warmup",
            node=L(make_dataloader)(dataset=dataset_gr00t_pretrain_warmup),
        )
        cs.store(
            group=f"data_{split}",
            package=f"dataloader_{split}",
            name="gr00t_old_gr1_dreamdojo_warmup",
            node=L(make_dataloader)(dataset=dataset_gr00t_old_gr1_dreamdojo_warmup),
        )
        cs.store(
            group=f"data_{split}",
            package=f"dataloader_{split}",
            name="gr00t_old_gr1_cosmos_warmup",
            node=L(make_dataloader)(dataset=dataset_gr00t_old_gr1_cosmos_warmup),
        )
        cs.store(
            group=f"data_{split}",
            package=f"dataloader_{split}",
            name="gr00t_fold_towel_agilex_3view_warmup",
            node=L(make_dataloader)(dataset=dataset_gr00t_fold_towel_agilex_3view_warmup),
        )
        cs.store(
            group=f"data_{split}",
            package=f"dataloader_{split}",
            name="gr00t_customized_gr1",
            node=L(make_dataloader)(dataset=gr00t_customized_gr1_dataset, num_workers=0, pin_memory=False),
        )
        cs.store(
            group=f"data_{split}",
            package=f"dataloader_{split}",
            name="gr00t_customized_gr1_long",
            node=L(make_dataloader)(dataset=gr00t_customized_gr1_dataset_long, num_workers=0, pin_memory=False),
        )
        cs.store(
            group=f"data_{split}",
            package=f"dataloader_{split}",
            name="gr00t_customized_g1",
            node=L(make_dataloader)(dataset=gr00t_customized_g1_dataset, num_workers=0, pin_memory=False),
        )
        cs.store(
            group=f"data_{split}",
            package=f"dataloader_{split}",
            name="gr00t_customized_g1_long",
            node=L(make_dataloader)(dataset=gr00t_customized_g1_dataset_long, num_workers=0, pin_memory=False),
        )
        cs.store(
            group=f"data_{split}",
            package=f"dataloader_{split}",
            name="gr00t_customized_agibot",
            node=L(make_dataloader)(dataset=gr00t_customized_agibot_dataset, num_workers=0, pin_memory=False),
        )
        cs.store(
            group=f"data_{split}",
            package=f"dataloader_{split}",
            name="gr00t_customized_agibot_long",
            node=L(make_dataloader)(dataset=gr00t_customized_agibot_dataset_long, num_workers=0, pin_memory=False),
        )
        cs.store(
            group=f"data_{split}",
            package=f"dataloader_{split}",
            name="gr00t_customized_agibot_fruit",
            node=L(make_dataloader)(dataset=gr00t_customized_agibot_fruit_dataset, num_workers=0, pin_memory=False),
        )
        cs.store(
            group=f"data_{split}",
            package=f"dataloader_{split}",
            name="gr00t_customized_agibot_fruit_long",
            node=L(make_dataloader)(dataset=gr00t_customized_agibot_fruit_dataset_long, num_workers=0, pin_memory=False),
        )
        cs.store(
            group=f"data_{split}",
            package=f"dataloader_{split}",
            name="gr00t_customized_yam",
            node=L(make_dataloader)(dataset=gr00t_customized_yam_dataset, num_workers=0, pin_memory=False),
        )
        cs.store(
            group=f"data_{split}",
            package=f"dataloader_{split}",
            name="gr00t_customized_yam_long",
            node=L(make_dataloader)(dataset=gr00t_customized_yam_dataset_long, num_workers=0, pin_memory=False),
        )
        cs.store(
            group=f"data_{split}",
            package=f"dataloader_{split}",
            name="gr00t_customized_fold_towel_agilex_3view",
            node=L(make_dataloader)(dataset=gr00t_customized_fold_towel_agilex_3view_dataset, num_workers=0, pin_memory=False),
        )
        cs.store(
            group=f"data_{split}",
            package=f"dataloader_{split}",
            name="gr00t_customized_fold_towel_agilex_3view_long",
            node=L(make_dataloader)(dataset=gr00t_customized_fold_towel_agilex_3view_dataset_long, num_workers=0, pin_memory=False),
        )
        cs.store(
            group=f"data_{split}",
            package=f"dataloader_{split}",
            name="gr00t_customized_pretrain",
            node=L(make_dataloader)(dataset=gr00t_customized_pretrain_dataset, num_workers=0, pin_memory=False),
        )
        cs.store(
            group=f"data_{split}",
            package=f"dataloader_{split}",
            name="gr00t_customized_pretrain_long",
            node=L(make_dataloader)(dataset=gr00t_customized_pretrain_dataset_long, num_workers=0, pin_memory=False),
        )
