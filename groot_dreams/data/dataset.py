import json
from collections import defaultdict
from pathlib import Path
from random import randint

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from einops import rearrange
from pydantic import BaseModel, ValidationError
from torch.utils.data import Dataset
from tqdm import tqdm

from groot_dreams.utils.video import get_all_frames, get_frames_by_timestamps
from groot_dreams.data.embodiment_tags import EmbodimentTag
from groot_dreams.data.schema import (
    DatasetMetadata,
    DatasetStatisticalValues,
    LeRobotModalityMetadata,
    LeRobotStateActionMetadata,
)
from groot_dreams.data.transform.base import ComposedModalityTransform

LE_ROBOT_MODALITY_FILENAME = "meta/modality.json"
LE_ROBOT_EPISODE_FILENAME = "meta/episodes.jsonl"
LE_ROBOT_TASKS_FILENAME = "meta/tasks.jsonl"
LE_ROBOT_INFO_FILENAME = "meta/info.json"
LE_ROBOT_STATS_FILENAME = "meta/stats.json"
LE_ROBOT_DATA_FILENAME = "data/*/*.parquet"


def _as_cthw_video(video) -> torch.Tensor:
    """Convert transformed video output to ``[C, T, H, W]`` or ``[V, C, T, H, W]``."""
    if isinstance(video, np.ndarray):
        video = torch.from_numpy(video)
    if video.ndim == 5:
        if video.shape[1] == 1:
            video = video[:, 0]  # [T, 1, C, H, W] -> [T, C, H, W]
        elif video.shape[0] == 1:
            video = video[0]  # [1, T, C, H, W] -> [T, C, H, W]
        elif video.shape[2] in (1, 3):
            return video.permute(1, 2, 0, 3, 4).contiguous()  # [T, V, C, H, W] -> [V, C, T, H, W]
        else:
            raise ValueError(f"Cannot coerce multi-view video shape {tuple(video.shape)} to [C, T, H, W]")
    if video.ndim != 4:
        raise ValueError(f"Expected video shape [T, C, H, W] or [C, T, H, W], got {tuple(video.shape)}")
    if video.shape[1] in (1, 3):
        return video.transpose(0, 1)
    if video.shape[0] in (1, 3):
        return video
    raise ValueError(f"Cannot infer channel axis from video shape {tuple(video.shape)}")


def _tile_video_list(tiles: list[torch.Tensor]) -> torch.Tensor:
    if len(tiles) == 1:
        return tiles[0]
    channels, num_frames, tile_h, tile_w = tiles[0].shape
    local_cols = int(np.ceil(np.sqrt(len(tiles))))
    local_rows = int(np.ceil(len(tiles) / local_cols))
    out = torch.zeros(
        channels,
        num_frames,
        local_rows * tile_h,
        local_cols * tile_w,
        dtype=tiles[0].dtype,
        device=tiles[0].device,
    )
    for i, tile in enumerate(tiles):
        row = i // local_cols
        col = i % local_cols
        out[:, :, row * tile_h : (row + 1) * tile_h, col * tile_w : (col + 1) * tile_w] = tile
    return out


def _tile_group_to_video(frames: torch.Tensor, grid_shape: tuple[int, int], tile_group: list[int]) -> torch.Tensor:
    """Extract one agent stream from a tiled ``[C, T, H, W]`` video."""
    rows, cols = grid_shape
    channels, num_frames, height, width = frames.shape
    if height % rows != 0 or width % cols != 0:
        raise ValueError(f"Video size {(height, width)} is not divisible by grid_shape={grid_shape}")
    tile_h = height // rows
    tile_w = width // cols
    tiles = []
    for tile_idx in tile_group:
        row = int(tile_idx) // cols
        col = int(tile_idx) % cols
        if row >= rows:
            raise ValueError(f"Tile index {tile_idx} out of range for grid_shape={grid_shape}")
        tiles.append(frames[:, :, row * tile_h : (row + 1) * tile_h, col * tile_w : (col + 1) * tile_w])
    return _tile_video_list(tiles)


def _view_group_to_video(views: torch.Tensor, view_group: list[int]) -> torch.Tensor:
    """Extract one agent stream from explicit ``[V, C, T, H, W]`` views."""
    tiles = []
    for view_idx in view_group:
        if int(view_idx) >= views.shape[0]:
            raise ValueError(f"View index {view_idx} out of range for {views.shape[0]} views")
        tiles.append(views[int(view_idx)])
    return _tile_video_list(tiles)


def _split_video_grid_by_agent(
    frames: torch.Tensor,
    *,
    grid_shape: tuple[int, int] | None,
    view_groups: list[list[int]] | None,
    tile_groups: list[list[int]] | None,
) -> torch.Tensor:
    if frames.ndim == 5:
        groups = view_groups if view_groups is not None else tile_groups
        if groups is None:
            return frames
        return torch.stack([_view_group_to_video(frames, group) for group in groups], dim=0)
    if grid_shape is None or tile_groups is None:
        return frames.unsqueeze(0)
    return torch.stack([_tile_group_to_video(frames, grid_shape, group) for group in tile_groups], dim=0)


def _extract_video_group(
    frames: torch.Tensor,
    *,
    grid_shape: tuple[int, int] | None,
    view_group: list[int] | None,
    tile_group: list[int] | None,
) -> torch.Tensor | None:
    group = view_group if frames.ndim == 5 else tile_group
    if group is None:
        return None
    if frames.ndim == 5:
        return _view_group_to_video(frames, group)
    if grid_shape is None:
        raise ValueError("shared_video_tile_group requires agent_video_grid for tiled-frame fallback")
    return _tile_group_to_video(frames, grid_shape, group)


def _resize_video_frames(frames: torch.Tensor, output_size: tuple[int, int] | None) -> torch.Tensor:
    if output_size is None:
        return frames
    dtype = frames.dtype
    if frames.ndim == 4:
        flat = rearrange(frames, "c t h w -> t c h w").float()
        resized = F.interpolate(flat, output_size, mode="bilinear", align_corners=False)
        if dtype == torch.uint8:
            resized = torch.clamp(resized, 0, 255).to(dtype)
        return rearrange(resized, "t c h w -> c t h w")
    if frames.ndim == 5:
        num_agents, _, num_frames, _, _ = frames.shape
        flat = rearrange(frames, "p c t h w -> (p t) c h w").float()
        resized = F.interpolate(flat, output_size, mode="bilinear", align_corners=False)
        if dtype == torch.uint8:
            resized = torch.clamp(resized, 0, 255).to(dtype)
        return rearrange(resized, "(p t) c h w -> p c t h w", p=num_agents, t=num_frames)
    raise ValueError(f"Expected frames [C,T,H,W] or [P,C,T,H,W], got {tuple(frames.shape)}")


def _split_action_by_agent_slots(
    action: torch.Tensor,
    agent_action_dims: list | None,
    *,
    action_dim: int,
) -> torch.Tensor:
    if agent_action_dims is None:
        return action.unsqueeze(0)
    per_agent = []
    for agent_slices in agent_action_dims:
        if len(agent_slices) == 2 and all(isinstance(v, int) for v in agent_slices):
            slices = [agent_slices]
        else:
            slices = agent_slices
        agent_action = torch.zeros(action.shape[0], action_dim, dtype=action.dtype, device=action.device)
        for start, end in slices:
            agent_action[:, start:end] = action[:, start:end]
        per_agent.append(agent_action)
    return torch.stack(per_agent, dim=0)


def _make_lam_video(frames: torch.Tensor) -> torch.Tensor:
    """Build LAM input from ``[C, T, H, W]`` or ``[P, C, T, H, W]``."""
    if frames.ndim == 4:
        lam_frames = F.interpolate(frames, (240, 320), mode="bilinear")
        lam_frames = torch.clamp(lam_frames / 255.0, 0, 1)
        lam_frames = torch.repeat_interleave(lam_frames, 2, dim=1)[:, 1:-1, :, :]
        return rearrange(lam_frames, "c t h w -> t h w c")
    if frames.ndim == 5:
        num_agents, channels, num_frames, height, width = frames.shape
        flat = rearrange(frames, "p c t h w -> (p t) c h w")
        lam_frames = F.interpolate(flat, (240, 320), mode="bilinear")
        lam_frames = torch.clamp(lam_frames / 255.0, 0, 1)
        lam_frames = rearrange(lam_frames, "(p t) c h w -> p c t h w", p=num_agents, t=num_frames)
        lam_frames = torch.repeat_interleave(lam_frames, 2, dim=2)[:, :, 1:-1, :, :]
        return rearrange(lam_frames, "p c t h w -> p t h w c")
    raise ValueError(f"Expected frames [C,T,H,W] or [P,C,T,H,W], got {tuple(frames.shape)}")


def calculate_dataset_statistics(parquet_paths: list[Path]) -> dict:
    """Calculate the dataset statistics of all columns for a list of parquet files."""
    # Dataset statistics
    all_low_dim_data_list = []
    # Collect all the data
    for parquet_path in tqdm(
            sorted(list(parquet_paths)),
            desc="Collecting all parquet files...",
    ):
        # Load the parquet file
        parquet_data = pd.read_parquet(parquet_path)
        parquet_data = parquet_data
        all_low_dim_data_list.append(parquet_data)
    all_low_dim_data = pd.concat(all_low_dim_data_list, axis=0)
    # Compute dataset statistics
    dataset_statistics = {}
    for le_modality in all_low_dim_data.columns:
        if "text" in le_modality:
            continue
        print(f"Computing statistics for {le_modality}...")
        np_data = np.vstack(
            [np.asarray(x, dtype=np.float32) for x in all_low_dim_data[le_modality]]
        )
        dataset_statistics[le_modality] = {
            "mean": np.mean(np_data, axis=0).tolist(),
            "std": np.std(np_data, axis=0).tolist(),
            "min": np.min(np_data, axis=0).tolist(),
            "max": np.max(np_data, axis=0).tolist(),
            "q01": np.quantile(np_data, 0.01, axis=0).tolist(),
            "q99": np.quantile(np_data, 0.99, axis=0).tolist(),
        }
    return dataset_statistics


class ModalityConfig(BaseModel):
    """Configuration for a modality."""

    delta_indices: list[int]
    """Delta indices to sample relative to the current index. The returned data will correspond to the original data at a sampled base index + delta indices."""
    modality_keys: list[str]
    """The keys to load for the modality in the dataset."""


class LeRobotSingleDataset(Dataset):
    """
    Base dataset class for LeRobot that supports sharding.
    """

    def __init__(
        self,
        dataset_path: Path | str,
        modality_configs: dict[str, ModalityConfig],
        embodiment_tag: str | EmbodimentTag,
        video_backend: str = "decord",
        video_backend_kwargs: dict | None = None,
        transforms: ComposedModalityTransform | None = None,
        single_base_index: bool = False,
        num_frames: int = 13,
    ):
        """
        Initialize the dataset.

        Args:
            dataset_path (Path | str): The path to the dataset.
            modality_configs (dict[str, ModalityConfig]): The configuration for each modality. The keys are the modality names, and the values are the modality configurations.
                See `ModalityConfig` for more details.
            video_backend (str): Backend for video reading.
            video_backend_kwargs (dict): Keyword arguments for the video backend when initializing the video reader.
            transforms (ComposedModalityTransform): The transforms to apply to the dataset.
            embodiment_tag (EmbodimentTag): Overload the embodiment tag for the dataset. e.g. define it as "new_embodiment"
        """
        # first check if the path directory exists
        if not Path(dataset_path).exists():
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist")

        self.modality_configs = modality_configs
        self.video_backend = video_backend
        self.video_backend_kwargs = video_backend_kwargs if video_backend_kwargs is not None else {}
        self.transforms = (
            transforms if transforms is not None else ComposedModalityTransform(transforms=[])
        )

        self._dataset_path = Path(dataset_path)
        self._dataset_name = self._dataset_path.name
        if isinstance(embodiment_tag, EmbodimentTag):
            self.tag = embodiment_tag.value
        else:
            self.tag = embodiment_tag

        self.num_frames = num_frames
        self.timestep_interval = self.modality_configs["video"].delta_indices[-1] - self.modality_configs["video"].delta_indices[-2]

        self._metadata = self._get_metadata(EmbodimentTag(self.tag))
        self._trajectory_ids, self._trajectory_lengths = self._get_trajectories()
        self._all_steps = self._get_all_steps(single_base_index=single_base_index)
        self._modality_keys = self._get_modality_keys()
        self._delta_indices = self._get_delta_indices()
        self.set_transforms_metadata(self.metadata)
        self.set_epoch(0)

        print(f"Initialized dataset {self.dataset_name} with {embodiment_tag}")

        # LeRobot-specific config
        self._lerobot_modality_meta = self._get_lerobot_modality_meta()
        self._lerobot_info_meta = self._get_lerobot_info_meta()
        self._data_path_pattern = self._get_data_path_pattern()
        self._video_path_pattern = self._get_video_path_pattern()
        self._chunk_size = self._get_chunk_size()
        self._tasks = self._get_tasks()
        self.curr_traj_data = None
        self.curr_traj_id = None

        # Check if the dataset is valid
        self._check_integrity()

    @property
    def dataset_path(self) -> Path:
        """The path to the dataset that contains the METADATA_FILENAME file."""
        return self._dataset_path

    @property
    def metadata(self) -> DatasetMetadata:
        """The metadata for the dataset, loaded from metadata.json in the dataset directory"""
        return self._metadata

    @property
    def trajectory_ids(self) -> np.ndarray:
        """The trajectory IDs in the dataset, stored as a 1D numpy array of strings."""
        return self._trajectory_ids

    @property
    def trajectory_lengths(self) -> np.ndarray:
        """The trajectory lengths in the dataset, stored as a 1D numpy array of integers.
        The order of the lengths is the same as the order of the trajectory IDs.
        """
        return self._trajectory_lengths

    @property
    def all_steps(self) -> list[tuple[int, int]]:
        """The trajectory IDs and base indices for all steps in the dataset.
        Example:
            self.trajectory_ids: [0, 1, 2]
            self.trajectory_lengths: [3, 2, 4]
            return: [
                ("traj_0", 0), ("traj_0", 1), ("traj_0", 2),
                ("traj_1", 0), ("traj_1", 1),
                ("traj_2", 0), ("traj_2", 1), ("traj_2", 2), ("traj_2", 3)
            ]
        """
        return self._all_steps

    @property
    def modality_keys(self) -> dict:
        """The modality keys for the dataset. The keys are the modality names, and the values are the keys for each modality.

        Example: {
            "video": ["video.image_side_0", "video.image_side_1"],
            "state": ["state.eef_position", "state.eef_rotation"],
            "action": ["action.eef_position", "action.eef_rotation"],
            "language": ["language.human.task"],
            "timestamp": ["timestamp"],
            "reward": ["reward"],
        }
        """
        return self._modality_keys

    @property
    def delta_indices(self) -> dict[str, np.ndarray]:
        """The delta indices for the dataset. The keys are the modality.key, and the values are the delta indices for each modality.key."""
        return self._delta_indices

    @property
    def dataset_name(self) -> str:
        """The name of the dataset."""
        return self._dataset_name

    @property
    def lerobot_modality_meta(self) -> LeRobotModalityMetadata:
        """The metadata for the LeRobot dataset."""
        return self._lerobot_modality_meta

    @property
    def lerobot_info_meta(self) -> dict:
        """The metadata for the LeRobot dataset."""
        return self._lerobot_info_meta

    @property
    def data_path_pattern(self) -> str:
        """The path pattern for the LeRobot dataset."""
        return self._data_path_pattern

    @property
    def video_path_pattern(self) -> str:
        """The path pattern for the LeRobot dataset."""
        return self._video_path_pattern

    @property
    def chunk_size(self) -> int:
        """The chunk size for the LeRobot dataset."""
        return self._chunk_size

    @property
    def tasks(self) -> pd.DataFrame:
        """The tasks for the dataset."""
        return self._tasks

    def _get_metadata(self, embodiment_tag: EmbodimentTag) -> DatasetMetadata:
        """Get the metadata for the dataset.

        Returns:
            dict: The metadata for the dataset.
        """

        # 1. Modality metadata
        modality_meta_path = self.dataset_path / LE_ROBOT_MODALITY_FILENAME
        if not (modality_meta_path.exists()):
            if embodiment_tag == EmbodimentTag.GR1:
                if "inhouse_human" in str(self.dataset_path):
                    modality_meta_path = Path("shared_meta/GR1_human_modality.json")
                    print("WARNING: Could not find modality.json in dataset path, falling back to shared_meta/GR1_human_modality.json")
                else:
                    modality_meta_path = Path("shared_meta/GR1_unified_modality.json")
                    print("WARNING: Could not find modality.json in dataset path, falling back to shared_meta/GR1_unified_modality.json")
            elif embodiment_tag == EmbodimentTag.AGIBOT:
                modality_meta_path = Path("shared_meta/AgiBot_modality.json")
                print("WARNING: Could not find modality.json in dataset path, falling back to shared_meta/AgiBot_modality.json")
            elif embodiment_tag == EmbodimentTag.G1:
                modality_meta_path = Path("shared_meta/G1_modality.json")
                print("WARNING: Could not find modality.json in dataset path, falling back to shared_meta/G1_modality.json")
            elif embodiment_tag == EmbodimentTag.YAM:
                modality_meta_path = Path("shared_meta/YAM_modality.json")
                print("WARNING: Could not find modality.json in dataset path, falling back to shared_meta/YAM_modality.json")
            else:
                raise ValueError(f"Embodiment tag {embodiment_tag} not supported")
        assert (
            modality_meta_path.exists()
        ), f"Please provide a {LE_ROBOT_MODALITY_FILENAME} file in {self.dataset_path}"

        # 1.1. State and action modalities
        simplified_modality_meta: dict[str, dict] = {}
        with open(modality_meta_path, "r") as f:
            le_modality_meta = LeRobotModalityMetadata.model_validate(json.load(f))
        for modality in ["state", "action"]:
            simplified_modality_meta[modality] = {}
            le_state_action_meta: dict[str, LeRobotStateActionMetadata] = getattr(
                le_modality_meta, modality
            )
            for subkey in le_state_action_meta:
                state_action_dtype = np.dtype(le_state_action_meta[subkey].dtype)
                if np.issubdtype(state_action_dtype, np.floating):
                    continuous = True
                else:
                    continuous = False
                simplified_modality_meta[modality][subkey] = {
                    "absolute": le_state_action_meta[subkey].absolute,
                    "rotation_type": le_state_action_meta[subkey].rotation_type,
                    "shape": [
                        le_state_action_meta[subkey].end - le_state_action_meta[subkey].start
                    ],
                    "continuous": continuous,
                }

        # 1.2. Video modalities
        le_info_path = self.dataset_path / LE_ROBOT_INFO_FILENAME
        assert (
            le_info_path.exists()
        ), f"Please provide a {LE_ROBOT_INFO_FILENAME} file in {self.dataset_path}"
        with open(le_info_path, "r") as f:
            le_info = json.load(f)
        simplified_modality_meta["video"] = {}
        for new_key in le_modality_meta.video:
            original_key = le_modality_meta.video[new_key].original_key
            if original_key is None:
                original_key = new_key
            try:
                le_video_meta = le_info["features"][original_key]
            except:
                le_video_meta = le_info["features"][f"observation.images.{original_key}"]
            height = le_video_meta["shape"][le_video_meta["names"].index("height")]
            width = le_video_meta["shape"][le_video_meta["names"].index("width")]
            # NOTE(FH): different lerobot dataset versions have different keys for the number of channels and fps
            try:
                channels = le_video_meta["shape"][le_video_meta["names"].index("channel")]
                fps = le_video_meta["video_info"]["video.fps"] if "video_info" in le_video_meta else le_video_meta["info"]["video.fps"]
            except ValueError:
                channels = le_video_meta["shape"][le_video_meta["names"].index("channels")]
                fps = le_video_meta["video_info"]["video.fps"] if "video_info" in le_video_meta else le_video_meta["info"]["video.fps"]
            simplified_modality_meta["video"][new_key] = {
                "resolution": [width, height],
                "channels": channels,
                "fps": fps,
            }

        # 2. Dataset statistics
        stats_path = self.dataset_path / LE_ROBOT_STATS_FILENAME
        has_default_meta = True
        if embodiment_tag == EmbodimentTag.GR1:
            if "inhouse_human" in str(self.dataset_path):
                default_stats_path = Path("shared_meta/GR1_human_stats.json")
                default_modality_meta_path = Path("shared_meta/GR1_human_modality.json")
            else:
                default_stats_path = Path("shared_meta/GR1_unified_stats.json")
                default_modality_meta_path = Path("shared_meta/GR1_unified_modality.json")
        elif embodiment_tag == EmbodimentTag.AGIBOT:
            default_stats_path = Path("shared_meta/AgiBot_stats.json")
            default_modality_meta_path = Path("shared_meta/AgiBot_modality.json")
        elif embodiment_tag == EmbodimentTag.G1:
            default_stats_path = Path("shared_meta/G1_stats.json")
            default_modality_meta_path = Path("shared_meta/G1_modality.json")
        elif embodiment_tag == EmbodimentTag.YAM:
            default_stats_path = Path("shared_meta/YAM_stats.json")
            default_modality_meta_path = Path("shared_meta/YAM_modality.json")
        else:
            raise ValueError(f"Embodiment tag {embodiment_tag} not supported")
        if has_default_meta:
            default_statistics = json.load(open(default_stats_path, "r"))
            default_modality_meta = LeRobotModalityMetadata.model_validate(json.load(open(default_modality_meta_path, "r")))
        try:
            with open(stats_path, "r") as f:
                le_statistics = json.load(f)
            for stat in le_statistics.values():
                if isinstance(stat, int):
                    continue
                DatasetStatisticalValues.model_validate(stat)
        except (FileNotFoundError, ValidationError) as e:
            print(f"Failed to load dataset statistics: {e}")
            print(f"Calculating dataset statistics for {self.dataset_name}")
            # Get all parquet files in the dataset paths
            parquet_files = list((self.dataset_path).glob(LE_ROBOT_DATA_FILENAME))
            le_statistics = calculate_dataset_statistics(parquet_files)
        dataset_statistics = {}
        for our_modality in ["state", "action"]:
            dataset_statistics[our_modality] = {}
            for subkey in simplified_modality_meta[our_modality]:
                dataset_statistics[our_modality][subkey] = {}
                state_action_meta = le_modality_meta.get_key_meta(f"{our_modality}.{subkey}")
                assert isinstance(state_action_meta, LeRobotStateActionMetadata)
                le_modality = state_action_meta.original_key
                for stat_name in le_statistics[le_modality]:
                    if has_default_meta:
                        try:
                            default_state_action_meta = default_modality_meta.get_key_meta(f"{our_modality}.{subkey}")
                            indices = np.arange(
                                default_state_action_meta.start,
                                default_state_action_meta.end,
                            )
                            stat = np.array(default_statistics[le_modality][stat_name])
                            print(f"NOTE: Using default statistics for {our_modality}.{subkey}")
                        except ValueError:
                            indices = np.arange(
                                state_action_meta.start,
                                state_action_meta.end,
                            )
                            stat = np.array(le_statistics[le_modality][stat_name])
                            print(f"NOTE: Using original statistics for {our_modality}.{subkey}")
                    else:
                        indices = np.arange(
                            state_action_meta.start,
                            state_action_meta.end,
                        )
                        stat = np.array(le_statistics[le_modality][stat_name])
                    dataset_statistics[our_modality][subkey][stat_name] = stat[indices].tolist()

        # 3. Full dataset metadata
        metadata = DatasetMetadata(
            statistics=dataset_statistics,  # type: ignore
            modalities=simplified_modality_meta,  # type: ignore
            embodiment_tag=embodiment_tag,
        )

        return metadata

    def _get_trajectories(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the trajectories in the dataset."""
        # Get trajectory lengths, IDs, and whitelist from dataset metadata
        episode_path = self.dataset_path / LE_ROBOT_EPISODE_FILENAME
        with open(episode_path, "r") as f:
            episode_metadata = [json.loads(line) for line in f]
        trajectory_ids = []
        trajectory_lengths = []
        for episode in episode_metadata:
            trajectory_ids.append(episode["episode_index"])
            trajectory_lengths.append(episode["length"])
        return np.array(trajectory_ids), np.array(trajectory_lengths)

    def _get_all_steps(self, single_base_index=False) -> list[tuple[int, int]]:
        """Get the trajectory IDs and base indices for all steps in the dataset.

        Returns:
            list[tuple[str, int]]: A list of (trajectory_id, base_index) tuples.

        Example:
            self.trajectory_ids: [0, 1, 2]
            self.trajectory_lengths: [3, 2, 4]
            return: [
                ("traj_0", 0), ("traj_0", 1), ("traj_0", 2),
                ("traj_1", 0), ("traj_1", 1),
                ("traj_2", 0), ("traj_2", 1), ("traj_2", 2), ("traj_2", 3)
            ]
        """
        all_steps: list[tuple[int, int]] = []
        for trajectory_id, trajectory_length in zip(self.trajectory_ids, self.trajectory_lengths):
            if single_base_index:
                all_steps.append((trajectory_id, 0))
            else:
                for base_index in range(trajectory_length):
                    if base_index + max(self.modality_configs["video"].delta_indices) >= trajectory_length:
                        break
                    all_steps.append((trajectory_id, base_index))
        return all_steps

    def _get_modality_keys(self) -> dict:
        """Get the modality keys for the dataset.
        The keys are the modality names, and the values are the keys for each modality.
        See property `modality_keys` for the expected format.
        """
        modality_keys = defaultdict(list)
        for modality, config in self.modality_configs.items():
            modality_keys[modality] = config.modality_keys
        return modality_keys

    def _get_delta_indices(self) -> dict[str, np.ndarray]:
        """Restructure the delta indices to use modality.key as keys instead of just the modalities."""
        delta_indices: dict[str, np.ndarray] = {}
        for config in self.modality_configs.values():
            for key in config.modality_keys:
                delta_indices[key] = np.array(config.delta_indices)
        return delta_indices

    def _get_lerobot_modality_meta(self) -> LeRobotModalityMetadata:
        """Get the metadata for the LeRobot dataset."""
        modality_meta_path = self.dataset_path / LE_ROBOT_MODALITY_FILENAME
        if not (modality_meta_path.exists()):
            if self.tag == "gr1":
                if "inhouse_human" in str(self.dataset_path):
                    modality_meta_path = Path("shared_meta/GR1_human_modality.json")
                    print("WARNING: Could not find modality.json in dataset path, falling back to shared_meta/GR1_human_modality.json")
                else:
                    modality_meta_path = Path("shared_meta/GR1_unified_modality.json")
                    print("WARNING: Could not find modality.json in dataset path, falling back to shared_meta/GR1_unified_modality.json")
            elif self.tag == "agibot":
                modality_meta_path = Path("shared_meta/AgiBot_modality.json")
                print("WARNING: Could not find modality.json in dataset path, falling back to shared_meta/AgiBot_modality.json")
            elif self.tag == "g1":
                modality_meta_path = Path("shared_meta/G1_modality.json")
                print("WARNING: Could not find modality.json in dataset path, falling back to shared_meta/G1_modality.json")
            elif self.tag == "yam":
                modality_meta_path = Path("shared_meta/YAM_modality.json")
                print("WARNING: Could not find modality.json in dataset path, falling back to shared_meta/YAM_modality.json")
            else:
                raise ValueError(f"Embodiment tag {self.tag} not supported")
        assert (
            modality_meta_path.exists()
        ), f"Please provide a {LE_ROBOT_MODALITY_FILENAME} file in {self.dataset_path}"
        with open(modality_meta_path, "r") as f:
            modality_meta = LeRobotModalityMetadata.model_validate(json.load(f))
        return modality_meta

    def _get_lerobot_info_meta(self) -> dict:
        """Get the metadata for the LeRobot dataset."""
        info_meta_path = self.dataset_path / LE_ROBOT_INFO_FILENAME
        with open(info_meta_path, "r") as f:
            info_meta = json.load(f)
        return info_meta

    def _get_data_path_pattern(self) -> str:
        """Get the data path pattern for the LeRobot dataset."""
        return self.lerobot_info_meta["data_path"]

    def _get_video_path_pattern(self) -> str:
        """Get the video path pattern for the LeRobot dataset."""
        return self.lerobot_info_meta["video_path"]

    def _get_chunk_size(self) -> int:
        """Get the chunk size for the LeRobot dataset."""
        return self.lerobot_info_meta["chunks_size"]

    def _get_tasks(self) -> pd.DataFrame:
        """Get the tasks for the dataset."""
        tasks_path = self.dataset_path / LE_ROBOT_TASKS_FILENAME
        with open(tasks_path, "r") as f:
            tasks = [json.loads(line) for line in f]
        df = pd.DataFrame(tasks)
        return df.set_index("task_index")

    def _check_integrity(self):
        """Use the config to check if the keys are valid and detect silent data corruption."""
        ERROR_MSG_HEADER = f"Error occurred in initializing dataset {self.dataset_name}:\n"

        for modality_config in self.modality_configs.values():
            for key in modality_config.modality_keys:
                if key == "lapa_action" or key == "dream_actions":
                    continue  # no need for any metadata for lapa actions because it comes normalized
                # Check if the key is valid
                try:
                    self.lerobot_modality_meta.get_key_meta(key)
                except Exception as e:
                    raise ValueError(
                        ERROR_MSG_HEADER + f"Unable to find key {key} in modality metadata:\n{e}"
                    )

    def set_transforms_metadata(self, metadata: DatasetMetadata):
        """Set the metadata for the transforms. This is useful for transforms that need to know the metadata, such as the normalization values."""
        self.transforms.set_metadata(metadata)

    def set_epoch(self, epoch: int):
        """Set the epoch for the dataset.

        Args:
            epoch (int): The epoch to set.
        """
        self.epoch = epoch

    def __len__(self) -> int:
        """Get the total number of data points in the dataset.

        Returns:
            int: the total number of data points in the dataset.
        """
        return len(self.all_steps)

    def __str__(self) -> str:
        """Get the description of the dataset."""
        return f"{self.dataset_name} ({len(self)} steps)"

    def __getitem__(self, index: int) -> dict:
        """Get the data for a single step in a trajectory.

        Args:
            index (int): The index of the step to get.

        Returns:
            dict: The data for the step.
        """
        trajectory_id, base_index = self.all_steps[index]
        return self.transforms(self.get_step_data(trajectory_id, base_index))

    def get_step_data(self, trajectory_id: int, base_index: int) -> dict:
        """Get the RAW data for a single step in a trajectory. No transforms are applied.

        Args:
            trajectory_id (int): The name of the trajectory.
            base_index (int): The base step index in the trajectory.

        Returns:
            dict: The RAW data for the step.

        Example return:
            {
                "video": {
                    "video.image_side_0": [B, T, H, W, C],
                    "video.image_side_1": [B, T, H, W, C],
                },
                "state": {
                    "state.eef_position": [B, T, state_dim],
                    "state.eef_rotation": [B, T, state_dim],
                },
                "action": {
                    "action.eef_position": [B, T, action_dim],
                    "action.eef_rotation": [B, T, action_dim],
                },
            }
        """
        data = {}
        # Get the data for all modalities
        self.curr_traj_data = self.get_trajectory_data(trajectory_id)
        for modality in self.modality_keys:
            # Get the data corresponding to each key in the modality
            for key in self.modality_keys[modality]:
                data[key] = self.get_data_by_modality(trajectory_id, modality, key, base_index)
        return data

    def get_trajectory_data(self, trajectory_id: int) -> pd.DataFrame:
        """Get the data for a trajectory."""
        if self.curr_traj_id == trajectory_id and self.curr_traj_data is not None:
            return self.curr_traj_data
        else:
            chunk_index = self.get_episode_chunk(trajectory_id)
            parquet_path = self.dataset_path / self.data_path_pattern.format(
                episode_chunk=chunk_index, episode_index=trajectory_id
            )
            assert parquet_path.exists(), f"Parquet file not found at {parquet_path}"
            return pd.read_parquet(parquet_path)

    def get_trajectory_index(self, trajectory_id: int) -> int:
        """Get the index of the trajectory in the dataset by the trajectory ID.
        This is useful when you need to get the trajectory length or sampling weight corresponding to the trajectory ID.

        Args:
            trajectory_id (str): The ID of the trajectory.

        Returns:
            int: The index of the trajectory in the dataset.
        """
        trajectory_indices = np.where(self.trajectory_ids == trajectory_id)[0]
        if len(trajectory_indices) != 1:
            raise ValueError(
                f"Error finding trajectory index for {trajectory_id}, found {trajectory_indices=}"
            )
        return trajectory_indices[0]

    def get_episode_chunk(self, ep_index: int) -> int:
        """Get the chunk index for an episode index."""
        return ep_index // self.chunk_size

    def retrieve_data_and_pad(
        self,
        array: np.ndarray,
        step_indices: np.ndarray,
        max_length: int,
        padding_strategy: str = "first_last",
    ) -> np.ndarray:
        """Retrieve the data from the dataset and pad it if necessary.
        Args:
            array (np.ndarray): The array to retrieve the data from.
            step_indices (np.ndarray): The step indices to retrieve the data for.
            max_length (int): The maximum length of the data.
            padding_strategy (str): The padding strategy, either "first" or "last".
        """
        # Get the padding indices
        front_padding_indices = step_indices < 0
        end_padding_indices = step_indices >= max_length
        padding_positions = np.logical_or(front_padding_indices, end_padding_indices)
        # Retrieve the data with the non-padding indices
        # If there exists some padding, Given T step_indices, the shape of the retrieved data will be (T', ...) where T' < T
        raw_data = array[step_indices[~padding_positions]]
        assert isinstance(raw_data, np.ndarray), f"{type(raw_data)=}"
        # This is the shape of the output, (T, ...)
        if raw_data.ndim == 1:
            expected_shape = (len(step_indices),)
        else:
            expected_shape = (len(step_indices), *array.shape[1:])

        # Pad the data
        output = np.zeros(expected_shape)
        # Assign the non-padded data
        output[~padding_positions] = raw_data
        # If there exists some padding, pad the data
        if padding_positions.any():
            if padding_strategy == "first_last":
                # Use first / last step data to pad
                front_padding_data = array[0]
                end_padding_data = array[-1]
                output[front_padding_indices] = front_padding_data
                output[end_padding_indices] = end_padding_data
            elif padding_strategy == "zero":
                # Use zero padding
                output[padding_positions] = 0
            else:
                raise ValueError(f"Invalid padding strategy: {padding_strategy}")
        return output

    def get_video_path(self, trajectory_id: int, key: str) -> Path:
        chunk_index = self.get_episode_chunk(trajectory_id)
        original_key = self.lerobot_modality_meta.video[key].original_key
        if original_key is None:
            original_key = key
        video_filename = self.video_path_pattern.format(
            episode_chunk=chunk_index, episode_index=trajectory_id, video_key=original_key
        )
        if not (self.dataset_path / video_filename).exists():
            original_key = f"observation.images.{original_key}"
            video_filename = self.video_path_pattern.format(
                episode_chunk=chunk_index, episode_index=trajectory_id, video_key=original_key
            )
        return self.dataset_path / video_filename

    def get_video(
        self,
        trajectory_id: int,
        key: str,
        base_index: int,
    ) -> np.ndarray:
        """Get the video frames for a trajectory by a base index.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (str): The ID of the trajectory.
            key (str): The key of the video.
            base_index (int): The base index of the trajectory.

        Returns:
            np.ndarray: The video frames for the trajectory and frame indices. Shape: (T, H, W, C)
        """
        # Get the step indices
        step_indices = self.delta_indices[key] + base_index
        # print(f"{step_indices=}")
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Ensure the indices are within the valid range
        # This is equivalent to padding the video with extra frames at the beginning and end
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, self.trajectory_lengths[trajectory_index] - 1)
        # need_cut = False
        # for i in range(len(step_indices)):
        #     if step_indices[i] >= self.trajectory_lengths[trajectory_index]:
        #         need_cut = True
        #         break
        # if need_cut:
        #     step_indices = step_indices[:i]  # Only keep the valid indices
        assert key.startswith("video."), f"Video key must start with 'video.', got {key}"
        # Get the sub-key
        key = key.replace("video.", "")
        video_path = self.get_video_path(trajectory_id, key)
        # Get the action/state timestamps for each frame in the video
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        assert "timestamp" in self.curr_traj_data.columns, f"No timestamp found in {trajectory_id=}"
        timestamp: np.ndarray = self.curr_traj_data["timestamp"].to_numpy()
        if np.all(timestamp == 0):
            # Check if timestamp is all 0, and if so, use the video fps to generate timestamps
            fps = self._metadata.modalities.video[key].fps
            timestamp = np.arange(len(timestamp)) / fps

        # Get the corresponding video timestamps from the step indices
        video_timestamp = timestamp[step_indices]

        try:
            return get_frames_by_timestamps(
                video_path.as_posix(),
                video_timestamp,
                video_backend=self.video_backend,
                video_backend_kwargs=self.video_backend_kwargs,
            )
        except:
            self.video_backend = "torchvision_av"
            return get_frames_by_timestamps(
                video_path.as_posix(),
                video_timestamp,
                video_backend=self.video_backend,
                video_backend_kwargs=self.video_backend_kwargs,
            )

    def get_state_or_action(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        base_index: int,
    ) -> np.ndarray:
        """Get the state or action data for a trajectory by a base index.
        If the step indices are out of range, pad with the data:
            if the data is stored in absolute format, pad with the first or last step data;
            otherwise, pad with zero.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            modality (str): The modality of the data.
            key (str): The key of the data.
            base_index (int): The base index of the trajectory.

        Returns:
            np.ndarray: The data for the trajectory and step indices.
        """
        # Get the step indices
        step_indices = self.delta_indices[key] + base_index
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Get the maximum length of the trajectory
        max_length = self.trajectory_lengths[trajectory_index]
        # need_cut = False
        # for i in range(len(step_indices)):
        #     if step_indices[i] >= max_length:
        #         need_cut = True
        #         break
        # if need_cut:
        #     step_indices = step_indices[:i]  # Only keep the valid indices
        assert key.startswith(modality + "."), f"{key} must start with {modality + '.'}, got {key}"
        # Get the sub-key, e.g. state.joint_angles -> joint_angles
        key = key.replace(modality + ".", "")
        # Get the lerobot key
        le_state_or_action_cfg = getattr(self.lerobot_modality_meta, modality)
        le_key = le_state_or_action_cfg[key].original_key
        if le_key is None:
            le_key = key
        # Get the data array, shape: (T, D)
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        assert le_key in self.curr_traj_data.columns, f"No {le_key} found in {trajectory_id=}"
        data_array: np.ndarray = np.stack(self.curr_traj_data[le_key])  # type: ignore
        assert data_array.ndim == 2, f"Expected 2D array, got {data_array.shape} array"
        le_indices = np.arange(
            le_state_or_action_cfg[key].start,
            le_state_or_action_cfg[key].end,
        )
        data_array = data_array[:, le_indices]
        # Get the state or action configuration
        state_or_action_cfg = getattr(self.metadata.modalities, modality)[key]

        # Pad the data
        return self.retrieve_data_and_pad(
            array=data_array,
            step_indices=step_indices,
            max_length=max_length,
            padding_strategy="first_last" if state_or_action_cfg.absolute else "zero",
        )

    def get_language(
        self,
        trajectory_id: int,
        key: str,
        base_index: int,
    ) -> list[str]:
        """Get the language annotation data for a trajectory by step indices.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            key (str): The key of the annotation.
            base_index (int): The base index of the trajectory.

        Returns:
            list[str]: The annotation data for the trajectory and step indices. If no matching data is found, return empty strings.
        """
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        # Get the step indices
        step_indices = self.delta_indices[key] + base_index
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Get the maximum length of the trajectory
        max_length = self.trajectory_lengths[trajectory_index]
        # Get the end times corresponding to the closest indices
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, max_length - 1)
        # Get the annotations
        task_indices: list[int] = []
        assert key.startswith(
            "annotation."
        ), f"Language key must start with 'annotation.', got {key}"
        subkey = key.replace("annotation.", "")
        annotation_meta = self.lerobot_modality_meta.annotation
        assert annotation_meta is not None, f"Annotation metadata is None for {subkey}"
        assert (
                subkey in annotation_meta
        ), f"Annotation key {subkey} not found in metadata, available annotation keys: {annotation_meta.keys()}"
        subkey_meta = annotation_meta[subkey]
        original_key = subkey_meta.original_key
        if original_key is None:
            original_key = key
        for i in range(len(step_indices)):
            task_indices.append(self.curr_traj_data[original_key][step_indices[i]].item())
        return self.tasks.loc[task_indices]["task"].tolist()

    def get_data_by_modality(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        base_index: int,
    ):
        """Get the data corresponding to the modality for a trajectory by a base index.
        This method will call the corresponding helper method based on the modality.
        See the helper methods for more details.
        NOTE: For the language modality, the data is padded with empty strings if no matching data is found.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            modality (str): The modality of the data.
            key (str): The key of the data.
            base_index (int): The base index of the trajectory.
        """
        if modality == "video":
            return self.get_video(trajectory_id, key, base_index)
        elif modality == "state" or modality == "action":
            return self.get_state_or_action(trajectory_id, modality, key, base_index)
        elif modality == "language":
            return self.get_language(trajectory_id, key, base_index)
        else:
            raise ValueError(f"Invalid modality: {modality}")


class CachedLeRobotSingleDataset(LeRobotSingleDataset):
    def __init__(self, img_resize: tuple[int, int] | None = None, *args, **kwargs):
        """
        This class caches the video frames for each trajectory and key.
        It is recommended to use this class if the video frames need to be accessed multiple times.

        Args:
            resize_img (tuple[int, int], optional): The size to resize the video frames to reduce memory usage.
        """
        # Convert img_resize to tuple if it is not already
        if img_resize is not None and not isinstance(img_resize, tuple):
            img_resize = tuple(img_resize)
            assert len(img_resize) == 2, f"Expected tuple of length 2, got {img_resize}"
        self.img_resize = img_resize

        # Initialize img_resize attribute first to ensure it exists
        super().__init__(*args, **kwargs)
        cached_frames: dict[str, np.ndarray] = {}

        for key in self.modality_keys["video"]:
            all_frames = []
            key = key.replace("video.", "")
            for trajectory_id, trajectory_length in tqdm(
                    zip(self.trajectory_ids, self.trajectory_lengths),
                    total=len(self.trajectory_ids),
                    desc=f"Caching {key} frames",
            ):
                video_path = self.get_video_path(trajectory_id, key)
                frames = get_all_frames(
                    video_path.as_posix(),
                    video_backend=self.video_backend,
                    video_backend_kwargs=self.video_backend_kwargs,
                    resize_size=img_resize,
                )
                assert frames.ndim == 4, f"Expected 4D array, got {frames.shape} array"
                assert frames.shape[3] == 3, f"Expected 3 channels, got {frames.shape[3]} channels"
                # assert (
                #     frames.shape[0] == trajectory_length
                # ), f"Expected {trajectory_length} frames, got {frames.shape[0]} frames"
                all_frames.append(frames)
            cached_frames[key] = np.concatenate(all_frames, axis=0)
            print(f"{key}: {cached_frames[key].shape}")
        self.cached_frames = cached_frames
        self.start_indices = np.cumsum(self.trajectory_lengths) - self.trajectory_lengths

    def get_video(self, trajectory_id: int, key: str, base_index: int) -> np.ndarray:
        step_indices = self.delta_indices[key] + base_index
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Ensure the indices are within the valid range
        # This is equivalent to padding the video with extra frames at the beginning and end
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, self.trajectory_lengths[trajectory_index] - 1)
        # need_cut = False
        # for i in range(len(step_indices)):
        #     if step_indices[i] >= self.trajectory_lengths[trajectory_index]:
        #         need_cut = True
        #         break
        # if need_cut:
        #     step_indices = step_indices[:i]  # Only keep the valid indices
        assert key.startswith("video."), f"Video key must start with 'video.', got {key}"
        # Get the sub-key
        key = key.replace("video.", "")
        # Calculate the absolute indices
        absolute_indices = self.start_indices[trajectory_index] + step_indices
        return self.cached_frames[key][absolute_indices]

    def get_step_data(self, trajectory_id: int, base_index: int) -> dict:
        """Get the RAW data for a single step. No transforms are applied.

        Args:
            trajectory_id (str): The ID of the trajectory.
            base_index (int): The base index of the step.

        Returns:
            dict: The data for the step.
        """
        data = {}
        self.curr_traj_data = self.get_trajectory_data(trajectory_id)
        # Get the data for all modalities
        for modality in self.modality_keys:
            # Get the data corresponding to each key in the modality
            for key in self.modality_keys[modality]:
                data[key] = self.get_data_by_modality(trajectory_id, modality, key, base_index)
        return data

    def set_transforms_metadata(self, metadata: DatasetMetadata):
        """Set the metadata for the transforms. This is useful for transforms that need to know the metadata, such as the normalization values."""
        if self.img_resize is not None:
            all_video_keys = [key for key in self.modality_keys["video"]]
            for key in metadata.modalities.video:
                if key in all_video_keys:
                    metadata.modalities.video[key].resolution = self.img_resize
        super().set_transforms_metadata(metadata)


class WrappedLeRobotSingleDataset(LeRobotSingleDataset):
    def __init__(
        self,
        *args,
        data_split="full",
        multi_agent_output: bool = False,
        agent_video_grid: list[int] | tuple[int, int] | None = None,
        agent_video_tile_groups: list[list[int]] | None = None,
        agent_video_views: list[list[int]] | None = None,
        agent_video_output_size: list[int] | tuple[int, int] | None = None,
        shared_video_tile_group: list[int] | None = None,
        shared_video_views: list[int] | None = None,
        shared_video_output_size: list[int] | tuple[int, int] | None = None,
        agent_action_dims: list | None = None,
        multi_agent_action_dim: int = 384,
        **kwargs,
    ):
        self.multi_agent_output = multi_agent_output
        self.agent_video_grid = tuple(agent_video_grid) if agent_video_grid is not None else None
        self.agent_video_tile_groups = agent_video_tile_groups
        self.agent_video_views = agent_video_views
        self.agent_video_output_size = tuple(agent_video_output_size) if agent_video_output_size is not None else None
        self.shared_video_tile_group = shared_video_tile_group
        self.shared_video_views = shared_video_views
        self.shared_video_output_size = (
            tuple(shared_video_output_size) if shared_video_output_size is not None else None
        )
        self.agent_action_dims = agent_action_dims
        self.multi_agent_action_dim = multi_agent_action_dim
        super().__init__(*args, **kwargs)

        if data_split == "full":
            pass
        elif data_split == "train":
            self._all_steps = self._all_steps[:-len(self) // 20]
        elif data_split == "test":
            self._all_steps = self._all_steps[-len(self) // 20:]

    def _get_trajectories(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the trajectories in the dataset."""
        # Get trajectory lengths, IDs, and whitelist from dataset metadata
        episode_path = self.dataset_path / LE_ROBOT_EPISODE_FILENAME
        with open(episode_path, "r") as f:
            episode_metadata = [json.loads(line) for line in f]
        trajectory_ids = []
        trajectory_lengths = []
        for episode in episode_metadata:
            trajectory_ids.append(episode["episode_index"])
            trajectory_lengths.append(episode["length"])
        return np.array(trajectory_ids), np.array(trajectory_lengths)

    def __getitem__(self, index: int) -> dict:
        """Get the data for a single step in a trajectory.

        Args:
            index (int): The index of the step to get.

        Returns:
            dict: The data for the step.
        """
        try:
            trajectory_id, base_index = self.all_steps[index]
            original_outputs = self.transforms(self.get_step_data(trajectory_id, base_index))

            frames = _as_cthw_video(original_outputs["video"])
            frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)
            if frames.ndim == 4:
                frames = frames[:, 1:, :, :]  # Skip first frame (used only as action baseline)
            elif frames.ndim == 5:
                frames = frames[:, :, 1:, :, :]
            else:
                raise ValueError(f"Unexpected video shape after conversion: {tuple(frames.shape)}")

            shared_frames = None
            if self.multi_agent_output:
                shared_frames = _extract_video_group(
                    frames,
                    grid_shape=self.agent_video_grid,
                    view_group=self.shared_video_views,
                    tile_group=self.shared_video_tile_group,
                )
                if shared_frames is not None:
                    shared_frames = _resize_video_frames(shared_frames, self.shared_video_output_size)
                frames = _split_video_grid_by_agent(
                    frames,
                    grid_shape=self.agent_video_grid,
                    view_groups=self.agent_video_views,
                    tile_groups=self.agent_video_tile_groups,
                )
                frames = _resize_video_frames(frames, self.agent_video_output_size)
            elif frames.ndim == 5:
                raise ValueError("Explicit multi-view video requires multi_agent_output=True")

            lam_frames = _make_lam_video(frames)
            
            delta_actions = []
            for t in range(1, len(original_outputs["action"]) - 1, 4):
                delta_actions.append(original_outputs["action"][t:t + 4] - original_outputs["action"][t - 1])
            delta_actions = torch.cat(delta_actions, dim=0).float()

            action_seq = torch.zeros(self.num_frames - 1, 384, dtype=torch.float32)
            # gt_actions = torch.zeros(self.num_frames - 1, 352, dtype=torch.float32)
            # latent_actions = torch.ones(self.num_frames - 1, 32, dtype=torch.float32)
            # latent_actions = latent_actions * torch.bernoulli(0.8 * torch.ones(1)).type_as(latent_actions)
            # action_seq = torch.cat([gt_actions, latent_actions], dim=-1)
            if "inhouse_human" in str(self.dataset_path).lower():
                gt_actions = torch.zeros(self.num_frames - 1, 352, dtype=torch.float32)
                latent_actions = torch.ones(self.num_frames - 1, 32, dtype=torch.float32)
                action_seq = torch.cat([gt_actions, latent_actions], dim=-1)
                # action_seq[:, 29:58] = delta_actions
            elif "gr1" in str(self.dataset_path).lower():
                action_seq[:, :29] = delta_actions
            elif "g1" in str(self.dataset_path).lower():
                action_seq[:, 58:101] = delta_actions
            elif "yam" in str(self.dataset_path).lower():
                action_seq[:, 101:147] = delta_actions
            elif "agibot" in str(self.dataset_path).lower():
                action_seq[:, 147:169] = delta_actions

            num_agents = frames.shape[0] if self.multi_agent_output else 1
            if self.multi_agent_output:
                action_seq = _split_action_by_agent_slots(
                    action_seq,
                    self.agent_action_dims,
                    action_dim=self.multi_agent_action_dim,
                )
            
            text = ""
            if "annotation.human.coarse_action" in original_outputs:
                text = original_outputs["annotation.human.coarse_action"][0].split(":")[-1].strip()

            base_action_seq = action_seq[0] if self.multi_agent_output else action_seq
            key = base_action_seq[0:1, :29]
            key[:, :min(original_outputs["state"].shape[1], 29)] = original_outputs["state"][:, :29]
            if self.multi_agent_output:
                key = key.unsqueeze(0).repeat(num_agents, 1, 1)

            data = {
                "__key__": key,
                "action": action_seq,
                "video": frames,
                "lam_video": lam_frames,
                "num_agents": num_agents,
            }
            if shared_frames is not None:
                data["shared_video"] = shared_frames
                data["shared_lam_video"] = _make_lam_video(shared_frames)

            # Just add these to fit the interface
            data["fps"] = 4
            data["image_size"] = 256 * torch.ones(4).cuda()
            data["num_frames"] = self.num_frames
            data["padding_mask"] = torch.zeros(1, 256, 256).cuda()
            # Ensure caption key exists for online text encoding
            # Many models expect `ai_caption` when text_encoder_config.compute_online=True.
            # Default to an empty string to avoid KeyError; training configs can override upstream.
            data.setdefault("ai_caption", "")  # Return a single string, not a list
            return data
        except Exception as e:
            print(f"Error occurred while getting item {index} in {self.dataset_name}: {e}")
            print("Retrying with a random index...")
            return self.__getitem__(randint(0, len(self) - 1))
