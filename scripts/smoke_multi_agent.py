import os

os.environ.setdefault("COSMOS_PREDICT2_SKIP_CUDA_EXTRA_CHECK", "1")

import torch

from cosmos_predict2._src.predict2.action.models.multi_agent import (
    encode_shared_video_condition_inplace,
    prepare_multi_agent_batch_inplace,
)
from cosmos_predict2._src.predict2.action.models.action_utils import (
    normalize_multi_agent_action,
)


AGENT_ACTION_DIMS = (
    ((0, 7), (14, 20)),
    ((7, 14), (20, 26)),
)


class FakeNet:
    action_dim = 384
    agent_action_dims = AGENT_ACTION_DIMS


class FakeModel:
    tensor_kwargs = {"device": "cpu", "dtype": torch.float32}

    def encode(self, shared_video):
        assert shared_video.shape == (2, 3, 13, 16, 16)
        assert torch.is_floating_point(shared_video)
        assert shared_video.min() >= -1.0 and shared_video.max() <= 1.0
        return torch.ones(shared_video.shape[0], 16, 4, 2, 2, dtype=torch.float32)


def assert_close(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_action_slot_preservation():
    fused = torch.zeros(2, 12, 384)
    fused[:, :, 0:7] = 1.0
    fused[:, :, 7:14] = 2.0
    fused[:, :, 14:20] = 3.0
    fused[:, :, 20:26] = 4.0
    per_agent = normalize_multi_agent_action(
        fused,
        action_dim=384,
        num_agents=2,
        agent_action_dims=AGENT_ACTION_DIMS,
    )
    assert per_agent.shape == (2, 2, 12, 384)
    assert_close(per_agent[:, 0, :, 0:7], torch.ones(2, 12, 7))
    assert_close(per_agent[:, 0, :, 7:14], torch.zeros(2, 12, 7))
    assert_close(per_agent[:, 0, :, 14:20], torch.full((2, 12, 6), 3.0))
    assert_close(per_agent[:, 0, :, 20:26], torch.zeros(2, 12, 6))
    assert_close(per_agent[:, 1, :, 0:7], torch.zeros(2, 12, 7))
    assert_close(per_agent[:, 1, :, 7:14], torch.full((2, 12, 7), 2.0))
    assert_close(per_agent[:, 1, :, 14:20], torch.zeros(2, 12, 6))
    assert_close(per_agent[:, 1, :, 20:26], torch.full((2, 12, 6), 4.0))
    assert_close(per_agent[:, :, :, 26:], torch.zeros(2, 2, 12, 358))
    return per_agent


def test_batch_flatten_and_shared_video(per_agent_action):
    batch = {
        "video": torch.zeros(2, 2, 3, 13, 16, 16, dtype=torch.uint8),
        "action": per_agent_action.clone(),
        "lam_video": torch.zeros(2, 2, 12, 8, 8, 3, dtype=torch.float32),
        "shared_video": torch.zeros(2, 3, 13, 16, 16, dtype=torch.uint8),
        "fps": torch.ones(2),
        "padding_mask": torch.zeros(2, 1, 16, 16),
        "image_size": torch.ones(2, 4) * 16,
        "ai_caption": ["left", "right"],
    }
    prepare_multi_agent_batch_inplace(batch, input_video_key="video", net=FakeNet())
    assert batch["video"].shape == (4, 3, 13, 16, 16)
    assert batch["action"].shape == (4, 12, 384)
    assert batch["lam_video"].shape == (4, 12, 8, 8, 3)
    assert batch["agent_ids"].tolist() == [0, 1, 0, 1]
    assert batch["fps"].shape == (4,)
    assert batch["padding_mask"].shape == (4, 1, 16, 16)
    assert batch["image_size"].shape == (4, 4)
    assert batch["ai_caption"] == ["left", "left", "right", "right"]
    encode_shared_video_condition_inplace(FakeModel(), batch)
    assert batch["shared_video_latent"].shape == (4, 16, 4, 2, 2)


def main():
    print(f"torch={torch.__version__}")
    per_agent_action = test_action_slot_preservation()
    test_batch_flatten_and_shared_video(per_agent_action)
    print("multi-agent smoke test passed")


if __name__ == "__main__":
    main()
