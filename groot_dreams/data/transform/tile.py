from typing import Any

import torch
from pydantic import Field

from groot_dreams.data.transform.base import ModalityTransform


class VideoTile(ModalityTransform):
    """Tile multiple video views into a single 2x2 grid image.

    Takes N video tensors (each [T, C, H, W]) specified by `apply_to`,
    places them into a 2x2 grid (row-major order, unfilled cells are black),
    and stores the result under `output_key`. The original keys are removed.
    """

    output_key: str = Field(..., description="The output key for the tiled video")

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        views = [data.pop(key) for key in self.apply_to]
        # Each view: [T, C, H, W] torch tensor after VideoToTensor + VideoResize
        T, C, H, W = views[0].shape

        # Create 2x2 canvas
        canvas = torch.zeros(T, C, H * 2, W * 2, dtype=views[0].dtype)
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for i, view in enumerate(views):
            r, c = positions[i]
            canvas[:, :, r * H:(r + 1) * H, c * W:(c + 1) * W] = view

        data[self.output_key] = canvas
        return data
