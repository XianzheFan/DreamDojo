#!/usr/bin/env python3
"""
DreamDojo-style Value Expert: DINOv2-initialized, multi-view, with pairwise ranking loss.

Architecture (following DreamDojo D.6 but extended for multi-view + outcome scoring):
  - Frozen DINOv2 backbone extracts per-frame per-view features
  - View fusion projects concatenated multi-view features
  - Temporal attention (global self-attention across frames)
  - Value head predicts a scalar value

Training losses:
  - MSE loss for individual outcome value prediction
  - Pairwise margin ranking loss: for pairs with different values, enforce correct ordering

Usage:
    python scripts/train_value_expert.py \
        --annotation_path outputs/critical_phase_annotations/annotations.json \
        --dataset_path datasets/fold_towel_0109_agilex \
        --output_dir outputs/value_expert \
        [--num_frames 4] [--batch_size 8] [--epochs 50] [--lr 1e-4]
"""

import argparse
import itertools
import json
import math
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# Model: DreamDojo-style Value Expert
class MultiViewTemporalAttention(nn.Module):
    """Global attention module over temporally fused multi-view features."""

    def __init__(self, feat_dim: int, num_heads: int = 8, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim,
            nhead=num_heads,
            dim_feedforward=feat_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, feat_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) temporal feature sequence.
        Returns:
            (B, D) pooled representation.
        """
        B = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, 1+T, D)
        x = self.encoder(x)
        return x[:, 0]  # CLS token output


class ValueExpert(nn.Module):
    """
    DINOv2-initialized value expert for multi-view robot manipulation scoring.

    Following DreamDojo D.6:
    - Frozen DINOv2 backbone independently extracts features per frame
    - Extended with multi-view fusion (3 camera views)
    - Global attention across temporal dimension
    - Scalar value prediction
    """

    def __init__(
        self,
        dinov2_model: str = "dinov2_vitb14",
        num_views: int = 3,
        num_frames: int = 4,
        num_attn_layers: int = 2,
        num_attn_heads: int = 8,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.num_views = num_views
        self.num_frames = num_frames

        # Load DINOv2 backbone
        self.backbone = torch.hub.load("facebookresearch/dinov2", dinov2_model)
        self.feat_dim = self.backbone.embed_dim  # 768 for vitb14

        if freeze_backbone:
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad = False

        # View fusion: project concatenated multi-view features to feat_dim
        self.view_fusion = nn.Sequential(
            nn.Linear(self.feat_dim * num_views, self.feat_dim),
            nn.LayerNorm(self.feat_dim),
            nn.GELU(),
        )

        # Temporal attention with global self-attention (DreamDojo D.6)
        self.temporal_attn = MultiViewTemporalAttention(
            feat_dim=self.feat_dim,
            num_heads=num_attn_heads,
            num_layers=num_attn_layers,
            dropout=dropout,
        )

        # Value prediction head
        self.value_head = nn.Sequential(
            nn.Linear(self.feat_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # Initialize weights (except frozen backbone)
        self._init_weights()

    def _init_weights(self):
        for m in [self.view_fusion, self.value_head]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.trunc_normal_(layer.weight, std=0.02)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: (B, T, V, C, H, W) multi-view video clip.
                T = num_frames, V = num_views (3 cameras).
        Returns:
            (B,) predicted value scores.
        """
        B, T, V, C, H, W = frames.shape

        # Flatten for backbone: (B*T*V, C, H, W)
        x = frames.reshape(B * T * V, C, H, W)

        # Extract features with frozen DINOv2
        with torch.no_grad():
            feats = self.backbone(x)  # (B*T*V, feat_dim)

        # Reshape: (B, T, V, feat_dim)
        feats = feats.reshape(B, T, V, self.feat_dim)

        # Fuse views: concat along view dim -> project
        feats = feats.reshape(B, T, V * self.feat_dim)  # (B, T, V*D)
        feats = self.view_fusion(feats)  # (B, T, D)

        # Temporal attention
        pooled = self.temporal_attn(feats)  # (B, D)

        # Value prediction
        value = self.value_head(pooled).squeeze(-1)  # (B,)
        return value


# Dataset: Critical Phase clips for value expert training
class CriticalPhaseDataset(Dataset):
    """
    Loads annotated critical phase data for value expert training.

    Each sample: 3-view video clip (num_frames frames) around the critical phase + value label.
    """

    def __init__(
        self,
        annotations: list[dict],
        dataset_path: Path,
        num_frames: int = 4,
        img_size: int = 224,
        augment: bool = True,
    ):
        # Filter out failed annotations
        self.annotations = [a for a in annotations if "outcome" in a and "error" not in a]
        self.dataset_path = Path(dataset_path)
        self.num_frames = num_frames
        self.img_size = img_size
        self.augment = augment

        self.view_keys = [
            "observation.images.cam_high",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        ]

        # DINOv2 normalization (ImageNet)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.annotations)

    def _load_frames(self, video_path: str, frame_indices: list[int]) -> np.ndarray:
        """Load specific frames from a video file."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.img_size, self.img_size))
            else:
                frame = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            frames.append(frame)
        cap.release()
        return np.stack(frames)  # (T, H, W, 3)

    def __getitem__(self, idx: int) -> dict:
        ann = self.annotations[idx]
        ep_idx = ann["episode_index"]

        # Determine frame indices to sample within the critical phase + context
        start = ann["extracted_start"]
        end = ann["extracted_end"]
        total = end - start

        if total <= self.num_frames:
            indices = list(range(start, end))
            # Pad to num_frames
            while len(indices) < self.num_frames:
                indices.append(indices[-1])
        else:
            # Uniformly sample num_frames from the extracted range
            if self.augment:
                # Random offset jitter during training
                jitter = random.randint(0, max(0, total - self.num_frames))
                step = max(1, (total - jitter) // self.num_frames)
                indices = [start + jitter + i * step for i in range(self.num_frames)]
            else:
                step = total // self.num_frames
                indices = [start + i * step for i in range(self.num_frames)]
            indices = [min(i, end - 1) for i in indices]

        # Load frames from all 3 views: (T, V, H, W, 3)
        all_views = []
        for view_key in self.view_keys:
            video_path = str(
                self.dataset_path / f"videos/chunk-000/{view_key}/episode_{ep_idx:06d}.mp4"
            )
            frames = self._load_frames(video_path, indices)  # (T, H, W, 3)
            all_views.append(frames)

        # Stack views: (T, V, H, W, 3)
        clip = np.stack(all_views, axis=1)

        # Normalize: (T, V, H, W, 3) -> float32 [0,1] -> ImageNet norm
        clip = clip.astype(np.float32) / 255.0
        clip = (clip - self.mean) / self.std

        # Convert to (T, V, 3, H, W) for PyTorch
        clip = np.transpose(clip, (0, 1, 4, 2, 3))
        clip = torch.from_numpy(clip).float()

        value = torch.tensor(ann["value"], dtype=torch.float32)

        return {
            "frames": clip,       # (T, V, C, H, W)
            "value": value,       # scalar
            "episode_index": ep_idx,
            "outcome": ann["outcome"],
        }


def rank_collate_fn(batch: list[dict]) -> dict:
    """
    Collate with exhaustive pairwise ranking within the batch.
    For all pairs (i, j) where value_i != value_j, generate a ranking target.
    """
    frames = torch.stack([b["frames"] for b in batch])
    values = torch.stack([b["value"] for b in batch])
    outcomes = [b["outcome"] for b in batch]
    episode_indices = [b["episode_index"] for b in batch]

    B = len(batch)
    pair_i, pair_j, pair_target = [], [], []
    for i in range(B):
        for j in range(i + 1, B):
            if values[i] != values[j]:
                pair_i.append(i)
                pair_j.append(j)
                pair_target.append(1.0 if values[i] > values[j] else -1.0)

    return {
        "frames": frames,
        "values": values,
        "outcomes": outcomes,
        "episode_indices": episode_indices,
        "pair_i": torch.tensor(pair_i, dtype=torch.long) if pair_i else torch.zeros(0, dtype=torch.long),
        "pair_j": torch.tensor(pair_j, dtype=torch.long) if pair_j else torch.zeros(0, dtype=torch.long),
        "pair_target": torch.tensor(pair_target, dtype=torch.float32) if pair_target else torch.zeros(0),
    }


def value_scoring_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE loss for individual value prediction."""
    return F.mse_loss(pred, target)


def pairwise_ranking_loss(
    pred: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    pair_target: torch.Tensor,
    margin: float = 0.1,
) -> torch.Tensor:
    """
    Margin ranking loss for pairwise comparisons.

    For each pair (i, j) with target t in {-1, +1}:
        loss = max(0, -t * (pred_i - pred_j) + margin)

    This enforces that if target_i > target_j (t=+1), then pred_i > pred_j + margin.
    """
    if len(pair_i) == 0:
        return torch.tensor(0.0, device=pred.device)

    pred_i = pred[pair_i]
    pred_j = pred[pair_j]
    return F.margin_ranking_loss(pred_i, pred_j, pair_target, margin=margin)


def train_one_epoch(
    model: ValueExpert,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    rank_loss_weight: float = 1.0,
    margin: float = 0.1,
) -> dict:
    model.train()
    # Keep backbone frozen
    model.backbone.eval()

    total_loss = 0.0
    total_score_loss = 0.0
    total_rank_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        frames = batch["frames"].to(device)  # (B, T, V, C, H, W)
        values = batch["values"].to(device)  # (B,)
        pair_i = batch["pair_i"].to(device)
        pair_j = batch["pair_j"].to(device)
        pair_target = batch["pair_target"].to(device)

        pred = model(frames)

        # Individual scoring loss
        loss_score = value_scoring_loss(pred, values)

        # Pairwise ranking loss
        loss_rank = pairwise_ranking_loss(pred, pair_i, pair_j, pair_target, margin=margin)

        loss = loss_score + rank_loss_weight * loss_rank

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_score_loss += loss_score.item()
        total_rank_loss += loss_rank.item()
        num_batches += 1

    return {
        "loss": total_loss / max(num_batches, 1),
        "score_loss": total_score_loss / max(num_batches, 1),
        "rank_loss": total_rank_loss / max(num_batches, 1),
    }


@torch.no_grad()
def evaluate(
    model: ValueExpert,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()

    all_preds = []
    all_targets = []
    all_outcomes = []

    for batch in dataloader:
        frames = batch["frames"].to(device)
        values = batch["values"].to(device)

        pred = model(frames)
        all_preds.append(pred.cpu())
        all_targets.append(values.cpu())
        all_outcomes.extend(batch["outcomes"])

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    mse = F.mse_loss(preds, targets).item()

    # Pairwise ranking accuracy: for all pairs with different values,
    # what fraction does the model rank correctly?
    correct = 0
    total_pairs = 0
    for i in range(len(preds)):
        for j in range(i + 1, len(preds)):
            if targets[i] != targets[j]:
                total_pairs += 1
                if (preds[i] - preds[j]).sign() == (targets[i] - targets[j]).sign():
                    correct += 1

    rank_acc = correct / max(total_pairs, 1)

    return {
        "mse": mse,
        "rank_accuracy": rank_acc,
        "num_samples": len(preds),
        "num_pairs": total_pairs,
    }


def main():
    parser = argparse.ArgumentParser(description="Train DreamDojo-style Value Expert")
    parser.add_argument("--annotation_path", type=str, required=True,
                        help="Path to annotations.json from annotate_critical_phases.py")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the LeRobot dataset")
    parser.add_argument("--output_dir", type=str, default="outputs/value_expert",
                        help="Directory to save model checkpoints")

    # Model config
    parser.add_argument("--dinov2_model", type=str, default="dinov2_vitb14",
                        choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"],
                        help="DINOv2 model variant")
    parser.add_argument("--num_frames", type=int, default=4,
                        help="Number of frames per clip (DreamDojo D.6 uses 4)")
    parser.add_argument("--num_attn_layers", type=int, default=2,
                        help="Number of transformer layers in temporal attention")
    parser.add_argument("--img_size", type=int, default=224,
                        help="Input image size for DINOv2")

    # Training config
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--rank_loss_weight", type=float, default=1.0,
                        help="Weight for pairwise ranking loss relative to MSE loss")
    parser.add_argument("--rank_margin", type=float, default=0.1,
                        help="Margin for pairwise ranking loss")
    parser.add_argument("--val_split", type=float, default=0.15,
                        help="Fraction of data for validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load annotations
    with open(args.annotation_path) as f:
        annotations = json.load(f)
    valid_annotations = [a for a in annotations if "outcome" in a and "error" not in a]
    print(f"Loaded {len(valid_annotations)} valid annotations out of {len(annotations)} total.")

    # Train/val split
    random.shuffle(valid_annotations)
    n_val = max(1, int(len(valid_annotations) * args.val_split))
    val_annotations = valid_annotations[:n_val]
    train_annotations = valid_annotations[n_val:]
    print(f"Train: {len(train_annotations)}, Val: {len(val_annotations)}")

    # Datasets
    train_dataset = CriticalPhaseDataset(
        train_annotations, args.dataset_path,
        num_frames=args.num_frames, img_size=args.img_size, augment=True,
    )
    val_dataset = CriticalPhaseDataset(
        val_annotations, args.dataset_path,
        num_frames=args.num_frames, img_size=args.img_size, augment=False,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=rank_collate_fn,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=rank_collate_fn,
        pin_memory=True,
    )

    # Model
    print(f"Building ValueExpert with {args.dinov2_model} backbone...")
    model = ValueExpert(
        dinov2_model=args.dinov2_model,
        num_views=3,
        num_frames=args.num_frames,
        num_attn_layers=args.num_attn_layers,
        freeze_backbone=True,
    ).to(device)

    # Only optimize trainable parameters (backbone is frozen)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    num_trainable = sum(p.numel() for p in trainable_params)
    num_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {num_trainable:,} / {num_total:,} ({num_trainable/num_total*100:.1f}%)")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Save config
    config = vars(args)
    config["num_trainable_params"] = num_trainable
    config["num_total_params"] = num_total
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Training loop
    best_rank_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device,
            rank_loss_weight=args.rank_loss_weight,
            margin=args.rank_margin,
        )
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"lr={lr:.2e} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"(score={train_metrics['score_loss']:.4f}, rank={train_metrics['rank_loss']:.4f}) | "
            f"val_mse={val_metrics['mse']:.4f} rank_acc={val_metrics['rank_accuracy']:.3f}"
        )

        # Save best model
        if val_metrics["rank_accuracy"] >= best_rank_acc:
            best_rank_acc = val_metrics["rank_accuracy"]
            ckpt_path = output_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": {k: v for k, v in model.state_dict().items() if "backbone" not in k},
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "config": config,
            }, ckpt_path)

        # Save periodic checkpoint
        if epoch % 10 == 0:
            ckpt_path = output_dir / f"checkpoint_epoch{epoch:03d}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": {k: v for k, v in model.state_dict().items() if "backbone" not in k},
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "config": config,
            }, ckpt_path)

    print(f"\nTraining complete. Best rank accuracy: {best_rank_acc:.3f}")
    print(f"Checkpoints saved to {output_dir}")


if __name__ == "__main__":
    main()
