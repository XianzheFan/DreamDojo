# Multi-Agent DreamDojo Notes

DreamDojo originally treats the robot observation as one fused stream: multiple
camera views are commonly tiled into a single 4x4 image, and all robot actions
are packed into one action vector. The multi-agent path keeps each actor stream
explicit while sharing the same DiT/VAE weights.

## Recommended View And Action Assignment

Use the main camera as shared scene context, not as its own agent. The preferred
path is to encode the main camera once with the shared tokenizer/VAE and provide
that latent as a conditioning signal to each agent stream.

- Shared video: main camera.
- Left agent video: left/wrist-left camera.
- Right agent video: right/wrist-right camera.
- Left agent action: left arm/action + left wrist/hand/gripper action.
- Right agent action: right arm/action + right wrist/hand/gripper action.

For a 4x4 tiled video, tile groups use row-major tile indices:

```yaml
shared_video_tile_group: [0]  # main camera
agent_video_grid: [4, 4]
agent_video_tile_groups:
  - [1]  # left agent: left/wrist-left camera
  - [2]  # right agent: right/wrist-right camera
```

If the original 4x4 order is different, only change these tile indices. If the
dataset provides explicit multi-camera keys instead of one tiled frame, use the
same grouping idea over view indices.

As a fallback, the groups may overlap if you want to keep main inside each
agent's target video:

```yaml
agent_video_tile_groups:
  - [0, 1]  # left agent: main + left/wrist-left camera
  - [0, 2]  # right agent: main + right/wrist-right camera
```

## Tensor Shapes

The dataset may emit either the original single-agent tensors or explicit
multi-agent tensors.

- Single-agent video: `[C, T, H, W]`
- Multi-agent video: `[P, C, T, H, W]`
- Shared video: `[C, T, H, W]`
- Single-agent action: `[T, D]`
- Multi-agent action: `[P, T, D]`
- Multi-agent LAM video: `[P, T, H, W, C]`

At model entry, `prepare_multi_agent_batch_inplace` flattens agent streams into
the batch dimension:

- Video: `[B, P, C, T, H, W] -> [B * P, C, T, H, W]`
- Action: `[B, P, T, D] -> [B * P, T, D]`

It also creates `agent_ids`, so the action-conditioned DiT can add a learned
role embedding for left vs. right streams.

If `shared_video` is present, the model encodes it once:

- Shared video: `[B, C, T, H, W] -> shared_video_latent=[B, C_latent, T_latent, H_latent, W_latent]`
- Shared latent is repeated to `[B * P, C_latent, T_latent, H_latent, W_latent]`
- The DiT pools it and injects it into the timestep/AdaLN condition.

Enable this path with:

```yaml
model:
  config:
    net:
      shared_video_conditioning: true
```

## Action Slice Examples

The action embedder keeps the existing 384-wide DreamDojo action layout. Each
agent receives a 384-wide action vector with only that agent's slices filled and
the rest set to zero. This preserves checkpoint compatibility with the current
`action_dim: 384` setup.

GR1 compact action is placed in slots `0:29` with concat order:
`left_arm, right_arm, left_hand, right_hand, waist`.

```yaml
agent_action_dims:
  - [[0, 7], [14, 20]]   # left arm + left hand/wrist
  - [[7, 14], [20, 26]]  # right arm + right hand/wrist
```

If waist should be shared by both agents, add `[26, 29]` to both groups.

G1 action is placed in slots `58:101`; left and right arm/hand slices are:

```yaml
agent_action_dims:
  - [[73, 87]]
  - [[87, 101]]
```

YAM action is placed in slots `101:147`; left and right slices are:

```yaml
agent_action_dims:
  - [[101, 117], [133, 134], [135, 141]]
  - [[117, 133], [134, 135], [141, 147]]
```

AgiBot action is placed in slots `147:169`; left and right slices are:

```yaml
agent_action_dims:
  - [[147, 154], [161, 162]]
  - [[154, 161], [162, 163]]
```

For global body, head, waist, or base actions, either duplicate those slices
into both agents or add a third body agent. The default examples keep only the
left/right manipulator actions per agent.

## Resolution

When extracting tiles from a 4x4 grid, each agent stream can be smaller than
the original frame. To keep the original training resolution, set:

```yaml
agent_video_output_size: [480, 640]
```

Leaving it unset preserves the raw extracted tile-group resolution.
