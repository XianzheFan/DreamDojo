"""DreamDojo API server.

Exposes a /generate endpoint that accepts an initial frame + action sequence
from an external policy (e.g. OpenPI running in Libero) and returns a
predicted future video saved to disk.

Single-GPU (default):
    python examples/dreamdojo_server.py \
        --checkpoint <ckpt> --experiment dreamdojo_2b_480_640_libero \
        --save-dir <dir> --port 8000

Multi-GPU with context parallelism (e.g. 4 GPUs per server):
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
        --master_port 29500 examples/dreamdojo_server.py \
        --checkpoint <ckpt> --experiment dreamdojo_2b_480_640_libero \
        --save-dir <dir> --port 8000 --context-parallel-size 4

Example client call (Python):
    import base64, numpy as np, requests
    frame_bytes = base64.b64encode(frame_hwc_uint8.tobytes()).decode()
    resp = requests.post("http://localhost:8000/generate", json={
        "frame": frame_bytes,
        "frame_height": 480,
        "frame_width": 640,
        "actions": actions.tolist(),   # shape (T, action_dim)
        "save_name": "step_0000",
    })
    print(resp.json()["save_path"])
"""

import argparse
import base64
import threading
from pathlib import Path

import mediapy
import numpy as np
import torch
import torch.distributed as dist
import torchvision
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from cosmos_oss.init import init_environment
from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference
from cosmos_predict2.config import DEFAULT_NEGATIVE_PROMPT


class GenerateRequest(BaseModel):
    frame: str          # base64-encoded raw bytes of a uint8 HWC RGB image
    frame_height: int   # image height in pixels
    frame_width: int    # image width in pixels
    actions: list[list[float]]  # shape (T, action_dim)
    save_name: str = "output"   # filename stem, e.g. "step_0000"
    prompt: str = ""            # task language instruction, e.g. "pick up the mug"
    seed: int = 0
    guidance: float = 0.0
    num_latent_conditional_frames: int = 1


class GenerateResponse(BaseModel):
    save_path: str


app = FastAPI(title="DreamDojo API")
_model: Video2WorldInference | None = None
_save_dir: Path | None = None
_cp_size: int = 1
# Action preprocessing config (set at startup)
_action_slot_start  = 169   # where raw actions go in the 384-dim embodiment vector
_action_slot_end    = 176   # exclusive end index (169:176 = LIBERO 7-DoF)
_model_action_dim   = 384   # total embodiment vector size
_model_action_chunk = 12    # num_action_per_chunk expected by model

# Signals for worker ranks in CP mode
_SIGNAL_EXIT     = torch.tensor([0], dtype=torch.int64)
_SIGNAL_GENERATE = torch.tensor([1], dtype=torch.int64)

# Mutex so only one request runs at a time (CP requires all ranks in sync)
_inference_lock = threading.Lock()


def _preprocess_actions(actions: np.ndarray) -> torch.Tensor:
    """Map raw (T, raw_dim) actions into (model_action_chunk, model_action_dim) embodiment vector.

    Raw actions are placed at [_action_slot_start:_action_slot_end].
    The sequence is zero-padded or truncated to exactly _model_action_chunk steps.
    """
    T, raw_dim = actions.shape
    expected_raw_dim = _action_slot_end - _action_slot_start
    assert raw_dim == expected_raw_dim, (
        f"Expected raw action dim {expected_raw_dim} "
        f"(slot {_action_slot_start}:{_action_slot_end}), got {raw_dim}"
    )
    action_seq = np.zeros((_model_action_chunk, _model_action_dim), dtype=np.float32)
    n = min(T, _model_action_chunk)
    action_seq[:n, _action_slot_start:_action_slot_end] = actions[:n]
    return torch.from_numpy(action_seq)


def _build_vid_input(frame_np: np.ndarray, num_frames: int) -> torch.Tensor:
    """Build (1, C, T, H, W) uint8 tensor from a single HWC numpy frame."""
    img_tensor = torchvision.transforms.functional.to_tensor(frame_np).unsqueeze(0) * 255.0
    img_tensor = img_tensor.to(torch.uint8)
    vid_input = torch.cat(
        [img_tensor, torch.zeros_like(img_tensor).repeat(num_frames - 1, 1, 1, 1)], dim=0
    ).unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()  # (1, C, T, H, W)
    return vid_input


def _broadcast_inputs(vid_input, action, seed, guidance, num_latent_conditional_frames, prompt=""):
    """Broadcast inference inputs from rank 0 to all CP ranks."""
    device = torch.device("cuda")

    # Pack scalar metadata into one tensor; include prompt byte length for broadcast
    prompt_bytes = prompt.encode("utf-8") if dist.get_rank() == 0 else b""
    if dist.get_rank() == 0:
        meta = torch.tensor(
            [vid_input.shape[2], action.shape[0], action.shape[1], seed, num_latent_conditional_frames,
             int(guidance * 1e6), len(prompt_bytes)],
            dtype=torch.int64, device=device,
        )
    else:
        meta = torch.zeros(7, dtype=torch.int64, device=device)
    dist.broadcast(meta, src=0)

    T            = int(meta[0].item())
    action_steps = int(meta[1].item())
    action_dim   = int(meta[2].item())
    seed         = int(meta[3].item())
    num_lcf      = int(meta[4].item())
    guidance     = float(meta[5].item()) / 1e6
    prompt_len   = int(meta[6].item())

    if dist.get_rank() != 0:
        vid_input = torch.zeros(1, 3, T, 480, 640, dtype=torch.uint8, device=device)
        action    = torch.zeros(action_steps, action_dim, dtype=torch.float32, device=device)
    else:
        vid_input = vid_input.to(device)
        action    = action.to(device)

    dist.broadcast(vid_input, src=0)
    dist.broadcast(action,    src=0)

    # Broadcast prompt string as a fixed-size byte tensor (padded to prompt_len)
    if prompt_len > 0:
        if dist.get_rank() == 0:
            prompt_tensor = torch.frombuffer(
                prompt_bytes.ljust(prompt_len, b"\x00"), dtype=torch.uint8
            ).to(device)
        else:
            prompt_tensor = torch.zeros(prompt_len, dtype=torch.uint8, device=device)
        dist.broadcast(prompt_tensor, src=0)
        prompt = prompt_tensor.cpu().numpy().tobytes().rstrip(b"\x00").decode("utf-8")
    else:
        prompt = ""

    return vid_input.cpu(), action.cpu(), seed, guidance, num_lcf, prompt


def _run_inference(vid_input, action, seed, guidance, num_latent_conditional_frames, prompt=""):
    with torch.no_grad():
        video = _model.generate_vid2world(
            prompt=prompt,
            input_path=vid_input,
            action=action.float(),
            guidance=guidance,
            num_video_frames=vid_input.shape[2],
            num_latent_conditional_frames=num_latent_conditional_frames,
            resolution="480,640",
            seed=seed,
            negative_prompt=DEFAULT_NEGATIVE_PROMPT,
            lam_video=None,
        )
    return video


def worker_loop():
    """Runs on ranks 1..N-1. Waits for rank 0 to signal work, then participates
    in the collective inference calls."""
    device = torch.device("cuda")
    while True:
        signal = _SIGNAL_EXIT.clone().to(device)
        dist.broadcast(signal, src=0)

        if signal.item() == _SIGNAL_EXIT.item():
            break

        # Receive inputs and participate in inference
        vid_input, action, seed, guidance, num_lcf, prompt = _broadcast_inputs(
            None, None, None, None, None
        )
        _run_inference(vid_input, action, seed, guidance, num_lcf, prompt)


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    # Decode and resize frame
    raw = base64.b64decode(req.frame)
    frame_np = np.frombuffer(raw, dtype=np.uint8).reshape(req.frame_height, req.frame_width, 3)
    if frame_np.shape[:2] != (480, 640):
        import cv2
        frame_np = cv2.resize(frame_np, (640, 480))

    actions   = np.array(req.actions, dtype=np.float32)   # (T, raw_action_dim)
    model_required_frames = _model.model.tokenizer.get_pixel_num_frames(_model.model.config.state_t)
    vid_input  = _build_vid_input(frame_np, model_required_frames)
    action_t   = _preprocess_actions(actions)              # (model_action_chunk, model_action_dim)

    with _inference_lock:
        if _cp_size > 1:
            # Signal workers to participate
            signal = _SIGNAL_GENERATE.clone().to("cuda")
            dist.broadcast(signal, src=0)
            vid_input, action_t, seed, guidance, num_lcf, prompt = _broadcast_inputs(
                vid_input, action_t, req.seed, req.guidance, req.num_latent_conditional_frames,
                req.prompt,
            )
        else:
            seed, guidance, num_lcf, prompt = (
                req.seed, req.guidance, req.num_latent_conditional_frames, req.prompt
            )

        video = _run_inference(vid_input, action_t, seed, guidance, num_lcf, prompt)

    # Decode: (1, C, T, H, W) -> (T, H, W, C) uint8
    video_np = (
        torch.clamp((video[0] + 1) / 2, 0, 1) * 255
    ).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()

    save_path = _save_dir / f"{req.save_name}.mp4"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(str(save_path), video_np, fps=10)

    return GenerateResponse(save_path=str(save_path))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--experiment",  required=True)
    parser.add_argument("--save-dir",    required=True)
    parser.add_argument("--config-file", default="cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py")
    parser.add_argument("--port",        type=int, default=8000)
    parser.add_argument("--host",        default="0.0.0.0")
    parser.add_argument("--context-parallel-size", type=int, default=1,
                        help="Number of GPUs per server for context parallelism")
    parser.add_argument("--action-slot-start", type=int, default=169,
                        help="Start index in 384-dim embodiment vector for raw actions (default: 169 = LIBERO)")
    parser.add_argument("--action-slot-end",   type=int, default=176,
                        help="End index (exclusive) in 384-dim embodiment vector (default: 176 = LIBERO 7-DoF)")
    parser.add_argument("--model-action-dim",  type=int, default=384,
                        help="Total action dim expected by the model (default: 384)")
    parser.add_argument("--model-action-chunk", type=int, default=12,
                        help="num_action_per_chunk expected by the model (default: 12)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    _save_dir = Path(args.save_dir)
    _cp_size  = args.context_parallel_size

    _action_slot_start  = args.action_slot_start
    _action_slot_end    = args.action_slot_end
    _model_action_dim   = args.model_action_dim
    _model_action_chunk = args.model_action_chunk

    init_environment()

    _model = Video2WorldInference(
        experiment_name=args.experiment,
        ckpt_path=Path(args.checkpoint),
        s3_credential_path="",
        context_parallel_size=_cp_size,
        config_file=args.config_file,
    )
    state_t = _model.model.config.state_t
    if _cp_size > 1 and state_t % _cp_size != 0:
        raise ValueError(
            f"context_parallel_size={_cp_size} must divide state_t={state_t} "
            f"(T_latent={state_t}). Use --context-parallel-size 1, 2, or 4."
        )
    print(f"[rank {dist.get_rank() if dist.is_initialized() else 0}] "
          f"Model loaded from {args.checkpoint}", flush=True)

    if _cp_size > 1 and dist.is_initialized() and dist.get_rank() != 0:
        # Non-rank-0: sit in worker loop
        worker_loop()
    else:
        # Rank 0 (or single-GPU): serve HTTP
        uvicorn.run(app, host=args.host, port=args.port)
