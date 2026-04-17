# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# -----------------------------------------------------------------------------

"""
Script for streaming action-conditioned video generation using KV cache in a naive way
TODO (kaichun): make this command up to date
Example:
python -m cosmos_predict2._src.predict2.interactive.inference.action_video2world \
    --config=cosmos_predict2/_src/predict2/interactive/configs/config_distill.py \
    --experiment=cosmos_predict2p5_2B_action_gr00t_gr1_self_forcing_no_s3 \
    --ckpt_path checkpoints/iter_000003000 \
    --input_json datasets/eval/info.json
"""

import argparse
import json
import os
from typing import Any, Dict, List, Tuple
from pathlib import Path

import mediapy
import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms.functional as TF
from loguru import logger
from PIL import Image
import time

try:
    from megatron.core import parallel_state
except Exception:

    class _DummyParallelState:
        def is_initialized(self):
            return False

        def get_context_parallel_group(self):
            return None

        def initialize_model_parallel(self, **kwargs):
            return None

        def destroy_model_parallel(self):
            return None

    parallel_state = _DummyParallelState()  # type: ignore

from cosmos_predict2._src.imaginaire.utils import distributed, misc
from cosmos_predict2._src.imaginaire.visualize.video import save_img_or_video
from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.distill.utils.model_loader import load_model_from_checkpoint
from cosmos_predict2._src.predict2.interactive.datasets.utils import extract_cr1_embedding
from cosmos_predict2._src.predict2.models.video2world_model_rectified_flow import (
    NUM_CONDITIONAL_FRAMES_KEY,
)

_DEFAULT_NEGATIVE_PROMPT = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Action-conditioned streaming Video2World inference script")
    parser.add_argument(
        "--config", type=str, default="cosmos_predict2/_src/predict2/interactive/configs/config.py", help="Config file"
    )
    parser.add_argument("--experiment", type=str, required=True, help="Experiment config name")
    parser.add_argument("--ckpt_path", type=str, default="", help="S3/local checkpoint path")
    parser.add_argument("--s3_cred", type=str, default="credentials/s3_checkpoint.secret")
    parser.add_argument("--input_json", type=str, required=True, help="Path to JSON entries list")
    parser.add_argument("--resolution", type=str, default="none", help="Optional resolution H,W (e.g. 256,320)")
    parser.add_argument("--fps", type=float, default=10.0, help="FPS for output")
    parser.add_argument("--num_steps", type=int, default=4, help="Student steps per frame")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--start_frame_idx", type=int, default=0, help="Start frame index for conditioning")
    parser.add_argument("--max_frames", type=int, default=13, help="Max frames for conditioning [default is 13]")
    # parser.add_argument("--cache_frame_size", type=int, default=-1, help="Cache frame size (-1 uses video frame size)")
    parser.add_argument(
        "--cr1_embeddings_path",
        type=str,
        default="datasets/cr1_empty_string_text_embeddings.pt",
        help="Local path to CR1 empty-string text embeddings (.pt)",
    )
    parser.add_argument("--context_parallel_size", type=int, default=1, help="Context parallel size")
    parser.add_argument("--enable_fsdp", action="store_true", help="Enable FSDP")
    parser.add_argument("--torch_compile", action="store_true", help="Enable torch.compile for DiT and VAE")
    return parser.parse_args()


class ActionStreamingInference:
    def __init__(
        self,
        config_path: str,
        experiment_name: str,
        ckpt_path: str,
        s3_credential_path: str,
        cr1_embeddings_path: str,
        context_parallel_size: int = 1,
        enable_fsdp: bool = False,
        torch_compile: bool = False,
    ) -> None:
        self.experiment_name = experiment_name
        self.ckpt_path = ckpt_path
        self.s3_credential_path = s3_credential_path
        self.cr1_embeddings_path = cr1_embeddings_path
        self.context_parallel_size = context_parallel_size
        self.process_group = None
        self.torch_compile = torch_compile

        if self.context_parallel_size > 1:
            self._init_distributed()

        model, config = load_model_from_checkpoint(
            experiment_name=self.experiment_name,
            s3_checkpoint_dir=self.ckpt_path,
            config_file=config_path,
            load_ema_to_reg=True,
            experiment_opts=["ckpt_type=dcp"],
            skip_teacher_init=True,
            enable_fsdp=enable_fsdp,
        )

        if self.context_parallel_size > 1:
            model.net.enable_context_parallel(self.process_group)  # type: ignore

        # assert isinstance(
        #     model, ActionVideo2WorldModelRFSelfForcingDMD2
        # ), "Loaded model is not ActionVideo2WorldModelRFSelfForcingDMD2; check experiment config."

        self.model = model
        self.config = config
        self.batch_size = 1

        # Load CR1 empty-string text embeddings once (CPU). Expected shapes: [B, T, D] or [T, D].
        extract_cr1_embedding(self.cr1_embeddings_path)
        _emb = torch.load(self.cr1_embeddings_path, map_location="cpu")
        if isinstance(_emb, (list, tuple)):
            _emb = _emb[0]
        if not torch.is_tensor(_emb):
            raise ValueError("Loaded CR1 embeddings are not a torch.Tensor")
        if _emb.dim() == 2:
            _emb = _emb.unsqueeze(0)  # [1, T, D]
        elif _emb.dim() != 3:
            raise ValueError(f"Unexpected CR1 embeddings dim: {_emb.dim()} (expected 2 or 3)")
        self.t5_text_embeddings_cpu = _emb  # cache on CPU; move per inference call
        logger.info(
            f"Loaded CR1 text embeddings: shape={tuple(self.t5_text_embeddings_cpu.shape)} from {self.cr1_embeddings_path}"
        )

        # Set up torch.compile for DiT and VAE if requested
        self._compiled_decode = None
        if self.torch_compile:
            self._setup_torch_compile()

    def _init_distributed(self) -> None:
        distributed.init()
        parallel_state.initialize_model_parallel(context_parallel_size=self.context_parallel_size)
        self.process_group = parallel_state.get_context_parallel_group()
        logger.info(f"Initialized context parallel with size {self.context_parallel_size}")
        logger.info(f"Current rank: {distributed.get_rank()}, World size: {distributed.get_world_size()}")

    def _setup_torch_compile(self) -> None:
        """Prepare the model for ``torch.compile`` and compile DiT blocks + VAE.

        Steps (order matters):
        1. Unwrap activation-checkpointing wrappers (training artefact).
        2. Replace **all** TransformerEngine C extensions with pure-PyTorch
           equivalents (RMSNorm → PyTorch, DotProductAttention → SDPA,
           apply_rotary_pos_emb → fused=False).
        3. Pre-install stateless KV cache wrappers with **tensor** state.
           Tensor-based caches use pure tensor ops (in-place writes + slicing)
           instead of Python lists, eliminating all graph breaks.
        4. Compile each DiT block individually (zero graph breaks per block).
        5. Compile the VAE decoder with ``mode="reduce-overhead"`` (CUDA graphs).

        After steps 1–3, every DiT block is fully traceable by
        ``torch.compile``, enabling the compiler to fuse all operations
        (AdalN + norms + QKV projections + RoPE + KV cache + attention +
        output proj + cross-attention + MLP) into a single optimised graph.
        """
        from cosmos_predict2._src.predict2.interactive.networks.utils import make_network_kv_cache

        # 1. Strip activation-checkpointing wrappers.
        self._unwrap_activation_checkpointing()

        # 2. Replace ALL TransformerEngine ops with pure-PyTorch equivalents.
        #    This eliminates ~170 graph breaks, leaving only the KV cache
        #    list management (~28 breaks, 1 per block).
        self._replace_te_ops_for_compile()

        # 3. Pre-install stateless KV cache wrappers with TENSOR state before
        #    compilation.  Tensor-based KV caches use pure tensor ops (in-place
        #    index writes + slicing) instead of Python lists, making the entire
        #    block forward fully traceable by torch.compile with ZERO graph
        #    breaks per block.
        cache_frame_size = self.config.model.config.cache_frame_size
        make_network_kv_cache(self.model.net, max_cache_size=cache_frame_size, stateless=True, use_tensor_state=True)

        # 4. Compile each DiT block individually.  With tensor KV cache there
        #    are NO graph breaks – each block is a single compiled graph.
        #    torch.compile will specialize on (store_kv, run_with_kv, current_idx)
        #    combinations; all specializations are traced during warmup.
        logger.info("Compiling DiT blocks with torch.compile ...")
        for i in range(len(self.model.net.blocks)):
            self.model.net.blocks[i] = torch.compile(self.model.net.blocks[i])
        logger.info(f"Compiled {len(self.model.net.blocks)} DiT blocks (tensor KV cache, no graph breaks)")

        # 5. Compile VAE encode (default mode) and decode (reduce-overhead).
        #    The encoder is called every chunk (inside get_data_and_condition)
        #    and is a major data-prep bottleneck; compiling it saves ~50-100ms.
        #    Using default mode (not reduce-overhead) for the encoder avoids
        #    CUDA graph capture stalls on early chunks.
        #    The decoder uses reduce-overhead (CUDA graphs) since it has fixed
        #    shapes and no in-place mutations.
        logger.info("Compiling VAE encode (default) + decode (reduce-overhead) ...")
        self.model.tokenizer.encode = torch.compile(self.model.tokenizer.encode)
        self._compiled_decode = torch.compile(self.model.decode, mode="reduce-overhead")

        logger.info("torch.compile setup complete (graphs will be traced lazily on first call)")

    def _replace_te_ops_for_compile(self) -> None:
        """Replace TransformerEngine ops with pure-PyTorch equivalents for ``torch.compile``.

        TE uses C/C++ extensions (PyCapsule functions) that ``torch._dynamo``
        cannot trace, causing the compiler to hang or create graph breaks.  We
        replace **all** TE ops in the DiT forward path:

        * ``te.pytorch.RMSNorm`` → pure-PyTorch ``_PyTorchRMSNorm`` that
          preserves exact dtype semantics (float32-in → float32-out).
          Weights are shared (not copied).
        * ``apply_rotary_pos_emb`` → monkey-patched to always use
          ``fused=False``, falling back to a PyTorch implementation.
        * ``TE DotProductAttention`` → ``_SDPAttention`` wrapping PyTorch's
          ``F.scaled_dot_product_attention``.  Applied to **all** attention
          ops (both self- and cross-attention).  Self-attention ops will
          subsequently be wrapped by ``AttentionOpWithKVCache`` (which has
          ``@torch.compiler.disable`` for the KV cache list management);
          cross-attention ops become fully traceable.

        After this, and combined with the tensor-based KV cache (step 3),
        there are **zero** graph breaks per block (compared to ~170+ before
        any replacements, or 28 with list-based KV cache).
        """
        import transformer_engine as te
        from cosmos_predict2._src.predict2.networks.minimal_v4_dit import (
            Attention as DITAttention,
            torch_attention_op,
        )

        net = self.model.net

        # ------------------------------------------------------------------
        # Helper: pure-PyTorch RMSNorm that torch.compile can trace.
        # Computes normalisation in float32 for stability, then casts back
        # to the *input* dtype so that attention q_norm/k_norm produce the
        # same dtype as v (required by F.scaled_dot_product_attention).
        # For the ``t_embedding_norm`` path (where ``use_wan_fp32_strategy``
        # requires float32 output), an explicit ``.float()`` cast is applied
        # in ``forward_seq`` after the norm — see ``dit_action_causal.py``
        # and ``dit_causal.py``.
        # ------------------------------------------------------------------
        class _PyTorchRMSNorm(torch.nn.Module):
            def __init__(self, dim: int, eps: float = 1e-6):
                super().__init__()
                self.eps = eps
                self.weight = torch.nn.Parameter(torch.ones(dim))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                in_dtype = x.dtype
                x_f32 = x.float()
                out = x_f32 * torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + self.eps)
                return (out * self.weight.float()).to(in_dtype)

            def reset_parameters(self) -> None:
                torch.nn.init.ones_(self.weight)

        # ------------------------------------------------------------------
        # Helper: SDPA-based attention that torch.compile can trace.
        # Matches the TE DotProductAttention interface: [B, S, H, D] → [B, S, H*D].
        # ------------------------------------------------------------------
        class _SDPAttention(torch.nn.Module):
            """Drop-in replacement for TE DotProductAttention using PyTorch SDPA."""

            def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs) -> torch.Tensor:
                return torch_attention_op(q, k, v)

            def set_context_parallel_group(self, *args, **kwargs) -> None:
                pass  # Not applicable for single-GPU SDPA

        # --- 1. Replace te.pytorch.RMSNorm → _PyTorchRMSNorm (share weights) ---
        rmsnorm_count = 0
        for name, module in list(net.named_modules()):
            if isinstance(module, te.pytorch.RMSNorm):
                dim = module.weight.shape[0]
                eps = module.eps if hasattr(module, "eps") else 1e-5
                new_norm = _PyTorchRMSNorm(dim, eps=eps)
                new_norm.weight = module.weight  # share trained parameter
                parts = name.split(".")
                parent = net
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], new_norm)
                rmsnorm_count += 1
        if rmsnorm_count:
            logger.info(f"Replaced {rmsnorm_count} TE RMSNorm → PyTorch RMSNorm")

        # --- 2. Monkey-patch apply_rotary_pos_emb to use fused=False ---
        import cosmos_predict2._src.predict2.networks.minimal_v4_dit as _dit_mod

        _original_rope = _dit_mod.apply_rotary_pos_emb

        def _compile_safe_rope(*args, **kwargs):
            kwargs["fused"] = False
            return _original_rope(*args, **kwargs)

        _dit_mod.apply_rotary_pos_emb = _compile_safe_rope
        logger.info("Patched apply_rotary_pos_emb to use fused=False for compile")

        # --- 3. Replace ALL TE DotProductAttention → _SDPAttention ---
        # Both self-attention and cross-attention ops are replaced.
        # Self-attention ops are later wrapped by AttentionOpWithKVCache
        # (step 3 in _setup_torch_compile), which adds @torch.compiler.disable
        # around the KV cache list management.  Cross-attention ops become
        # fully traceable, eliminating their graph breaks entirely.
        try:
            from transformer_engine.pytorch.attention import DotProductAttention as TEDotProductAttention
        except ImportError:
            TEDotProductAttention = None  # type: ignore[misc,assignment]

        te_attn_replaced = 0
        if TEDotProductAttention is not None:
            for module in net.modules():
                if isinstance(module, DITAttention):
                    if isinstance(module.attn_op, TEDotProductAttention):
                        module.attn_op = _SDPAttention()
                        te_attn_replaced += 1
        if te_attn_replaced:
            logger.info(f"Replaced {te_attn_replaced} TE DotProductAttention → PyTorch SDPA")
        else:
            logger.info("No TE DotProductAttention found (attention already uses PyTorch SDPA)")

    def _unwrap_activation_checkpointing(self) -> None:
        """Remove activation-checkpointing wrappers from DiT blocks and final_layer.

        During training, blocks may be wrapped with ``ptd_checkpoint_wrapper``
        which adds a ``CheckpointWrapper`` around the module.  This is
        unnecessary for inference and introduces graph breaks under
        ``torch.compile``.
        """
        net = self.model.net
        unwrapped = 0

        # Unwrap blocks
        if hasattr(net, "blocks"):
            for block_id, block in list(net.blocks.named_children()):
                if hasattr(block, "_checkpoint_wrapped_module"):
                    net.blocks.register_module(block_id, block._checkpoint_wrapped_module)
                    unwrapped += 1

        # Unwrap final_layer
        if hasattr(net, "final_layer") and hasattr(net.final_layer, "_checkpoint_wrapped_module"):
            net.final_layer = net.final_layer._checkpoint_wrapped_module
            unwrapped += 1

        if unwrapped > 0:
            logger.info(f"Unwrapped {unwrapped} activation-checkpointing wrappers for inference")

    def _decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents using compiled or eager VAE."""
        if self._compiled_decode is not None:
            return self._compiled_decode(latents)
        return self.model.decode(latents)

    def _prepare_data_batch(
        self,
        video_b_c_t_h_w: torch.Tensor,
        actions_np: np.ndarray,
        fps: float,
        num_latent_conditional_frames: int = 1,
    ) -> Dict[str, Any]:
        _, _, _, H, W = video_b_c_t_h_w.shape
        data_batch: Dict[str, Any] = {
            "dataset_name": "video_data",
            "video": video_b_c_t_h_w,
            "action": torch.from_numpy(actions_np).float(),  # [B, T-1, A]
            "fps": torch.tensor([fps], dtype=torch.float32),
            "padding_mask": torch.zeros(1, 1, H, W, dtype=torch.float32),
            NUM_CONDITIONAL_FRAMES_KEY: num_latent_conditional_frames,
        }

        # Move tensors to GPU and convert to bfloat16 if floating point
        for k, v in list(data_batch.items()):
            if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                data_batch[k] = v.cuda().to(dtype=torch.bfloat16)
            elif isinstance(v, torch.Tensor):
                data_batch[k] = v.cuda()
        return data_batch

    @torch.inference_mode()
    def generate_action_streaming(
        self,
        video_path: str | np.ndarray,
        actions_np: np.ndarray,
        resolution_hw: Tuple[int, int] | None,
        num_steps: int,
        seed: int,
        start_frame_idx: int,
        max_frames: int,
        # cache_frame_size: int,
    ) -> torch.Tensor:
        # Load input video and extract conditioning frames
        if isinstance(video_path, str):
            video_array = mediapy.read_video(video_path)  # [T, H, W, C]
        else:
            video_array = video_path  # Assume already loaded numpy array
        if video_array.ndim == 4:
            video_array = video_array[None, ...]  # [B, T, H, W, C]
        if actions_np.ndim == 2:
            actions_np = actions_np[None, ...]  # [B, T, A]

        # Determine target resolution
        Ht, Wt = video_array.shape[-3], video_array.shape[-2]
        if resolution_hw is not None:
            Ht, Wt = resolution_hw

        # Prepare first conditioning frame
        frame = video_array[:, start_frame_idx]  # [B, H, W, C]
        if (frame.shape[1], frame.shape[2]) != (Ht, Wt):
            frame = np.stack([
                mediapy.resize_image(i, (Ht, Wt)) for i in frame
            ])

        frame_uint8 = np.clip(np.round(frame), 0, 255).astype(np.uint8)
        frame_tensor = torch.from_numpy(frame_uint8).permute(0, 3, 1, 2).contiguous()

        cond_frames = frame_tensor.unsqueeze(2)
        # Collect decoded video chunks in a list; concatenate once at the end
        # to avoid O(n²) reallocation from repeated torch.cat in the loop.
        _video_chunks: list[torch.Tensor] = [frame_tensor.float().unsqueeze(2) / 128 - 1]

        cache_frame_size = self.config.model.config.cache_frame_size
        chunk_size = cache_frame_size * 4
        num_chunks = (actions_np.shape[-2] - chunk_size) // 4 + 1

        # Always use stateless KV cache – required for torch.compile
        # compatibility, and also works correctly in eager mode.
        use_stateless_kv = self.torch_compile

        # Pre-cache T5 embeddings on GPU (avoids CPU→GPU copy every chunk)
        _t5_gpu = self.t5_text_embeddings_cpu.to(
            device=self.model.tensor_kwargs["device"], dtype=torch.bfloat16
        )
        _t5_mask_gpu = torch.ones(
            (_t5_gpu.shape[0], _t5_gpu.shape[1]),
            device=self.model.tensor_kwargs["device"],
            dtype=torch.bfloat16,
        )

        warmup_done = False
        _chunk_timings: list[tuple[float, float, float]] = []  # (prep, gen, dec) per chunk
        _new_frames_per_chunk: list[int] = []
        _t_chunk_end = time.time()
        start_time = 0.0
        for chunk_idx in range(num_chunks):

            # Prepare video input
            first_stack = cond_frames.permute(0, 2, 1, 3, 4)  # [B, T, 3, H, W]
            zeros_tail = torch.zeros_like(first_stack[:, :1]).repeat(1, chunk_size - first_stack.shape[1] + 1, 1, 1, 1)
            vid_chw_t = torch.cat([first_stack, zeros_tail], dim=1)  # [B, T, 3, H, W]
            video_b_c_t_h_w = vid_chw_t.permute(0, 2, 1, 3, 4).contiguous()  # [B, 3, T, H, W]

            # Prepare data batch
            chunk_actions = actions_np[:, chunk_idx * 4 : chunk_idx * 4 + chunk_size, :]
            data_batch = self._prepare_data_batch(
                video_b_c_t_h_w=video_b_c_t_h_w,
                actions_np=chunk_actions,
                fps=4,
                num_latent_conditional_frames=0,
            )

            # Normalize and augment dims to match training pipeline
            self.model._normalize_video_databatch_inplace(data_batch)
            self.model._augment_image_dim_inplace(data_batch)

            # Inject pre-cached GPU T5 embeddings and mask
            data_batch["t5_text_embeddings"] = _t5_gpu
            data_batch["t5_text_mask"] = _t5_mask_gpu

            # Final safety: ensure all floating inputs are bf16 before conditioning
            for k, v in list(data_batch.items()):
                if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                    data_batch[k] = v.to(dtype=self.model.tensor_kwargs["dtype"])  # typically bf16

            # Use model helper to construct condition on this batch
            _, x0, condition, _ = self.model.get_data_and_condition(data_batch)

            # Ensure latent frames used for conditioning match model precision (bf16)
            x0 = x0.to(dtype=self.model.tensor_kwargs["dtype"])
            condition = condition.edit_data_type(DataType.VIDEO)
            condition = condition.set_video_condition(
                gt_frames=x0,
                random_min_num_conditional_frames=None,
                random_max_num_conditional_frames=None,
                num_conditional_frames=data_batch[NUM_CONDITIONAL_FRAMES_KEY],
            )

            # Init noise with correct latent shape
            _T, _H, _W = data_batch[self.model.input_data_key].shape[-3:]
            state_shape = (
                self.model.config.state_ch,
                int(self.model.tokenizer.get_latent_num_frames(_T)),
                _H // self.model.tokenizer.spatial_compression_factor,
                _W // self.model.tokenizer.spatial_compression_factor,
            )
            noise = misc.arch_invariant_rand(
                (1, *state_shape),
                torch.float32,
                self.model.tensor_kwargs["device"],
                seed + chunk_idx,
            )

            # Clamp steps to the configured student schedule length
            if hasattr(self.model, "config") and hasattr(self.model.config, "selected_sampling_time"):
                K = len(self.model.config.selected_sampling_time)
            else:
                raise AttributeError("Model does not define 'config.selected_sampling_time' to determine steps")
            n_steps = max(1, min(int(num_steps), K))

            start_idx = (cond_frames.shape[2]-1)//4+1

            # ---- torch.compile warmup pass (first chunk only) ----
            # Run the full pipeline (DiT generate → VAE decode) TWICE so that
            # all compiled paths and CUDA graph trees (reduce-overhead) are
            # fully traced and stabilised before we start timing.
            #   Pass 1: triggers compilation / CUDA graph capture.
            #   Pass 2: ensures the CUDA graph tree is stable (reduce-overhead
            #           may need 2+ calls to lock in the graph).
            if self.torch_compile and chunk_idx == 0 and not warmup_done:
                logger.info("torch.compile warmup: tracing all graphs (this will be slow) ...")
                warmup_t0 = time.time()
                for _warmup_pass in range(2):
                    _warmup_latents = self.model.generate_streaming_video(
                        condition, noise, n_steps=n_steps, cache_frame_size=cache_frame_size,
                        start_idx=start_idx, stateless_kv=use_stateless_kv,
                    )
                    _ = self._decode(_warmup_latents)
                torch.cuda.synchronize()
                warmup_t1 = time.time()
                logger.info(f"torch.compile warmup complete in {warmup_t1 - warmup_t0:.1f}s")
                warmup_done = True

            if chunk_idx == 0:
                start_time = time.time()
                _t_chunk_end = start_time

            # --- Run streaming generation in latent space and decode ---
            torch.cuda.synchronize()
            _t_gen0 = time.time()
            latents = self.model.generate_streaming_video(
                condition, noise, n_steps=n_steps, cache_frame_size=cache_frame_size,
                start_idx=start_idx, stateless_kv=use_stateless_kv,
            )  # type: ignore[arg-type]
            torch.cuda.synchronize()
            _t_gen1 = time.time()
            video = self._decode(latents)
            video = video.clip(min=-1, max=1)
            torch.cuda.synchronize()
            _t_dec1 = time.time()

            _prep = _t_gen0 - _t_chunk_end
            _gen = _t_gen1 - _t_gen0
            _dec = _t_dec1 - _t_gen1
            _chunk_timings.append((_prep, _gen, _dec))
            _t_chunk_end = _t_dec1

            new_frames = video[:, :, ((start_idx-1)*4+1):].cpu()
            _nf = new_frames.shape[2]
            _video_chunks.append(new_frames)
            _new_frames_per_chunk.append(_nf)
            logger.info(
                f"chunk {chunk_idx}: prep={_prep*1000:.0f}ms  gen={_gen*1000:.0f}ms  "
                f"dec={_dec*1000:.0f}ms  total={(_prep+_gen+_dec)*1000:.0f}ms  "
                f"→ {_nf} new frames"
            )

            cond_frames = ((1.0 + video[:, :, -((cache_frame_size-1)*4+1):]) / 2 * 255.0).to(torch.uint8)

        end_time = time.time()
        total_time = end_time - start_time
        final_video = torch.cat(_video_chunks, dim=2)
        total_generated = final_video.shape[2] - 1  # exclude the initial conditioning frame

        # ---- Timing summary ----
        # Chunk 0 produces more frames (includes the initial context);
        # chunks 1+ each produce a fixed number of new frames (steady state).
        ss_timings = _chunk_timings[1:]  # steady-state = chunks 1+
        ss_nf = _new_frames_per_chunk[1:] if len(_new_frames_per_chunk) > 1 else _new_frames_per_chunk
        new_frames_ss = ss_nf[0] if ss_nf else _new_frames_per_chunk[0]

        avg_prep = sum(t[0] for t in ss_timings) / max(len(ss_timings), 1) * 1000
        avg_gen  = sum(t[1] for t in ss_timings) / max(len(ss_timings), 1) * 1000
        avg_dec  = sum(t[2] for t in ss_timings) / max(len(ss_timings), 1) * 1000
        avg_chunk = avg_prep + avg_gen + avg_dec

        chunk0_total = sum(_chunk_timings[0]) * 1000 if _chunk_timings else 0
        chunk0_frames = _new_frames_per_chunk[0] if _new_frames_per_chunk else 0

        logger.info("=" * 60)
        logger.info("TIMING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total video: {total_generated} generated frames + 1 conditioning = {final_video.shape[2]} frames")
        logger.info(f"Chunks: {num_chunks} (chunk 0 → {chunk0_frames} frames, chunks 1+ → {new_frames_ss} frames each)")
        logger.info(f"")
        logger.info(f"Video generation time (end-to-end): {total_time:.3f}s")
        logger.info(f"Overall FPS: {total_generated / total_time:.2f}")
        logger.info(f"")
        logger.info(f"Chunk 0:  {chunk0_total:.0f}ms → {chunk0_frames} frames")
        logger.info(f"Steady-state per chunk (avg of chunks 1–{num_chunks - 1}):")
        logger.info(f"  Data prep + VAE encode:  {avg_prep:.0f}ms")
        logger.info(f"  DiT generation:          {avg_gen:.0f}ms")
        logger.info(f"  VAE decode:              {avg_dec:.0f}ms")
        logger.info(f"  Total per chunk:         {avg_chunk:.0f}ms → {new_frames_ss} new frames")
        logger.info(f"")
        playback_budget_ms = new_frames_ss / 10.0 * 1000  # budget at 10 FPS
        logger.info(f"Real-time check (10 FPS playback, {new_frames_ss} frames = {playback_budget_ms:.0f}ms budget):")
        if avg_chunk <= playback_budget_ms:
            logger.info(f"  ✓ {avg_chunk:.0f}ms < {playback_budget_ms:.0f}ms — runs in REAL-TIME ({avg_chunk/playback_budget_ms:.2f}x)")
        else:
            logger.info(f"  ✗ {avg_chunk:.0f}ms > {playback_budget_ms:.0f}ms — {avg_chunk/playback_budget_ms:.2f}x real-time")
        logger.info("=" * 60)

        return final_video

    def cleanup(self) -> None:
        if self.context_parallel_size > 1:
            import torch.distributed as dist

            if parallel_state.is_initialized():
                parallel_state.destroy_model_parallel()
            dist.destroy_process_group()


def _process_entries(
    entries: List[Dict[str, Any]], args: argparse.Namespace, infer: ActionStreamingInference, rank0: bool
):
    if not isinstance(entries, list):
        raise ValueError("input_json must contain a list of entries")

    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    dp_size = world_size // args.context_parallel_size
    dp_rank = rank // args.context_parallel_size

    # Distribute work using strided slicing for even load balancing
    my_entries = entries[dp_rank::dp_size]

    import math
    total_len = len(entries)
    max_per_rank = math.ceil(total_len / dp_size)
    padded_cnt = max_per_rank - len(my_entries)

    if padded_cnt > 0:
        # Pad with the last actual assignment (or a safe default if empty)
        pad_entry = my_entries[-1].copy() if my_entries else entries[0].copy()
        pad_entry["is_padding"] = True
        my_entries.extend([pad_entry] * padded_cnt)

    entries = my_entries
    logger.info(f"Rank {rank} (DP {dp_rank}/{dp_size}) processing {len(entries)} entries (padding: {padded_cnt})")

    for entry in entries:
        input_video = entry.get("input_video")
        input_action_path = entry.get("input_action")
        output_video_path = entry.get("output_video")

        is_padding = entry.get("is_padding", False)

        if os.path.exists(output_video_path) and not is_padding:
            logger.info(f"Skipping {output_video_path} as it already exists")
            continue

        if input_video is None or input_action_path is None or output_video_path is None:
            logger.warning(
                "Entry missing one of required keys: 'input_video', 'input_action', 'output_video'; skipping"
            )
            continue

        # Ensure output directory exists
        out_dir = os.path.dirname(output_video_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        # Load actions from a .npy file and use directly
        actions = np.load(input_action_path)

        # Optional per-entry overrides
        if "resolution" in entry and isinstance(entry["resolution"], list) and len(entry["resolution"]) == 2:
            res_hw = (int(entry["resolution"][0]), int(entry["resolution"][1]))
        elif args.resolution != "none":
            try:
                h, w = map(int, args.resolution.split(","))
                res_hw = (h, w)
            except Exception:
                res_hw = None
        else:
            res_hw = None

        # Per-entry override for start_frame_idx if provided
        if "start_frame_idx" in entry:
            try:
                start_frame_idx = int(entry["start_frame_idx"])  # type: ignore[arg-type]
            except Exception:
                start_frame_idx = args.start_frame_idx
        else:
            start_frame_idx = args.start_frame_idx

        video = infer.generate_action_streaming(
            video_path=input_video,
            actions_np=actions,
            resolution_hw=res_hw,
            num_steps=args.num_steps,
            seed=args.seed,
            start_frame_idx=start_frame_idx,
            max_frames=args.max_frames,
            # cache_frame_size=args.cache_frame_size,
        )

        if not is_padding:
            save_fp_wo_ext = output_video_path[:-4] if output_video_path.endswith(".mp4") else output_video_path
            save_img_or_video((1.0 + video[0]) / 2, save_fp_wo_ext, fps=args.fps)
            logger.info(f"Saved video to {output_video_path}")

            # Save GT vs Pred side-by-side comparison
            try:
                import cv2
                gt_video_array = mediapy.read_video(input_video)  # (T, H, W, C) uint8
                pred_video = ((1.0 + video[0]) / 2).clamp(0, 1).cpu()  # (C, T, H, W)
                pred_video = (pred_video.permute(1, 2, 3, 0).numpy() * 255).astype(np.uint8)  # (T, H, W, C)
                min_t = min(len(gt_video_array), len(pred_video))
                h, w = pred_video.shape[1], pred_video.shape[2]
                cmp_path = save_fp_wo_ext + "_compare.mp4"
                writer = cv2.VideoWriter(cmp_path, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w * 2, h))
                for t in range(min_t):
                    gt_f = cv2.resize(gt_video_array[t], (w, h)) if gt_video_array[t].shape[:2] != (h, w) else gt_video_array[t]
                    gt_f = cv2.putText(gt_f.copy(), "GT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                    pred_f = cv2.putText(pred_video[t].copy(), "Pred", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                    writer.write(cv2.cvtColor(np.concatenate([gt_f, pred_f], axis=1), cv2.COLOR_RGB2BGR))
                writer.release()
                logger.info(f"Saved comparison to {cmp_path}")
            except Exception as e:
                logger.warning(f"Failed to save comparison: {e}")


def main() -> None:
    args = parse_arguments()

    # The KV cache's current_idx is a Python int that changes each chunk,
    # causing torch.compile guard specialisation and recompilations.
    # Raise the limit so all unique idx values get compiled without warnings.
    torch._dynamo.config.recompile_limit = 64  # type: ignore[attr-defined]

    if "RANK" in os.environ:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank)
        logger.info(f"Initialized distributed: rank {rank}/{world_size}")
    else:
        rank = 0
        world_size = 1
        logger.info("Running in single-GPU mode")

    infer = ActionStreamingInference(
        config_path=args.config,
        experiment_name=args.experiment,
        ckpt_path=args.ckpt_path,
        s3_credential_path=args.s3_cred,
        cr1_embeddings_path=args.cr1_embeddings_path,
        context_parallel_size=args.context_parallel_size,
        enable_fsdp=args.enable_fsdp,
        torch_compile=args.torch_compile,
    )

    mem_bytes = torch.cuda.memory_allocated(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"GPU memory usage after model load: {mem_bytes / (1024**3):.2f} GB")

    rank0 = True
    if args.context_parallel_size > 1:
        rank0 = distributed.get_rank() == 0

    with open(args.input_json, "r") as f:
        entries = json.load(f)

    with torch.inference_mode():
        _process_entries(entries, args, infer, rank0)
    if args.context_parallel_size > 1:
        torch.distributed.barrier()

    infer.cleanup()


if __name__ == "__main__":
    main()
