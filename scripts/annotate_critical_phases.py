#!/usr/bin/env python3
"""
Auto-annotate critical phases in robot manipulation episodes using Gemini 3.1 Flash Lite.

For each episode, this script:
1. Samples frames from 3 camera views and sends them to Gemini
2. Identifies critical phase time ranges (start_frame, end_frame)
3. Labels outcome: success(1), easy_recovery(0.5), reset_then_success(-0.5), failure(-1)
4. Extracts action chunks + video clips around critical phases

Usage:
    python scripts/annotate_critical_phases.py \
        --dataset_path datasets/fold_towel_0109_agilex \
        --output_dir outputs/critical_phase_annotations \
        --gemini_api_key YOUR_API_KEY \
        [--sample_interval 15] [--context_window 30] [--num_workers 4]
"""

import argparse
import base64
import io
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def init_gemini(api_key: str, model_name: str = "gemini-3.1-flash-lite-preview"):
    """Initialize the Gemini client."""
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        return client, model_name
    except ImportError:
        raise ImportError(
            "google-genai is required. Install with: pip install google-genai"
        )


def encode_frames_to_images(frames: np.ndarray, max_size: int = 512) -> list:
    """Encode numpy frames to PIL Images, resized for API efficiency."""
    images = []
    for frame in frames:
        img = Image.fromarray(frame)
        # Resize to save tokens
        h, w = img.size
        scale = min(max_size / max(h, w), 1.0)
        if scale < 1.0:
            img = img.resize((int(h * scale), int(w * scale)), Image.LANCZOS)
        images.append(img)
    return images


def build_annotation_prompt(task_description: str, num_frames: int, fps: int, sample_interval: int) -> str:
    """Build the prompt for Gemini to annotate critical phases."""
    return f"""You are analyzing a robot manipulation video for the task: "{task_description}".

The video is recorded at {fps} FPS. You are shown {num_frames} sampled frames at every {sample_interval} frames (i.e., one frame every {sample_interval / fps:.1f} seconds). The frame indices are labeled on each image.

Please analyze the episode and provide:

1. **Critical Phase**: Identify the most critical phase of the manipulation — the moment where success or failure is determined. This could be a grasping moment, a folding action, a placement, etc. Provide the approximate start and end frame indices (in the original video, not the sampled indices).

2. **Outcome**: Classify the overall episode outcome:
   - "success": The task is completed successfully without major issues.
   - "easy_recovery": The robot encounters a minor issue but recovers smoothly and completes the task.
   - "reset_then_success": The robot fails or nearly fails, resets/retries, and eventually succeeds.
   - "failure": The robot fails to complete the task.

3. **Reasoning**: Brief explanation of your assessment.

Respond ONLY in the following JSON format:
```json
{{
    "critical_phase_start_frame": <int>,
    "critical_phase_end_frame": <int>,
    "outcome": "<success|easy_recovery|reset_then_success|failure>",
    "reasoning": "<string>"
}}
```"""


def annotate_episode_with_gemini(
    client,
    model_name: str,
    video_paths: dict[str, str],
    task_description: str,
    fps: int,
    total_frames: int,
    sample_interval: int = 15,
    max_retries: int = 3,
) -> dict:
    """Send sampled frames from all views to Gemini and get annotation."""
    from google import genai
    from google.genai import types

    # Sample frame indices
    sample_indices = list(range(0, total_frames, sample_interval))
    if len(sample_indices) > 60:
        # Cap at 60 frames to stay within token limits
        step = len(sample_indices) // 60
        sample_indices = sample_indices[::step]

    # Build interleaved multi-view grid images
    # For each sampled time step, create a side-by-side image of all 3 views
    content_parts = []
    view_names = sorted(video_paths.keys())

    # Load videos lazily
    caps = {}
    for view_name, vpath in video_paths.items():
        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {vpath}")
        caps[view_name] = cap

    grid_images = []
    for idx in sample_indices:
        row_frames = []
        for view_name in view_names:
            cap = caps[view_name]
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                row_frames.append(np.zeros((120, 160, 3), dtype=np.uint8))
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize each view for API efficiency
                frame = cv2.resize(frame, (160, 120))
                row_frames.append(frame)
        # Concatenate views horizontally
        grid = np.concatenate(row_frames, axis=1)  # (120, 480, 3)
        # Add frame index label
        grid_pil = Image.fromarray(grid)
        grid_images.append((idx, grid_pil))

    for cap in caps.values():
        cap.release()

    # Build content: send a batch of grid images with frame labels
    prompt = build_annotation_prompt(task_description, len(sample_indices), fps, sample_interval)
    content_parts.append(prompt)

    # Send images in batches to avoid token overflow
    # Select up to 20 evenly spaced grid images
    if len(grid_images) > 20:
        step = len(grid_images) // 20
        grid_images = grid_images[::step][:20]

    for idx, img in grid_images:
        content_parts.append(f"\n[Frame {idx}] (views: {', '.join(view_names)})")
        content_parts.append(img)

    # Call Gemini API with retries
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=content_parts,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=1024,
                ),
            )
            text = response.text.strip()
            # Extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                # Validate fields
                assert "critical_phase_start_frame" in result
                assert "critical_phase_end_frame" in result
                assert result["outcome"] in ("success", "easy_recovery", "reset_then_success", "failure")
                return result
            else:
                raise ValueError(f"No JSON found in Gemini response: {text}")
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"Gemini annotation failed after {max_retries} retries: {e}")


OUTCOME_TO_VALUE = {
    "success": 1.0,
    "easy_recovery": 0.5,
    "reset_then_success": -0.5,
    "failure": -1.0,
}


def extract_critical_chunk(
    dataset_path: Path,
    episode_index: int,
    annotation: dict,
    context_window: int = 30,
    fps: int = 30,
) -> dict:
    """Extract action chunk and video clip paths around the critical phase.

    Args:
        dataset_path: Root path to the dataset.
        episode_index: Episode index.
        annotation: Gemini annotation dict with critical_phase_start/end_frame.
        context_window: Number of extra frames before/after the critical phase.
        fps: Video FPS.

    Returns:
        Dict with extracted data paths and metadata.
    """
    # Load parquet
    parquet_path = dataset_path / f"data/chunk-000/episode_{episode_index:06d}.parquet"
    df = pd.read_parquet(parquet_path)

    start = max(0, annotation["critical_phase_start_frame"] - context_window)
    end = min(len(df), annotation["critical_phase_end_frame"] + context_window)

    # Extract action chunk
    action_chunk = np.vstack([np.asarray(x, dtype=np.float32) for x in df["action"].iloc[start:end]])
    state_chunk = np.vstack([np.asarray(x, dtype=np.float32) for x in df["observation.state"].iloc[start:end]])

    return {
        "episode_index": episode_index,
        "start_frame": int(start),
        "end_frame": int(end),
        "critical_start": annotation["critical_phase_start_frame"],
        "critical_end": annotation["critical_phase_end_frame"],
        "outcome": annotation["outcome"],
        "value": OUTCOME_TO_VALUE[annotation["outcome"]],
        "reasoning": annotation.get("reasoning", ""),
        "action_chunk": action_chunk,
        "state_chunk": state_chunk,
    }


def extract_video_clip(
    dataset_path: Path,
    episode_index: int,
    start_frame: int,
    end_frame: int,
    output_dir: Path,
):
    """Extract video clips for all 3 views around the critical phase."""
    view_names = ["observation.images.cam_high",
                  "observation.images.cam_left_wrist",
                  "observation.images.cam_right_wrist"]
    clip_paths = {}

    for view in view_names:
        video_path = dataset_path / f"videos/chunk-000/{view}/episode_{episode_index:06d}.mp4"
        if not video_path.exists():
            print(f"Warning: video not found: {video_path}")
            continue

        short_view = view.split(".")[-1]  # cam_high, cam_left_wrist, cam_right_wrist
        out_path = output_dir / f"episode_{episode_index:06d}_{short_view}_critical.mp4"

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for fi in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)

        writer.release()
        cap.release()
        clip_paths[short_view] = str(out_path)

    return clip_paths


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def load_dataset_metadata(dataset_path: Path) -> tuple[list[dict], str, int]:
    """Load episodes metadata, task description, and FPS."""
    episodes = []
    with open(dataset_path / "meta/episodes.jsonl") as f:
        for line in f:
            episodes.append(json.loads(line))

    tasks = []
    with open(dataset_path / "meta/tasks.jsonl") as f:
        for line in f:
            tasks.append(json.loads(line))
    task_description = tasks[0]["task"] if tasks else "Unknown task"

    with open(dataset_path / "meta/info.json") as f:
        info = json.load(f)
    fps = info.get("fps", 30)

    return episodes, task_description, fps


def get_video_paths(dataset_path: Path, episode_index: int) -> dict[str, str]:
    """Get paths to all 3 camera view videos for an episode."""
    views = {
        "cam_high": f"videos/chunk-000/observation.images.cam_high/episode_{episode_index:06d}.mp4",
        "cam_left_wrist": f"videos/chunk-000/observation.images.cam_left_wrist/episode_{episode_index:06d}.mp4",
        "cam_right_wrist": f"videos/chunk-000/observation.images.cam_right_wrist/episode_{episode_index:06d}.mp4",
    }
    return {k: str(dataset_path / v) for k, v in views.items()}


def process_episode(
    client,
    model_name: str,
    dataset_path: Path,
    episode: dict,
    task_description: str,
    fps: int,
    output_dir: Path,
    sample_interval: int = 15,
    context_window: int = 30,
) -> dict:
    """Full pipeline for a single episode: annotate -> extract."""
    ep_idx = episode["episode_index"]
    total_frames = episode["length"]

    video_paths = get_video_paths(dataset_path, ep_idx)

    # Step 1: Gemini annotation
    annotation = annotate_episode_with_gemini(
        client=client,
        model_name=model_name,
        video_paths=video_paths,
        task_description=task_description,
        fps=fps,
        total_frames=total_frames,
        sample_interval=sample_interval,
    )

    # Step 2: Extract action/state chunk
    chunk_data = extract_critical_chunk(
        dataset_path=dataset_path,
        episode_index=ep_idx,
        annotation=annotation,
        context_window=context_window,
        fps=fps,
    )

    # Step 3: Extract video clips
    clips_dir = output_dir / "video_clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    clip_paths = extract_video_clip(
        dataset_path=dataset_path,
        episode_index=ep_idx,
        start_frame=chunk_data["start_frame"],
        end_frame=chunk_data["end_frame"],
        output_dir=clips_dir,
    )

    # Step 4: Save action/state chunk as npz
    chunks_dir = output_dir / "action_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    npz_path = chunks_dir / f"episode_{ep_idx:06d}.npz"
    np.savez_compressed(
        npz_path,
        action=chunk_data["action_chunk"],
        state=chunk_data["state_chunk"],
    )

    result = {
        "episode_index": ep_idx,
        "total_frames": total_frames,
        "critical_start": chunk_data["critical_start"],
        "critical_end": chunk_data["critical_end"],
        "extracted_start": chunk_data["start_frame"],
        "extracted_end": chunk_data["end_frame"],
        "outcome": chunk_data["outcome"],
        "value": chunk_data["value"],
        "reasoning": chunk_data["reasoning"],
        "video_clips": clip_paths,
        "action_chunk_path": str(npz_path),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Annotate critical phases using Gemini")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the LeRobot dataset (e.g., datasets/fold_towel_0109_agilex)")
    parser.add_argument("--output_dir", type=str, default="outputs/critical_phase_annotations",
                        help="Directory to save annotations and extracted data")
    parser.add_argument("--gemini_api_key", type=str, default=None,
                        help="Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--gemini_model", type=str, default="gemini-3.1-flash-lite-preview",
                        help="Gemini model name")
    parser.add_argument("--sample_interval", type=int, default=15,
                        help="Frame sampling interval for Gemini (default: every 15 frames)")
    parser.add_argument("--context_window", type=int, default=30,
                        help="Extra frames before/after critical phase to extract")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of parallel workers for Gemini API calls")
    parser.add_argument("--episodes", type=str, default=None,
                        help="Comma-separated episode indices to process (default: all)")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    api_key = args.gemini_api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Gemini API key required. Pass --gemini_api_key or set GEMINI_API_KEY env var.")

    client, model_name = init_gemini(api_key, args.gemini_model)
    episodes, task_description, fps = load_dataset_metadata(dataset_path)

    # Filter episodes if specified
    if args.episodes:
        ep_indices = set(int(x) for x in args.episodes.split(","))
        episodes = [e for e in episodes if e["episode_index"] in ep_indices]

    print(f"Dataset: {dataset_path}")
    print(f"Task: {task_description}")
    print(f"FPS: {fps}")
    print(f"Episodes to process: {len(episodes)}")

    all_results = []

    if args.num_workers <= 1:
        for ep in tqdm(episodes, desc="Annotating episodes"):
            try:
                result = process_episode(
                    client, model_name, dataset_path, ep, task_description,
                    fps, output_dir, args.sample_interval, args.context_window,
                )
                all_results.append(result)
                print(f"  Episode {ep['episode_index']}: {result['outcome']} (value={result['value']})")
            except Exception as e:
                print(f"  Episode {ep['episode_index']}: FAILED - {e}")
                all_results.append({
                    "episode_index": ep["episode_index"],
                    "error": str(e),
                })
    else:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {}
            for ep in episodes:
                future = executor.submit(
                    process_episode,
                    client, model_name, dataset_path, ep, task_description,
                    fps, output_dir, args.sample_interval, args.context_window,
                )
                futures[future] = ep["episode_index"]

            for future in tqdm(as_completed(futures), total=len(futures), desc="Annotating"):
                ep_idx = futures[future]
                try:
                    result = future.result()
                    all_results.append(result)
                    print(f"  Episode {ep_idx}: {result['outcome']} (value={result['value']})")
                except Exception as e:
                    print(f"  Episode {ep_idx}: FAILED - {e}")
                    all_results.append({"episode_index": ep_idx, "error": str(e)})

    # Save all annotations
    annotations_path = output_dir / "annotations.json"
    with open(annotations_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAnnotations saved to {annotations_path}")

    # Print summary
    outcomes = [r["outcome"] for r in all_results if "outcome" in r]
    from collections import Counter
    print("\nOutcome distribution:")
    for outcome, count in Counter(outcomes).most_common():
        print(f"  {outcome}: {count} ({count/len(outcomes)*100:.1f}%)")


if __name__ == "__main__":
    main()
