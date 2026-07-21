"""
Render test poses for BTS Digital Twin pipeline.

Loads a trained 3DGS checkpoint and renders all test views from test_poses.csv
at the exact specified resolution, saving as PNG files for submission.

Uses Mip-Splatting (antialiased) rendering for alias-free output at test resolution.
Appearance embeddings are disabled (no per-image embedding for test views).
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


def render_test_poses(
    scene_dir: str,
    checkpoint_path: str,
    output_dir: str,
    config_path: Optional[str] = None,
    device: str = "cuda",
):
    """Render all test poses for a scene from a trained checkpoint.

    Args:
        scene_dir: Path to scene directory (contains test/test_poses.csv)
        checkpoint_path: Path to trained model checkpoint (.pt)
        output_dir: Directory to save rendered PNG images
        config_path: Optional YAML config file
        device: CUDA device
    """
    from scripts.dataset import SceneDataset, SceneNormalization
    from scripts.gaussian_model import GaussianModel
    from scripts.train import _load_config

    scene_name = Path(scene_dir).name
    logger.info(f"Rendering test poses for scene: {scene_name}")

    # Load config
    cfg = _load_config(config_path)

    # Load dataset (for normalization and test poses)
    dataset = SceneDataset(
        scene_dir=scene_dir,
        test_every=0,  # Don't need val split for rendering
        normalize=cfg.get("normalize_world_space", True),
        data_factor=1,
    )

    if dataset.num_test == 0:
        logger.error(f"No test poses found for scene {scene_name}")
        return

    logger.info(f"Found {dataset.num_test} test poses")

    # Initialize model with dummy points (will be overwritten by checkpoint)
    # We need at least 1 point for initialization
    dummy_points = dataset.points if len(dataset.points) > 0 else np.zeros((1, 3))
    dummy_colors = dataset.point_colors if len(dataset.point_colors) > 0 else np.zeros((1, 3))

    model = GaussianModel(
        points=dummy_points,
        colors=dummy_colors,
        num_train_images=dataset.num_train,
        sh_degree=cfg.get("sh_degree", 3),
        scene_scale=dataset.scene_scale,
        app_opt=False,  # Disable appearance for test rendering
        device=device,
    )

    # Load checkpoint
    step = model.load_checkpoint(checkpoint_path)
    model.set_eval()
    logger.info(f"Loaded checkpoint from step {step} with {model.num_gaussians} Gaussians")

    # Activate all SH degrees for best quality
    model.active_sh_degree = cfg.get("sh_degree", 3)

    # Create output directory
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Rendering settings
    antialiased = cfg.get("antialiased", True)
    near_plane = cfg.get("near_plane", 0.01)
    far_plane = cfg.get("far_plane", 1000.0)

    t_start = time.time()

    with torch.no_grad():
        for i, cam in enumerate(dataset.test_cameras):
            w2c = torch.from_numpy(cam.w2c).float().to(device)
            K = torch.from_numpy(cam.K).float().to(device)

            # Render at exact test resolution
            result = model.render(
                viewmat=w2c,
                K=K,
                width=cam.width,
                height=cam.height,
                cam_idx=None,  # No appearance embedding
                near_plane=near_plane,
                far_plane=far_plane,
                render_mode="RGB",
                antialiased=antialiased,
                absgrad=False,
            )

            # Convert to uint8 PNG
            rgb = result["rgb"].clamp(0, 1)  # [H, W, 3]
            img_np = (rgb.cpu().numpy() * 255).astype(np.uint8)

            # Save with exact name from test_poses.csv (e.g., DJI_xxx.JPG or frame_xxx.png)
            out_name = cam.image_name
            out_file = out_path / out_name

            suffix = out_file.suffix.lower()
            if suffix in (".jpg", ".jpeg"):
                Image.fromarray(img_np).save(out_file, quality=95, subsampling=0)
            else:
                Image.fromarray(img_np).save(out_file)

            if (i + 1) % 10 == 0 or (i + 1) == len(dataset.test_cameras):
                elapsed = time.time() - t_start
                logger.info(
                    f"[{scene_name}] Rendered {i+1}/{len(dataset.test_cameras)} "
                    f"({cam.width}x{cam.height}) | {elapsed:.1f}s"
                )

    total_time = time.time() - t_start
    n_rendered = len(dataset.test_cameras)
    logger.info(
        f"[{scene_name}] Rendered {n_rendered} test images in {total_time:.1f}s "
        f"({total_time/max(n_rendered,1):.2f}s/image) → {out_path}"
    )


def render_all_scenes(
    data_dir: str,
    results_dir: str,
    renders_dir: str,
    config_path: Optional[str] = None,
    device: str = "cuda",
):
    """Render test poses for all scenes.

    Args:
        data_dir: Directory containing all scene subdirectories
        results_dir: Directory containing per-scene training results/checkpoints
        renders_dir: Directory to save rendered images (per-scene subdirectories)
        config_path: Optional YAML config file
        device: CUDA device
    """
    data_path = Path(data_dir)
    results_path = Path(results_dir)
    renders_path = Path(renders_dir)

    # Find all scenes
    scenes = sorted([
        d.name for d in data_path.iterdir()
        if d.is_dir() and (d / "test" / "test_poses.csv").exists()
    ])

    if not scenes:
        logger.error(f"No scenes found in {data_dir}")
        return

    logger.info(f"Found {len(scenes)} scenes to render: {scenes}")

    for scene_name in scenes:
        scene_dir = str(data_path / scene_name)
        result_dir = results_path / scene_name

        # Find latest checkpoint
        ckpt_dir = result_dir / "checkpoints"
        if not ckpt_dir.exists():
            logger.error(f"No checkpoints found for {scene_name}")
            continue

        ckpts = sorted(ckpt_dir.glob("ckpt_*.pt"))
        if not ckpts:
            logger.error(f"No checkpoint files found for {scene_name}")
            continue

        checkpoint_path = str(ckpts[-1])
        output_dir = str(renders_path / scene_name)

        try:
            render_test_poses(
                scene_dir=scene_dir,
                checkpoint_path=checkpoint_path,
                output_dir=output_dir,
                config_path=config_path,
                device=device,
            )
        except Exception as e:
            logger.error(f"Failed to render {scene_name}: {e}", exc_info=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Render test poses for submission")
    subparsers = parser.add_subparsers(dest="command")

    # Single scene
    single = subparsers.add_parser("single", help="Render a single scene")
    single.add_argument("--scene-dir", required=True)
    single.add_argument("--checkpoint", required=True)
    single.add_argument("--output-dir", required=True)
    single.add_argument("--config", default=None)
    single.add_argument("--device", default="cuda")

    # All scenes
    all_cmd = subparsers.add_parser("all", help="Render all scenes")
    all_cmd.add_argument("--data-dir", required=True)
    all_cmd.add_argument("--results-dir", required=True)
    all_cmd.add_argument("--renders-dir", required=True)
    all_cmd.add_argument("--config", default=None)
    all_cmd.add_argument("--device", default="cuda")

    args = parser.parse_args()

    if args.command == "single":
        render_test_poses(
            scene_dir=args.scene_dir,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            config_path=args.config,
            device=args.device,
        )
    elif args.command == "all":
        render_all_scenes(
            data_dir=args.data_dir,
            results_dir=args.results_dir,
            renders_dir=args.renders_dir,
            config_path=args.config,
            device=args.device,
        )
    else:
        parser.print_help()
