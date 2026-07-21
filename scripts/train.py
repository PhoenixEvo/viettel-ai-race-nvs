"""
Training loop for BTS Digital Twin 3DGS pipeline.

Implements a 30k-iteration training schedule with:
- L1 + D-SSIM loss (tuned for competition SSIM/PSNR/LPIPS scoring)
- Depth distortion regularization to suppress floaters
- AbsGrad densification for thin structures
- Mip-Splatting antialiased rendering
- Per-image appearance embeddings
- Periodic validation with PSNR/SSIM/LPIPS logging
- Checkpointing for resume capability
"""

import json
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------
def l1_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Pixel-wise L1 loss."""
    return (pred - target).abs().mean()


def ssim_loss(
    pred: Tensor,
    target: Tensor,
    window_size: int = 11,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> Tensor:
    """Structural similarity loss (1 - SSIM).

    Args:
        pred, target: [H, W, 3] float tensors in [0, 1]

    Returns:
        scalar loss value (1 - SSIM)
    """
    # Rearrange to [1, 3, H, W]
    pred = pred.permute(2, 0, 1).unsqueeze(0)
    target = target.permute(2, 0, 1).unsqueeze(0)

    # Gaussian kernel
    channel = pred.shape[1]
    kernel = _fspecial_gauss(window_size, 1.5, channel, pred.device)

    mu1 = F.conv2d(pred, kernel, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(target, kernel, padding=window_size // 2, groups=channel)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, kernel, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target * target, kernel, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred * target, kernel, padding=window_size // 2, groups=channel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return 1.0 - ssim_map.mean()


def _fspecial_gauss(size: int, sigma: float, channels: int, device) -> Tensor:
    """Create a Gaussian kernel for SSIM computation."""
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = g.unsqueeze(0) * g.unsqueeze(1)  # [size, size]
    kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)  # [C, 1, size, size]
    return kernel


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_psnr(pred: Tensor, target: Tensor) -> float:
    """Compute PSNR between pred and target [H, W, 3] in [0, 1]."""
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


@torch.no_grad()
def compute_ssim_metric(pred: Tensor, target: Tensor) -> float:
    """Compute SSIM metric (not loss) between images."""
    return 1.0 - ssim_loss(pred, target).item()


@torch.no_grad()
def compute_lpips(pred: Tensor, target: Tensor, lpips_fn) -> float:
    """Compute LPIPS distance between images.

    Args:
        pred, target: [H, W, 3] float tensors in [0, 1]
        lpips_fn: LPIPS model instance
    """
    # LPIPS expects [B, 3, H, W] in [-1, 1]
    p = pred.permute(2, 0, 1).unsqueeze(0) * 2 - 1
    t = target.permute(2, 0, 1).unsqueeze(0) * 2 - 1
    return lpips_fn(p, t).item()


def compute_competition_score(psnr: float, ssim: float, lpips: float,
                              psnr_max: float = 50.0) -> float:
    """Approximate the competition scoring function.

    Score = 0.4 * (1 - LPIPS) + 0.3 * SSIM + 0.3 * PSNR_norm
    PSNR_norm = clamp(PSNR / psnr_max, 0, 1)
    """
    psnr_norm = min(max(psnr / psnr_max, 0.0), 1.0)
    return 0.4 * (1.0 - lpips) + 0.3 * ssim + 0.3 * psnr_norm


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_scene(
    scene_dir: str,
    result_dir: str,
    config_path: Optional[str] = None,
    resume: bool = True,
    device: str = "cuda",
    checkpoint_callback = None,
):
    """Train a 3DGS model for a single scene.

    Args:
        scene_dir: Path to scene directory (contains train/ and test/ subdirs)
        result_dir: Path to save checkpoints, logs, and renders
        config_path: Optional YAML config file path
        resume: Whether to resume from last checkpoint
        device: CUDA device
        checkpoint_callback: Optional callable called when a checkpoint is saved
    """
    from scripts.dataset import SceneDataset
    from scripts.gaussian_model import GaussianModel

    # Load config
    cfg = _load_config(config_path)

    scene_name = Path(scene_dir).name
    logger.info(f"=" * 60)
    logger.info(f"Training scene: {scene_name}")
    logger.info(f"Config: {cfg}")
    logger.info(f"=" * 60)

    # Setup directories
    result_path = Path(result_dir)
    ckpt_dir = result_path / "checkpoints"
    log_dir = result_path / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = SceneDataset(
        scene_dir=scene_dir,
        test_every=cfg.get("test_every", 8),
        normalize=cfg.get("normalize_world_space", True),
        data_factor=1,
    )

    # Initialize model
    model = GaussianModel(
        points=dataset.points,
        colors=dataset.point_colors,
        num_train_images=dataset.num_train,
        sh_degree=cfg.get("sh_degree", 3),
        init_opacity=cfg.get("init_opacity", 0.1),
        init_scale=cfg.get("init_scale", 1.0),
        means_lr=cfg.get("means_lr", 1.6e-4),
        scales_lr=cfg.get("scales_lr", 5e-3),
        quats_lr=cfg.get("quats_lr", 1e-3),
        opacities_lr=cfg.get("opacities_lr", 5e-2),
        sh0_lr=cfg.get("sh0_lr", 2.5e-3),
        shN_lr=cfg.get("shN_lr", 1.25e-4),
        scene_scale=dataset.scene_scale,
        app_opt=cfg.get("app_opt", True),
        app_embed_dim=cfg.get("app_embed_dim", 32),
        app_mlp_width=cfg.get("app_mlp_width", 64),
        app_mlp_depth=cfg.get("app_mlp_depth", 2),
        app_opt_lr=cfg.get("app_opt_lr", 1e-3),
        app_opt_reg=cfg.get("app_opt_reg", 1e-6),
        device=device,
    )

    # Resume from checkpoint if available
    start_step = 0
    if resume:
        latest_ckpt = _find_latest_checkpoint(ckpt_dir)
        if latest_ckpt is not None:
            start_step = model.load_checkpoint(str(latest_ckpt))
            logger.info(f"Resumed from step {start_step}")

    # Initialize LPIPS for validation
    lpips_fn = None
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net=cfg.get("lpips_net", "alex")).to(device)
        lpips_fn.eval()
    except ImportError:
        logger.warning("lpips not installed, LPIPS metric will be unavailable")

    # Training config
    max_steps = cfg.get("max_steps", 30000)
    ssim_lambda = cfg.get("ssim_lambda", 0.2)
    opacity_reg_weight = cfg.get("opacity_reg", 0.001)
    scale_reg_weight = cfg.get("scale_reg", 0.01)
    depth_loss_enabled = cfg.get("depth_loss", True)
    depth_lambda = cfg.get("depth_lambda", 0.01)
    depth_rampup = cfg.get("depth_rampup_iters", 3000)
    densify_start = cfg.get("densify_start_iter", 500)
    densify_stop = cfg.get("densify_stop_iter", 15000)
    densify_every = cfg.get("densify_every", 100)
    densify_grad_thresh = cfg.get("densify_grad_thresh", 0.0002)
    opacity_reset_every = cfg.get("opacity_reset_every", 3000)
    max_gaussians = cfg.get("max_gaussians", 5_000_000)
    val_every = cfg.get("val_every", 1000)
    checkpoint_every = cfg.get("checkpoint_every", 5000)
    sh_degree_interval = cfg.get("sh_degree_interval", 1000)
    antialiased = cfg.get("antialiased", True)
    near_plane = cfg.get("near_plane", 0.01)
    far_plane = cfg.get("far_plane", 1000.0)
    render_mode = "RGB+ED" if depth_loss_enabled else "RGB"

    # Training log
    metrics_file = log_dir / "metrics.jsonl"

    # Save config
    config_save = log_dir / "config.yaml"
    with open(config_save, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # Initialize densification stats
    model.init_densification_stats()

    # Random training order
    train_indices = list(range(dataset.num_train))

    logger.info(f"Starting training: steps {start_step}→{max_steps}, "
                f"{dataset.num_train} train images, {model.num_gaussians} Gaussians")

    t_start = time.time()

    for step in range(start_step, max_steps):
        # Update SH degree
        model.step_sh_degree(step, sh_degree_interval)

        # Update learning rate
        model.update_learning_rate(step, max_steps)

        # Sample random training image
        idx = random.choice(train_indices)
        batch = dataset.get_train_batch(idx)

        gt_image = batch["image"].to(device)       # [H, W, 3]
        w2c = batch["w2c"].to(device)               # [4, 4]
        K = batch["K"].to(device)                    # [3, 3]
        width = batch["width"]
        height = batch["height"]
        cam_idx = batch["cam_idx"]

        # Forward: render
        result = model.render(
            viewmat=w2c,
            K=K,
            width=width,
            height=height,
            cam_idx=cam_idx,
            near_plane=near_plane,
            far_plane=far_plane,
            render_mode=render_mode,
            antialiased=antialiased,
            absgrad=(densify_start <= step < densify_stop),
        )

        rendered = result["rgb"]  # [H, W, 3]

        # Compute loss
        loss_l1 = l1_loss(rendered, gt_image)
        loss_ssim = ssim_loss(rendered, gt_image)
        loss = (1.0 - ssim_lambda) * loss_l1 + ssim_lambda * loss_ssim

        # Opacity regularization
        if opacity_reg_weight > 0:
            opacity = torch.sigmoid(model.splats["opacities"])
            loss_opa_reg = opacity.mean()
            loss = loss + opacity_reg_weight * loss_opa_reg

        # Scale regularization (prevent bloated Gaussians)
        if scale_reg_weight > 0:
            scales = torch.exp(model.splats["scales"])
            loss_scale_reg = scales.max(dim=-1).values.mean()
            loss = loss + scale_reg_weight * loss_scale_reg

        # Depth distortion loss (ramp up)
        if depth_loss_enabled and "depth" in result and step > depth_rampup:
            depth_weight = min(1.0, (step - depth_rampup) / depth_rampup) * depth_lambda
            # Simple depth variance regularization
            depth_map = result["depth"]  # [H, W, 1]
            alpha_map = result["alpha"]  # [H, W, 1]
            # Penalize high variance in depth where alpha is high
            if depth_weight > 0:
                depth_var = depth_map.var()
                loss = loss + depth_weight * depth_var * 0.01

        # Backward
        loss.backward()

        # Update densification stats
        if densify_start <= step < densify_stop:
            model.update_densification_stats(result["info"])

        # Optimizer step
        for opt in model.optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)

        if model.app_optimizer is not None:
            model.app_optimizer.step()
            model.app_optimizer.zero_grad(set_to_none=True)

        # Densification
        if densify_start <= step < densify_stop and step % densify_every == 0:
            model.densify_and_split(
                grad_thresh=densify_grad_thresh,
                max_gaussians=max_gaussians,
            )

        # Opacity reset
        if step > 0 and step % opacity_reset_every == 0 and step < densify_stop:
            model.reset_opacity(new_opacity=0.01)

        # Logging
        if step % 100 == 0:
            elapsed = time.time() - t_start
            its_per_sec = (step - start_step + 1) / max(elapsed, 1e-8)
            logger.info(
                f"[{scene_name}] Step {step}/{max_steps} | "
                f"Loss: {loss.item():.5f} (L1={loss_l1.item():.5f}, "
                f"SSIM={loss_ssim.item():.5f}) | "
                f"N={model.num_gaussians} | "
                f"{its_per_sec:.1f} it/s"
            )

        # Validation
        if step > 0 and step % val_every == 0 and dataset.num_val > 0:
            val_metrics = _validate(model, dataset, device, lpips_fn,
                                    antialiased, near_plane, far_plane)
            val_metrics["step"] = step
            val_metrics["train_loss"] = loss.item()
            val_metrics["num_gaussians"] = model.num_gaussians
            val_metrics["elapsed_sec"] = time.time() - t_start

            # Log to file
            with open(metrics_file, "a") as f:
                f.write(json.dumps(val_metrics) + "\n")

            comp_score = compute_competition_score(
                val_metrics["psnr"], val_metrics["ssim"], val_metrics["lpips"]
            )
            logger.info(
                f"[{scene_name}] VAL Step {step}: "
                f"PSNR={val_metrics['psnr']:.2f} "
                f"SSIM={val_metrics['ssim']:.4f} "
                f"LPIPS={val_metrics['lpips']:.4f} "
                f"Score={comp_score:.4f}"
            )

        # Checkpointing
        if step > 0 and step % checkpoint_every == 0:
            ckpt_path = ckpt_dir / f"ckpt_{step:06d}.pt"
            model.save_checkpoint(
                str(ckpt_path), step,
                extra={"scene_norm": dataset.norm.to_dict()},
            )
            if checkpoint_callback is not None:
                try:
                    checkpoint_callback()
                except Exception as e:
                    logger.warning(f"Failed to run checkpoint callback: {e}")

    # Final checkpoint
    final_ckpt = ckpt_dir / f"ckpt_{max_steps:06d}.pt"
    model.save_checkpoint(
        str(final_ckpt), max_steps,
        extra={"scene_norm": dataset.norm.to_dict()},
    )
    if checkpoint_callback is not None:
        try:
            checkpoint_callback()
        except Exception as e:
            logger.warning(f"Failed to run final checkpoint callback: {e}")

    # Final validation
    if dataset.num_val > 0:
        val_metrics = _validate(model, dataset, device, lpips_fn,
                                antialiased, near_plane, far_plane)
        val_metrics["step"] = max_steps
        val_metrics["num_gaussians"] = model.num_gaussians
        val_metrics["elapsed_sec"] = time.time() - t_start

        with open(metrics_file, "a") as f:
            f.write(json.dumps(val_metrics) + "\n")

        comp_score = compute_competition_score(
            val_metrics["psnr"], val_metrics["ssim"], val_metrics["lpips"]
        )
        logger.info(
            f"[{scene_name}] FINAL: "
            f"PSNR={val_metrics['psnr']:.2f} "
            f"SSIM={val_metrics['ssim']:.4f} "
            f"LPIPS={val_metrics['lpips']:.4f} "
            f"Score={comp_score:.4f} "
            f"Gaussians={model.num_gaussians} "
            f"Time={val_metrics['elapsed_sec']:.0f}s"
        )

    total_time = time.time() - t_start
    logger.info(f"Training complete: {max_steps} steps in {total_time:.0f}s")

    return str(final_ckpt)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
@torch.no_grad()
def _validate(
    model,
    dataset,
    device: str,
    lpips_fn,
    antialiased: bool,
    near_plane: float,
    far_plane: float,
    max_images: int = 8,
) -> dict:
    """Run validation on held-out images."""
    from scripts.dataset import SceneDataset

    psnrs = []
    ssims = []
    lpipses = []

    n_val = min(len(dataset.val_cameras), max_images)
    for i in range(n_val):
        cam = dataset.val_cameras[i]
        gt_image = dataset.load_image(cam).to(device)  # [H, W, 3]
        w2c = torch.from_numpy(cam.w2c).float().to(device)
        K = torch.from_numpy(cam.K).float().to(device)

        result = model.render(
            viewmat=w2c,
            K=K,
            width=cam.width,
            height=cam.height,
            cam_idx=None,  # No appearance embedding for validation
            near_plane=near_plane,
            far_plane=far_plane,
            render_mode="RGB",
            antialiased=antialiased,
            absgrad=False,
        )

        rendered = result["rgb"].clamp(0, 1)  # [H, W, 3]

        psnrs.append(compute_psnr(rendered, gt_image))
        ssims.append(compute_ssim_metric(rendered, gt_image))
        if lpips_fn is not None:
            lpipses.append(compute_lpips(rendered, gt_image, lpips_fn))

    return {
        "psnr": np.mean(psnrs) if psnrs else 0.0,
        "ssim": np.mean(ssims) if ssims else 0.0,
        "lpips": np.mean(lpipses) if lpipses else 0.0,
    }


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def _load_config(config_path: Optional[str] = None) -> dict:
    """Load training configuration from YAML file."""
    # Defaults
    cfg = {
        "max_steps": 30000,
        "batch_size": 1,
        "val_every": 1000,
        "checkpoint_every": 5000,
        "sh_degree": 3,
        "sh_degree_interval": 1000,
        "init_opacity": 0.1,
        "init_scale": 1.0,
        "ssim_lambda": 0.2,
        "opacity_reg": 0.001,
        "scale_reg": 0.01,
        "depth_loss": True,
        "depth_lambda": 0.01,
        "depth_rampup_iters": 3000,
        "densify_start_iter": 500,
        "densify_stop_iter": 15000,
        "densify_every": 100,
        "densify_grad_thresh": 0.0002,
        "opacity_reset_every": 3000,
        "max_gaussians": 5_000_000,
        "means_lr": 1.6e-4,
        "means_lr_final": 1.6e-6,
        "scales_lr": 5e-3,
        "quats_lr": 1e-3,
        "opacities_lr": 5e-2,
        "sh0_lr": 2.5e-3,
        "shN_lr": 1.25e-4,
        "app_opt": True,
        "app_embed_dim": 32,
        "app_mlp_width": 64,
        "app_mlp_depth": 2,
        "app_opt_lr": 1e-3,
        "app_opt_reg": 1e-6,
        "antialiased": True,
        "near_plane": 0.01,
        "far_plane": 1000.0,
        "normalize_world_space": True,
        "test_every": 8,
        "lpips_net": "alex",
    }

    if config_path and Path(config_path).exists():
        with open(config_path, "r") as f:
            user_cfg = yaml.safe_load(f)
        if user_cfg:
            cfg.update(user_cfg)
            logger.info(f"Loaded config from {config_path}")

    return cfg


def _find_latest_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    """Find the latest checkpoint file in a directory."""
    ckpts = sorted(ckpt_dir.glob("ckpt_*.pt"))
    return ckpts[-1] if ckpts else None


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

    parser = argparse.ArgumentParser(description="Train 3DGS model for a scene")
    parser.add_argument("--scene-dir", required=True, help="Path to scene directory")
    parser.add_argument("--result-dir", required=True, help="Path to save results")
    parser.add_argument("--config", default=None, help="Path to YAML config")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from checkpoint")
    parser.add_argument("--device", default="cuda", help="CUDA device")
    args = parser.parse_args()

    train_scene(
        scene_dir=args.scene_dir,
        result_dir=args.result_dir,
        config_path=args.config,
        resume=not args.no_resume,
        device=args.device,
    )
