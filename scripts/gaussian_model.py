"""
3D Gaussian Splatting model for BTS Digital Twin pipeline.

Implements the full Gaussian representation with quality-maximizing enhancements:
- Mip-Splatting (antialiased rasterization) for alias-free rendering
- AbsGrad densification (Pixel-GS style) for thin structures
- Per-image appearance embeddings to absorb exposure variation
- Depth rendering for consistency regularization

Uses gsplat as the CUDA rasterization backend.
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def rgb_to_sh(rgb: Tensor) -> Tensor:
    """Convert linear RGB to 0th-order spherical harmonic coefficient."""
    C0 = 0.28209479177387814  # 1 / (2 * sqrt(pi))
    return (rgb - 0.5) / C0


def sh_to_rgb(sh: Tensor) -> Tensor:
    """Convert 0th-order SH coefficient to linear RGB."""
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


def inverse_sigmoid(x: Tensor) -> Tensor:
    """Numerically stable inverse sigmoid (logit)."""
    return torch.log(x / (1 - x + 1e-8) + 1e-8)


def knn(points: Tensor, k: int) -> Tensor:
    """K-nearest neighbors using sklearn (CPU).

    Args:
        points: (N, 3) point positions
        k: number of neighbors

    Returns:
        distances: (N, k) distances to k nearest neighbors
    """
    from sklearn.neighbors import NearestNeighbors

    pts_np = points.detach().cpu().numpy()
    nn_model = NearestNeighbors(n_neighbors=k, algorithm="auto", metric="euclidean")
    nn_model.fit(pts_np)
    distances, _ = nn_model.kneighbors(pts_np)
    return torch.from_numpy(distances).float().to(points.device)




# ---------------------------------------------------------------------------
# Gaussian Model
# ---------------------------------------------------------------------------
class GaussianModel:
    """3D Gaussian Splatting model with quality-maximizing enhancements.

    This manages:
    - Gaussian parameters (means, scales, quats, opacities, SH)
    - Per-parameter optimizers
    - Appearance model
    - Densification logic (AbsGrad / Pixel-GS style)
    - Rendering via gsplat
    """

    def __init__(
        self,
        points: np.ndarray,        # (N, 3) initial 3D positions
        colors: np.ndarray,        # (N, 3) initial RGB in [0, 1]
        num_train_images: int,
        sh_degree: int = 3,
        init_opacity: float = 0.1,
        init_scale: float = 1.0,
        # Learning rates
        means_lr: float = 1.6e-4,
        scales_lr: float = 5e-3,
        quats_lr: float = 1e-3,
        opacities_lr: float = 5e-2,
        sh0_lr: float = 2.5e-3,
        shN_lr: float = 1.25e-4,
        scene_scale: float = 1.0,
        # Appearance
        app_opt: bool = True,
        app_opt_lr: float = 1e-3,
        app_opt_reg: float = 1e-6,
        device: str = "cuda",
    ):
        self.device = device
        self._is_training = True
        self.sh_degree = sh_degree
        self.max_sh_degree = sh_degree
        self.active_sh_degree = 0
        self.scene_scale = scene_scale
        self.app_opt = app_opt

        # Initialize Gaussian parameters from point cloud
        points_t = torch.from_numpy(points).float()
        colors_t = torch.from_numpy(colors).float()

        N = points_t.shape[0]
        logger.info(f"Initializing {N} Gaussians from point cloud")

        # Compute initial scales from KNN distances
        if N > 1:
            dist_k = knn(points_t, 4)  # [N, 4]
            dist2_avg = (dist_k[:, 1:] ** 2).mean(dim=-1)  # [N]
            dist_avg = torch.sqrt(dist2_avg)
            scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)
        else:
            scales = torch.zeros((N, 3))

        # Random quaternions (uniform on SO(3))
        quats = torch.randn((N, 4))
        quats = F.normalize(quats, dim=-1)

        # Opacities in logit space
        opacities = inverse_sigmoid(torch.full((N,), init_opacity))

        # SH coefficients
        num_sh = (sh_degree + 1) ** 2
        sh_all = torch.zeros((N, num_sh, 3))
        sh_all[:, 0, :] = rgb_to_sh(colors_t)

        # Create parameters
        self.splats = nn.ParameterDict({
            "means": nn.Parameter(points_t),
            "scales": nn.Parameter(scales),
            "quats": nn.Parameter(quats),
            "opacities": nn.Parameter(opacities),
            "sh0": nn.Parameter(sh_all[:, :1, :].contiguous()),
            "shN": nn.Parameter(sh_all[:, 1:, :].contiguous()),
        }).to(device)

        # Create optimizers
        self.optimizers = {
            "means": torch.optim.Adam(
                [{"params": self.splats["means"], "lr": means_lr * scene_scale, "name": "means"}],
                eps=1e-15, betas=(0.9, 0.999), fused=True,
            ),
            "scales": torch.optim.Adam(
                [{"params": self.splats["scales"], "lr": scales_lr, "name": "scales"}],
                eps=1e-15, betas=(0.9, 0.999), fused=True,
            ),
            "quats": torch.optim.Adam(
                [{"params": self.splats["quats"], "lr": quats_lr, "name": "quats"}],
                eps=1e-15, betas=(0.9, 0.999), fused=True,
            ),
            "opacities": torch.optim.Adam(
                [{"params": self.splats["opacities"], "lr": opacities_lr, "name": "opacities"}],
                eps=1e-15, betas=(0.9, 0.999), fused=True,
            ),
            "sh0": torch.optim.Adam(
                [{"params": self.splats["sh0"], "lr": sh0_lr, "name": "sh0"}],
                eps=1e-15, betas=(0.9, 0.999), fused=True,
            ),
            "shN": torch.optim.Adam(
                [{"params": self.splats["shN"], "lr": shN_lr, "name": "shN"}],
                eps=1e-15, betas=(0.9, 0.999), fused=True,
            ),
        }

        # Appearance model
        if app_opt and num_train_images > 0:
            # Per-image affine color transform: 3x3 matrix + 3 bias = 12 params per image
            # Initialized to identity transform (no color change)
            self.app_affine = nn.Embedding(num_train_images, 12).to(device)
            nn.init.zeros_(self.app_affine.weight)
            # We'll store as delta from identity: actual A = I + delta_A, b = delta_b
            self.app_optimizer = torch.optim.Adam(
                self.app_affine.parameters(),
                lr=app_opt_lr,
                weight_decay=app_opt_reg,
            )
        else:
            self.app_affine = None
            self.app_optimizer = None

        # Densification state
        self._grad_accum = None
        self._grad_count = None
        self._max_radii2d = None

        # LR scheduling
        self._means_lr_init = means_lr * scene_scale
        self._means_lr_final = 1.6e-6 * scene_scale

    @property
    def num_gaussians(self) -> int:
        return self.splats["means"].shape[0]

    def step_sh_degree(self, step: int, interval: int = 1000):
        """Increment active SH degree based on training step."""
        new_degree = min(step // interval, self.max_sh_degree)
        if new_degree != self.active_sh_degree:
            self.active_sh_degree = new_degree
            logger.info(f"Step {step}: SH degree → {self.active_sh_degree}")

    def update_learning_rate(self, step: int, max_steps: int):
        """Cosine decay for means learning rate."""
        import math
        if max_steps <= 0:
            return
        progress = min(step / max_steps, 1.0)
        lr = self._means_lr_final + 0.5 * (self._means_lr_init - self._means_lr_final) * (1 + math.cos(math.pi * progress))
        for pg in self.optimizers["means"].param_groups:
            pg["lr"] = lr

    def render(
        self,
        viewmat: Tensor,       # [4, 4] world-to-camera
        K: Tensor,             # [3, 3] intrinsic matrix
        width: int,
        height: int,
        cam_idx: Optional[int] = None,
        near_plane: float = 0.01,
        far_plane: float = 1000.0,
        render_mode: str = "RGB",
        antialiased: bool = True,
        absgrad: bool = True,
        background: Optional[Tensor] = None,
        sh_degree: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Render the scene from a given viewpoint.

        Args:
            viewmat: [4, 4] world-to-camera matrix
            K: [3, 3] camera intrinsic matrix
            width, height: render resolution
            cam_idx: training image index (for appearance model, None at test time)
            near_plane, far_plane: clipping planes
            render_mode: "RGB", "RGB+D", "RGB+ED"
            antialiased: Mip-Splatting alias-free filtering
            absgrad: compute absolute gradients for densification
            background: [3] background color (default: black)
            sh_degree: override active SH degree

        Returns:
            Dictionary with 'rgb' [H, W, 3], optionally 'depth' [H, W, 1],
            and 'info' dict from gsplat.
        """
        from gsplat.rendering import rasterization

        means = self.splats["means"]     # [N, 3]
        quats = self.splats["quats"]     # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N]

        # SH coefficients — only use active degrees
        if sh_degree is None:
            sh_degree = self.active_sh_degree
        num_active_sh = (sh_degree + 1) ** 2
        sh0 = self.splats["sh0"]       # [N, 1, 3]
        shN = self.splats["shN"][:, :num_active_sh - 1, :]  # [N, K-1, 3]
        colors = torch.cat([sh0, shN], dim=1)  # [N, K, 3]

        # Format for gsplat: viewmats [C, 4, 4], Ks [C, 3, 3]
        viewmats = viewmat.unsqueeze(0)  # [1, 4, 4]
        Ks = K.unsqueeze(0)              # [1, 3, 3]

        if background is None:
            background = torch.zeros(3, device=self.device)
        backgrounds = background.unsqueeze(0)  # [1, 3]

        # Rasterize
        rasterize_mode = "antialiased" if antialiased else "classic"

        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=width,
            height=height,
            near_plane=near_plane,
            far_plane=far_plane,
            sh_degree=sh_degree,
            render_mode=render_mode,
            rasterize_mode=rasterize_mode,
            absgrad=absgrad,
            backgrounds=backgrounds,
        )

        # render_colors: [1, H, W, C], render_alphas: [1, H, W, 1]
        result = {"info": info, "alpha": render_alphas[0]}  # [H, W, 1]

        if render_mode == "RGB":
            result["rgb"] = render_colors[0]  # [H, W, 3]
        elif render_mode in ("RGB+D", "RGB+ED"):
            result["rgb"] = render_colors[0, ..., :3]  # [H, W, 3]
            result["depth"] = render_colors[0, ..., 3:]  # [H, W, 1]
        elif render_mode in ("D", "ED"):
            result["depth"] = render_colors[0]  # [H, W, 1]

        # Apply per-image affine color correction (training only)
        if (
            self.app_affine is not None
            and cam_idx is not None
            and self._is_training
        ):
            cam_idx_t = torch.tensor(cam_idx, device=self.device, dtype=torch.long)
            delta = self.app_affine(cam_idx_t)  #[1]
            # delta_A:  additive to identity, delta_b:[2]
            delta_A = delta[:9].reshape(3, 3)
            delta_b = delta[9:]
            A = torch.eye(3, device=self.device) + delta_A * 0.1  # small residual
            rgb = result["rgb"]  # [H, W, 3]
            # Apply affine: out[h,w] = A @ rgb[h,w] + b
            result["rgb"] = (rgb @ A.T + delta_b * 0.1).clamp(0, 1)

        return result

    @property
    def training_mode(self) -> bool:
        return self._is_training

    def set_train(self):
        self._is_training = True

    def set_eval(self):
        self._is_training = False

    # ------------------------------------------------------------------
    # Densification (AbsGrad / Pixel-GS style)
    # ------------------------------------------------------------------
    def init_densification_stats(self):
        """Initialize gradient accumulation buffers."""
        N = self.num_gaussians
        self._grad_accum = torch.zeros(N, device=self.device)
        self._grad_count = torch.zeros(N, device=self.device, dtype=torch.int32)
        self._max_radii2d = torch.zeros(N, device=self.device)

    def update_densification_stats(self, info: dict):
        """Accumulate absolute gradients from rasterization.

        With absgrad=True, gsplat stores absolute gradients on means2d.
        """
        if self._grad_accum is None:
            self.init_densification_stats()

        # Get 2D gradient norms
        if "means2d" in info and info["means2d"].grad is not None:
            means2d_grad = info["means2d"].grad  # [1, N, 2] or [N, 2]
            if means2d_grad.dim() == 3:
                means2d_grad = means2d_grad[0]  # [N, 2]
            grad_norms = means2d_grad.norm(dim=-1)  # [N]

            # Handle size mismatch (in case of dynamic Gaussian count)
            if grad_norms.shape[0] == self._grad_accum.shape[0]:
                self._grad_accum += grad_norms
                self._grad_count += (grad_norms > 0).int()

        # Track max screen-space radii
        if "radii" in info:
            radii = info["radii"]  # [1, N] or [N]
            if radii.dim() == 2:
                radii = radii[0]
            if radii.shape[0] == self._max_radii2d.shape[0]:
                self._max_radii2d = torch.max(self._max_radii2d, radii.float())

    def densify_and_split(
        self,
        grad_thresh: float = 0.0002,
        max_screen_size: float = 20.0,
        min_opacity: float = 0.005,
        max_gaussians: int = 5_000_000,
    ):
        """Perform densification: clone small under-reconstructed Gaussians,
        split large ones, and prune near-transparent ones.

        Uses absolute gradient (Pixel-GS / AbsGrad) instead of averaged gradient.
        """
        if self._grad_accum is None or self._grad_count is None:
            return

        N = self.num_gaussians
        if N >= max_gaussians:
            logger.info(f"Max Gaussian count reached ({N}), skipping densification")
            return

        # Average gradient
        avg_grad = self._grad_accum / (self._grad_count.float() + 1e-8)

        # Mask: Gaussians with high gradient
        grad_mask = avg_grad >= grad_thresh

        # Split large Gaussians (scale > threshold)
        scales = torch.exp(self.splats["scales"])
        max_scale = scales.max(dim=-1).values
        scale_thresh = 0.01  # fixed, in normalized world coordinates
        split_mask = grad_mask & (max_scale > scale_thresh)
        clone_mask = grad_mask & (max_scale <= scale_thresh)

        # Prune near-transparent
        opacity = torch.sigmoid(self.splats["opacities"])
        prune_mask = opacity < min_opacity

        # Prune large screen-space Gaussians
        if self._max_radii2d is not None:
            big_screen_mask = self._max_radii2d > max_screen_size
            prune_mask = prune_mask | big_screen_mask

        n_split = split_mask.sum().item()
        n_clone = clone_mask.sum().item()
        n_prune = prune_mask.sum().item()

        if n_split + n_clone + n_prune == 0:
            self._reset_densification_stats()
            return

        logger.debug(
            f"Densify: split={n_split}, clone={n_clone}, prune={n_prune}, "
            f"total={N} → ~{N + n_clone + n_split - n_prune}"
        )

        # --- Clone: duplicate small Gaussians ---
        if n_clone > 0:
            clone_means = self.splats["means"][clone_mask].data.clone()
            clone_scales = self.splats["scales"][clone_mask].data.clone()
            clone_quats = self.splats["quats"][clone_mask].data.clone()
            clone_opacities = self.splats["opacities"][clone_mask].data.clone()
            clone_sh0 = self.splats["sh0"][clone_mask].data.clone()
            clone_shN = self.splats["shN"][clone_mask].data.clone()
        else:
            clone_means = torch.empty(0, 3, device=self.device)
            clone_scales = torch.empty(0, 3, device=self.device)
            clone_quats = torch.empty(0, 4, device=self.device)
            clone_opacities = torch.empty(0, device=self.device)
            clone_sh0 = torch.empty(0, 1, 3, device=self.device)
            clone_shN = torch.empty(0, self.splats["shN"].shape[1], 3, device=self.device)

        # --- Split: replace large Gaussians with 2 smaller ones ---
        if n_split > 0:
            split_means = self.splats["means"][split_mask].data.clone()
            split_scales = self.splats["scales"][split_mask].data.clone()
            split_quats = self.splats["quats"][split_mask].data.clone()
            split_opacities = self.splats["opacities"][split_mask].data.clone()
            split_sh0 = self.splats["sh0"][split_mask].data.clone()
            split_shN = self.splats["shN"][split_mask].data.clone()

            # Reduce scale by factor of 1.6
            split_scales_new = split_scales - math.log(1.6)

            # Offset positions along the principal axis
            stds = torch.exp(split_scales)
            samples = torch.randn_like(split_means) * stds
            # Two children offset in opposite directions
            split_means_a = split_means + samples
            split_means_b = split_means - samples

            new_split_means = torch.cat([split_means_a, split_means_b], dim=0)
            new_split_scales = split_scales_new.repeat(2, 1)
            new_split_quats = split_quats.repeat(2, 1)
            new_split_opacities = split_opacities.repeat(2)
            new_split_sh0 = split_sh0.repeat(2, 1, 1)
            new_split_shN = split_shN.repeat(2, 1, 1)
        else:
            new_split_means = torch.empty(0, 3, device=self.device)
            new_split_scales = torch.empty(0, 3, device=self.device)
            new_split_quats = torch.empty(0, 4, device=self.device)
            new_split_opacities = torch.empty(0, device=self.device)
            new_split_sh0 = torch.empty(0, 1, 3, device=self.device)
            new_split_shN = torch.empty(0, self.splats["shN"].shape[1], 3, device=self.device)

        # --- Build keep mask (everything except pruned and split originals) ---
        keep_mask = ~(prune_mask | split_mask)

        # Reconstruct parameters
        new_means = torch.cat([
            self.splats["means"][keep_mask].data,
            clone_means,
            new_split_means,
        ], dim=0)
        new_scales = torch.cat([
            self.splats["scales"][keep_mask].data,
            clone_scales,
            new_split_scales,
        ], dim=0)
        new_quats = torch.cat([
            self.splats["quats"][keep_mask].data,
            clone_quats,
            new_split_quats,
        ], dim=0)
        new_opacities = torch.cat([
            self.splats["opacities"][keep_mask].data,
            clone_opacities,
            new_split_opacities,
        ], dim=0)
        new_sh0 = torch.cat([
            self.splats["sh0"][keep_mask].data,
            clone_sh0,
            new_split_sh0,
        ], dim=0)
        new_shN = torch.cat([
            self.splats["shN"][keep_mask].data,
            clone_shN,
            new_split_shN,
        ], dim=0)

        # Update parameters and re-create optimizers
        self._replace_params({
            "means": new_means,
            "scales": new_scales,
            "quats": new_quats,
            "opacities": new_opacities,
            "sh0": new_sh0,
            "shN": new_shN,
        }, keep_mask=keep_mask, n_clone=n_clone, n_split=n_split)

        self._reset_densification_stats()

        logger.info(
            f"After densification: {self.num_gaussians} Gaussians "
            f"(+{n_clone} cloned, +{2*n_split} from splits, -{n_prune} pruned)"
        )

    def reset_opacity(self, new_opacity: float = 0.01):
        """Reset all opacities to a low value (prevent over-accumulation)."""
        with torch.no_grad():
            opacities = torch.sigmoid(self.splats["opacities"])
            new_opa = torch.clamp(opacities, max=new_opacity)
            self.splats["opacities"].data = inverse_sigmoid(new_opa)
        logger.info(f"Reset opacities to max={new_opacity}")

    def _replace_params(self, new_params: Dict[str, torch.Tensor],
                        keep_mask: Optional[torch.Tensor] = None,
                        n_clone: int = 0,
                        n_split: int = 0):
        """Replace all Gaussian parameters and preserve optimizer states."""
        for name in new_params.keys():
            old_param = self.splats[name]
            old_opt = self.optimizers[name]
            old_lr = old_opt.param_groups[0]["lr"]
            
            new_param = nn.Parameter(new_params[name].to(self.device))
            new_opt = torch.optim.Adam(
                [{"params": new_param, "lr": old_lr, "name": name}],
                eps=1e-15, betas=(0.9, 0.999), fused=True
            )
            
            if keep_mask is not None and old_param in old_opt.state:
                old_state = old_opt.state[old_param]
                if "exp_avg" in old_state:
                    def _migrate(t):
                        kept = t[keep_mask]
                        z_c = torch.zeros((n_clone,) + t.shape[1:], dtype=t.dtype, device=t.device)
                        z_s = torch.zeros((n_split * 2,) + t.shape[1:], dtype=t.dtype, device=t.device)
                        return torch.cat([kept, z_c, z_s], dim=0)
                    
                    new_opt.state[new_param] = {
                        "step": old_state["step"],
                        "exp_avg": _migrate(old_state["exp_avg"]),
                        "exp_avg_sq": _migrate(old_state["exp_avg_sq"]),
                    }
                    
            self.splats[name] = new_param
            self.optimizers[name] = new_opt

    def _reset_densification_stats(self):
        """Reset gradient accumulation buffers."""
        N = self.num_gaussians
        self._grad_accum = torch.zeros(N, device=self.device)
        self._grad_count = torch.zeros(N, device=self.device, dtype=torch.int32)
        self._max_radii2d = torch.zeros(N, device=self.device)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def save_checkpoint(self, path: str, step: int, extra: dict = None):
        """Save model checkpoint."""
        state = {
            "step": step,
            "splats": {k: v.data for k, v in self.splats.items()},
            "optimizer_states": {
                k: v.state_dict() for k, v in self.optimizers.items()
            },
            "active_sh_degree": self.active_sh_degree,
            "num_gaussians": self.num_gaussians,
        }
        if self.app_affine is not None:
            state["app_affine"] = self.app_affine.state_dict()
            state["app_optimizer"] = self.app_optimizer.state_dict()
        if extra:
            state.update(extra)
        torch.save(state, path)
        logger.info(f"Saved checkpoint: {path} (step={step}, N={self.num_gaussians})")

    def load_checkpoint(self, path: str) -> int:
        """Load model checkpoint, returns the step number."""
        state = torch.load(path, map_location=self.device, weights_only=False)

        # Restore Gaussian parameters
        for name, data in state["splats"].items():
            if name in self.splats:
                self.splats[name] = nn.Parameter(data.to(self.device))

        # Re-create optimizers
        lr_map = {}
        for name, opt in self.optimizers.items():
            lr_map[name] = opt.param_groups[0]["lr"]

        self.optimizers = {
            name: torch.optim.Adam(
                [{"params": self.splats[name], "lr": lr_map.get(name, 1e-4), "name": name}],
                eps=1e-15, betas=(0.9, 0.999), fused=True,
            )
            for name in self.splats.keys()
        }

        # Try to restore optimizer states
        if "optimizer_states" in state:
            for name, opt_state in state["optimizer_states"].items():
                if name in self.optimizers:
                    try:
                        self.optimizers[name].load_state_dict(opt_state)
                    except Exception as e:
                        logger.warning(f"Could not restore optimizer state for {name}: {e}")

        self.active_sh_degree = state.get("active_sh_degree", 0)

        if self.app_affine is not None and "app_affine" in state:
            self.app_affine.load_state_dict(state["app_affine"])
        if self.app_optimizer is not None and "app_optimizer" in state:
            try:
                self.app_optimizer.load_state_dict(state["app_optimizer"])
            except Exception:
                pass

        step = state.get("step", 0)
        logger.info(
            f"Loaded checkpoint: {path} (step={step}, N={self.num_gaussians})"
        )
        return step
