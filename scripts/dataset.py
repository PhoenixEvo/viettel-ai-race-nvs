"""
Dataset module for BTS Digital Twin pipeline.

Loads COLMAP-posed training images, creates train/val splits,
and provides a consistent camera representation for training and test-time rendering.
Scene normalization (centering + scaling) is applied consistently to all poses.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from scripts.colmap_parser import (
    COLMAPScene,
    CameraModel,
    ImageInfo,
    TestPose,
    load_colmap_scene,
    load_test_poses,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scene normalization
# ---------------------------------------------------------------------------
@dataclass
class SceneNormalization:
    """Records the normalization transform applied to a scene.

    The transform is: p_normalized = scale * (p_world - translate)
    Applied consistently to all camera positions and 3D points.
    """
    translate: np.ndarray  # (3,) center offset
    scale: float           # scaling factor

    def normalize_point(self, p: np.ndarray) -> np.ndarray:
        """Normalize a world-space point."""
        return self.scale * (p - self.translate)

    def normalize_w2c(self, w2c: np.ndarray) -> np.ndarray:
        """Normalize a world-to-camera 4x4 matrix.

        For w2c = [R | t], the normalized version is:
          R_norm = R  (rotation unchanged)
          t_norm = s * (t + R @ translate)
        """
        w2c_norm = w2c.copy()
        R = w2c[:3, :3]
        t = w2c[:3, 3]
        w2c_norm[:3, 3] = self.scale * (t + R @ self.translate)
        return w2c_norm

    def to_dict(self) -> dict:
        return {
            "translate": self.translate.tolist(),
            "scale": float(self.scale),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SceneNormalization":
        return cls(
            translate=np.array(d["translate"], dtype=np.float64),
            scale=float(d["scale"]),
        )


def compute_scene_normalization(
    camera_centers: np.ndarray,
    points: Optional[np.ndarray] = None,
) -> SceneNormalization:
    """Compute normalization that centers cameras and scales to unit sphere.

    Args:
        camera_centers: (N, 3) camera centers in world space.
        points: Optional (M, 3) 3D points for better centering.

    Returns:
        SceneNormalization with translate and scale.
    """
    # Center on camera centroid
    translate = camera_centers.mean(axis=0)

    # Scale so that camera positions fit in a unit sphere
    centered = camera_centers - translate
    dists = np.linalg.norm(centered, axis=1)
    scale = 1.0 / (np.percentile(dists, 95) + 1e-8)

    return SceneNormalization(translate=translate, scale=scale)


# ---------------------------------------------------------------------------
# Camera data structure for training
# ---------------------------------------------------------------------------
@dataclass
class CameraData:
    """Unified camera data for a single view (training or test)."""
    image_name: str
    w2c: np.ndarray        # (4, 4) world-to-camera (normalized)
    K: np.ndarray          # (3, 3) intrinsic matrix
    width: int
    height: int
    image_path: Optional[Path] = None  # None for test poses


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class SceneDataset:
    """Dataset for a single scene, providing train/val splits and test poses."""

    def __init__(
        self,
        scene_dir: str | Path,
        test_every: int = 8,
        normalize: bool = True,
        data_factor: int = 1,
    ):
        """
        Args:
            scene_dir: Path to scene directory containing train/ and test/ subdirs.
            test_every: Every Nth image becomes validation (0 = no val split).
            normalize: Whether to normalize the scene.
            data_factor: Downsample factor for images (1 = original).
        """
        self.scene_dir = Path(scene_dir)
        self.test_every = test_every
        self.data_factor = data_factor

        # Load COLMAP reconstruction
        sparse_dir = self.scene_dir / "train" / "sparse" / "0"
        self.colmap_scene = load_colmap_scene(sparse_dir)

        # Get sorted image list
        all_images = self.colmap_scene.get_image_list()
        image_dir = self.scene_dir / "train" / "images"

        # Load test poses to identify expected test views
        test_csv = self.scene_dir / "test" / "test_poses.csv"
        test_names = set()
        if test_csv.exists():
            try:
                test_names = {tp.image_name for tp in load_test_poses(test_csv)}
            except Exception as e:
                logger.debug(f"Could not load test poses for filtering: {e}")

        # Filter to images that actually exist on disk
        existing_images = []
        for img in all_images:
            img_path = image_dir / img.name
            if img_path.exists():
                existing_images.append(img)
            else:
                if img.name in test_names:
                    logger.debug(f"Image {img.name} is a test pose (expectedly absent from train/images)")
                else:
                    logger.warning(f"Training image not found on disk: {img_path}")

        if len(existing_images) == 0:
            raise ValueError(f"No training images found in {image_dir}")

        logger.info(
            f"Found {len(existing_images)}/{len(all_images)} images on disk"
        )

        # Compute camera centers for normalization
        camera_centers = np.array([img.camera_center for img in existing_images])
        points, point_colors = self.colmap_scene.get_points_array()

        # Compute normalization
        if normalize:
            self.norm = compute_scene_normalization(camera_centers, points)
        else:
            self.norm = SceneNormalization(
                translate=np.zeros(3, dtype=np.float64), scale=1.0
            )

        logger.info(
            f"Scene normalization: translate={self.norm.translate}, "
            f"scale={self.norm.scale:.4f}"
        )

        # Normalize 3D points
        if len(points) > 0:
            self.points = np.array([
                self.norm.normalize_point(p) for p in points
            ])
            self.point_colors = point_colors
        else:
            self.points = np.zeros((0, 3), dtype=np.float64)
            self.point_colors = np.zeros((0, 3), dtype=np.float64)

        # Build camera data for all images
        all_cameras = []
        for img in existing_images:
            cam = self.colmap_scene.cameras[img.camera_id]
            w2c = img.w2c_matrix()
            w2c_norm = self.norm.normalize_w2c(w2c)
            K = cam.K(scale=1.0 / data_factor)
            width = cam.width // data_factor
            height = cam.height // data_factor

            all_cameras.append(CameraData(
                image_name=img.name,
                w2c=w2c_norm,
                K=K,
                width=width,
                height=height,
                image_path=image_dir / img.name,
            ))

        # Train/val split
        if test_every > 0:
            self.train_cameras = [
                c for i, c in enumerate(all_cameras) if i % test_every != 0
            ]
            self.val_cameras = [
                c for i, c in enumerate(all_cameras) if i % test_every == 0
            ]
        else:
            self.train_cameras = all_cameras
            self.val_cameras = []

        logger.info(
            f"Split: {len(self.train_cameras)} train, "
            f"{len(self.val_cameras)} val"
        )

        # Load test poses
        test_csv = self.scene_dir / "test" / "test_poses.csv"
        if test_csv.exists():
            raw_test_poses = load_test_poses(test_csv)
            self.test_cameras = []
            for tp in raw_test_poses:
                w2c = tp.w2c_matrix()
                w2c_norm = self.norm.normalize_w2c(w2c)
                K = tp.K()
                self.test_cameras.append(CameraData(
                    image_name=tp.image_name,
                    w2c=w2c_norm,
                    K=K,
                    width=tp.width,
                    height=tp.height,
                ))
            logger.info(f"Loaded {len(self.test_cameras)} test poses")
        else:
            self.test_cameras = []
            logger.warning(f"No test_poses.csv found at {test_csv}")

    def load_image(self, cam: CameraData) -> torch.Tensor:
        """Load and resize image as float32 tensor [H, W, 3] in [0, 1]."""
        if cam.image_path is None:
            raise ValueError(f"No image path for {cam.image_name}")

        img = Image.open(cam.image_path).convert("RGB")

        # Resize if needed
        target_size = (cam.width, cam.height)  # PIL uses (W, H)
        if img.size != target_size:
            img = img.resize(target_size, Image.BICUBIC)

        img_np = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(img_np)

    def get_train_batch(self, idx: int) -> dict:
        """Get a single training sample."""
        cam = self.train_cameras[idx % len(self.train_cameras)]
        image = self.load_image(cam)
        return {
            "image": image,           # [H, W, 3]
            "w2c": torch.from_numpy(cam.w2c).float(),      # [4, 4]
            "K": torch.from_numpy(cam.K).float(),           # [3, 3]
            "width": cam.width,
            "height": cam.height,
            "image_name": cam.image_name,
            "cam_idx": idx % len(self.train_cameras),
        }

    @property
    def scene_scale(self) -> float:
        """The reciprocal of the normalization scale — used for LR scaling."""
        return 1.0 / self.norm.scale

    @property
    def num_train(self) -> int:
        return len(self.train_cameras)

    @property
    def num_val(self) -> int:
        return len(self.val_cameras)

    @property
    def num_test(self) -> int:
        return len(self.test_cameras)


# ---------------------------------------------------------------------------
# CLI for standalone testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Test dataset loading")
    parser.add_argument("scene_dir", type=str, help="Path to scene directory")
    parser.add_argument("--test-every", type=int, default=8)
    parser.add_argument("--data-factor", type=int, default=1)
    args = parser.parse_args()

    dataset = SceneDataset(
        args.scene_dir,
        test_every=args.test_every,
        data_factor=args.data_factor,
    )

    print(f"\nDataset summary:")
    print(f"  Train: {dataset.num_train}")
    print(f"  Val:   {dataset.num_val}")
    print(f"  Test:  {dataset.num_test}")
    print(f"  Points: {len(dataset.points)}")
    print(f"  Scene scale: {dataset.scene_scale:.4f}")

    # Test loading one image
    if dataset.num_train > 0:
        batch = dataset.get_train_batch(0)
        print(f"\n  Sample batch:")
        print(f"    Image: {batch['image_name']} shape={batch['image'].shape}")
        print(f"    W2C:\n{batch['w2c']}")
        print(f"    K:\n{batch['K']}")
        print(f"    Resolution: {batch['width']}x{batch['height']}")
