"""
COLMAP binary parser for BTS Digital Twin pipeline.

Reads cameras.bin, images.bin, and points3D.bin without pycolmap dependency.
Handles both standard COLMAP naming and competition-specific 'frames.bin'.

References:
  - COLMAP binary format: https://colmap.github.io/format.html
  - Quaternion convention: (qw, qx, qy, qz) for world-to-camera rotation
"""

import struct
import collections
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# COLMAP camera model IDs (from colmap/src/base/camera_models.h)
# ---------------------------------------------------------------------------
CAMERA_MODEL_IDS = {
    0: "SIMPLE_PINHOLE",   # f, cx, cy
    1: "PINHOLE",          # fx, fy, cx, cy
    2: "SIMPLE_RADIAL",    # f, cx, cy, k
    3: "RADIAL",           # f, cx, cy, k1, k2
    4: "OPENCV",           # fx, fy, cx, cy, k1, k2, p1, p2
    5: "OPENCV_FISHEYE",   # fx, fy, cx, cy, k1, k2, k3, k4
    6: "FULL_OPENCV",      # fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
    7: "FOV",              # fx, fy, cx, cy, omega
    8: "SIMPLE_RADIAL_FISHEYE",  # f, cx, cy, k
    9: "RADIAL_FISHEYE",   # f, cx, cy, k1, k2
    10: "THIN_PRISM_FISHEYE",  # fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sx2
}

CAMERA_MODEL_NUM_PARAMS = {
    "SIMPLE_PINHOLE": 3,
    "PINHOLE": 4,
    "SIMPLE_RADIAL": 4,
    "RADIAL": 5,
    "OPENCV": 8,
    "OPENCV_FISHEYE": 8,
    "FULL_OPENCV": 12,
    "FOV": 5,
    "SIMPLE_RADIAL_FISHEYE": 4,
    "RADIAL_FISHEYE": 5,
    "THIN_PRISM_FISHEYE": 12,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class CameraModel:
    """A single COLMAP camera (shared intrinsics for multiple images)."""
    camera_id: int
    model_name: str
    width: int
    height: int
    params: np.ndarray  # raw parameter array

    @property
    def fx(self) -> float:
        if self.model_name == "SIMPLE_PINHOLE":
            return float(self.params[0])
        return float(self.params[0])

    @property
    def fy(self) -> float:
        if self.model_name == "SIMPLE_PINHOLE":
            return float(self.params[0])  # same as fx
        return float(self.params[1])

    @property
    def cx(self) -> float:
        if self.model_name == "SIMPLE_PINHOLE":
            return float(self.params[1])
        return float(self.params[2])

    @property
    def cy(self) -> float:
        if self.model_name == "SIMPLE_PINHOLE":
            return float(self.params[2])
        return float(self.params[3])

    def K(self, scale: float = 1.0) -> np.ndarray:
        """3x3 intrinsic matrix, optionally scaled."""
        K = np.array([
            [self.fx * scale, 0.0, self.cx * scale],
            [0.0, self.fy * scale, self.cy * scale],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        return K

    def distortion_params(self) -> np.ndarray:
        """Return distortion coefficients (empty for PINHOLE)."""
        if self.model_name in ("SIMPLE_PINHOLE", "PINHOLE"):
            return np.zeros(0, dtype=np.float64)
        elif self.model_name == "SIMPLE_RADIAL":
            return np.array([self.params[3], 0.0, 0.0, 0.0], dtype=np.float64)
        elif self.model_name == "RADIAL":
            return np.array([self.params[3], self.params[4], 0.0, 0.0], dtype=np.float64)
        elif self.model_name in ("OPENCV", "OPENCV_FISHEYE"):
            return self.params[4:8].astype(np.float64)
        else:
            logger.warning(f"Unsupported camera model for distortion: {self.model_name}")
            return np.zeros(0, dtype=np.float64)


@dataclass
class ImageInfo:
    """A single COLMAP registered image."""
    image_id: int
    qvec: np.ndarray       # (qw, qx, qy, qz) — world-to-camera quaternion
    tvec: np.ndarray       # (tx, ty, tz) — world-to-camera translation
    camera_id: int
    name: str              # image filename
    point3D_ids: np.ndarray  # per-2D-point: which 3D point it observes (-1 if none)

    def w2c_matrix(self) -> np.ndarray:
        """4x4 world-to-camera transformation matrix."""
        R = qvec_to_rotmat(self.qvec)
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = R
        w2c[:3, 3] = self.tvec
        return w2c

    def c2w_matrix(self) -> np.ndarray:
        """4x4 camera-to-world transformation matrix."""
        return np.linalg.inv(self.w2c_matrix())

    @property
    def camera_center(self) -> np.ndarray:
        """Camera center in world coordinates: -R^T @ t"""
        R = qvec_to_rotmat(self.qvec)
        return -R.T @ self.tvec


@dataclass
class Point3D:
    """A single COLMAP 3D point."""
    point3D_id: int
    xyz: np.ndarray        # (3,) world coordinates
    rgb: np.ndarray        # (3,) uint8 color
    error: float           # reprojection error
    track_length: int      # number of images observing this point


@dataclass
class COLMAPScene:
    """Complete parsed COLMAP reconstruction."""
    cameras: Dict[int, CameraModel]
    images: Dict[int, ImageInfo]
    points3D: Dict[int, Point3D]

    @property
    def num_images(self) -> int:
        return len(self.images)

    @property
    def num_points(self) -> int:
        return len(self.points3D)

    def get_points_array(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (positions [N,3], colors [N,3] float in [0,1])."""
        if len(self.points3D) == 0:
            return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)
        pts = []
        colors = []
        for pid in sorted(self.points3D.keys()):
            p = self.points3D[pid]
            pts.append(p.xyz)
            colors.append(p.rgb.astype(np.float64) / 255.0)
        return np.stack(pts, axis=0), np.stack(colors, axis=0)

    def get_image_list(self) -> List[ImageInfo]:
        """Return images sorted by image_id."""
        return [self.images[k] for k in sorted(self.images.keys())]


# ---------------------------------------------------------------------------
# Quaternion utilities
# ---------------------------------------------------------------------------
def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    """Convert COLMAP quaternion (qw, qx, qy, qz) to 3x3 rotation matrix.

    COLMAP convention: quaternion represents world-to-camera rotation.
    """
    qw, qx, qy, qz = qvec
    R = np.array([
        [1 - 2*qy*qy - 2*qz*qz,   2*qx*qy - 2*qz*qw,       2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,        1 - 2*qx*qx - 2*qz*qz,   2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,        2*qy*qz + 2*qx*qw,        1 - 2*qx*qx - 2*qy*qy],
    ], dtype=np.float64)
    return R


def rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to COLMAP quaternion (qw, qx, qy, qz)."""
    # Shepperd's method for numerical stability
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return np.array([qw, qx, qy, qz], dtype=np.float64)


# ---------------------------------------------------------------------------
# Binary parsers
# ---------------------------------------------------------------------------
def read_cameras_binary(path: Path) -> Dict[int, CameraModel]:
    """Read COLMAP cameras.bin file.

    Binary format:
      num_cameras: uint64
      per camera:
        camera_id: uint32
        model_id: int32
        width: uint64
        height: uint64
        params: float64[num_params]
    """
    cameras = {}
    with open(path, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cameras):
            camera_id = struct.unpack("<I", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]

            model_name = CAMERA_MODEL_IDS.get(model_id, f"UNKNOWN_{model_id}")
            num_params = CAMERA_MODEL_NUM_PARAMS.get(model_name, 0)
            params = np.array(
                struct.unpack(f"<{num_params}d", f.read(8 * num_params)),
                dtype=np.float64,
            )
            cameras[camera_id] = CameraModel(
                camera_id=camera_id,
                model_name=model_name,
                width=int(width),
                height=int(height),
                params=params,
            )
    logger.info(f"Read {len(cameras)} cameras from {path}")
    for cid, cam in cameras.items():
        logger.info(
            f"  Camera {cid}: {cam.model_name} {cam.width}x{cam.height} "
            f"fx={cam.fx:.2f} fy={cam.fy:.2f} cx={cam.cx:.2f} cy={cam.cy:.2f}"
        )
    return cameras


def read_images_binary(path: Path) -> Dict[int, ImageInfo]:
    """Read COLMAP images.bin file.

    Binary format:
      num_images: uint64
      per image:
        image_id: uint32
        qw, qx, qy, qz: float64[4]     (world-to-camera quaternion)
        tx, ty, tz: float64[3]           (world-to-camera translation)
        camera_id: uint32
        name: null-terminated string
        num_points2D: uint64
        per 2D point:
          x, y: float64[2]
          point3D_id: int64 (-1 if not triangulated)
    """
    images = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("<I", f.read(4))[0]
            qvec = np.array(struct.unpack("<4d", f.read(32)), dtype=np.float64)
            tvec = np.array(struct.unpack("<3d", f.read(24)), dtype=np.float64)
            camera_id = struct.unpack("<I", f.read(4))[0]

            # Read null-terminated name
            name_bytes = b""
            while True:
                ch = f.read(1)
                if ch == b"\x00" or ch == b"":
                    break
                name_bytes += ch
            name = name_bytes.decode("utf-8")

            num_points2D = struct.unpack("<Q", f.read(8))[0]
            # Each 2D point: x(f64), y(f64), point3D_id(i64) = 24 bytes
            points2D_data = f.read(24 * num_points2D)
            point3D_ids = np.zeros(num_points2D, dtype=np.int64)
            for j in range(num_points2D):
                offset = j * 24
                # Skip x, y (16 bytes), read point3D_id
                pid = struct.unpack_from("<q", points2D_data, offset + 16)[0]
                point3D_ids[j] = pid

            images[image_id] = ImageInfo(
                image_id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=name,
                point3D_ids=point3D_ids,
            )
    logger.info(f"Read {len(images)} images from {path}")
    return images


def read_points3D_binary(path: Path) -> Dict[int, Point3D]:
    """Read COLMAP points3D.bin file.

    Binary format:
      num_points3D: uint64
      per point:
        point3D_id: uint64
        x, y, z: float64[3]
        r, g, b: uint8[3]
        error: float64
        track_length: uint64
        per track element:
          image_id: uint32
          point2D_idx: uint32
    """
    points3D = {}
    with open(path, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_points):
            point3D_id = struct.unpack("<Q", f.read(8))[0]
            xyz = np.array(struct.unpack("<3d", f.read(24)), dtype=np.float64)
            rgb = np.array(struct.unpack("<3B", f.read(3)), dtype=np.uint8)
            error = struct.unpack("<d", f.read(8))[0]
            track_length = struct.unpack("<Q", f.read(8))[0]
            # Skip track data: each element is image_id(u32) + point2D_idx(u32) = 8 bytes
            f.read(8 * track_length)

            points3D[point3D_id] = Point3D(
                point3D_id=int(point3D_id),
                xyz=xyz,
                rgb=rgb,
                error=error,
                track_length=int(track_length),
            )
    logger.info(f"Read {len(points3D)} points from {path}")
    return points3D


def load_colmap_scene(sparse_dir: str | Path) -> COLMAPScene:
    """Load a complete COLMAP scene from a sparse reconstruction directory.

    Handles both standard COLMAP naming and competition-specific naming:
    - cameras.bin → camera intrinsics
    - images.bin → image poses (falls back to frames.bin)
    - points3D.bin → 3D points

    Args:
        sparse_dir: Path to sparse/0/ directory containing COLMAP binary files.

    Returns:
        COLMAPScene with cameras, images, and points3D.
    """
    sparse_dir = Path(sparse_dir)

    # Read cameras
    cameras_path = sparse_dir / "cameras.bin"
    if not cameras_path.exists():
        raise FileNotFoundError(f"cameras.bin not found in {sparse_dir}")
    cameras = read_cameras_binary(cameras_path)

    # Read images (try images.bin first, fall back to frames.bin)
    images_path = sparse_dir / "images.bin"
    if not images_path.exists():
        images_path = sparse_dir / "frames.bin"
        if not images_path.exists():
            raise FileNotFoundError(
                f"Neither images.bin nor frames.bin found in {sparse_dir}"
            )
        logger.info("Using frames.bin instead of images.bin")
    images = read_images_binary(images_path)

    # Read points3D
    points_path = sparse_dir / "points3D.bin"
    if not points_path.exists():
        logger.warning(f"points3D.bin not found in {sparse_dir}, using empty point cloud")
        points3D = {}
    else:
        points3D = read_points3D_binary(points_path)

    scene = COLMAPScene(cameras=cameras, images=images, points3D=points3D)
    logger.info(
        f"Loaded COLMAP scene: {scene.num_images} images, "
        f"{len(cameras)} cameras, {scene.num_points} points"
    )
    return scene


# ---------------------------------------------------------------------------
# Test CSV parser
# ---------------------------------------------------------------------------
@dataclass
class TestPose:
    """A single test camera pose from test_poses.csv."""
    image_name: str
    qvec: np.ndarray   # (qw, qx, qy, qz) — world-to-camera quaternion
    tvec: np.ndarray   # (tx, ty, tz) — world-to-camera translation
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def w2c_matrix(self) -> np.ndarray:
        """4x4 world-to-camera transformation matrix."""
        R = qvec_to_rotmat(self.qvec)
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = R
        w2c[:3, 3] = self.tvec
        return w2c

    def K(self) -> np.ndarray:
        """3x3 intrinsic matrix."""
        return np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

    @property
    def output_name(self) -> str:
        """Output filename (PNG extension)."""
        stem = Path(self.image_name).stem
        return f"{stem}.png"


def load_test_poses(csv_path: str | Path) -> List[TestPose]:
    """Parse test_poses.csv file.

    Expected columns:
      image_name, qw, qx, qy, qz, tx, ty, tz, fx, fy, cx, cy, width, height

    Returns:
        List of TestPose objects.
    """
    csv_path = Path(csv_path)
    poses = []
    with open(csv_path, "r") as f:
        header = f.readline().strip()
        expected_cols = "image_name,qw,qx,qy,qz,tx,ty,tz,fx,fy,cx,cy,width,height"
        if header != expected_cols:
            logger.warning(
                f"CSV header mismatch.\n  Expected: {expected_cols}\n  Got: {header}"
            )

        for line_num, line in enumerate(f, start=2):
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 14:
                logger.warning(f"Skipping malformed line {line_num}: {line}")
                continue

            image_name = parts[0].strip()
            qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
            fx, fy = float(parts[8]), float(parts[9])
            cx, cy = float(parts[10]), float(parts[11])
            width, height = int(parts[12]), int(parts[13])

            poses.append(TestPose(
                image_name=image_name,
                qvec=np.array([qw, qx, qy, qz], dtype=np.float64),
                tvec=np.array([tx, ty, tz], dtype=np.float64),
                fx=fx, fy=fy, cx=cx, cy=cy,
                width=width, height=height,
            ))

    logger.info(f"Loaded {len(poses)} test poses from {csv_path}")
    return poses


# ---------------------------------------------------------------------------
# CLI for standalone testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Parse COLMAP binary files")
    parser.add_argument("sparse_dir", type=str, help="Path to sparse/0/ directory")
    parser.add_argument("--test-csv", type=str, default=None,
                        help="Optional: path to test_poses.csv")
    args = parser.parse_args()

    scene = load_colmap_scene(args.sparse_dir)
    print(f"\nScene summary:")
    print(f"  Cameras: {len(scene.cameras)}")
    print(f"  Images:  {scene.num_images}")
    print(f"  Points:  {scene.num_points}")

    # Print first 3 images
    img_list = scene.get_image_list()[:3]
    for img in img_list:
        cam = scene.cameras[img.camera_id]
        print(f"\n  Image {img.image_id}: {img.name}")
        print(f"    Camera {img.camera_id}: {cam.model_name}")
        print(f"    qvec: {img.qvec}")
        print(f"    tvec: {img.tvec}")
        print(f"    center: {img.camera_center}")

    # Point cloud stats
    pts, colors = scene.get_points_array()
    if len(pts) > 0:
        print(f"\n  Point cloud bounds:")
        print(f"    min: {pts.min(axis=0)}")
        print(f"    max: {pts.max(axis=0)}")
        print(f"    center: {pts.mean(axis=0)}")

    if args.test_csv:
        test_poses = load_test_poses(args.test_csv)
        print(f"\n  Test poses: {len(test_poses)}")
        if test_poses:
            tp = test_poses[0]
            print(f"    First: {tp.image_name} {tp.width}x{tp.height}")
            print(f"    qvec: {tp.qvec}")
            print(f"    tvec: {tp.tvec}")
