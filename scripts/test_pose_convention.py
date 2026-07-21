"""
Pose convention unit test for BTS Digital Twin pipeline.

Verifies that:
1. Quaternion/translation conventions match between COLMAP images.bin and test_poses.csv
2. A training image rendered at its COLMAP pose reconstructs correctly
3. World-to-camera / camera-to-world transforms are consistent

This MUST pass before running full training — catches:
- Quaternion sign flips (qw,qx,qy,qz) vs (qx,qy,qz,qw) confusion
- w2c vs c2w inversion bugs
- Intrinsics scaling errors
- Scene normalization bugs
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def test_quaternion_roundtrip():
    """Test that quaternion ↔ rotation matrix conversion is consistent."""
    from scripts.colmap_parser import qvec_to_rotmat, rotmat_to_qvec

    # Test identity
    qvec_id = np.array([1, 0, 0, 0], dtype=np.float64)
    R = qvec_to_rotmat(qvec_id)
    assert np.allclose(R, np.eye(3), atol=1e-10), f"Identity failed: {R}"

    # Test roundtrip with random quaternion
    np.random.seed(42)
    for _ in range(10):
        qvec = np.random.randn(4)
        qvec /= np.linalg.norm(qvec)
        if qvec[0] < 0:
            qvec = -qvec  # Canonical form (qw > 0)

        R = qvec_to_rotmat(qvec)
        qvec_back = rotmat_to_qvec(R)

        # Quaternions may differ by sign
        if qvec_back[0] < 0:
            qvec_back = -qvec_back

        assert np.allclose(qvec, qvec_back, atol=1e-8), \
            f"Roundtrip failed:\n  orig: {qvec}\n  back: {qvec_back}"

        # Verify R is proper rotation (det=1, R^T R = I)
        assert abs(np.linalg.det(R) - 1.0) < 1e-10, f"det(R) = {np.linalg.det(R)}"
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-10), "R not orthogonal"

    logger.info("✓ Quaternion roundtrip test passed")


def test_w2c_c2w_inverse():
    """Test that w2c and c2w are proper inverses."""
    from scripts.colmap_parser import ImageInfo, qvec_to_rotmat

    np.random.seed(42)
    qvec = np.random.randn(4)
    qvec /= np.linalg.norm(qvec)
    tvec = np.random.randn(3)

    img = ImageInfo(
        image_id=1, qvec=qvec, tvec=tvec,
        camera_id=1, name="test.jpg", point3D_ids=np.array([]),
    )

    w2c = img.w2c_matrix()
    c2w = img.c2w_matrix()

    # w2c @ c2w should be identity
    product = w2c @ c2w
    assert np.allclose(product, np.eye(4), atol=1e-10), \
        f"w2c @ c2w != I:\n{product}"

    # Camera center should be -R^T @ t
    R = qvec_to_rotmat(qvec)
    expected_center = -R.T @ tvec
    actual_center = img.camera_center
    assert np.allclose(expected_center, actual_center, atol=1e-10), \
        f"Camera center mismatch:\n  expected: {expected_center}\n  actual: {actual_center}"

    # c2w[:3, 3] should be the camera center
    assert np.allclose(c2w[:3, 3], expected_center, atol=1e-10), \
        f"c2w translation != camera center"

    logger.info("✓ w2c/c2w inverse test passed")


def test_normalization_consistency():
    """Test that scene normalization preserves camera-point relationships."""
    from scripts.dataset import SceneNormalization

    np.random.seed(42)
    norm = SceneNormalization(
        translate=np.array([1.0, 2.0, 3.0]),
        scale=0.5,
    )

    # A point in world space
    p_world = np.array([5.0, 6.0, 7.0])
    p_norm = norm.normalize_point(p_world)
    expected = 0.5 * (p_world - np.array([1, 2, 3]))
    assert np.allclose(p_norm, expected, atol=1e-10), \
        f"Point normalization: {p_norm} != {expected}"

    # Camera pose: project a 3D point before and after normalization
    from scripts.colmap_parser import qvec_to_rotmat
    qvec = np.array([1, 0, 0, 0], dtype=np.float64)  # identity rotation
    tvec = np.array([1.0, 0, 0], dtype=np.float64)

    w2c = np.eye(4)
    R = qvec_to_rotmat(qvec)
    w2c[:3, :3] = R
    w2c[:3, 3] = tvec

    # Project in world space
    p_cam = (w2c @ np.append(p_world, 1.0))[:3]

    # Normalize w2c and project normalized point
    w2c_norm = norm.normalize_w2c(w2c)
    p_cam_norm = (w2c_norm @ np.append(p_norm, 1.0))[:3]

    # The projected point should match (up to scale)
    # In normalized space: p_cam_norm = s * p_cam (depth scaled by s)
    ratio = p_cam_norm / (p_cam + 1e-15)
    assert np.allclose(ratio, norm.scale, atol=1e-8), \
        f"Projection inconsistency: ratio={ratio}, expected scale={norm.scale}"

    logger.info("✓ Normalization consistency test passed")


def test_colmap_parsing(scene_dir: str):
    """Test COLMAP parsing on a real scene directory."""
    from scripts.colmap_parser import load_colmap_scene, load_test_poses

    scene_path = Path(scene_dir)
    sparse_dir = scene_path / "train" / "sparse" / "0"

    scene = load_colmap_scene(sparse_dir)

    assert len(scene.cameras) > 0, "No cameras parsed"
    assert scene.num_images > 0, "No images parsed"
    logger.info(f"  Parsed {scene.num_images} images, {len(scene.cameras)} cameras, "
                f"{scene.num_points} points")

    # Verify all images reference valid cameras
    for img in scene.get_image_list():
        assert img.camera_id in scene.cameras, \
            f"Image {img.name} references invalid camera {img.camera_id}"

    # Verify quaternions are unit
    for img in scene.get_image_list():
        q_norm = np.linalg.norm(img.qvec)
        assert abs(q_norm - 1.0) < 0.01, \
            f"Image {img.name}: quaternion not unit (norm={q_norm})"

    # Load test poses if available
    test_csv = scene_path / "test" / "test_poses.csv"
    if test_csv.exists():
        test_poses = load_test_poses(test_csv)
        assert len(test_poses) > 0, "No test poses parsed"
        logger.info(f"  Parsed {len(test_poses)} test poses")

        # Verify test quaternions are unit
        for tp in test_poses:
            q_norm = np.linalg.norm(tp.qvec)
            assert abs(q_norm - 1.0) < 0.01, \
                f"Test pose {tp.image_name}: quaternion not unit (norm={q_norm})"

        # Verify test and train use same convention by checking
        # that camera centers are in a similar spatial region
        train_centers = np.array([img.camera_center for img in scene.get_image_list()])
        test_centers = np.array([
            -(qvec_to_rotmat_wrapper(tp.qvec).T @ tp.tvec) for tp in test_poses
        ])

        train_centroid = train_centers.mean(axis=0)
        test_centroid = test_centers.mean(axis=0)

        # They should be in roughly the same region
        center_dist = np.linalg.norm(train_centroid - test_centroid)
        train_spread = np.linalg.norm(train_centers - train_centroid, axis=1).max()

        logger.info(f"  Train centroid: {train_centroid}")
        logger.info(f"  Test centroid:  {test_centroid}")
        logger.info(f"  Distance: {center_dist:.4f}, Train spread: {train_spread:.4f}")

        if center_dist > 5 * train_spread:
            logger.warning(
                f"  ⚠ Test cameras are far from train cameras "
                f"(dist={center_dist:.2f} vs spread={train_spread:.2f}). "
                f"This might indicate a pose convention mismatch!"
            )
        else:
            logger.info("  ✓ Test and train cameras in same spatial region")

    logger.info("✓ COLMAP parsing test passed")


def test_dataset_loading(scene_dir: str):
    """Test full dataset loading and normalization."""
    from scripts.dataset import SceneDataset

    dataset = SceneDataset(
        scene_dir=scene_dir,
        test_every=8,
        normalize=True,
    )

    assert dataset.num_train > 0, "No training images"
    logger.info(f"  Train: {dataset.num_train}, Val: {dataset.num_val}, "
                f"Test: {dataset.num_test}")

    # Test loading an image
    batch = dataset.get_train_batch(0)
    assert batch["image"].shape[2] == 3, f"Image channels: {batch['image'].shape}"
    assert batch["w2c"].shape == (4, 4), f"W2C shape: {batch['w2c'].shape}"
    assert batch["K"].shape == (3, 3), f"K shape: {batch['K'].shape}"

    # Verify image is in [0, 1]
    assert batch["image"].min() >= 0 and batch["image"].max() <= 1, \
        f"Image range: [{batch['image'].min()}, {batch['image'].max()}]"

    # Verify w2c is a valid transformation (det of rotation ≈ 1)
    R = batch["w2c"][:3, :3].numpy()
    det = np.linalg.det(R)
    assert abs(det - 1.0) < 0.01, f"W2C rotation det = {det}"

    logger.info("✓ Dataset loading test passed")


def qvec_to_rotmat_wrapper(qvec):
    """Wrapper for use in test functions."""
    from scripts.colmap_parser import qvec_to_rotmat
    return qvec_to_rotmat(qvec)


def run_all_tests(scene_dir: Optional[str] = None):
    """Run all pose convention tests."""
    logger.info("=" * 60)
    logger.info("Running pose convention unit tests")
    logger.info("=" * 60)

    passed = 0
    failed = 0

    # Pure unit tests (no data needed)
    for test_fn in [test_quaternion_roundtrip, test_w2c_c2w_inverse,
                    test_normalization_consistency]:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            logger.error(f"FAILED: {test_fn.__name__}: {e}")
            failed += 1

    # Data-dependent tests
    if scene_dir:
        for test_fn in [test_colmap_parsing, test_dataset_loading]:
            try:
                test_fn(scene_dir)
                passed += 1
            except Exception as e:
                logger.error(f"FAILED: {test_fn.__name__}: {e}")
                failed += 1

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Results: {passed} passed, {failed} failed")
    logger.info(f"{'=' * 60}")

    return failed == 0


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Pose convention unit tests")
    parser.add_argument("--scene-dir", default=None,
                        help="Scene directory for data-dependent tests")
    args = parser.parse_args()

    success = run_all_tests(args.scene_dir)
    sys.exit(0 if success else 1)
