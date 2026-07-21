#!/usr/bin/env python3
"""
package_submission.py — Validate and package rendered images into submission.zip.

For each scene in the renders directory, this script:
  1. Reads test_poses.csv from the corresponding data directory.
  2. Verifies that every expected PNG exists with correct dimensions.
  3. Checks that no extraneous files are present.
  4. Assembles submission.zip with structure: <scene>/<image>.png

Usage:
    # Full validation + packaging
    python package_submission.py --renders_dir renders/ --data_dir datasets/ --output submission.zip

    # Validation only (no zip created)
    python package_submission.py --renders_dir renders/ --data_dir datasets/ --validate-only
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

from PIL import Image

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ExpectedImage:
    """One row parsed from test_poses.csv."""
    image_name: str  # original name from CSV (e.g. DJI_xxx.JPG)
    png_name: str    # derived PNG filename (e.g. DJI_xxx.png)
    width: int
    height: int


@dataclass
class SceneReport:
    """Validation results for a single scene."""
    scene_name: str
    expected_count: int = 0
    found_count: int = 0
    missing: List[str] = field(default_factory=list)
    extra: List[str] = field(default_factory=list)
    dimension_errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (
            not self.missing
            and not self.extra
            and not self.dimension_errors
            and self.found_count == self.expected_count
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_png_name(image_name: str) -> str:
    """Convert any image filename to its .png equivalent."""
    return Path(image_name).with_suffix(".png").name


def parse_test_poses(csv_path: Path) -> List[ExpectedImage]:
    """Parse test_poses.csv and return a list of ExpectedImage records.

    Args:
        csv_path: Path to the test_poses.csv file.

    Returns:
        List of ExpectedImage with name and expected dimensions.

    Raises:
        FileNotFoundError: If the CSV does not exist.
        ValueError: If the CSV is malformed or empty.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"test_poses.csv not found: {csv_path}")

    expected: List[ExpectedImage] = []
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)

        # Validate header
        required_cols = {"image_name", "width", "height"}
        if reader.fieldnames is None or not required_cols.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"CSV {csv_path} is missing required columns. "
                f"Expected at least {required_cols}, got {reader.fieldnames}"
            )

        for row_idx, row in enumerate(reader, start=2):  # row 2 = first data row
            image_name = row["image_name"].strip()
            if not image_name:
                raise ValueError(f"Empty image_name at row {row_idx} in {csv_path}")

            try:
                width = int(row["width"])
                height = int(row["height"])
            except (ValueError, KeyError) as exc:
                raise ValueError(
                    f"Invalid width/height at row {row_idx} in {csv_path}: {exc}"
                ) from exc

            expected.append(
                ExpectedImage(
                    image_name=image_name,
                    png_name=_to_png_name(image_name),
                    width=width,
                    height=height,
                )
            )

    if not expected:
        raise ValueError(f"test_poses.csv is empty: {csv_path}")

    return expected


def validate_scene(
    scene_name: str,
    renders_dir: Path,
    data_dir: Path,
) -> SceneReport:
    """Validate a single scene's rendered outputs against test_poses.csv.

    Args:
        scene_name: Name of the scene subdirectory.
        renders_dir: Root directory containing per-scene render folders.
        data_dir: Root directory containing per-scene dataset folders.

    Returns:
        A SceneReport describing the validation outcome.
    """
    report = SceneReport(scene_name=scene_name)

    csv_path = data_dir / scene_name / "test" / "test_poses.csv"
    scene_renders = renders_dir / scene_name

    # Parse expected images --------------------------------------------------
    try:
        expected_images = parse_test_poses(csv_path)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Scene '%s': %s", scene_name, exc)
        report.expected_count = -1
        report.missing.append(f"CSV ERROR: {exc}")
        return report

    report.expected_count = len(expected_images)
    expected_png_names = {img.png_name for img in expected_images}

    # Gather actual files ----------------------------------------------------
    if not scene_renders.is_dir():
        logger.error(
            "Scene '%s': renders directory does not exist: %s",
            scene_name,
            scene_renders,
        )
        report.missing = [img.png_name for img in expected_images]
        return report

    actual_files = {f.name for f in scene_renders.iterdir() if f.is_file()}

    # Missing / Extra --------------------------------------------------------
    report.missing = sorted(expected_png_names - actual_files)
    report.extra = sorted(actual_files - expected_png_names)
    report.found_count = len(expected_png_names & actual_files)

    if report.missing:
        logger.warning(
            "Scene '%s': %d missing file(s):\n  %s",
            scene_name,
            len(report.missing),
            "\n  ".join(report.missing[:10]),
        )
    if report.extra:
        logger.warning(
            "Scene '%s': %d extra file(s):\n  %s",
            scene_name,
            len(report.extra),
            "\n  ".join(report.extra[:10]),
        )

    # Dimension checks -------------------------------------------------------
    for img_spec in expected_images:
        img_path = scene_renders / img_spec.png_name
        if not img_path.exists():
            continue  # already captured as missing

        try:
            with Image.open(img_path) as im:
                actual_w, actual_h = im.size
        except Exception as exc:  # noqa: BLE001
            report.dimension_errors.append(
                f"{img_spec.png_name}: failed to open — {exc}"
            )
            continue

        if actual_w != img_spec.width or actual_h != img_spec.height:
            msg = (
                f"{img_spec.png_name}: expected {img_spec.width}x{img_spec.height}, "
                f"got {actual_w}x{actual_h}"
            )
            report.dimension_errors.append(msg)
            logger.warning("Scene '%s': %s", scene_name, msg)

    return report


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(reports: List[SceneReport]) -> None:
    """Print a human-readable validation summary table."""
    col_scene = "Scene"
    col_expected = "Expected"
    col_found = "Found"
    col_missing = "Missing"
    col_extra = "Extra"
    col_dim = "Dim Errs"
    col_status = "Status"

    # Compute column widths
    w_scene = max(len(col_scene), *(len(r.scene_name) for r in reports))
    w_num = 8  # width for numeric columns

    header = (
        f"  {col_scene:<{w_scene}}  "
        f"{col_expected:>{w_num}}  "
        f"{col_found:>{w_num}}  "
        f"{col_missing:>{w_num}}  "
        f"{col_extra:>{w_num}}  "
        f"{col_dim:>{w_num}}  "
        f"{col_status}"
    )
    sep = "  " + "-" * (len(header) - 2)

    print("\n" + sep)
    print(header)
    print(sep)

    for r in reports:
        status = "OK" if r.ok else "FAIL"
        status_icon = "✓" if r.ok else "✗"
        print(
            f"  {r.scene_name:<{w_scene}}  "
            f"{r.expected_count:>{w_num}}  "
            f"{r.found_count:>{w_num}}  "
            f"{len(r.missing):>{w_num}}  "
            f"{len(r.extra):>{w_num}}  "
            f"{len(r.dimension_errors):>{w_num}}  "
            f"{status_icon} {status}"
        )

    print(sep + "\n")


# ---------------------------------------------------------------------------
# Zip creation
# ---------------------------------------------------------------------------

def create_submission_zip(
    output_path: Path,
    renders_dir: Path,
    reports: List[SceneReport],
) -> None:
    """Create submission.zip containing only validated scene renders.

    Only scenes that passed validation are included.  Each file is stored
    as ``<scene_name>/<image_name>.png`` inside the archive.

    Args:
        output_path: Destination path for the zip file.
        renders_dir: Root directory with per-scene render folders.
        reports: Validation reports (only ``ok`` scenes are zipped).
    """
    ok_reports = [r for r in reports if r.ok]
    if not ok_reports:
        logger.error("No scenes passed validation — zip will NOT be created.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_files = 0
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_STORED) as zf:
        for report in ok_reports:
            scene_dir = renders_dir / report.scene_name
            for png in sorted(scene_dir.glob("*.png")):
                arcname = f"{report.scene_name}/{png.name}"
                zf.write(png, arcname)
                total_files += 1

    logger.info(
        "Created %s — %d files from %d scene(s), size %.2f MB",
        output_path,
        total_files,
        len(ok_reports),
        output_path.stat().st_size / (1024 * 1024),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        description="Validate rendered images and package submission.zip.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python package_submission.py \\\n"
            "      --renders_dir renders/ --data_dir datasets/ --output submission.zip\n\n"
            "  python package_submission.py \\\n"
            "      --renders_dir renders/ --data_dir datasets/ --validate-only\n"
        ),
    )
    parser.add_argument(
        "--renders_dir",
        type=Path,
        required=True,
        help="Directory containing per-scene rendered images (one subfolder per scene).",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Directory containing per-scene datasets (with test/test_poses.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submission.zip"),
        help="Output path for submission.zip (default: submission.zip).",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        default=False,
        help="Only validate — do not create the zip file.",
    )
    return parser


def main(argv: List[str] | None = None) -> None:
    """Entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    renders_dir: Path = args.renders_dir.resolve()
    data_dir: Path = args.data_dir.resolve()
    output: Path = args.output.resolve()
    validate_only: bool = args.validate_only

    # Sanity-check inputs
    if not renders_dir.is_dir():
        logger.error("Renders directory does not exist: %s", renders_dir)
        sys.exit(1)
    if not data_dir.is_dir():
        logger.error("Data directory does not exist: %s", data_dir)
        sys.exit(1)

    # Discover scenes (subdirectories of renders_dir)
    scene_dirs = sorted(
        [d for d in renders_dir.iterdir() if d.is_dir()],
        key=lambda p: p.name,
    )
    if not scene_dirs:
        logger.error("No scene subdirectories found in %s", renders_dir)
        sys.exit(1)

    logger.info("Found %d scene(s) in %s", len(scene_dirs), renders_dir)

    # Validate each scene
    reports: List[SceneReport] = []
    for scene_dir in scene_dirs:
        scene_name = scene_dir.name
        logger.info("Validating scene: %s", scene_name)
        report = validate_scene(scene_name, renders_dir, data_dir)
        reports.append(report)

    # Print summary
    print_summary(reports)

    # Check overall status
    all_ok = all(r.ok for r in reports)
    failed_scenes = [r.scene_name for r in reports if not r.ok]

    if not all_ok:
        logger.error(
            "VALIDATION FAILED — %d scene(s) have errors: %s",
            len(failed_scenes),
            ", ".join(failed_scenes),
        )
        # Print detailed error breakdown for each failed scene
        for r in reports:
            if not r.ok:
                if r.missing:
                    logger.error(
                        "  [%s] %d missing file(s): %s%s",
                        r.scene_name,
                        len(r.missing),
                        ", ".join(r.missing[:5]),
                        " ..." if len(r.missing) > 5 else "",
                    )
                if r.extra:
                    logger.error(
                        "  [%s] %d extra file(s): %s%s",
                        r.scene_name,
                        len(r.extra),
                        ", ".join(r.extra[:5]),
                        " ..." if len(r.extra) > 5 else "",
                    )
                if r.dimension_errors:
                    logger.error(
                        "  [%s] %d dimension error(s): %s%s",
                        r.scene_name,
                        len(r.dimension_errors),
                        "; ".join(r.dimension_errors[:3]),
                        " ..." if len(r.dimension_errors) > 3 else "",
                    )
        sys.exit(1)

    logger.info("All %d scene(s) passed validation.", len(reports))

    # Create zip if requested
    if validate_only:
        logger.info("--validate-only flag set — skipping zip creation.")
    else:
        create_submission_zip(output, renders_dir, reports)
        logger.info("Done. Submission ready: %s", output)


if __name__ == "__main__":
    main()
