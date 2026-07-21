# BTS Digital Twin — Novel View Synthesis

**Viettel AI Race 2026**

An end-to-end 3D Gaussian Splatting (3DGS) pipeline that reconstructs BTS (Base Transceiver Station) scenes from drone/handheld RGB images and synthesizes photorealistic novel views at unseen camera poses. The pipeline is optimized for **image quality** (PSNR, SSIM, LPIPS) and runs on NVIDIA L4 GPUs via [Modal](https://modal.com).

---

## Pipeline Overview

```
COLMAP data ──▶ gsplat training (30k iters) ──▶ render test poses ──▶ submission.zip
     │                    │                            │
     ▼                    ▼                            ▼
 cameras.bin        checkpoint.pt              <scene>/<image>.png
 images.bin         (persisted on Modal)
 points3D.bin
```

Key techniques:
- **Mip-Splatting** — alias-free rendering across varying focal lengths / resolutions.
- **Pixel-GS / AbsGrad densification** — better Gaussian placement on thin structures (antennas, cabling).
- **Appearance embeddings** — per-image latent codes to handle exposure / lighting variation in outdoor captures.
- **Depth regularization** — reduces floaters at extrapolated test viewpoints.

---

## Environment Requirements

| Package | Version |
|---------|---------|
| Python | 3.11 |
| PyTorch | 2.4.0 + CUDA 12.1 |
| gsplat | 1.5.0 |
| modal | latest |
| numpy | ≥ 1.24 |
| Pillow | ≥ 10.0 |
| imageio | ≥ 2.31 |
| tqdm | ≥ 4.65 |
| pyyaml | ≥ 6.0 |
| lpips | ≥ 0.1.4 |
| torchmetrics | ≥ 1.0 |
| scikit-learn | ≥ 1.3 |
| opencv-python-headless | ≥ 4.8 |

### Quick install

```bash
# Create and activate environment
conda create -n bts-nvs python=3.11 -y
conda activate bts-nvs

# PyTorch + CUDA 12.1
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# gsplat
pip install gsplat==1.5.0

# Other dependencies
pip install numpy Pillow imageio tqdm pyyaml lpips torchmetrics scikit-learn opencv-python-headless modal
```

---

## Data Preparation

### Expected directory structure

```
datasets/
├── HCM0421/
│   ├── train/
│   │   ├── images/
│   │   │   ├── DJI_xxx.JPG
│   │   │   └── ...
│   │   └── sparse/
│   │       └── 0/
│   │           ├── cameras.bin
│   │           ├── images.bin
│   │           └── points3D.bin
│   └── test/
│       └── test_poses.csv
├── HCM0539/
│   └── ...
└── ...
```

### `test_poses.csv` format

| Column | Description |
|--------|-------------|
| `image_name` | Target output filename (e.g. `DJI_xxx.JPG`) |
| `qw,qx,qy,qz` | Quaternion rotation (COLMAP world-to-camera convention) |
| `tx,ty,tz` | Translation vector (COLMAP world-to-camera convention) |
| `fx,fy` | Focal lengths in pixels |
| `cx,cy` | Principal point in pixels |
| `width,height` | Required output image dimensions |

Rendered images must match the `width` and `height` exactly, saved as PNG with the same base name (`.png` extension).

---

## Command Sequence

### 1. Authenticate with Modal

```bash
modal setup
```

### 2. Upload datasets to Modal volume

```bash
modal run modal_app.py::upload_data
```

This uploads the `datasets/` directory to a persistent Modal volume so GPU workers can access it.

### 3. Train a single scene

```bash
modal run modal_app.py::train_scene --scene-name HCM0421
```

Trains a 3DGS model for one scene on an L4 GPU. Checkpoints are saved to the Modal volume every 5,000 iterations.

### 4. Train all scenes

```bash
modal run modal_app.py::train_all_scenes
```

Trains all scenes in parallel (one L4 GPU per scene). Equivalent to running `train_scene` for each subdirectory in `datasets/`.

### 5. Render test poses

```bash
modal run modal_app.py::render_all_scenes
```

Loads the final checkpoint for each scene and renders all test poses from `test_poses.csv`. Output images are saved to the Modal volume under `renders/<scene>/`.

### 6. Validate and package submission

```bash
modal run modal_app.py::package_submission
```

Downloads renders from the Modal volume, validates them against `test_poses.csv`, and creates `submission.zip`. Fails loudly if any scene has missing/extra files or wrong dimensions.

You can also validate locally:

```bash
python scripts/package_submission.py \
    --renders_dir renders/ \
    --data_dir datasets/ \
    --output submission.zip
```

Or validate without creating the zip:

```bash
python scripts/package_submission.py \
    --renders_dir renders/ \
    --data_dir datasets/ \
    --validate-only
```

---

## Local Development

For testing the pipeline locally without Modal (e.g., on a workstation with a GPU):

### Train locally

```bash
# Set PYTHONPATH so python can locate scripts/
$env:PYTHONPATH="."  # Windows Powershell
# export PYTHONPATH="."  # Linux/macOS

python scripts/train.py \
    --scene-dir datasets/HCM0421 \
    --result-dir checkpoints/HCM0421 \
    --config configs/default.yaml
```

### Render test poses locally

```bash
python scripts/render_test_poses.py single \
    --scene-dir datasets/HCM0421 \
    --checkpoint checkpoints/HCM0421/checkpoints/ckpt_030000.pt \
    --output-dir renders/HCM0421 \
    --config configs/default.yaml
```

### Validate submission locally

```bash
python scripts/package_submission.py \
    --renders_dir renders/ \
    --data_dir datasets/ \
    --validate-only
```

---

## Troubleshooting

### CUDA out of memory during densification

The default config allows up to 5M Gaussians. If VRAM is exhausted on the L4 (24 GB):
- Reduce `max_gaussians` in `configs/default.yaml`.
- Lower `densify_stop_iter` to halt densification earlier.
- The pipeline uses `gsplat`'s memory-efficient rasterizer, which should handle most scenes within budget.

### Wrong image dimensions in submission

The packaging script validates every rendered image against `width` and `height` from `test_poses.csv`. If dimensions don't match:
- Check that the rendering script reads `fx, fy, cx, cy, width, height` from the CSV and uses them directly — do **not** hard-code resolution.
- Ensure no accidental resizing or cropping in the rendering pipeline.

### COLMAP pose convention mismatch

Both `images.bin` (training poses) and `test_poses.csv` use **COLMAP's world-to-camera** convention: the quaternion `(qw, qx, qy, qz)` and translation `(tx, ty, tz)` transform a world point into camera coordinates. If you see mirrored or rotated renders, verify that:
- The quaternion-to-rotation conversion matches COLMAP's convention.
- No accidental double-inversion of the camera extrinsics.

### Modal container times out

Default Modal timeout may be insufficient for 30k-iteration training. Ensure:
- The Modal function has an appropriate `timeout` setting (e.g., 7200 seconds for a single scene).
- Checkpointing (`checkpoint_every: 5000`) is enabled so training can resume.

### Submission invalidated — missing scenes

The competition invalidates the **entire** submission if any scene is missing. Always run the validation step before uploading. The `package_submission.py` script will `sys.exit(1)` if any scene fails checks.

---

## Project Structure

```
AI-RACE26/
├── AGENT.md                    # Agent instructions & constraints
├── README.md                   # This file
├── configs/
│   └── default.yaml            # Default training hyperparameters
├── datasets/                   # Per-scene COLMAP data + test poses
│   ├── HCM0421/
│   ├── HCM0539/
│   └── ...
├── scripts/
│   ├── __init__.py
│   ├── train.py                # Training script
│   ├── render_test_poses.py    # Test-pose rendering script
│   └── package_submission.py   # Submission validation & packaging
├── modal_app.py                # Modal application (GPU training/rendering)
├── renders/                    # Rendered test-pose images (generated)
├── checkpoints/                # Model checkpoints (generated)
├── logs/                       # Training logs (generated)
└── submission.zip              # Final submission (generated)
```

---

## License

This project is developed for the Viettel AI Race 2026 competition. All code is proprietary and intended solely for competition use. The datasets are provided by the competition organizers and subject to their terms of use.
