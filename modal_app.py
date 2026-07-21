"""
Modal application orchestrator for BTS Digital Twin Novel View Synthesis.

Manages data upload, GPU training (L4), batch rendering of test poses,
and submission packaging/validation.
"""

import os
import shutil
from pathlib import Path
import modal

# Define the Modal App
app = modal.App("bts-digital-twin")

# Define Persistent Volumes
data_volume = modal.Volume.from_name("bts-data", create_if_missing=True)
results_volume = modal.Volume.from_name("bts-results", create_if_missing=True)

# Build custom container image with CUDA development tools & gsplat dependencies
# In Modal 1.0+, local directory inclusion is done using .add_local_dir()
image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        "numpy<2.0.0",
        "Pillow",
        "imageio",
        "tqdm",
        "pyyaml",
        "lpips",
        "torchmetrics",
        "scikit-learn",
        "opencv-python-headless",
    )
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .pip_install("gsplat==1.4.0", extra_index_url="https://download.pytorch.org/whl/cu121")
    .add_local_dir("./scripts", "/root/scripts")
    .add_local_dir("./configs", "/root/configs")
)

# Lightweight image for uploading datasets (mounts datasets folder at runtime)
upload_image = modal.Image.debian_slim().add_local_dir("./datasets", "/root/datasets")


# ---------------------------------------------------------------------------
# Stage 1: Data Upload
# ---------------------------------------------------------------------------
@app.function(
    image=upload_image,
    volumes={"/data": data_volume},
    timeout=1800,  # 30 minutes
)
def upload_data():
    """Upload local datasets directory to persistent Volume `/data`."""
    print("Uploading datasets from local workspace Mount to Volume `/data`...")
    dest_dir = Path("/data/datasets")
    
    # Copy from the mounted directory to the volume
    if dest_dir.exists():
        print(f"Destination {dest_dir} already exists. Merging/overwriting...")
    
    shutil.copytree("/root/datasets", dest_dir, dirs_exist_ok=True)
    
    # Commit changes to Volume
    data_volume.commit()
    print("Data upload completed successfully and Volume committed.")


# ---------------------------------------------------------------------------
# Stage 2: Scene Training (GPU L4)
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="A10G",
    volumes={"/data": data_volume, "/results": results_volume},
    timeout=10800,  # 3 hours
)
def train_scene(scene_name: str, config_name: str = "default.yaml", force_retrain: bool = False):
    """Train 3DGS model for a single scene on a L4 GPU."""
    import sys
    sys.path.append("/root")
    from scripts.train import train_scene as run_train
    
    scene_dir = f"/data/datasets/{scene_name}"
    result_dir = f"/results/runs/{scene_name}"
    config_path = f"/root/configs/{config_name}"
    
    if not os.path.exists(scene_dir):
        raise FileNotFoundError(f"Scene data not found: {scene_dir}. Please upload data first.")
        
    print(f"Starting training for scene: {scene_name}")
    
    # Callback to commit checkpoints to the persistent volume periodically
    def commit_callback():
        print("Commit callback triggered: flushing checkpoints to persistent volume...")
        results_volume.commit()

    run_train(
        scene_dir=scene_dir,
        result_dir=result_dir,
        config_path=config_path,
        resume=not force_retrain,
        device="cuda",
        checkpoint_callback=commit_callback,
    )
    
    # Final commit
    results_volume.commit()
    print(f"Scene {scene_name} training and final commit completed successfully!")


# ---------------------------------------------------------------------------
# Stage 3: Train All Scenes Orchestrator
# ---------------------------------------------------------------------------
@app.function(
    volumes={"/data": data_volume},
    timeout=28800,  # 8 hours
)
def train_all_scenes(config_name: str = "default.yaml", force_retrain: bool = False):
    """Orchestrate training of all scenes in parallel."""
    import sys
    
    # Find all scenes in volume
    datasets_dir = Path("/data/datasets")
    if not datasets_dir.exists():
        raise FileNotFoundError("Datasets folder not found on volume. Run upload_data first.")
        
    scenes = sorted([
        d.name for d in datasets_dir.iterdir()
        if d.is_dir() and (d / "test" / "test_poses.csv").exists()
    ])
    
    print(f"Found {len(scenes)} scenes to train: {scenes}")
    
    # Launch training for all scenes concurrently (non-blocking spawn)
    print(f"Spawning parallel training containers for {len(scenes)} scenes...")
    calls = []
    for scene in scenes:
        try:
            calls.append((scene, train_scene.spawn(
                scene_name=scene,
                config_name=config_name,
                force_retrain=force_retrain
            )))
        except Exception as e:
            print(f"Failed to spawn training for scene {scene}: {e}")
            
    # Collect results (orchestrator waits, but all containers train in parallel)
    print(f"Waiting for {len(calls)} scenes to finish in parallel...")
    failed_scenes = []
    for scene, call in calls:
        try:
            result = call.get()
            print(f"Scene {scene} completed successfully: {result}")
        except Exception as e:
            print(f"Error training scene {scene}: {e}")
            failed_scenes.append(scene)
            
    if failed_scenes:
        print(f"Retrying {len(failed_scenes)} failed scenes...")
        retry_calls = []
        for scene in failed_scenes:
            retry_calls.append((scene, train_scene.spawn(
                scene_name=scene,
                config_name=config_name,
                force_retrain=force_retrain
            )))
        for scene, call in retry_calls:
            try:
                call.get()
                print(f"Scene {scene} retry completed successfully.")
            except Exception as e:
                print(f"Scene {scene} retry failed again: {e}")
            
    print("All scene training loops completed.")


# ---------------------------------------------------------------------------
# Stage 4: Test Pose Batch Rendering (GPU L4)
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="A10G",
    volumes={"/data": data_volume, "/results": results_volume},
    timeout=3600,  # 1 hour
)
def render_all_scenes(config_name: str = "default.yaml"):
    """Render test poses for all scenes from the latest checkpoints."""
    import sys
    sys.path.append("/root")
    from scripts.render_test_poses import render_all_scenes as run_renders
    
    data_dir = "/data/datasets"
    results_dir = "/results/runs"
    renders_dir = "/results/renders"
    config_path = f"/root/configs/{config_name}"
    
    print("Cleaning up old renders from volume to prevent stale extra files...")
    import shutil
    shutil.rmtree(renders_dir, ignore_errors=True)
    
    print("Starting rendering process for all test poses...")
    
    run_renders(
        data_dir=data_dir,
        results_dir=results_dir,
        renders_dir=renders_dir,
        config_path=config_path,
        device="cuda"
    )
    
    results_volume.commit()
    print("Rendering of all test poses completed and Volume committed.")


# ---------------------------------------------------------------------------
# Stage 5: Package & Validate Submission
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    volumes={"/data": data_volume, "/results": results_volume},
    timeout=1200,  # 20 minutes
)
def package_submission():
    """Validate renders against test_poses.csv and build submission.zip."""
    import sys
    import subprocess
    sys.path.append("/root")
    
    renders_dir = "/results/renders"
    data_dir = "/data/datasets"
    output_zip = "/results/submission.zip"
    
    # Run the package_submission.py utility script
    print("Running packaging & validation script...")
    
    cmd = [
        "python", "/root/scripts/package_submission.py",
        "--renders_dir", renders_dir,
        "--data_dir", data_dir,
        "--output", output_zip
    ]
    
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    
    if res.returncode != 0:
        print(res.stderr)
        raise RuntimeError("Submission validation failed! Please check missing/faulty files.")
        
    results_volume.commit()
    print(f"Validation passed and submission packaged at: {output_zip}")


# ---------------------------------------------------------------------------
# Stage 6: Quick Clean (Remove old DJI_*.png renders)
# ---------------------------------------------------------------------------
@app.function(
    image=upload_image,
    volumes={"/results": results_volume},
)
def clean_extra():
    """Completely wipe the renders directory to avoid mixed files."""
    import shutil
    renders_dir = "/results/renders"
    print(f"Completely wiping {renders_dir}...")
    shutil.rmtree(renders_dir, ignore_errors=True)
    results_volume.commit()
    print("Cleaned!")


# ---------------------------------------------------------------------------
# Local Entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(action: str, scene: str = None, force_retrain: bool = False):
    """
    Main orchestrator entrypoint.
    
    Usage:
      modal run modal_app.py --action upload
      modal run modal_app.py --action train --scene HCM0421
      modal run modal_app.py --action train_all
      modal run modal_app.py --action render
      modal run modal_app.py --action package
      modal run modal_app.py --action clean
    """
    if action == "upload":
        upload_data.remote()
    elif action == "train":
        if not scene:
            print("Please specify a scene name: --scene <scene_name>")
            return
        train_scene.remote(scene_name=scene, force_retrain=force_retrain)
    elif action == "train_all":
        train_all_scenes.remote(force_retrain=force_retrain)
    elif action == "render":
        render_all_scenes.remote()
    elif action == "package":
        package_submission.remote()
    elif action == "clean":
        clean_extra.remote()
    else:
        print(f"Unknown action: {action}")
        print("Supported actions: upload, train, train_all, render, package, clean")
