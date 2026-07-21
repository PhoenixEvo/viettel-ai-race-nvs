import sys
import subprocess
from pathlib import Path
import yaml
from PIL import Image

def train_scene(
    scene_dir: str,
    result_dir: str,
    config_path: str,
    resume: bool = False,
    device: str = "cuda",
    checkpoint_callback=None,
):
    """Thin wrapper around simple_trainer.py"""
    
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # 1. Compute data_factor
    train_images_dir = Path(scene_dir) / "train" / "images"
    image_files = list(train_images_dir.glob("*.jpg")) + list(train_images_dir.glob("*.JPG")) + \
                  list(train_images_dir.glob("*.png")) + list(train_images_dir.glob("*.PNG"))
    
    if not image_files:
        raise FileNotFoundError(f"No images found in {train_images_dir}")
        
    sample_img = Image.open(image_files[0])
    w, h = sample_img.size
    max_dim = max(w, h)
    
    if max_dim > 3200:
        data_factor = 8
    elif max_dim > 1600:
        data_factor = 4
    elif max_dim > 800:
        data_factor = 2
    else:
        data_factor = 1
        
    print(f"Sample image {sample_img.size}. Using data_factor={data_factor}")
    
    # 2. Build subprocess command
    cmd = [
        sys.executable, "/root/scripts/simple_trainer.py", "default",
        "--data_dir", str(scene_dir),
        "--result_dir", str(result_dir),
        "--data_factor", str(data_factor),
        "--max_steps", str(cfg["max_steps"]),
        "--eval_steps", "7000", "15000", "30000", str(cfg["max_steps"]),
        "--save_steps", "7000", "15000", "30000", str(cfg["max_steps"]),
    ]
    
    if cfg.get("sh_degree") is not None:
        cmd.extend(["--sh_degree", str(cfg["sh_degree"])])
        
    if cfg.get("ssim_lambda") is not None:
        cmd.extend(["--ssim_lambda", str(cfg["ssim_lambda"])])
        
    if cfg.get("antialiased"):
        cmd.append("--antialiased")
        
    if cfg.get("app_opt"):
        cmd.append("--app_opt")
        
    if cfg.get("packed"):
        cmd.append("--packed")
        
    # Check for resume
    latest_ckpt_path = None
    checkpoints_dir = Path(result_dir) / "checkpoints"
    if resume and checkpoints_dir.exists():
        ckpts = list(checkpoints_dir.glob("ckpt_*.pt"))
        if ckpts:
            latest_ckpt_path = max(ckpts, key=lambda p: int(p.stem.split("_")[1].split(".")[0]))
            cmd.extend(["--ckpt_path", str(latest_ckpt_path)])
            print(f"Resuming from {latest_ckpt_path}")
            
    # 3. Run subprocess
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        raise RuntimeError(f"simple_trainer failed with code {result.returncode}")
        
    # 4. Callback
    if checkpoint_callback is not None:
        checkpoint_callback()
        
    # 5. Return final checkpoint
    final_ckpt = Path(result_dir) / "checkpoints" / f"ckpt_{cfg['max_steps']:05d}.pt"
    if not final_ckpt.exists():
        final_ckpt = Path(result_dir) / f"ckpt_{cfg['max_steps']:05d}.pt" # alternative pattern
    return str(final_ckpt)
