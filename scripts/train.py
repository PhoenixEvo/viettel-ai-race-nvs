import sys
import os
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
    subcommand = "mcmc" if cfg.get("use_mcmc") else "default"
    cmd = [
        sys.executable, "/root/scripts/simple_trainer.py", subcommand,
        "--data_dir", str(scene_dir),
        "--result_dir", str(result_dir),
        "--data_factor", str(data_factor),
        "--max_steps", str(cfg["max_steps"]),
        "--eval_steps", "7000", "15000", "30000", str(cfg["max_steps"]),
        "--save_steps", "7000", "15000", "30000", str(cfg["max_steps"]),
        "--disable_viewer",
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
    checkpoints_dir = Path(result_dir) / "ckpts"
    if resume and checkpoints_dir.exists():
        ckpts = list(checkpoints_dir.glob("ckpt_*_rank0.pt"))
        if ckpts:
            latest_ckpt_path = max(ckpts, key=lambda p: int(p.stem.split("_")[1]))
            cmd.extend(["--ckpt", str(latest_ckpt_path)])
            print(f"Resuming from {latest_ckpt_path}")
            
    # 3. Run subprocess
    print(f"Running command: {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parent) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(cmd, env=env)
    
    if result.returncode != 0:
        raise RuntimeError(f"simple_trainer failed with code {result.returncode}")
        
    # 4. Callback
    if checkpoint_callback is not None:
        checkpoint_callback()
        
    # 5. Return final checkpoint
    # simple_trainer saves to {result_dir}/ckpts/ckpt_{step}_rank0.pt
    ckpts_dir = Path(result_dir) / "ckpts"
    found = sorted(ckpts_dir.glob("ckpt_*_rank0.pt")) if ckpts_dir.exists() else []
    if not found:
        found = sorted(Path(result_dir).glob("ckpt_*.pt"))
    final_ckpt = found[-1] if found else ckpts_dir / f"ckpt_{cfg['max_steps']-1}_rank0.pt"
    return str(final_ckpt)
