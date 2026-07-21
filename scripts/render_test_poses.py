import os
from pathlib import Path

def render_all_scenes(
    data_dir: str,
    results_dir: str,
    renders_dir: str,
    config_path: str,
    device: str = "cuda"
):
    from scripts.colmap_parser import load_test_poses
    from scripts.simple_trainer import render_from_checkpoint
    
    data_path = Path(data_dir)
    results_path = Path(results_dir)
    renders_path = Path(renders_dir)
    
    scenes = sorted([
        d.name for d in data_path.iterdir() 
        if d.is_dir() and (d / "test" / "test_poses.csv").exists()
    ])
    
    total_images = 0
    processed_scenes = 0
    for scene in scenes:
        scene_results = results_path / scene
        
        # Check both save patterns (with and without /checkpoints/ subdir)
        ckpts = list((scene_results / "checkpoints").glob("ckpt_*.pt"))
        if not ckpts:
            ckpts = list(scene_results.glob("ckpt_*.pt"))
            
        if not ckpts:
            print(f"No checkpoints found for {scene} in {scene_results}")
            continue
            
        latest_ckpt = max(ckpts, key=lambda p: int(p.stem.split("_")[1].split(".")[0].split("-")[0]))
        
        test_csv = data_path / scene / "test" / "test_poses.csv"
        test_poses = load_test_poses(test_csv)
        
        out_dir = renders_path / scene
        print(f"Rendering {scene} from {latest_ckpt}...")
        render_from_checkpoint(str(latest_ckpt), test_poses, str(out_dir), device)
        
        total_images += len(test_poses)
        processed_scenes += 1
        
    print(f"Summary: processed {processed_scenes}/{len(scenes)} scenes, rendered {total_images} images.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--renders_dir", required=True)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    render_all_scenes(
        args.data_dir,
        args.results_dir,
        args.renders_dir,
        args.config_path,
        args.device
    )
