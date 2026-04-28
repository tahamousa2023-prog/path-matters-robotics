import argparse, os, subprocess
from pathlib import Path

def is_scene_dir(p: Path) -> bool:
    # skip junk folders like _batch_run, _logs_*, etc.
    if p.name.startswith("_"):
        return False
    return (p / "images").is_dir()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--out_name", default="recon_outputs")  # created inside each scene
    ap.add_argument("--max_images", type=int, default=120)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    scenes = [p for p in sorted(dataset_root.iterdir()) if p.is_dir() and is_scene_dir(p)]

    for scene in scenes:
        img_dir = scene / "images"
        out_dir = scene / args.out_name
        vggt_ply = out_dir / "vggt" / "points.ply"
        fast3r_ply = out_dir / "fast3r" / "points.ply"
        os.makedirs(vggt_ply.parent, exist_ok=True)
        os.makedirs(fast3r_ply.parent, exist_ok=True)

        if not vggt_ply.exists():
            subprocess.check_call([
                "python", "/home/AP_PathMatters/vggt/export_vggt_ply.py",
                "--image_dir", str(img_dir),
                "--out_ply", str(vggt_ply),
                "--max_images", str(args.max_images),
                "--stride", str(args.stride),
            ])

        if not fast3r_ply.exists():
            subprocess.check_call([
                "python", "/home/AP_PathMatters/fast3r/export_fast3r_ply.py",
                "--image_dir", str(img_dir),
                "--out_ply", str(fast3r_ply),
                "--max_images", str(args.max_images),
                "--stride", str(args.stride),
                "--image_size", "512",
            ])

    # Now run your evaluator + plots.
    # This evaluator uses global init (FPFH+RANSAC) then robust point-to-plane ICP.
    # Open3D: global registration + robust kernels docs:
    # - global init: https://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html
    # - robust kernels only for PointToPlane ICP: https://www.open3d.org/docs/release/tutorial/pipelines/robust_kernels.html

    subprocess.check_call([
        "python", "/home/AP_PathMatters/vggt/Last_attempt/compare_all_scenes_and_plot.py",
        "--dataset_root", str(dataset_root),
        "--methods", "vggt", "fast3r",
        "--out_dir", "/home/AP_PathMatters/vggt/Last_attempt/runs/reallife_compare",
        "--method_pattern",
        f"vggt=**/{args.out_name}/vggt/points.ply",
        f"fast3r=**/{args.out_name}/fast3r/points.ply",
        "--mesh_sample_points", "1000000",
        "--max_eval_points", "1500000",
    ])

if __name__ == "__main__":
    main()