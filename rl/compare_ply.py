"""
experiments/taha/compare_ply.py
Direct comparison between any two .ply files.
Usage:
    python experiments/taha/compare_ply.py \
        --recon reconstruction.ply \
        --gt    ground_truth.ply \
        --output-dir /path/to/output/
"""
import argparse, json, subprocess, sys
from pathlib import Path
import open3d as o3d

def compare(recon, gt, output_dir, auto_crop_recon=True,
            cluster_eps=0.05, min_fraction=0.05, max_fraction=0.15):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if auto_crop_recon:
        cleaned = out / 'recon_cleaned.ply'
        result = subprocess.run([
            sys.executable,
            str(Path(__file__).parent / 'auto_crop.py'),
            '--input',         str(recon),
            '--output',        str(cleaned),
            '--cluster-eps',   str(cluster_eps),
            '--min-fraction',  str(min_fraction),
            '--max-fraction',  str(max_fraction),
        ])
        if result.returncode != 0:
            print('Auto-crop failed — using raw file')
            cleaned = Path(recon)
    else:
        cleaned = Path(recon)

    src = o3d.io.read_point_cloud(str(cleaned))
    tgt = o3d.io.read_point_cloud(str(gt))
    print(f'Reconstruction: {len(src.points):,} points')
    print(f'Ground truth:   {len(tgt.points):,} points')

    voxel = max(src.get_axis_aligned_bounding_box().get_max_extent() / 50, 0.005)
    src_d = src.voxel_down_sample(voxel)
    tgt_d = tgt.voxel_down_sample(voxel)
    src_d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*4, max_nn=30))
    tgt_d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*4, max_nn=30))

    icp = o3d.pipelines.registration.registration_icp(
        src_d, tgt_d,
        max_correspondence_distance=voxel * 4,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )

    metrics = {
        'icp_fitness':     round(icp.fitness, 6),
        'icp_inlier_rmse': round(icp.inlier_rmse, 6),
        'recon_points':    len(src.points),
        'gt_points':       len(tgt.points),
        'voxel_size':      voxel,
    }

    with open(out / 'comparison_result.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    src_aligned = src.transform(icp.transformation)
    o3d.io.write_point_cloud(str(out / 'aligned_recon.ply'), src_aligned)
    o3d.io.write_point_cloud(str(out / 'ground_truth.ply'),  tgt)

    print(f'\nICP fitness:     {metrics["icp_fitness"]}')
    print(f'ICP inlier RMSE: {metrics["icp_inlier_rmse"]} m')
    print(f'Results saved to: {out}')
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--recon',        required=True)
    parser.add_argument('--gt',           required=True)
    parser.add_argument('--output-dir',   required=True)
    parser.add_argument('--no-crop',      action='store_true')
    parser.add_argument('--cluster-eps',  type=float, default=0.05)
    parser.add_argument('--min-fraction', type=float, default=0.05)
    parser.add_argument('--max-fraction', type=float, default=0.15)
    args = parser.parse_args()
    compare(args.recon, args.gt, args.output_dir,
            auto_crop_recon=not args.no_crop,
            cluster_eps=args.cluster_eps,
            min_fraction=args.min_fraction,
            max_fraction=args.max_fraction)
