"""
Usage:
  python compare.py <groundtruth.ply> <reconstruction.ply>

Example:
  python compare.py Test/Baby_Yoda.ply Test/points.ply
"""

import sys
import numpy as np
import open3d as o3d

if len(sys.argv) != 3:
    print("Usage: python compare.py <groundtruth.ply> <reconstruction.ply>")
    sys.exit(1)

gt_path  = sys.argv[1]
rec_path = sys.argv[2]

print(f"\nLoading: {gt_path}")
print(f"Loading: {rec_path}")

gt  = o3d.io.read_point_cloud(gt_path)
rec = o3d.io.read_point_cloud(rec_path)

print(f"\nGround truth points : {len(gt.points)}")
print(f"Reconstruction points: {len(rec.points)}")

# Color for visualization
gt.paint_uniform_color([1, 0, 0])    # Red   = ground truth
rec.paint_uniform_color([0, 0.6, 1]) # Blue  = reconstruction

# Show BEFORE alignment
print("\nBEFORE alignment (Red=GT, Blue=Reconstruction) — Press Q to continue")
o3d.visualization.draw_geometries([gt, rec], window_name="BEFORE", width=1280, height=720)

# Align using ICP
voxel = 0.01
gt_d  = gt.voxel_down_sample(voxel)
rec_d = rec.voxel_down_sample(voxel)
gt_d.estimate_normals()
rec_d.estimate_normals()

result = o3d.pipelines.registration.registration_icp(
    rec_d, gt_d, 0.05, np.eye(4),
    o3d.pipelines.registration.TransformationEstimationPointToPlane()
)

print(f"\n--- Results ---")
print(f"Fitness (1.0 = perfect): {result.fitness:.4f}")
print(f"RMSE   (0.0 = perfect) : {result.inlier_rmse:.6f}")

rec.transform(result.transformation)
rec.paint_uniform_color([0, 1, 0])   # Green = aligned

# Show AFTER alignment
print("\nAFTER alignment (Red=GT, Green=Aligned) — Press Q to close")
o3d.visualization.draw_geometries([gt, rec], window_name="AFTER", width=1280, height=720)