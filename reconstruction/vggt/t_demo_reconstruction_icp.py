# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
this code reconstruct and then use icp directly 

Anleitung
# 1. Just reconstruction (original behavior)
python /path/to/z_demo_reconstruction_icp.py  --scene_dir /path/to/scene/

# 2. Reconstruction + ICP with visualization
python /path/to/z_demo_reconstruction_icp.py \
    --scene_dir /path/to/scene/ \
    --run_icp \
    --object_ply /path/to/object.ply \
    --visualize_icp \
    --show_before

# 3. Full pipeline with custom parameters
python /path/to/z_demo_reconstruction_icp.py  \
    --scene_dir /path/to/scene/ \
    --conf_thres_value 3.0 \
    --run_icp \
    --object_ply object.ply \
    --voxel_size_object 0.002 \
    --voxel_size_scene 0.005 \
    --max_correspondence_dist 0.02 \
    --visualize_icp \
    --show_before


python /home/AP_PathMatters/vggt/z_demo_reconstruction_icp.py \
    --scene_dir /home/AP_PathMatters/path_matters/datasets/yoda \
    --run_icp \
    --object_ply /home/AP_PathMatters/path_matters/datasets/yoda/Baby_Yoda_v2.2.ply\
    --visualize_icp \
    --show_before
"""


import random
import numpy as np
import glob
import os
import copy
import torch
import torch.nn.functional as F
import json

# Configure CUDA settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

import argparse
from pathlib import Path
import trimesh
import pycolmap
import open3d as o3d

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.dependency.track_predict import predict_tracks
from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap, batch_np_matrix_to_pycolmap_wo_track


def parse_args():
    parser = argparse.ArgumentParser(description="VGGT Demo with ICP")
    
    # Original VGGT args
    parser.add_argument("--scene_dir", type=str, required=True, help="Directory containing the scene images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--use_ba", action="store_true", default=False, help="Use BA for reconstruction")
    
    # BA parameters
    parser.add_argument("--max_reproj_error", type=float, default=8.0, help="Maximum reprojection error")
    parser.add_argument("--shared_camera", action="store_true", default=False, help="Use shared camera")
    parser.add_argument("--camera_type", type=str, default="SIMPLE_PINHOLE", help="Camera type")
    parser.add_argument("--vis_thresh", type=float, default=0.2, help="Visibility threshold for tracks")
    parser.add_argument("--query_frame_num", type=int, default=8, help="Number of frames to query")
    parser.add_argument("--max_query_pts", type=int, default=4096, help="Maximum number of query points")
    parser.add_argument("--fine_tracking", action="store_true", default=True, help="Use fine tracking")
    parser.add_argument("--conf_thres_value", type=float, default=5.0, help="Confidence threshold (wo BA)")
    
    # ICP parameters
    parser.add_argument("--run_icp", action="store_true", help="Run ICP alignment")
    parser.add_argument("--object_ply", type=str, default=None, help="Object PLY path")
    parser.add_argument("--voxel_size_object", type=float, default=0.05, help="Object voxel size")
    parser.add_argument("--voxel_size_scene", type=float, default=0.001, help="Scene voxel size")
    parser.add_argument("--max_correspondence_dist", type=float, default=0.05, help="Max correspondence distance")
    parser.add_argument("--use_point_to_point", action="store_true", help="Use point-to-point ICP")
    parser.add_argument("--no_scale", action="store_true", help="Disable scale estimation")  # NEW
    parser.add_argument("--visualize_icp", action="store_true", help="Visualize ICP")
    parser.add_argument("--show_before", action="store_true", help="Show before/after")
    # In parse_args()
    parser.add_argument("--scale_method", type=str, default="auto",
                        choices=["auto", "ransac", "umeyama", "multi_scale", "bbox"],
                        help="Scale estimation method")
    parser.add_argument("--initial_alignment", type=str, default=None,
                    choices=[None, "pca", "center"],
                    help="Initial alignment method before ICP")
    parser.add_argument("--no_global", action="store_true", 
                    help="Skip global registration (use direct ICP only)")
    
    return parser.parse_args()


def run_VGGT(model, images, dtype, resolution=518):
    """Run VGGT inference"""
    assert len(images.shape) == 4
    assert images.shape[1] == 3

    images = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images = images[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images)

        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    depth_map = depth_map.squeeze(0).cpu().numpy()
    depth_conf = depth_conf.squeeze(0).cpu().numpy()
    
    return extrinsic, intrinsic, depth_map, depth_conf


def preprocess_scene_point_cloud(
    scene_ply_path: str,
    output_path: str = None,
    # Filtering options
    remove_statistical_outliers: bool = True,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
    # Cropping options
    crop_box: tuple = None,
    # Color filtering
    remove_color_range: list = None,
    # Downsampling
    voxel_size: float = None,
    # Plane removal
    remove_table_plane: bool = False,
    plane_distance_threshold: float = 0.01,
    plane_num_iterations: int = 1000,
    remove_points_below: bool = True,
    plane_offset: float = 0.02  # NEW: Offset above plane to remove entire plane
) -> o3d.geometry.PointCloud:
    """
    Preprocess reconstructed scene point cloud before ICP
    
    Args:
        scene_ply_path: Path to scene point cloud
        output_path: Where to save preprocessed cloud
        remove_statistical_outliers: Remove statistical outliers
        nb_neighbors: Number of neighbors for outlier removal
        std_ratio: Standard deviation ratio for outlier removal
        crop_box: Bounding box to crop ((min_x,y,z), (max_x,y,z))
        remove_color_range: List of RGB ranges to remove
        voxel_size: Voxel size for downsampling
        remove_table_plane: Remove table/floor plane
        plane_distance_threshold: Distance threshold for plane detection
        plane_num_iterations: RANSAC iterations for plane fitting
        remove_points_below: Remove points below plane (not just plane itself)
        plane_offset: Distance to shift plane upward before removal (removes entire plane)
        
    Returns:
        Preprocessed point cloud
    """

    print(f"\n{'='*70}")
    print(f"  PREPROCESSING SCENE POINT CLOUD")
    print(f"{'='*70}")
    
    # Load
    pcd = o3d.io.read_point_cloud(scene_ply_path)
    original_count = len(pcd.points)
    print(f"[PREP] Loaded: {original_count} points")
    
    # 1. Remove statistical outliers
    if remove_statistical_outliers:
        print(f"[PREP] Removing statistical outliers...")
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        print(f"       {original_count} → {len(pcd.points)} points")
    
    # 2. Crop to bounding box
    if crop_box is not None:
        print(f"[PREP] Cropping to bounding box...")
        min_bound, max_bound = crop_box
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=np.array(min_bound),
            max_bound=np.array(max_bound)
        )
        pcd = pcd.crop(bbox)
        print(f"       → {len(pcd.points)} points")
    
    # 3. Remove by color
    if remove_color_range is not None and pcd.has_colors():
        print(f"[PREP] Filtering by color...")
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) * 255
        
        mask = np.ones(len(points), dtype=bool)
        
        for color_min, color_max in remove_color_range:
            color_min = np.array(color_min)
            color_max = np.array(color_max)
            in_range = np.all((colors >= color_min) & (colors <= color_max), axis=1)
            mask &= ~in_range
        
        pcd = pcd.select_by_index(np.where(mask)[0])
        print(f"       → {len(pcd.points)} points")
    
    # 4. Remove table/floor plane
    if remove_table_plane:
        print(f"[PREP] Removing table/floor plane...")
        
        # Estimate normals
        print(f"       Estimating normals...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=10)
        )
        
        # Compute average normal (majority direction)
        pcd_downsampled = copy.deepcopy(pcd).voxel_down_sample(voxel_size=10)
        normals = np.asarray(pcd_downsampled.normals)
        average_normal = np.mean(normals, axis=0)
        average_normal /= np.linalg.norm(average_normal)
        print(f"       Average normal: {average_normal}")
        
        # Perform plane segmentation
        print(f"       Running RANSAC plane detection...")
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=plane_distance_threshold,
            ransac_n=3,
            num_iterations=plane_num_iterations
        )
        
        [a, b, c, d] = plane_model
        print(f"       Plane equation: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")
        print(f"       Inliers: {len(inliers)} points")
        
        # Flip plane normal if needed (to ensure correct orientation)
        plane_normal = np.array([a, b, c])
        plane_normal /= np.linalg.norm(plane_normal)
        
        dot_product = np.dot(plane_normal, average_normal)
        if dot_product > 0:
            print(f"       ⚠️  Flipping plane normal to match majority")
            plane_model = [-a, -b, -c, -d]
            [a, b, c, d] = plane_model
            plane_normal = -plane_normal
        
        if remove_points_below:
            # Apply offset to shift cutting plane upward
            # This ensures the entire physical plane is removed
            print(f"       Applying plane offset: {plane_offset} units upward")
            
            # Shift the plane by moving d
            # Plane equation: ax + by + cz + d = 0
            # To shift by offset along normal: d_new = d - offset * sqrt(a² + b² + c²)
            plane_norm = np.sqrt(a**2 + b**2 + c**2)
            d_offset = d - plane_offset * plane_norm
            
            print(f"       Original d: {d:.3f}, Offset d: {d_offset:.3f}")
            
            # Remove plane AND all points below the OFFSET plane
            print(f"       Removing plane and points below offset plane...")
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors) if pcd.has_colors() else None
            
            # Calculate signed distance to OFFSET plane
            distances = (a * points[:, 0] + 
                        b * points[:, 1] + 
                        c * points[:, 2] + d_offset) / plane_norm
            
            # Keep only points ABOVE the offset plane
            above_plane_mask = distances <= 0
            
            pcd_filtered = o3d.geometry.PointCloud()
            pcd_filtered.points = o3d.utility.Vector3dVector(points[above_plane_mask])
            
            if colors is not None:
                pcd_filtered.colors = o3d.utility.Vector3dVector(colors[above_plane_mask])
            
            pcd = pcd_filtered
            print(f"       → {len(pcd.points)} points (removed plane + below + offset)")
        else:
            # Remove only the plane itself
            print(f"       Removing plane only...")
            pcd = pcd.select_by_index(inliers, invert=True)
            print(f"       → {len(pcd.points)} points")
    
    # 5. Downsample
    if voxel_size is not None:
        print(f"[PREP] Downsampling (voxel_size={voxel_size})...")
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"       → {len(pcd.points)} points")
    
    # Save if requested
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(output_path), pcd)
        print(f"[PREP] ✓ Saved preprocessed cloud: {output_path}")
    
    print(f"[PREP] ✓ Preprocessing complete: {original_count} → {len(pcd.points)} points")
    
    return pcd


def run_icp_alignment(
    object_ply_path: str,
    scene_pcd: o3d.geometry.PointCloud,
    output_dir: str,
    voxel_size_object: float = 0.005,
    voxel_size_scene: float = 0.01,
    max_correspondence_distance: float = 0.05,
    use_point_to_plane: bool = True,
    estimate_scale: bool = True,
    scale_method: str = "auto",
    use_global_registration: bool = True,  # NEW
    visualize: bool = True,
    show_before: bool = True
):
    """
    Run ICP alignment with optional global registration first
    """

    print(f"\n{'='*70}")
    print(f"  ICP ALIGNMENT")
    print(f"{'='*70}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load object
    print(f"[ICP] Loading object: {object_ply_path}")
    object_pcd = o3d.io.read_point_cloud(object_ply_path)
    print(f"      {len(object_pcd.points)} points")
    
    # Estimate scale
    if estimate_scale:
        print(f"\n[SCALE] Estimating scale...")
        scale = estimate_scale_robust(
            source=object_pcd,
            target=scene_pcd,
            method=scale_method,
            correspondence_distance=max_correspondence_distance
        )
        
        # Apply scale to object
        print(f"\n[ICP] Applying scale {scale:.6f} to object...")
        object_pcd_scaled = copy.deepcopy(object_pcd)
        object_pcd_scaled.scale(scale, center=object_pcd_scaled.get_center())
    else:
        scale = 1.0
        object_pcd_scaled = object_pcd
    
    # Run registration
    if use_global_registration:
        # Global + Local registration
        result_global, result_local = run_global_then_local_registration(
            source=object_pcd_scaled,
            target=scene_pcd,
            voxel_size_global=max_correspondence_distance * 2,  # Coarser for global
            voxel_size_local=voxel_size_object,  # Fine for local
            use_point_to_plane=use_point_to_plane
        )
        
        transformation = result_local.transformation
        result = result_local  # Use local result for metrics
        
    else:
        # Direct ICP (old method)
        print(f"\n[ICP] Preprocessing for direct ICP...")
        
        object_pcd_processed = object_pcd_scaled.voxel_down_sample(voxel_size=voxel_size_object)
        scene_pcd_processed = scene_pcd.voxel_down_sample(voxel_size=voxel_size_scene)
        
        # Estimate normals
        object_pcd_processed.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel_size_object * 5, max_nn=30
            )
        )
        scene_pcd_processed.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel_size_scene * 5, max_nn=30
            )
        )
        
        print(f"\n[ICP] Running direct ICP...")
        
        if use_point_to_plane:
            estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        else:
            estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        
        result = o3d.pipelines.registration.registration_icp(
            source=object_pcd_processed,
            target=scene_pcd_processed,
            max_correspondence_distance=max_correspondence_distance,
            init=np.eye(4),
            estimation_method=estimation_method,
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=200,
                relative_fitness=1e-6,
                relative_rmse=1e-6
            )
        )
        
        transformation = result.transformation
    
    # Metrics
    metrics = {
        'fitness': float(result.fitness),
        'inlier_rmse': float(result.inlier_rmse),
        'num_correspondences': len(result.correspondence_set),
        'scale': float(scale),
        'transformation': transformation.tolist(),
        'used_global_registration': use_global_registration
    }
    
    print(f"\n[ICP] ✓ Alignment complete")
    print(f"      Scale:           {metrics['scale']:.6f}")
    print(f"      Fitness:         {metrics['fitness']:.4f}")
    print(f"      Inlier RMSE:     {metrics['inlier_rmse']:.6f}")
    print(f"      Correspondences: {metrics['num_correspondences']}")
    
    # Save results
    print(f"\n[ICP] Saving results to {output_dir}")
    
    np.save(output_dir / "transformation.npy", transformation)
    np.save(output_dir / "scale.npy", np.array([scale]))
    
    with open(output_dir / "icp_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Transform original object
    object_aligned = copy.deepcopy(object_pcd)
    object_aligned.scale(scale, center=object_aligned.get_center())
    object_aligned.transform(transformation)
    o3d.io.write_point_cloud(str(output_dir / "object_aligned.ply"), object_aligned)
    
    print(f"      ✓ transformation.npy")
    print(f"      ✓ scale.npy")
    print(f"      ✓ icp_metrics.json")
    print(f"      ✓ object_aligned.ply")
    
    # Visualization
    if visualize:
        print(f"\n[VIS] Preparing visualization...")
        
        target_colored = scene_pcd.paint_uniform_color([0.8, 0.8, 0.8])
        source_colored = copy.deepcopy(object_pcd).paint_uniform_color([1.0, 0.0, 0.0])
        source_aligned_colored = copy.deepcopy(object_pcd)
        source_aligned_colored.scale(scale, center=source_aligned_colored.get_center())
        source_aligned_colored.transform(transformation)
        source_aligned_colored = source_aligned_colored.paint_uniform_color([0.0, 1.0, 0.0])
        
        if show_before:
            print("[VIS] Showing BEFORE (Red=Object, Gray=Scene)")
            o3d.visualization.draw_geometries(
                [source_colored, target_colored],
                window_name="Before Alignment",
                width=1280,
                height=720
            )
        
        print(f"[VIS] Showing AFTER (Green=Aligned, Gray=Scene)")
        o3d.visualization.draw_geometries(
            [source_aligned_colored, target_colored],
            window_name="After Alignment",
            width=1280,
            height=720
        )
    
    return metrics


def compute_fpfh_features(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
    radius_multiplier: float = 5.0
):
    """
    Compute FPFH features for point cloud
    
    Args:
        pcd: Point cloud
        voxel_size: Voxel size for downsampling
        radius_multiplier: Multiplier for search radius
        
    Returns:
        FPFH feature object
    """
    radius_normal = voxel_size * 2
    radius_feature = voxel_size * radius_multiplier
    
    # Estimate normals if not present
    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius_normal, max_nn=30
            )
        )
    
    # Compute FPFH features
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_feature, max_nn=100
        )
    )
    
    return fpfh


def execute_global_registration(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    source_fpfh,
    target_fpfh,
    voxel_size: float,
    distance_threshold: float = None,
    edge_length_ratio: float = 0.9,
    angle_threshold: float = np.deg2rad(30),
    ransac_iterations: int = 100000,
    ransac_confidence: float = 0.999
):
    """
    Execute global registration using RANSAC-based feature matching
    
    Args:
        source: Source point cloud
        target: Target point cloud
        source_fpfh: Source FPFH features
        target_fpfh: Target FPFH features
        voxel_size: Voxel size used for downsampling
        distance_threshold: Max correspondence distance
        edge_length_ratio: Edge length ratio for correspondence checker
        angle_threshold: Normal angle threshold (radians)
        ransac_iterations: Number of RANSAC iterations
        ransac_confidence: RANSAC confidence
        
    Returns:
        Registration result
    """
    if distance_threshold is None:
        distance_threshold = voxel_size * 1.5
    
    print(f"\n[GLOBAL] Running RANSAC global registration...")
    print(f"         Distance threshold: {distance_threshold:.6f}")
    print(f"         RANSAC iterations: {ransac_iterations}")
    print(f"         Edge length ratio: {edge_length_ratio}")
    print(f"         Normal angle threshold: {np.rad2deg(angle_threshold):.1f}°")
    
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, target,
        source_fpfh, target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                edge_length_ratio
            ),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold
            ),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(
                angle_threshold
            )
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            ransac_iterations,
            ransac_confidence
        )
    )
    
    print(f"[GLOBAL] ✓ Global registration complete")
    print(f"         Fitness: {result.fitness:.4f}")
    print(f"         Inlier RMSE: {result.inlier_rmse:.6f}")
    print(f"         Correspondences: {len(result.correspondence_set)}")
    
    return result


def refine_registration(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    initial_transformation: np.ndarray,
    distance_threshold: float,
    use_point_to_plane: bool = True,
    max_iterations: int = 200
):
    """
    Refine registration using ICP
    
    Args:
        source: Source point cloud
        target: Target point cloud
        initial_transformation: Initial transformation from global registration
        distance_threshold: Max correspondence distance
        use_point_to_plane: Use point-to-plane ICP
        max_iterations: Maximum ICP iterations
        
    Returns:
        Refined registration result
    """
    print(f"\n[REFINE] Running ICP refinement...")
    print(f"         Distance threshold: {distance_threshold:.6f}")
    print(f"         Method: {'Point-to-Plane' if use_point_to_plane else 'Point-to-Point'}")
    print(f"         Max iterations: {max_iterations}")
    
    # Ensure normals for point-to-plane
    if use_point_to_plane:
        if not source.has_normals():
            source.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=distance_threshold * 2, max_nn=30
                )
            )
        if not target.has_normals():
            target.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=distance_threshold * 2, max_nn=30
                )
            )
        
        estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    
    result = o3d.pipelines.registration.registration_icp(
        source, target,
        distance_threshold,
        initial_transformation,
        estimation_method,
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iterations,
            relative_fitness=1e-6,
            relative_rmse=1e-6
        )
    )
    
    print(f"[REFINE] ✓ ICP refinement complete")
    print(f"         Fitness: {result.fitness:.4f}")
    print(f"         Inlier RMSE: {result.inlier_rmse:.6f}")
    print(f"         Correspondences: {len(result.correspondence_set)}")
    
    return result


def run_global_then_local_registration(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    voxel_size_global: float = 0.05,
    voxel_size_local: float = 0.01,
    global_distance_mult: float = 1.5,
    local_distance_mult: float = 0.4,
    use_point_to_plane: bool = True,
    ransac_iterations: int = 100000,
    icp_iterations: int = 200
):
    """
    Complete registration pipeline: Global (RANSAC+FPFH) → Local (ICP)
    
    Args:
        source: Source point cloud (object)
        target: Target point cloud (scene)
        voxel_size_global: Voxel size for global registration
        voxel_size_local: Voxel size for local refinement
        global_distance_mult: Distance threshold multiplier for global
        local_distance_mult: Distance threshold multiplier for local
        use_point_to_plane: Use point-to-plane ICP for refinement
        ransac_iterations: RANSAC iterations
        icp_iterations: ICP iterations
        
    Returns:
        (global_result, local_result)
    """
    print(f"\n{'='*70}")
    print(f"  GLOBAL + LOCAL REGISTRATION PIPELINE")
    print(f"{'='*70}")
    
    # Step 1: Downsample for global registration
    print(f"\n[PREP] Downsampling for global registration...")
    print(f"       Voxel size: {voxel_size_global}")
    
    source_down_global = source.voxel_down_sample(voxel_size_global)
    target_down_global = target.voxel_down_sample(voxel_size_global)
    
    print(f"       Source: {len(source.points)} → {len(source_down_global.points)} points")
    print(f"       Target: {len(target.points)} → {len(target_down_global.points)} points")
    
    # Step 2: Compute FPFH features
    print(f"\n[PREP] Computing FPFH features...")
    source_fpfh = compute_fpfh_features(source_down_global, voxel_size_global)
    target_fpfh = compute_fpfh_features(target_down_global, voxel_size_global)
    print(f"       ✓ Features computed")
    
    # Step 3: Global registration (RANSAC)
    global_distance_threshold = voxel_size_global * global_distance_mult
    
    result_global = execute_global_registration(
        source=source_down_global,
        target=target_down_global,
        source_fpfh=source_fpfh,
        target_fpfh=target_fpfh,
        voxel_size=voxel_size_global,
        distance_threshold=global_distance_threshold,
        ransac_iterations=ransac_iterations
    )
    
    # Check if global registration succeeded
    if result_global.fitness < 0.1:
        print(f"\n⚠️  WARNING: Global registration has low fitness ({result_global.fitness:.4f})")
        print(f"   This may indicate:")
        print(f"   - Object and scene don't overlap")
        print(f"   - Scale mismatch is too large")
        print(f"   - Need to adjust voxel_size_global or distance thresholds")
    
    # Step 4: Downsample for local refinement
    print(f"\n[PREP] Downsampling for local refinement...")
    print(f"       Voxel size: {voxel_size_local}")
    
    source_down_local = source.voxel_down_sample(voxel_size_local)
    target_down_local = target.voxel_down_sample(voxel_size_local)
    
    print(f"       Source: {len(source.points)} → {len(source_down_local.points)} points")
    print(f"       Target: {len(target.points)} → {len(target_down_local.points)} points")
    
    # Step 5: Local refinement (ICP)
    local_distance_threshold = voxel_size_local * local_distance_mult
    
    result_local = refine_registration(
        source=source_down_local,
        target=target_down_local,
        initial_transformation=result_global.transformation,
        distance_threshold=local_distance_threshold,
        use_point_to_plane=use_point_to_plane,
        max_iterations=icp_iterations
    )
    
    print(f"\n{'='*70}")
    print(f"  REGISTRATION COMPLETE")
    print(f"{'='*70}")
    print(f"Global → Local fitness: {result_global.fitness:.4f} → {result_local.fitness:.4f}")
    print(f"Global → Local RMSE:    {result_global.inlier_rmse:.6f} → {result_local.inlier_rmse:.6f}")
    print(f"{'='*70}")
    
    return result_global, result_local


def get_initial_alignment(source, target, method="center"):
    """Get initial transformation for better ICP convergence"""
    if method == "center":
        # Simple center alignment
        source_center = source.get_center()
        target_center = target.get_center()
        
        transformation = np.eye(4)
        transformation[:3, 3] = target_center - source_center
        
        print(f"[INIT] Center alignment: {transformation[:3, 3]}")
        return transformation
    
    elif method == "pca":
        # PCA-based alignment
        source_centered = copy.deepcopy(source)
        target_centered = copy.deepcopy(target)
        
        source_center = source.get_center()
        target_center = target.get_center()
        
        source_centered.translate(-source_center)
        target_centered.translate(-target_center)
        
        # Compute PCA
        source_points = np.asarray(source_centered.points)
        target_points = np.asarray(target_centered.points)
        
        _, _, source_v = np.linalg.svd(source_points.T @ source_points)
        _, _, target_v = np.linalg.svd(target_points.T @ target_points)
        
        # Rotation to align principal axes
        R = target_v.T @ source_v
        
        transformation = np.eye(4)
        transformation[:3, :3] = R
        transformation[:3, 3] = target_center - R @ source_center
        
        print(f"[INIT] PCA alignment computed")
        return transformation
    
    return np.eye(4)


def estimate_scale_from_bbox(source, target):
    """Fallback: estimate scale from bounding box diagonal"""
    source_bbox = source.get_axis_aligned_bounding_box()
    target_bbox = target.get_axis_aligned_bounding_box()
    
    source_diagonal = np.linalg.norm(source_bbox.get_extent())
    target_diagonal = np.linalg.norm(target_bbox.get_extent())
    
    scale = target_diagonal / source_diagonal
    print(f"[SCALE] Bounding box diagonal ratio: {scale:.6f}")
    print(f"        Source extent: {source_bbox.get_extent()}")
    print(f"        Target extent: {target_bbox.get_extent()}")
    
    return scale


def estimate_scale_ransac(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    correspondence_distance: float = 0.1,
    ransac_iterations: int = 1000,
    confidence: float = 0.99
):
    """
    Estimate scale using RANSAC on feature correspondences
    Based on: "Least-Squares Fitting of Two 3-D Point Sets" (Arun et al., 1987)
    """
    print(f"[SCALE] Estimating scale with RANSAC...")
    
    # 1. Extract features (FPFH - Fast Point Feature Histograms)
    print(f"[SCALE] Computing FPFH features...")
    
    # Downsample for feature extraction
    source_down = source.voxel_down_sample(voxel_size=correspondence_distance * 2)
    target_down = target.voxel_down_sample(voxel_size=correspondence_distance * 2)
    
    print(f"        Source: {len(source.points)} → {len(source_down.points)} points")
    print(f"        Target: {len(target.points)} → {len(target_down.points)} points")
    
    # Estimate normals
    source_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=correspondence_distance * 5, max_nn=30
        )
    )
    target_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=correspondence_distance * 5, max_nn=30
        )
    )
    
    # Compute FPFH features
    print(f"[SCALE] Computing features...")
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=correspondence_distance * 5, max_nn=100
        )
    )
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=correspondence_distance * 5, max_nn=100
        )
    )
    
    # 2. RANSAC feature matching
    print(f"[SCALE] Running RANSAC feature matching...")
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down,
        source_fpfh, target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=correspondence_distance,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(correspondence_distance)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(ransac_iterations, confidence)
    )
    
    # 3. Estimate scale from correspondences
    correspondences = np.asarray(result.correspondence_set)
    
    print(f"[SCALE] Found {len(correspondences)} correspondences (fitness: {result.fitness:.4f})")
    
    if len(correspondences) < 10:
        print(f"[SCALE] ⚠️  Too few correspondences, falling back to bbox method")
        return estimate_scale_from_bbox(source, target)
    
    source_points = np.asarray(source_down.points)[correspondences[:, 0]]
    target_points = np.asarray(target_down.points)[correspondences[:, 1]]
    
    # Calculate distances from centroid
    source_center = source_points.mean(axis=0)
    target_center = target_points.mean(axis=0)
    
    source_distances = np.linalg.norm(source_points - source_center, axis=1)
    target_distances = np.linalg.norm(target_points - target_center, axis=1)
    
    # Filter out very small distances
    valid_mask = (source_distances > 0.001) & (target_distances > 0.001)
    
    if valid_mask.sum() < 10:
        print(f"[SCALE] ⚠️  Too few valid distances, falling back to bbox method")
        return estimate_scale_from_bbox(source, target)
    
    # Robust scale estimation using median
    scale_ratios = target_distances[valid_mask] / source_distances[valid_mask]
    scale = np.median(scale_ratios)
    
    # Additional validation: use MAD (Median Absolute Deviation) to filter outliers
    mad = np.median(np.abs(scale_ratios - scale))
    inlier_mask = np.abs(scale_ratios - scale) < 3 * mad
    
    if inlier_mask.sum() > 10:
        scale_refined = np.median(scale_ratios[inlier_mask])
        print(f"[SCALE] ✓ Refined scale with {inlier_mask.sum()}/{len(scale_ratios)} inliers: {scale_refined:.6f}")
        return scale_refined
    else:
        print(f"[SCALE] ✓ Scale from {len(scale_ratios)} correspondences: {scale:.6f}")
        return scale


def estimate_scale_umeyama(source_points, target_points):
    """
    Umeyama algorithm: Closed-form solution for similarity transformation
    Returns: scale, rotation, translation
    
    Reference: "Least-squares estimation of transformation parameters 
                between two point patterns" (Umeyama, 1991)
    """
    assert source_points.shape == target_points.shape
    
    m, n = source_points.shape  # m = num_points, n = dimension (3)
    
    # Center the point sets
    source_mean = source_points.mean(axis=0)
    target_mean = target_points.mean(axis=0)
    
    source_centered = source_points - source_mean
    target_centered = target_points - target_mean
    
    # Compute variances
    source_var = np.sum(source_centered ** 2) / m
    
    # Covariance matrix
    cov = (target_centered.T @ source_centered) / m
    
    # SVD
    U, D, Vt = np.linalg.svd(cov)
    
    # Construct S matrix
    S = np.eye(n)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[n-1, n-1] = -1
    
    # Rotation
    R = U @ S @ Vt
    
    # Scale
    scale = np.trace(np.diag(D) @ S) / source_var
    
    # Translation
    t = target_mean - scale * R @ source_mean
    
    return scale, R, t


def apply_umeyama_registration(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    correspondence_distance: float = 0.1
):
    """Apply Umeyama registration with scale"""
    print(f"[SCALE] Running Umeyama algorithm...")
    
    # Get initial correspondences via nearest neighbors
    source_down = source.voxel_down_sample(voxel_size=correspondence_distance)
    target_down = target.voxel_down_sample(voxel_size=correspondence_distance)
    
    # Build KD-tree for target
    target_tree = o3d.geometry.KDTreeFlann(target_down)
    
    # Find correspondences
    source_points = np.asarray(source_down.points)
    target_points = np.asarray(target_down.points)
    
    correspondences = []
    for i, point in enumerate(source_points):
        [_, idx, dist] = target_tree.search_knn_vector_3d(point, 1)
        if dist[0] < correspondence_distance ** 2:
            correspondences.append((i, idx[0]))
    
    print(f"[SCALE] Found {len(correspondences)} nearest neighbor correspondences")
    
    if len(correspondences) < 10:
        print(f"[SCALE] ⚠️  Too few correspondences, falling back to bbox method")
        return estimate_scale_from_bbox(source, target), np.eye(3), np.zeros(3)
    
    correspondences = np.array(correspondences)
    source_corr = source_points[correspondences[:, 0]]
    target_corr = target_points[correspondences[:, 1]]
    
    # Apply Umeyama
    scale, R, t = estimate_scale_umeyama(source_corr, target_corr)
    
    print(f"[SCALE] ✓ Umeyama scale: {scale:.6f}")
    
    return scale, R, t


def multi_scale_icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    scales: list = None,
    max_correspondence_distance: float = 0.05
):
    """
    Try multiple scales and pick the best one based on fitness
    """
    if scales is None:
        # Auto-generate scale candidates around bbox estimate
        bbox_scale = estimate_scale_from_bbox(source, target)
        scales = [
            bbox_scale * 0.5,
            bbox_scale * 0.75,
            bbox_scale * 1.0,
            bbox_scale * 1.25,
            bbox_scale * 1.5,
            bbox_scale * 2.0
        ]
    
    print(f"[SCALE] Testing multiple scales...")
    print(f"        Candidates: {[f'{s:.3f}' for s in scales]}")
    
    best_fitness = -1
    best_scale = 1.0
    best_transformation = np.eye(4)
    best_rmse = float('inf')
    
    for scale in scales:
        # Scale source
        source_scaled = copy.deepcopy(source)
        source_scaled.scale(scale, center=source_scaled.get_center())
        
        # Run ICP
        result = o3d.pipelines.registration.registration_icp(
            source=source_scaled,
            target=target,
            max_correspondence_distance=max_correspondence_distance,
            init=np.eye(4),
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
        )
        
        print(f"        Scale {scale:.3f}: fitness={result.fitness:.4f}, rmse={result.inlier_rmse:.6f}")
        
        # Pick best based on fitness (or could use rmse)
        if result.fitness > best_fitness:
            best_fitness = result.fitness
            best_scale = scale
            best_transformation = result.transformation
            best_rmse = result.inlier_rmse
    
    print(f"[SCALE] ✓ Best scale: {best_scale:.6f} (fitness={best_fitness:.4f}, rmse={best_rmse:.6f})")
    
    return best_scale, best_transformation


def estimate_scale_robust(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    method: str = "auto",  # "auto", "ransac", "umeyama", "multi_scale", "bbox"
    correspondence_distance: float = None
):
    """
    Robust scale estimation with multiple methods
    
    Args:
        source: Object point cloud
        target: Scene point cloud
        method: Scale estimation method
            - "auto": Try ransac, fallback to multi_scale if fails
            - "ransac": RANSAC-based (most robust for noisy data)
            - "umeyama": Closed-form solution (fast, needs good correspondences)
            - "multi_scale": Brute force search (guaranteed result)
            - "bbox": Bounding box ratio (fast approximation)
        correspondence_distance: Max distance for correspondences (auto if None)
    
    Returns:
        scale: Estimated scale factor
    """
    print(f"\n{'='*70}")
    print(f"  SCALE ESTIMATION ({method.upper()})")
    print(f"{'='*70}")
    
    # Auto-determine correspondence distance if not provided
    if correspondence_distance is None:
        target_bbox = target.get_axis_aligned_bounding_box()
        correspondence_distance = np.linalg.norm(target_bbox.get_extent()) * 0.05
        print(f"[SCALE] Auto correspondence distance: {correspondence_distance:.6f}")
    
    if method == "auto":
        # Try RANSAC first (most robust)
        try:
            scale = estimate_scale_ransac(
                source, target,
                correspondence_distance=correspondence_distance
            )
            # Sanity check: scale should be reasonable (0.01 to 100)
            if 0.01 < scale < 100:
                return scale
            else:
                print(f"[SCALE] ⚠️  RANSAC scale {scale:.6f} seems unreasonable, trying multi-scale")
        except Exception as e:
            print(f"[SCALE] ⚠️  RANSAC failed: {e}, trying multi-scale")
        
        # Fallback to multi-scale
        scale, _ = multi_scale_icp(
            source, target,
            max_correspondence_distance=correspondence_distance
        )
        return scale
    
    elif method == "ransac":
        return estimate_scale_ransac(
            source, target,
            correspondence_distance=correspondence_distance
        )
    
    elif method == "umeyama":
        scale, _, _ = apply_umeyama_registration(
            source, target,
            correspondence_distance=correspondence_distance
        )
        return scale
    
    elif method == "multi_scale":
        scale, _ = multi_scale_icp(
            source, target,
            max_correspondence_distance=correspondence_distance
        )
        return scale
    
    elif method == "bbox":
        return estimate_scale_from_bbox(source, target)
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'auto', 'ransac', 'umeyama', 'multi_scale', or 'bbox'")


def demo_fn(args):
    """Main VGGT reconstruction with optional ICP"""
    
    # Print configuration
    print("Arguments:", vars(args))

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    print(f"Setting seed as: {args.seed}")

    # Set device and dtype
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")

    # Load VGGT model
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval()
    model = model.to(device)
    print(f"Model loaded")

    # Get image paths
    image_dir = os.path.join(args.scene_dir, "images")
    image_path_list = glob.glob(os.path.join(image_dir, "*"))
    if len(image_path_list) == 0:
        raise ValueError(f"No images found in {image_dir}")
    base_image_path_list = [os.path.basename(path) for path in image_path_list]

    # Load images
    vggt_fixed_resolution = 518
    img_load_resolution = 1024

    images, original_coords = load_and_preprocess_images_square(image_path_list, img_load_resolution)
    images = images.to(device)
    original_coords = original_coords.to(device)
    print(f"Loaded {len(images)} images from {image_dir}")

    # Run VGGT
    extrinsic, intrinsic, depth_map, depth_conf = run_VGGT(model, images, dtype, vggt_fixed_resolution)
    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)

    # Bundle Adjustment or feedforward
    if args.use_ba:
        image_size = np.array(images.shape[-2:])
        scale = img_load_resolution / vggt_fixed_resolution
        shared_camera = args.shared_camera

        with torch.cuda.amp.autocast(dtype=dtype):
            pred_tracks, pred_vis_scores, pred_confs, points_3d, points_rgb = predict_tracks(
                images,
                conf=depth_conf,
                points_3d=points_3d,
                masks=None,
                max_query_pts=args.max_query_pts,
                query_frame_num=args.query_frame_num,
                keypoint_extractor="aliked+sp",
                fine_tracking=args.fine_tracking,
            )
            torch.cuda.empty_cache()

        intrinsic[:, :2, :] *= scale
        track_mask = pred_vis_scores > args.vis_thresh

        reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
            points_3d,
            extrinsic,
            intrinsic,
            pred_tracks,
            image_size,
            masks=track_mask,
            max_reproj_error=args.max_reproj_error,
            shared_camera=shared_camera,
            camera_type=args.camera_type,
            points_rgb=points_rgb,
        )

        if reconstruction is None:
            raise ValueError("No reconstruction can be built with BA")

        ba_options = pycolmap.BundleAdjustmentOptions()
        pycolmap.bundle_adjustment(reconstruction, ba_options)
        reconstruction_resolution = img_load_resolution
        
    else:
        conf_thres_value = args.conf_thres_value
        max_points_for_colmap = 100000
        shared_camera = False
        camera_type = "PINHOLE"

        image_size = np.array([vggt_fixed_resolution, vggt_fixed_resolution])
        num_frames, height, width, _ = points_3d.shape

        points_rgb = F.interpolate(
            images, size=(vggt_fixed_resolution, vggt_fixed_resolution), mode="bilinear", align_corners=False
        )
        points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8)
        points_rgb = points_rgb.transpose(0, 2, 3, 1)

        points_xyf = create_pixel_coordinate_grid(num_frames, height, width)

        conf_mask = depth_conf >= conf_thres_value
        conf_mask = randomly_limit_trues(conf_mask, max_points_for_colmap)

        points_3d = points_3d[conf_mask]
        points_xyf = points_xyf[conf_mask]
        points_rgb = points_rgb[conf_mask]

        print("Converting to COLMAP format")
        reconstruction = batch_np_matrix_to_pycolmap_wo_track(
            points_3d,
            points_xyf,
            points_rgb,
            extrinsic,
            intrinsic,
            image_size,
            shared_camera=shared_camera,
            camera_type=camera_type,
        )
        reconstruction_resolution = vggt_fixed_resolution

    # Rescale camera
    reconstruction = rename_colmap_recons_and_rescale_camera(
        reconstruction,
        base_image_path_list,
        original_coords.cpu().numpy(),
        img_size=reconstruction_resolution,
        shift_point2d_to_original_res=True,
        shared_camera=shared_camera,
    )

    # Save reconstruction
    print(f"Saving reconstruction to {args.scene_dir}/sparse")
    sparse_reconstruction_dir = os.path.join(args.scene_dir, "sparse")
    os.makedirs(sparse_reconstruction_dir, exist_ok=True)
    reconstruction.write(sparse_reconstruction_dir)

    # Save point cloud
    scene_ply_path = os.path.join(args.scene_dir, "sparse/points.ply")
    trimesh.PointCloud(points_3d, colors=points_rgb).export(scene_ply_path)
    print(f"✓ Saved point cloud: {scene_ply_path}")

    # ICP alignment if requested
    if args.run_icp:
        
        if args.object_ply is None:
            print("❌ --object_ply required for ICP")
            return True
        
        scene_pcd = preprocess_scene_point_cloud(
            scene_ply_path=scene_ply_path,
            output_path=os.path.join(args.scene_dir, "sparse/points_preprocessed.ply"),
            remove_statistical_outliers=True,
            remove_table_plane=True,
            plane_distance_threshold=0.01,
            plane_num_iterations=1000,
            remove_points_below=True,
            plane_offset=-0.02,  # NEW: 2cm offset to remove entire plane
        )
        
        # Run ICP
        icp_output_dir = os.path.join(args.scene_dir, "icp_results")
        metrics = run_icp_alignment(
            object_ply_path=args.object_ply,
            scene_pcd=scene_pcd,
            output_dir=icp_output_dir,
            voxel_size_object=args.voxel_size_object,
            voxel_size_scene=args.voxel_size_scene,
            max_correspondence_distance=args.max_correspondence_dist,
            use_point_to_plane=not args.use_point_to_point,
            estimate_scale=not args.no_scale,
            scale_method=args.scale_method,
            use_global_registration=not args.no_global,  # NEW
            visualize=args.visualize_icp,
            show_before=args.show_before
        )

        print(f"\n{'='*70}")
        print(f"  ICP COMPLETE")
        print(f"{'='*70}")
        print(f"Scale:           {metrics['scale']:.6f}")  # NEW
        print(f"Fitness:         {metrics['fitness']:.4f}")
        print(f"Inlier RMSE:     {metrics['inlier_rmse']:.6f}")
        print(f"Correspondences: {metrics['num_correspondences']}")

    return True


def rename_colmap_recons_and_rescale_camera(
    reconstruction, image_paths, original_coords, img_size, shift_point2d_to_original_res=False, shared_camera=False
):
    """Rename and rescale camera parameters"""
    rescale_camera = True

    for pyimageid in reconstruction.images:
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = image_paths[pyimageid - 1]

        if rescale_camera:
            pred_params = copy.deepcopy(pycamera.params)
            real_image_size = original_coords[pyimageid - 1, -2:]
            resize_ratio = max(real_image_size) / img_size
            pred_params = pred_params * resize_ratio
            real_pp = real_image_size / 2
            pred_params[-2:] = real_pp

            pycamera.params = pred_params
            pycamera.width = real_image_size[0]
            pycamera.height = real_image_size[1]

        if shift_point2d_to_original_res:
            top_left = original_coords[pyimageid - 1, :2]
            for point2D in pyimage.points2D:
                point2D.xy = (point2D.xy - top_left) * resize_ratio

        if shared_camera:
            rescale_camera = False

    return reconstruction


if __name__ == "__main__":
    args = parse_args()
    with torch.no_grad():
        demo_fn(args)