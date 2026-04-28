#!/usr/bin/env python3
"""
Improved ICP Alignment Pipeline for UR5e Robot Scanning
========================================================

This pipeline performs robust point cloud alignment with:
- Adaptive preprocessing
- Multi-method scale estimation
- Global + Local registration
- Iterative refinement with adaptive search

Author: Ziad
Date: 2025
"""

import numpy as np
import copy
import time
import json
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict

import open3d as o3d
import torch
import torch.nn.functional as F


# ============================================================================
# CONFIGURATION
# ============================================================================

# ============================================================================
# CONFIGURATION PARAMETERS (from ICP original JSON)
# ============================================================================

class ICPConfig:
    """Configuration parameters - extracted from ICP original"""
    
    # Visualization
    DEBUG_VIS = False
    VIS_RESULT = True
    
    # Scene Preprocessing
    SCENE_DOWNSAMPLE_VOXEL = 0.02
    SCENE_OUTLIER_NEIGHBORS = 5
    SCENE_OUTLIER_STD = 2.0
    
    # Object Preprocessing  
    OBJECT_DOWNSAMPLE_VOXEL = 05.5 # dont change
    OBJECT_OUTLIER_NEIGHBORS = 10
    OBJECT_OUTLIER_STD = 10.0
    
    # Plane Removal (keep your settings)
    PLANE_DISTANCE_THRESHOLD = 0.0015
    PLANE_RANSAC_ITERATIONS = 1000
    PLANE_OFFSET = 0.005 
    
    # Scale Estimation
    SCALE_METHOD = "bbox"
    SCALE_CORRESPONDENCE_DIST = 0.01
    SCALE_CANDIDATES = [0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25]

    # Global Registration - FROM ICP ORIGINAL JSON
    GLOBAL_VOXEL_SIZE = 0.005
    GLOBAL_FPFH_RADIUS_MULTIPLIER = 5.0
    GLOBAL_FPFH_RADIUS = 10  # From JSON
    GLOBAL_FPFH_MAX_NN = 100  # From JSON
    GLOBAL_RANSAC_DIST = 0.05  # From JSON distance_threshold
    GLOBAL_RANSAC_ITERATIONS = 20000  # From JSON
    GLOBAL_EDGE_LENGTH_RATIO = 0.8  # From JSON
    GLOBAL_NORMAL_ANGLE_DEG = 4.0  # From JSON angle_threshold
    max_correspondence_distance = 0.5

    # Local Refinement - FROM ICP ORIGINAL JSON
    LOCAL_ICP_DIST = 0.025  # From JSON refine_registration distance_threshold
    LOCAL_ICP_ITERATIONS = 2000
    
    # Adaptive Refinement - FROM ICP ORIGINAL JSON
    ADAPTIVE_MAX_ITERATIONS = 50
    ADAPTIVE_FITNESS_THRESHOLD = 0.95  # From JSON run_icp fitness_threshold
    ADAPTIVE_RMSE_THRESHOLD = 0.005  # From JSON run_icp rmse_threshold
    ADAPTIVE_NOISE_ROTATION_RANGE = 0.0010 # From ICP original improve_icp_result
    ADAPTIVE_NOISE_TRANSLATION_START = 0.01 # From ICP original
    
    # Normal Estimation - FROM ICP ORIGINAL
    NORMALS_RADIUS = 50  # From JSON
    NORMALS_MAX_NN = 10  # From JSON
    
    # Visualization
    VISUALIZE_STEPS = False
    VISUALIZE_FINAL = True

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def setup_logging(log_file: Optional[Path] = None, level=logging.INFO):
    """Setup logging configuration"""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=handlers
    )


def timeit(func):
    """Decorator to measure function execution time"""
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logging.info(f"⏱️  {func.__name__}: {elapsed:.3f}s")
        return result
    return wrapper


def visualize_pcd(pcd: o3d.geometry.PointCloud, 
                  title: str = "Point Cloud",
                  width: int = 1280,
                  height: int = 720):
    """Visualize single point cloud"""
    if not ICPConfig.VISUALIZE_STEPS:
        return
    
    logging.info(f"🔍 Visualizing: {title} ({len(pcd.points)} points)")
    o3d.visualization.draw_geometries(
        [pcd],
        window_name=title,
        width=width,
        height=height
    )


def visualize_alignment(source: o3d.geometry.PointCloud,
                       target: o3d.geometry.PointCloud,
                       transformation: Optional[np.ndarray] = None,
                       title: str = "Alignment"):
    """Visualize source and target alignment"""
    if not ICPConfig.VISUALIZE_STEPS:
        return
    
    source_vis = copy.deepcopy(source).paint_uniform_color([1, 0, 0])  # Red
    target_vis = copy.deepcopy(target).paint_uniform_color([0, 1, 0])  # Green
    
    geometries = [source_vis, target_vis]
    
    if transformation is not None:
        source_transformed = copy.deepcopy(source)
        source_transformed.transform(transformation)
        source_transformed.paint_uniform_color([0, 0, 1])  # Blue
        geometries.append(source_transformed)
        logging.info(f"🔍 {title}: Red=Source, Green=Target, Blue=Aligned")
    else:
        logging.info(f"🔍 {title}: Red=Source, Green=Target")
    
    o3d.visualization.draw_geometries(
        geometries,
        window_name=title,
        width=1280,
        height=720
    )


def print_pcd_stats(pcd: o3d.geometry.PointCloud, name: str = "Point Cloud"):
    """Print point cloud statistics"""
    bbox = pcd.get_axis_aligned_bounding_box()
    center = pcd.get_center()
    
    logging.info(f"\n{'='*70}")
    logging.info(f"  {name.upper()} STATISTICS")
    logging.info(f"{'='*70}")
    logging.info(f"Points:        {len(pcd.points)}")
    logging.info(f"Has normals:   {pcd.has_normals()}")
    logging.info(f"Has colors:    {pcd.has_colors()}")
    logging.info(f"Center:        [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]")
    logging.info(f"BBox min:      [{bbox.min_bound[0]:.4f}, {bbox.min_bound[1]:.4f}, {bbox.min_bound[2]:.4f}]")
    logging.info(f"BBox max:      [{bbox.max_bound[0]:.4f}, {bbox.max_bound[1]:.4f}, {bbox.max_bound[2]:.4f}]")
    logging.info(f"BBox extent:   [{bbox.get_extent()[0]:.4f}, {bbox.get_extent()[1]:.4f}, {bbox.get_extent()[2]:.4f}]")
    logging.info(f"{'='*70}\n")


# ============================================================================
# PREPROCESSING - SCENE POINT CLOUD
# ============================================================================

@timeit
def preprocess_scene(pcd_path: Path,
                     config: ICPConfig = ICPConfig(),
                     save_path: Optional[Path] = None) -> o3d.geometry.PointCloud:
    """
    Preprocess reconstructed scene point cloud
    
    Steps:
    1. Load point cloud
    2. Remove statistical outliers (noise removal)
    3. Estimate normals for plane detection
    4. Detect and remove table/floor plane
    5. Conservative downsampling
    
    Args:
        pcd_path: Path to scene PLY file
        config: Configuration object
        save_path: Optional path to save preprocessed cloud
    
    Returns:
        Preprocessed point cloud
    """
    logging.info("\n" + "="*70)
    logging.info("  SCENE PREPROCESSING")
    logging.info("="*70)
    
    # Step 1: Load
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    original_count = len(pcd.points)
    logging.info(f"✓ Loaded: {original_count} points from {pcd_path.name}")
    
    if config.VISUALIZE_STEPS:
        visualize_pcd(pcd, "Step 1: Original Scene")
    
    # Step 2: Remove outliers (CRITICAL for noisy reconstruction)
    logging.info(f"🔧 Removing statistical outliers...")
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=config.SCENE_OUTLIER_NEIGHBORS,
        std_ratio=config.SCENE_OUTLIER_STD
    )
    logging.info(f"   {original_count} → {len(pcd.points)} points")
    
    if config.VISUALIZE_STEPS:
        visualize_pcd(pcd, "Step 2: After Outlier Removal")
    
    # Step 3: Estimate normals (needed for plane detection)
    logging.info(f"🔧 Estimating normals...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.05, max_nn=30
        )
    )
    logging.info(f"   ✓ Normals estimated")
    
    # Step 4: Remove table/floor plane (CONSERVATIVE)
    logging.info(f"🔧 Removing table/floor plane (CONSERVATIVE)...")
    
    # Get average normal direction
    pcd_sample = pcd.voxel_down_sample(voxel_size=0.01)
    normals = np.asarray(pcd_sample.normals)
    avg_normal = np.mean(normals, axis=0)
    avg_normal /= np.linalg.norm(avg_normal)
    logging.info(f"   Average normal: [{avg_normal[0]:.3f}, {avg_normal[1]:.3f}, {avg_normal[2]:.3f}]")
    
    # RANSAC plane detection
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=config.PLANE_DISTANCE_THRESHOLD,
        ransac_n=3,
        num_iterations=config.PLANE_RANSAC_ITERATIONS
    )
    
    [a, b, c, d] = plane_model
    plane_normal = np.array([a, b, c]) / np.linalg.norm([a, b, c])
    logging.info(f"   Plane: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")
    logging.info(f"   Inliers: {len(inliers)} points")
    
    # Flip if needed
    if np.dot(plane_normal, avg_normal) > 0:
        logging.info(f"   ⚠️  Flipping plane normal")
        plane_model = [-a, -b, -c, -d]
        [a, b, c, d] = plane_model
    
    # Remove points below plane with small offset
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    
    plane_norm = np.sqrt(a**2 + b**2 + c**2)
    d_offset = d - config.PLANE_OFFSET * plane_norm
    
    distances = (a * points[:, 0] + 
                 b * points[:, 1] + 
                 c * points[:, 2] + d_offset) / plane_norm
    
    above_mask = distances <= 0
    
    pcd_filtered = o3d.geometry.PointCloud()
    pcd_filtered.points = o3d.utility.Vector3dVector(points[above_mask])
    if colors is not None:
        pcd_filtered.colors = o3d.utility.Vector3dVector(colors[above_mask])
    
    logging.info(f"   {len(pcd.points)} → {len(pcd_filtered.points)} points (kept {100*len(pcd_filtered.points)/len(pcd.points):.1f}%)")
    
    # Re-estimate normals after filtering
    logging.info(f"🔧 Re-estimating normals...")
    pcd_filtered.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.05, max_nn=30
        )
    )
    
    if config.VISUALIZE_STEPS:
        visualize_pcd(pcd_filtered, "Step 3: After Plane Removal")
    
    # Step 5: Conservative downsampling
    logging.info(f"🔧 Downsampling (voxel={config.SCENE_DOWNSAMPLE_VOXEL})...")
    pcd_final = pcd_filtered.voxel_down_sample(voxel_size=config.SCENE_DOWNSAMPLE_VOXEL)
    logging.info(f"   {len(pcd_filtered.points)} → {len(pcd_final.points)} points")
    
    if config.VISUALIZE_STEPS:
        visualize_pcd(pcd_final, "Step 4: Final Preprocessed Scene")
    
    # Save if requested
    if save_path:
        o3d.io.write_point_cloud(str(save_path), pcd_final)
        logging.info(f"💾 Saved: {save_path}")
    
    logging.info(f"\n✅ SCENE PREPROCESSING COMPLETE: {original_count} → {len(pcd_final.points)} points")
    print_pcd_stats(pcd_final, "Preprocessed Scene")
    
    return pcd_final


# ============================================================================
# PREPROCESSING - OBJECT POINT CLOUD
# ============================================================================

@timeit
def preprocess_object(pcd_path: Path,
                      config: ICPConfig = ICPConfig()) -> o3d.geometry.PointCloud:
    """
    Preprocess object CAD model point cloud
    
    Steps:
    1. Load point cloud
    2. Remove statistical outliers
    3. Downsample to match scene resolution
    4. Estimate normals
    
    Args:
        pcd_path: Path to object PLY file
        config: Configuration object
    
    Returns:
        Preprocessed point cloud
    """
    logging.info("\n" + "="*70)
    logging.info("  OBJECT PREPROCESSING")
    logging.info("="*70)
    
    # Load
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    original_count = len(pcd.points)
    logging.info(f"✓ Loaded: {original_count} points from {pcd_path.name}")
    
    if config.VISUALIZE_STEPS:
        visualize_pcd(pcd, "Object: Original")
    
    # Remove outliers
    # logging.info(f"🔧 Removing outliers...")
    # pcd, _ = pcd.remove_statistical_outlier(
    #     nb_neighbors=config.OBJECT_OUTLIER_NEIGHBORS,
    #     std_ratio=config.OBJECT_OUTLIER_STD
    # )
    logging.info(f"   {original_count} → {len(pcd.points)} points")
    
    # Downsample
    logging.info(f"🔧 Downsampling (voxel={config.OBJECT_DOWNSAMPLE_VOXEL})...")
    pcd = pcd.voxel_down_sample(voxel_size=config.OBJECT_DOWNSAMPLE_VOXEL)
    logging.info(f"   → {len(pcd.points)} points")
    
    # Estimate normals
    logging.info(f"🔧 Estimating normals...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=config.OBJECT_DOWNSAMPLE_VOXEL * 5,
            max_nn=30
        )
    )
    
    if config.VISUALIZE_STEPS:
        visualize_pcd(pcd, "Object: Preprocessed")
    
    logging.info(f"\n✅ OBJECT PREPROCESSING COMPLETE")
    print_pcd_stats(pcd, "Preprocessed Object")
    
    return pcd


# ============================================================================
# SCALE ESTIMATION
# ============================================================================

def estimate_scale_bbox(source: o3d.geometry.PointCloud,
                       target: o3d.geometry.PointCloud) -> float:
    """Estimate scale from bounding box diagonal ratio"""
    source_bbox = source.get_axis_aligned_bounding_box()
    target_bbox = target.get_axis_aligned_bounding_box()
    
    source_diag = np.linalg.norm(source_bbox.get_extent())
    target_diag = np.linalg.norm(target_bbox.get_extent())
    
    scale = target_diag / source_diag
    logging.info(f"   BBox diagonal ratio: {scale:.6f}")
    logging.info(f"   Source diagonal: {source_diag:.4f}")
    logging.info(f"   Target diagonal: {target_diag:.4f}")
    
    return scale


def estimate_scale_multi_scale(source: o3d.geometry.PointCloud,
                               target: o3d.geometry.PointCloud,
                               candidates: list,
                               max_corr_dist: float = 0.01) -> Tuple[float, np.ndarray]:
    """
    Try multiple scale candidates and pick best based on ICP fitness
    
    This is MORE ROBUST than RANSAC for small/sparse scenes
    """
    logging.info(f"   Testing {len(candidates)} scale candidates...")
    
    best_fitness = -1
    best_scale = 1.0
    best_transformation = np.eye(4)
    
    results = []
    
    for scale in candidates:
        # Scale source
        source_scaled = copy.deepcopy(source)
        source_scaled.scale(scale, center=source_scaled.get_center())
        
        # Align centers
        translation = target.get_center() - source_scaled.get_center()
        init_transform = np.eye(4)
        init_transform[:3, 3] = translation
        source_scaled.transform(init_transform)
        
        # Quick ICP test
        result = o3d.pipelines.registration.registration_icp(
            source=source_scaled,
            target=target,
            max_correspondence_distance=max_corr_dist,
            init=np.eye(4),
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
        )
        
        results.append({
            'scale': scale,
            'fitness': result.fitness,
            'rmse': result.inlier_rmse
        })
        
        logging.info(f"      Scale {scale:.3f}: fitness={result.fitness:.4f}, rmse={result.inlier_rmse:.6f}")
        
        if result.fitness > best_fitness:
            best_fitness = result.fitness
            best_scale = scale
            best_transformation = init_transform @ result.transformation
    
    logging.info(f"   ✅ Best scale: {best_scale:.6f} (fitness={best_fitness:.4f})")
    
    return best_scale, best_transformation


@timeit
def estimate_scale(source: o3d.geometry.PointCloud,
                   target: o3d.geometry.PointCloud,
                   config: ICPConfig = ICPConfig()) -> Tuple[float, np.ndarray]:
    """
    Robust scale estimation with fallback strategy
    
    Returns:
        scale: Estimated scale factor
        initial_transform: Initial transformation (includes center alignment)
    """
    logging.info("\n" + "="*70)
    logging.info("  SCALE ESTIMATION")
    logging.info("="*70)
    
    if config.SCALE_METHOD == "multi_scale":
        scale, init_transform = estimate_scale_multi_scale(
            source, target,
            candidates=config.SCALE_CANDIDATES,
            max_corr_dist=config.SCALE_CORRESPONDENCE_DIST
        )
    elif config.SCALE_METHOD == "bbox":
        scale = estimate_scale_bbox(source, target)
        # Create initial transformation (scale + center alignment)
        source_scaled = copy.deepcopy(source)
        source_scaled.scale(scale, center=source_scaled.get_center())
        translation = target.get_center() - source_scaled.get_center()
        init_transform = np.eye(4)
        init_transform[:3, 3] = translation
    else:
        raise ValueError(f"Unknown scale method: {config.SCALE_METHOD}")
    
    logging.info(f"\n✅ SCALE ESTIMATION COMPLETE: {scale:.6f}")
    
    return scale, init_transform


# ============================================================================
# GLOBAL REGISTRATION (FPFH + RANSAC)
# ============================================================================

# @timeit
# def compute_fpfh(pcd: o3d.geometry.PointCloud,
#                  voxel_size: float,
#                  radius_multiplier: float = 5.0):
#     """Compute FPFH features for point cloud"""
#     radius_normal = voxel_size * 2
#     radius_feature = voxel_size * radius_multiplier
    
#     # Estimate normals if not present
#     if not pcd.has_normals():
#         pcd.estimate_normals(
#             search_param=o3d.geometry.KDTreeSearchParamHybrid(
#                 radius=radius_normal, max_nn=30
#             )
#         )
    
#     fpfh = o3d.pipelines.registration.compute_fpfh_feature(
#         pcd,
#         o3d.geometry.KDTreeSearchParamHybrid(
#             radius=radius_feature, max_nn=100
#         )
#     )
    
#     return fpfh


@timeit
def global_registration(source: o3d.geometry.PointCloud,
                        target: o3d.geometry.PointCloud,
                        config: ICPConfig = ICPConfig()) -> o3d.pipelines.registration.RegistrationResult:
    """Global registration - using ICP original parameters"""
    logging.info("\n" + "="*70)
    logging.info("  GLOBAL REGISTRATION (FPFH + RANSAC)")
    logging.info("="*70)
    
    # Downsample
    source_down = source.voxel_down_sample(config.GLOBAL_VOXEL_SIZE)
    target_down = target.voxel_down_sample(config.GLOBAL_VOXEL_SIZE)
    logging.info(f"   Source: {len(source.points)} → {len(source_down.points)}")
    logging.info(f"   Target: {len(target.points)} → {len(target_down.points)}")
    
    # Estimate normals - like ICP original
    if not source_down.has_normals():
        source_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=config.NORMALS_RADIUS, max_nn=config.NORMALS_MAX_NN
            )
        )
    if not target_down.has_normals():
        target_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=config.NORMALS_RADIUS, max_nn=config.NORMALS_MAX_NN
            )
        )
    
    # FPFH features - using ICP original parameters
    logging.info(f"🔧 Computing FPFH features...")
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=config.GLOBAL_FPFH_RADIUS,
            max_nn=config.GLOBAL_FPFH_MAX_NN
        )
    )
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=config.GLOBAL_FPFH_RADIUS,
            max_nn=config.GLOBAL_FPFH_MAX_NN
        )
    )
    
    # RANSAC - using ICP original parameters
    logging.info(f"🔧 Running RANSAC...")
    logging.info(f"   Distance threshold: {config.GLOBAL_RANSAC_DIST}")
    
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down,
        source_fpfh, target_fpfh,
        mutual_filter=False,  # Like ICP original
        max_correspondence_distance=config.max_correspondence_distance,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                config.GLOBAL_EDGE_LENGTH_RATIO
            ),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                config.GLOBAL_RANSAC_DIST
            ),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(
                np.deg2rad(config.GLOBAL_NORMAL_ANGLE_DEG)
            )
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            config.GLOBAL_RANSAC_ITERATIONS, 0.999
        )
    )
    
    logging.info(f"\n✅ GLOBAL: fitness={result.fitness:.4f}, rmse={result.inlier_rmse:.6f}")
    
    return result


# ============================================================================
# LOCAL REFINEMENT (ICP)
# ============================================================================

@timeit
def local_refinement(source: o3d.geometry.PointCloud,
                     target: o3d.geometry.PointCloud,
                     initial_transform: np.ndarray,
                     config: ICPConfig = ICPConfig()) -> o3d.pipelines.registration.RegistrationResult:
    """
    Local ICP refinement - EXACTLY like ICP original refine_registration
    NO downsampling, simple ICP call
    """
    logging.info("\n" + "="*70)
    logging.info("  LOCAL REFINEMENT (ICP)")
    logging.info("="*70)
    
    # Ensure normals exist - like ICP original estimate_normals
    if not source.has_normals():
        source.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=config.NORMALS_RADIUS, 
                max_nn=config.NORMALS_MAX_NN
            )
        )
    if not target.has_normals():
        target.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=config.NORMALS_RADIUS,
                max_nn=config.NORMALS_MAX_NN
            )
        )
    
    # ICP - Point-to-Plane EXACTLY like ICP original refine_registration
    logging.info(f"🔧 Running point-to-plane ICP (like ICP original)...")
    logging.info(f"   Distance threshold: {config.LOCAL_ICP_DIST}")
    
    result = o3d.pipelines.registration.registration_icp(
        source, target,
        config.LOCAL_ICP_DIST,
        initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=200
        )
    )

    logging.info(f"\n✅ LOCAL REFINEMENT COMPLETE")
    logging.info(f"   Fitness: {result.fitness:.4f}")
    logging.info(f"   RMSE: {result.inlier_rmse:.6f}")
    logging.info(f"   Correspondences: {len(result.correspondence_set)}")
    
    return result


# ============================================================================
# ADAPTIVE REFINEMENT (from your second code)
# ============================================================================

@timeit
def adaptive_refinement(source: o3d.geometry.PointCloud,
                        target: o3d.geometry.PointCloud,
                        initial_result: o3d.pipelines.registration.RegistrationResult,
                        config: ICPConfig = ICPConfig()) -> o3d.pipelines.registration.RegistrationResult:
    """
    Adaptive iterative refinement with random perturbations
    
    This method from your second code is excellent for escaping local minima!
    
    Strategy:
    - Start from current best transformation
    - Apply small random perturbations (rotation + translation)
    - Try ICP refinement
    - Keep result if better
    - Increase perturbation if stuck
    
    Args:
        source: Source point cloud
        target: Target point cloud
        initial_result: Initial registration result
        config: Configuration
    
    Returns:
        Improved registration result
    """
    logging.info("\n" + "="*70)
    logging.info("  ADAPTIVE REFINEMENT")
    logging.info("="*70)
    
    best_fitness = initial_result.fitness
    best_rmse = initial_result.inlier_rmse
    best_transformation = initial_result.transformation
    
    iteration = 0
    noise_translation = config.ADAPTIVE_NOISE_TRANSLATION_START
    
    logging.info(f"🎯 Target: fitness > {config.ADAPTIVE_FITNESS_THRESHOLD}, RMSE < {config.ADAPTIVE_RMSE_THRESHOLD}")
    logging.info(f"🔧 Starting from: fitness={best_fitness:.4f}, RMSE={best_rmse:.6f}")
    
    while (iteration < config.ADAPTIVE_MAX_ITERATIONS and
           (best_fitness < config.ADAPTIVE_FITNESS_THRESHOLD or
            best_rmse > config.ADAPTIVE_RMSE_THRESHOLD)):
        
        # Generate random perturbation
        noise_rotation = o3d.geometry.get_rotation_matrix_from_xyz(
            [np.random.uniform(-config.ADAPTIVE_NOISE_ROTATION_RANGE, 
                              config.ADAPTIVE_NOISE_ROTATION_RANGE) for _ in range(3)]
        )
        noise_trans_vec = np.random.uniform(-noise_translation, noise_translation, 3)
        
        noise_transform = np.eye(4)
        noise_transform[:3, :3] = noise_rotation
        noise_transform[:3, 3] = noise_trans_vec
        
        current_transform = noise_transform @ best_transformation
        
        # Try refinement
        try:
            result = o3d.pipelines.registration.registration_icp(
                source, target,
                config.LOCAL_ICP_DIST * 0.5,
                current_transform,
                o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )
            
            # Check if improved
            if result.fitness > 0 and result.inlier_rmse > 0:
                if (result.fitness > best_fitness or
                    (result.fitness == best_fitness and result.inlier_rmse < best_rmse)):
                    
                    improvement = result.fitness - best_fitness
                    best_fitness = result.fitness
                    best_rmse = result.inlier_rmse
                    best_transformation = result.transformation
                    
                    logging.info(f"   ✅ Iter {iteration+1}: fitness={best_fitness:.4f} (+{improvement:.4f}), RMSE={best_rmse:.6f}")
                    
                    # Check if reached target
                    if best_fitness >= config.ADAPTIVE_FITNESS_THRESHOLD and best_rmse <= config.ADAPTIVE_RMSE_THRESHOLD:
                        logging.info(f"   🎉 Target reached!")
                        break
            else:
                # Increase noise if stuck
                noise_translation += 0.75
                
        except Exception as e:
            logging.debug(f"   ⚠️  Iter {iteration+1} error: {e}")
            noise_translation += 0.1
        
        iteration += 1
        
        if iteration % 10 == 0:
            logging.info(f"   📊 Progress: iter={iteration}, best_fitness={best_fitness:.4f}, noise={noise_translation:.2f}")
    
    final_result = o3d.pipelines.registration.RegistrationResult()
    final_result.fitness = best_fitness
    final_result.inlier_rmse = best_rmse
    final_result.transformation = best_transformation
    
    logging.info(f"\n✅ ADAPTIVE REFINEMENT COMPLETE ({iteration} iterations)")
    logging.info(f"   Final fitness: {best_fitness:.4f}")
    logging.info(f"   Final RMSE: {best_rmse:.6f}")
    
    return final_result


@timeit
def initial_alignment_ransac(source: o3d.geometry.PointCloud,
                             target: o3d.geometry.PointCloud,
                             config: ICPConfig,
                             n_tries: int = 10) -> Tuple[o3d.pipelines.registration.RegistrationResult, list]:
    """
    Find initial alignment using FPFH + RANSAC (like ICP original)
    Try multiple times with different RANSAC seeds
    """
    logging.info(f"\n🔄 RANSAC Initial Alignment ({n_tries} attempts)")
    
    # Downsample for FPFH
    source_down = source.voxel_down_sample(0.01)
    target_down = target.voxel_down_sample(0.01)
    
    # Estimate normals
    source_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
    )
    target_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
    )
    
    # Compute FPFH features
    logging.info(f"   Computing FPFH features...")
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=100)
    )
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=100)
    )
    
    all_results = []
    best_result = None
    best_fitness = -1
    
    for i in range(n_tries):
        # Run RANSAC
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down,
            source_fpfh, target_fpfh,
            mutual_filter=True,
            max_correspondence_distance=0.05,  # 5cm
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=3,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(0.05)
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
        )
        
        all_results.append({
            'attempt': i + 1,
            'fitness': result.fitness,
            'rmse': result.inlier_rmse
        })
        
        logging.info(f"   Try {i+1}/{n_tries}: fitness={result.fitness:.4f}, RMSE={result.inlier_rmse:.6f}")
        
        if result.fitness > best_fitness:
            best_fitness = result.fitness
            best_result = result
            logging.info(f"      ✅ NEW BEST!")
    
    logging.info(f"\n   🎯 Best fitness: {best_fitness:.4f}")
    return best_result, all_results
# ============================================================================
# MAIN PIPELINE
# ============================================================================

@timeit
def run_icp_pipeline(object_ply_path: Path,
                     scene_ply_path: Path,
                     output_dir: Path,
                     config: ICPConfig = ICPConfig()) -> Dict:
    """
    Complete ICP alignment pipeline
    
    Pipeline stages:
    1. Preprocess scene (remove plane, outliers, downsample)
    2. Preprocess object (outliers, downsample, normals)
    3. Estimate scale (multi-scale search)
    4. Global registration (FPFH + RANSAC)
    5. Local refinement (point-to-plane ICP)
    6. Adaptive refinement (iterative improvement)
    
    Args:
        object_ply_path: Path to object CAD model
        scene_ply_path: Path to reconstructed scene
        output_dir: Directory to save results
        config: Configuration object
    
    Returns:
        Dictionary with metrics and results
    """
    start_time = time.perf_counter()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info("\n" + "="*70)
    logging.info("  ICP ALIGNMENT PIPELINE")
    logging.info("="*70)
    logging.info(f"Object: {object_ply_path.name}")
    logging.info(f"Scene:  {scene_ply_path.name}")
    logging.info(f"Output: {output_dir}")
    logging.info("="*70 + "\n")
    
    # ========================================================================
    # STAGE 1: PREPROCESSING
    # ========================================================================
    
    scene_pcd = preprocess_scene(
        scene_ply_path,
        config=config,
        save_path=output_dir / "scene_preprocessed.ply"
    )
    
    object_pcd = preprocess_object(
        object_ply_path,
        config=config
    )
    
    # Visualize before alignment
    if config.VISUALIZE_STEPS:
        visualize_alignment(object_pcd, scene_pcd, title="Before Alignment")
    
    # ========================================================================
    # STAGE 2: SCALE ESTIMATION
    # ========================================================================
    
    scale, scale_transform = estimate_scale(object_pcd, scene_pcd, config)
    
    # Apply scale
    object_pcd_scaled = copy.deepcopy(object_pcd)
    object_pcd_scaled.scale(scale, center=object_pcd_scaled.get_center())
    object_pcd_scaled.transform(scale_transform)
    
    if config.VISUALIZE_STEPS:
        visualize_alignment(object_pcd_scaled, scene_pcd, title="After Scale + Center Alignment")
    
    # ========================================================================
    # STAGE 3: GLOBAL REGISTRATION
    # ========================================================================
    # NEW (USE SCALE TRANSFORM AS INIT):
    logging.info("\n" + "="*70)
    logging.info("  SKIPPING GLOBAL - Using scale alignment as init")
    logging.info("="*70)

    # Create fake result using scale transform as initialization
    global_result = o3d.pipelines.registration.RegistrationResult()
    global_result.transformation = np.eye(4)  # Identity - object already aligned by scale
    global_result.fitness = 0.5
    global_result.inlier_rmse = 0.01
        
    if config.VISUALIZE_STEPS:
        visualize_alignment(object_pcd_scaled, scene_pcd, 
                          global_result.transformation,
                          title="After Global Registration")
    
    # ========================================================================
    # STAGE 4: LOCAL REFINEMENT
    # ========================================================================
    
    # local_result = local_refinement(
    #     object_pcd_scaled, scene_pcd,
    #     global_result.transformation,
    #     config
    # )

    # ========================================================================
    # STAGE 4: MULTI-START LOCAL REFINEMENT
    # ========================================================================

    logging.info("\n" + "="*70)
    logging.info("  MULTI-START ICP REFINEMENT")
    logging.info("="*70)

    # Try multiple random initial poses
    local_result, all_attempts = initial_alignment_ransac(
        object_pcd_scaled, scene_pcd,
        config,
        n_tries=20
    )

    # ✅ ADD THIS: Refine RANSAC result with ICP
    logging.info(f"\n🔧 Refining RANSAC result with ICP...")
    local_result = o3d.pipelines.registration.registration_icp(
        object_pcd_scaled, scene_pcd,
        config.LOCAL_ICP_DIST,
        local_result.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=500)  # More iterations
    )
    logging.info(f"   After ICP: fitness={local_result.fitness:.4f}, RMSE={local_result.inlier_rmse:.6f}")

    logging.info(f"\n📊 All {len(all_attempts)} attempts summary:")

    # Sort by fitness (descending)
    sorted_attempts = sorted([a for a in all_attempts if 'error' not in a], 
                            key=lambda x: x['fitness'], reverse=True)
    for i, attempt in enumerate(sorted_attempts[:5]):  # Show top 5
        logging.info(f"   #{i+1}: fitness={attempt['fitness']:.4f}, RMSE={attempt['rmse']:.6f}")

    if config.VISUALIZE_STEPS:
        visualize_alignment(object_pcd_scaled, scene_pcd,
                        local_result.transformation,
                        title="After Multi-Start Refinement (BEST)")
    
    # ========================================================================
    # STAGE 5: ADAPTIVE REFINEMENT
    # ========================================================================
    
    final_result = adaptive_refinement(
        object_pcd_scaled, scene_pcd,
        local_result,
        config
    )
    
    # ========================================================================
    # FINALIZE
    # ========================================================================
    
    # NEW (CORRECT):
    # Combine scale + ICP transformations
    final_transformation = np.dot(final_result.transformation, scale_transform)

    # Transform ORIGINAL object with COMBINED transformation
    object_aligned = copy.deepcopy(object_pcd)
    object_aligned.scale(scale, center=object_aligned.get_center())
    object_aligned.transform(final_transformation)
    
    # NEW (only 2 point clouds):
    if config.VISUALIZE_FINAL:
        # Only show: Scene (gray) + Aligned object (green)
        target_vis = copy.deepcopy(scene_pcd).paint_uniform_color([1, 0, 0])  # Red
        aligned_vis = copy.deepcopy(object_aligned).paint_uniform_color([0, 1, 0])  # Green
        
        logging.info("\n🎬 Final Visualization")
        logging.info("   Red = Scene")
        logging.info("   Green = Aligned Object")
        
        o3d.visualization.draw_geometries(
            [target_vis, aligned_vis],  # ✅ Only these 2!
            window_name="Final Result",
            width=1280,
            height=720
        )
        
    # Save results
    logging.info("\n💾 Saving results...")
    
    np.save(output_dir / "transformation.npy", final_transformation)
    np.save(output_dir / "scale.npy", np.array([scale]))
    o3d.io.write_point_cloud(str(output_dir / "object_aligned.ply"), object_aligned)
    
    metrics = {
        'scale': float(scale),
        'global_fitness': float(global_result.fitness),
        'global_rmse': float(global_result.inlier_rmse),
        'local_fitness': float(local_result.fitness),
        'local_rmse': float(local_result.inlier_rmse),
        'final_fitness': float(final_result.fitness),
        'final_rmse': float(final_result.inlier_rmse),
        'final_correspondences': len(final_result.correspondence_set),
        'transformation': final_transformation.tolist(),
        'elapsed_time': time.perf_counter() - start_time
    }
    
    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logging.info(f"   ✓ transformation.npy")
    logging.info(f"   ✓ scale.npy")
    logging.info(f"   ✓ object_aligned.ply")
    logging.info(f"   ✓ metrics.json")
    
    # Final summary
    elapsed = time.perf_counter() - start_time
    
    logging.info("\n" + "="*70)
    logging.info("  PIPELINE COMPLETE")
    logging.info("="*70)
    logging.info(f"⏱️  Total time: {elapsed:.2f}s")
    logging.info(f"📏 Scale: {scale:.6f}")
    logging.info(f"📊 Final fitness: {final_result.fitness:.4f}")
    logging.info(f"📊 Final RMSE: {final_result.inlier_rmse:.6f}")
    logging.info(f"📊 Correspondences: {len(final_result.correspondence_set)}")
    
    if final_result.fitness >= 0.85:
        logging.info(f"✅ EXCELLENT alignment!")
    elif final_result.fitness >= 0.6:
        logging.info(f"✓  GOOD alignment")
    elif final_result.fitness >= 0.3:
        logging.info(f"⚠️  MODERATE alignment - may need tuning")
    else:
        logging.info(f"❌ POOR alignment - check input data")
    
    logging.info("="*70 + "\n")
    
    return metrics


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage for UR5e robot scanning pipeline
    """
    
    # Setup logging
    setup_logging(level=logging.INFO)
    
    # Paths (MODIFY THESE)
    SCENE_DIR = Path("/home/AP_PathMatters/path_matters/datasets/yoda")
    OBJECT_PLY = SCENE_DIR / "Baby_Yoda.ply"
    SCENE_PLY = SCENE_DIR / "sparse/points.ply"
    OUTPUT_DIR = SCENE_DIR / "icp_results_improved"
    
    # Configuration
    config = ICPConfig()
    
    # Adjust for your scene (IMPORTANT!)
    # If scene is very small/sparse, use smaller values
    config.SCENE_DOWNSAMPLE_VOXEL = 0.002  # 2mm
    config.OBJECT_DOWNSAMPLE_VOXEL = 0.002
    config.PLANE_OFFSET = -0.02  # Keep more points
    config.SCALE_CANDIDATES = [0.1, 0.15, 0.2, 0.25, 0.3]  # Adjust range
    config.VISUALIZE_STEPS = True
    config.DEBUG_MODE = True
    
    # Run pipeline
    metrics = run_icp_pipeline(
        object_ply_path=OBJECT_PLY,
        scene_ply_path=SCENE_PLY,
        output_dir=OUTPUT_DIR,
        config=config
    )
    
    print(f"\n✅ Pipeline complete! Results saved to: {OUTPUT_DIR}")
    print(f"   Final fitness: {metrics['final_fitness']:.4f}")
    print(f"   Final RMSE: {metrics['final_rmse']:.6f}")







# #!/usr/bin/env python3
# """
# Improved ICP Alignment Pipeline for UR5e Robot Scanning
# ========================================================

# This pipeline performs robust point cloud alignment with:
# - Adaptive preprocessing
# - Multi-method scale estimation
# - Global + Local registration
# - Iterative refinement with adaptive search

# Author: Ziad
# Date: 2025
# """

# import numpy as np
# import copy
# import time
# import json
# import logging
# from pathlib import Path
# from typing import Tuple, Optional, Dict

# import open3d as o3d
# import torch
# import torch.nn.functional as F


# # ============================================================================
# # CONFIGURATION
# # ============================================================================

# class ICPConfig:
#     """Configuration parameters for ICP pipeline"""
    
#     # Preprocessing - Scene
#     SCENE_DOWNSAMPLE_VOXEL = 0.002  # Fine downsampling (2mm)
#     SCENE_OUTLIER_NEIGHBORS = 20
#     SCENE_OUTLIER_STD = 2.0
    
#     # Preprocessing - Object
#     OBJECT_DOWNSAMPLE_VOXEL = 0.002  # Match scene resolution
#     OBJECT_OUTLIER_NEIGHBORS = 20
#     OBJECT_OUTLIER_STD = 2.0
    
#     # Plane Removal (MORE CONSERVATIVE)
#     PLANE_DISTANCE_THRESHOLD = 0.015  # Increased from 0.01
#     PLANE_RANSAC_ITERATIONS = 1000
#     PLANE_OFFSET = -0.02  # Reduced from 0.02 to keep more points
    
#     # Scale Estimation
#     SCALE_METHOD = "multi_scale"  # More robust than RANSAC for small scenes
#     SCALE_CORRESPONDENCE_DIST = 0.01
#     SCALE_CANDIDATES = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]  # Fine-grained search
    
#     # Global Registration (FPFH + RANSAC)
#     GLOBAL_VOXEL_SIZE = 0.001  # Reduced from 0.02 for better features
#     GLOBAL_FPFH_RADIUS_MULTIPLIER = 5.0
#     GLOBAL_RANSAC_DIST = 0.0015  # Reduced from 0.03
#     GLOBAL_RANSAC_ITERATIONS = 100000
#     GLOBAL_EDGE_LENGTH_RATIO = 0.9
#     GLOBAL_NORMAL_ANGLE_DEG = 30
    
#     # Local Refinement (ICP)
#     LOCAL_VOXEL_SIZE = 0.002  # Fine resolution
#     LOCAL_ICP_DIST = 0.001  # Reduced from 0.01
#     LOCAL_ICP_ITERATIONS = 200
    
#     # Adaptive Refinement
#     ADAPTIVE_MAX_ITERATIONS = 250
#     ADAPTIVE_FITNESS_THRESHOLD = 0.85  # Increased from default
#     ADAPTIVE_RMSE_THRESHOLD = 0.005
#     ADAPTIVE_NOISE_ROTATION_RANGE = 0.1  # radians
#     ADAPTIVE_NOISE_TRANSLATION_START = 0.25  # mm
    
#     # Visualization
#     VISUALIZE_STEPS = False
#     VISUALIZE_FINAL = True
#     DEBUG_MODE = True


# # ============================================================================
# # UTILITY FUNCTIONS
# # ============================================================================

# def setup_logging(log_file: Optional[Path] = None, level=logging.INFO):
#     """Setup logging configuration"""
#     handlers = [logging.StreamHandler()]
#     if log_file:
#         handlers.append(logging.FileHandler(log_file))
    
#     logging.basicConfig(
#         level=level,
#         format='%(asctime)s [%(levelname)s] %(message)s',
#         handlers=handlers
#     )


# def timeit(func):
#     """Decorator to measure function execution time"""
#     def wrapper(*args, **kwargs):
#         start = time.perf_counter()
#         result = func(*args, **kwargs)
#         elapsed = time.perf_counter() - start
#         logging.info(f"⏱️  {func.__name__}: {elapsed:.3f}s")
#         return result
#     return wrapper


# def visualize_pcd(pcd: o3d.geometry.PointCloud, 
#                   title: str = "Point Cloud",
#                   width: int = 1280,
#                   height: int = 720):
#     """Visualize single point cloud"""
#     if not ICPConfig.VISUALIZE_STEPS:
#         return
    
#     logging.info(f"🔍 Visualizing: {title} ({len(pcd.points)} points)")
#     o3d.visualization.draw_geometries(
#         [pcd],
#         window_name=title,
#         width=width,
#         height=height
#     )


# def visualize_alignment(source: o3d.geometry.PointCloud,
#                        target: o3d.geometry.PointCloud,
#                        transformation: Optional[np.ndarray] = None,
#                        title: str = "Alignment"):
#     """Visualize source and target alignment"""
#     if not ICPConfig.VISUALIZE_STEPS:
#         return
    
#     source_vis = copy.deepcopy(source).paint_uniform_color([1, 0, 0])  # Red
#     target_vis = copy.deepcopy(target).paint_uniform_color([0, 1, 0])  # Green
    
#     geometries = [source_vis, target_vis]
    
#     if transformation is not None:
#         source_transformed = copy.deepcopy(source)
#         source_transformed.transform(transformation)
#         source_transformed.paint_uniform_color([0, 0, 1])  # Blue
#         geometries.append(source_transformed)
#         logging.info(f"🔍 {title}: Red=Source, Green=Target, Blue=Aligned")
#     else:
#         logging.info(f"🔍 {title}: Red=Source, Green=Target")
    
#     o3d.visualization.draw_geometries(
#         geometries,
#         window_name=title,
#         width=1280,
#         height=720
#     )


# def print_pcd_stats(pcd: o3d.geometry.PointCloud, name: str = "Point Cloud"):
#     """Print point cloud statistics"""
#     bbox = pcd.get_axis_aligned_bounding_box()
#     center = pcd.get_center()
    
#     logging.info(f"\n{'='*70}")
#     logging.info(f"  {name.upper()} STATISTICS")
#     logging.info(f"{'='*70}")
#     logging.info(f"Points:        {len(pcd.points)}")
#     logging.info(f"Has normals:   {pcd.has_normals()}")
#     logging.info(f"Has colors:    {pcd.has_colors()}")
#     logging.info(f"Center:        [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]")
#     logging.info(f"BBox min:      [{bbox.min_bound[0]:.4f}, {bbox.min_bound[1]:.4f}, {bbox.min_bound[2]:.4f}]")
#     logging.info(f"BBox max:      [{bbox.max_bound[0]:.4f}, {bbox.max_bound[1]:.4f}, {bbox.max_bound[2]:.4f}]")
#     logging.info(f"BBox extent:   [{bbox.get_extent()[0]:.4f}, {bbox.get_extent()[1]:.4f}, {bbox.get_extent()[2]:.4f}]")
#     logging.info(f"{'='*70}\n")


# # ============================================================================
# # PREPROCESSING - SCENE POINT CLOUD
# # ============================================================================

# @timeit
# def preprocess_scene(pcd_path: Path,
#                      config: ICPConfig = ICPConfig(),
#                      save_path: Optional[Path] = None) -> o3d.geometry.PointCloud:
#     """
#     Preprocess reconstructed scene point cloud
    
#     Steps:
#     1. Load point cloud
#     2. Remove statistical outliers (noise removal)
#     3. Estimate normals for plane detection
#     4. Detect and remove table/floor plane
#     5. Conservative downsampling
    
#     Args:
#         pcd_path: Path to scene PLY file
#         config: Configuration object
#         save_path: Optional path to save preprocessed cloud
    
#     Returns:
#         Preprocessed point cloud
#     """
#     logging.info("\n" + "="*70)
#     logging.info("  SCENE PREPROCESSING")
#     logging.info("="*70)
    
#     # Step 1: Load
#     pcd = o3d.io.read_point_cloud(str(pcd_path))
#     original_count = len(pcd.points)
#     logging.info(f"✓ Loaded: {original_count} points from {pcd_path.name}")
    
#     if config.VISUALIZE_STEPS:
#         visualize_pcd(pcd, "Step 1: Original Scene")
    
#     # Step 2: Remove outliers (CRITICAL for noisy reconstruction)
#     logging.info(f"🔧 Removing statistical outliers...")
#     pcd, _ = pcd.remove_statistical_outlier(
#         nb_neighbors=config.SCENE_OUTLIER_NEIGHBORS,
#         std_ratio=config.SCENE_OUTLIER_STD
#     )
#     logging.info(f"   {original_count} → {len(pcd.points)} points")
    
#     if config.VISUALIZE_STEPS:
#         visualize_pcd(pcd, "Step 2: After Outlier Removal")
    
#     # Step 3: Estimate normals (needed for plane detection)
#     logging.info(f"🔧 Estimating normals...")
#     pcd.estimate_normals(
#         search_param=o3d.geometry.KDTreeSearchParamHybrid(
#             radius=0.05, max_nn=30
#         )
#     )
#     logging.info(f"   ✓ Normals estimated")
    
#     # Step 4: Remove table/floor plane (CONSERVATIVE)
#     logging.info(f"🔧 Removing table/floor plane (CONSERVATIVE)...")
    
#     # Get average normal direction
#     pcd_sample = pcd.voxel_down_sample(voxel_size=0.01)
#     normals = np.asarray(pcd_sample.normals)
#     avg_normal = np.mean(normals, axis=0)
#     avg_normal /= np.linalg.norm(avg_normal)
#     logging.info(f"   Average normal: [{avg_normal[0]:.3f}, {avg_normal[1]:.3f}, {avg_normal[2]:.3f}]")
    
#     # RANSAC plane detection
#     plane_model, inliers = pcd.segment_plane(
#         distance_threshold=config.PLANE_DISTANCE_THRESHOLD,
#         ransac_n=3,
#         num_iterations=config.PLANE_RANSAC_ITERATIONS
#     )
    
#     [a, b, c, d] = plane_model
#     plane_normal = np.array([a, b, c]) / np.linalg.norm([a, b, c])
#     logging.info(f"   Plane: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")
#     logging.info(f"   Inliers: {len(inliers)} points")
    
#     # Flip if needed
#     if np.dot(plane_normal, avg_normal) > 0:
#         logging.info(f"   ⚠️  Flipping plane normal")
#         plane_model = [-a, -b, -c, -d]
#         [a, b, c, d] = plane_model
    
#     # Remove points below plane with small offset
#     points = np.asarray(pcd.points)
#     colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    
#     plane_norm = np.sqrt(a**2 + b**2 + c**2)
#     d_offset = d - config.PLANE_OFFSET * plane_norm
    
#     distances = (a * points[:, 0] + 
#                  b * points[:, 1] + 
#                  c * points[:, 2] + d_offset) / plane_norm
    
#     above_mask = distances <= 0
    
#     pcd_filtered = o3d.geometry.PointCloud()
#     pcd_filtered.points = o3d.utility.Vector3dVector(points[above_mask])
#     if colors is not None:
#         pcd_filtered.colors = o3d.utility.Vector3dVector(colors[above_mask])
    
#     logging.info(f"   {len(pcd.points)} → {len(pcd_filtered.points)} points (kept {100*len(pcd_filtered.points)/len(pcd.points):.1f}%)")
    
#     # Re-estimate normals after filtering
#     logging.info(f"🔧 Re-estimating normals...")
#     pcd_filtered.estimate_normals(
#         search_param=o3d.geometry.KDTreeSearchParamHybrid(
#             radius=0.05, max_nn=30
#         )
#     )
    
#     if config.VISUALIZE_STEPS:
#         visualize_pcd(pcd_filtered, "Step 3: After Plane Removal")
    
#     # Step 5: Conservative downsampling
#     logging.info(f"🔧 Downsampling (voxel={config.SCENE_DOWNSAMPLE_VOXEL})...")
#     pcd_final = pcd_filtered.voxel_down_sample(voxel_size=config.SCENE_DOWNSAMPLE_VOXEL)
#     logging.info(f"   {len(pcd_filtered.points)} → {len(pcd_final.points)} points")
    
#     if config.VISUALIZE_STEPS:
#         visualize_pcd(pcd_final, "Step 4: Final Preprocessed Scene")
    
#     # Save if requested
#     if save_path:
#         o3d.io.write_point_cloud(str(save_path), pcd_final)
#         logging.info(f"💾 Saved: {save_path}")
    
#     logging.info(f"\n✅ SCENE PREPROCESSING COMPLETE: {original_count} → {len(pcd_final.points)} points")
#     print_pcd_stats(pcd_final, "Preprocessed Scene")
    
#     return pcd_final


# # ============================================================================
# # PREPROCESSING - OBJECT POINT CLOUD
# # ============================================================================

# @timeit
# def preprocess_object(pcd_path: Path,
#                       config: ICPConfig = ICPConfig()) -> o3d.geometry.PointCloud:
#     """
#     Preprocess object CAD model point cloud
    
#     Steps:
#     1. Load point cloud
#     2. Remove statistical outliers
#     3. Downsample to match scene resolution
#     4. Estimate normals
    
#     Args:
#         pcd_path: Path to object PLY file
#         config: Configuration object
    
#     Returns:
#         Preprocessed point cloud
#     """
#     logging.info("\n" + "="*70)
#     logging.info("  OBJECT PREPROCESSING")
#     logging.info("="*70)
    
#     # Load
#     pcd = o3d.io.read_point_cloud(str(pcd_path))
#     original_count = len(pcd.points)
#     logging.info(f"✓ Loaded: {original_count} points from {pcd_path.name}")
    
#     if config.VISUALIZE_STEPS:
#         visualize_pcd(pcd, "Object: Original")
    
#     # Remove outliers
#     logging.info(f"🔧 Removing outliers...")
#     pcd, _ = pcd.remove_statistical_outlier(
#         nb_neighbors=config.OBJECT_OUTLIER_NEIGHBORS,
#         std_ratio=config.OBJECT_OUTLIER_STD
#     )
#     logging.info(f"   {original_count} → {len(pcd.points)} points")
    
#     # Downsample
#     logging.info(f"🔧 Downsampling (voxel={config.OBJECT_DOWNSAMPLE_VOXEL})...")
#     pcd = pcd.voxel_down_sample(voxel_size=config.OBJECT_DOWNSAMPLE_VOXEL)
#     logging.info(f"   → {len(pcd.points)} points")
    
#     # Estimate normals
#     logging.info(f"🔧 Estimating normals...")
#     pcd.estimate_normals(
#         search_param=o3d.geometry.KDTreeSearchParamHybrid(
#             radius=config.OBJECT_DOWNSAMPLE_VOXEL * 5,
#             max_nn=30
#         )
#     )
    
#     if config.VISUALIZE_STEPS:
#         visualize_pcd(pcd, "Object: Preprocessed")
    
#     logging.info(f"\n✅ OBJECT PREPROCESSING COMPLETE")
#     print_pcd_stats(pcd, "Preprocessed Object")
    
#     return pcd


# # ============================================================================
# # SCALE ESTIMATION
# # ============================================================================

# def estimate_scale_bbox(source: o3d.geometry.PointCloud,
#                        target: o3d.geometry.PointCloud) -> float:
#     """Estimate scale from bounding box diagonal ratio"""
#     source_bbox = source.get_axis_aligned_bounding_box()
#     target_bbox = target.get_axis_aligned_bounding_box()
    
#     source_diag = np.linalg.norm(source_bbox.get_extent())
#     target_diag = np.linalg.norm(target_bbox.get_extent())
    
#     scale = target_diag / source_diag
#     logging.info(f"   BBox diagonal ratio: {scale:.6f}")
#     logging.info(f"   Source diagonal: {source_diag:.4f}")
#     logging.info(f"   Target diagonal: {target_diag:.4f}")
    
#     return scale


# def estimate_scale_multi_scale(source: o3d.geometry.PointCloud,
#                                target: o3d.geometry.PointCloud,
#                                candidates: list,
#                                max_corr_dist: float = 0.01) -> Tuple[float, np.ndarray]:
#     """
#     Try multiple scale candidates and pick best based on ICP fitness
    
#     This is MORE ROBUST than RANSAC for small/sparse scenes
#     """
#     logging.info(f"   Testing {len(candidates)} scale candidates...")
    
#     best_fitness = -1
#     best_scale = 1.0
#     best_transformation = np.eye(4)
    
#     results = []
    
#     for scale in candidates:
#         # Scale source
#         source_scaled = copy.deepcopy(source)
#         source_scaled.scale(scale, center=source_scaled.get_center())
        
#         # Align centers
#         translation = target.get_center() - source_scaled.get_center()
#         init_transform = np.eye(4)
#         init_transform[:3, 3] = translation
#         source_scaled.transform(init_transform)
        
#         # Quick ICP test
#         result = o3d.pipelines.registration.registration_icp(
#             source=source_scaled,
#             target=target,
#             max_correspondence_distance=max_corr_dist,
#             init=np.eye(4),
#             estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
#             criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
#         )
        
#         results.append({
#             'scale': scale,
#             'fitness': result.fitness,
#             'rmse': result.inlier_rmse
#         })
        
#         logging.info(f"      Scale {scale:.3f}: fitness={result.fitness:.4f}, rmse={result.inlier_rmse:.6f}")
        
#         if result.fitness > best_fitness:
#             best_fitness = result.fitness
#             best_scale = scale
#             best_transformation = init_transform @ result.transformation
    
#     logging.info(f"   ✅ Best scale: {best_scale:.6f} (fitness={best_fitness:.4f})")
    
#     return best_scale, best_transformation


# @timeit
# def estimate_scale(source: o3d.geometry.PointCloud,
#                    target: o3d.geometry.PointCloud,
#                    config: ICPConfig = ICPConfig()) -> Tuple[float, np.ndarray]:
#     """
#     Robust scale estimation with fallback strategy
    
#     Returns:
#         scale: Estimated scale factor
#         initial_transform: Initial transformation (includes center alignment)
#     """
#     logging.info("\n" + "="*70)
#     logging.info("  SCALE ESTIMATION")
#     logging.info("="*70)
    
#     if config.SCALE_METHOD == "multi_scale":
#         scale, init_transform = estimate_scale_multi_scale(
#             source, target,
#             candidates=config.SCALE_CANDIDATES,
#             max_corr_dist=config.SCALE_CORRESPONDENCE_DIST
#         )
#     elif config.SCALE_METHOD == "bbox":
#         scale = estimate_scale_bbox(source, target)
#         # Create initial transformation (scale + center alignment)
#         source_scaled = copy.deepcopy(source)
#         source_scaled.scale(scale, center=source_scaled.get_center())
#         translation = target.get_center() - source_scaled.get_center()
#         init_transform = np.eye(4)
#         init_transform[:3, 3] = translation
#     else:
#         raise ValueError(f"Unknown scale method: {config.SCALE_METHOD}")
    
#     logging.info(f"\n✅ SCALE ESTIMATION COMPLETE: {scale:.6f}")
    
#     return scale, init_transform


# # ============================================================================
# # GLOBAL REGISTRATION (FPFH + RANSAC)
# # ============================================================================

# @timeit
# def compute_fpfh(pcd: o3d.geometry.PointCloud,
#                  voxel_size: float,
#                  radius_multiplier: float = 5.0):
#     """Compute FPFH features for point cloud"""
#     radius_normal = voxel_size * 2
#     radius_feature = voxel_size * radius_multiplier
    
#     # Estimate normals if not present
#     if not pcd.has_normals():
#         pcd.estimate_normals(
#             search_param=o3d.geometry.KDTreeSearchParamHybrid(
#                 radius=radius_normal, max_nn=30
#             )
#         )
    
#     fpfh = o3d.pipelines.registration.compute_fpfh_feature(
#         pcd,
#         o3d.geometry.KDTreeSearchParamHybrid(
#             radius=radius_feature, max_nn=100
#         )
#     )
    
#     return fpfh


# @timeit
# def global_registration(source: o3d.geometry.PointCloud,
#                         target: o3d.geometry.PointCloud,
#                         config: ICPConfig = ICPConfig()) -> o3d.pipelines.registration.RegistrationResult:
#     """
#     Global registration using FPFH features + RANSAC
    
#     Steps:
#     1. Downsample for feature extraction
#     2. Compute FPFH features
#     3. RANSAC-based feature matching
    
#     Returns:
#         Registration result with initial transformation
#     """
#     logging.info("\n" + "="*70)
#     logging.info("  GLOBAL REGISTRATION (FPFH + RANSAC)")
#     logging.info("="*70)
    
#     # Downsample
#     logging.info(f"🔧 Downsampling (voxel={config.GLOBAL_VOXEL_SIZE})...")
#     source_down = source.voxel_down_sample(config.GLOBAL_VOXEL_SIZE)
#     target_down = target.voxel_down_sample(config.GLOBAL_VOXEL_SIZE)
#     logging.info(f"   Source: {len(source.points)} → {len(source_down.points)}")
#     logging.info(f"   Target: {len(target.points)} → {len(target_down.points)}")
    
#     # Compute FPFH features
#     logging.info(f"🔧 Computing FPFH features...")
#     source_fpfh = compute_fpfh(source_down, config.GLOBAL_VOXEL_SIZE, 
#                                config.GLOBAL_FPFH_RADIUS_MULTIPLIER)
#     target_fpfh = compute_fpfh(target_down, config.GLOBAL_VOXEL_SIZE,
#                                config.GLOBAL_FPFH_RADIUS_MULTIPLIER)
#     logging.info(f"   ✓ Features computed")
    
#     # RANSAC
#     logging.info(f"🔧 Running RANSAC...")
#     logging.info(f"   Distance threshold: {config.GLOBAL_RANSAC_DIST}")
#     logging.info(f"   Max iterations: {config.GLOBAL_RANSAC_ITERATIONS}")
    
#     result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
#         source_down, target_down,
#         source_fpfh, target_fpfh,
#         mutual_filter=True,
#         max_correspondence_distance=config.GLOBAL_RANSAC_DIST,
#         estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
#         ransac_n=3,
#         checkers=[
#             o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
#                 config.GLOBAL_EDGE_LENGTH_RATIO
#             ),
#             o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
#                 config.GLOBAL_RANSAC_DIST
#             ),
#             o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(
#                 np.deg2rad(config.GLOBAL_NORMAL_ANGLE_DEG)
#             )
#         ],
#         criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
#             config.GLOBAL_RANSAC_ITERATIONS, 0.999
#         )
#     )
    
#     logging.info(f"\n✅ GLOBAL REGISTRATION COMPLETE")
#     logging.info(f"   Fitness: {result.fitness:.4f}")
#     logging.info(f"   RMSE: {result.inlier_rmse:.6f}")
#     logging.info(f"   Correspondences: {len(result.correspondence_set)}")
    
#     if result.fitness < 0.1:
#         logging.warning(f"   ⚠️  Low fitness - global registration may have failed")
    
#     return result


# # ============================================================================
# # LOCAL REFINEMENT (ICP)
# # ============================================================================

# @timeit
# def local_refinement(source: o3d.geometry.PointCloud,
#                      target: o3d.geometry.PointCloud,
#                      initial_transform: np.ndarray,
#                      config: ICPConfig = ICPConfig()) -> o3d.pipelines.registration.RegistrationResult:
#     """
#     Local ICP refinement (point-to-plane)
    
#     Args:
#         source: Source point cloud
#         target: Target point cloud
#         initial_transform: Initial transformation from global registration
#         config: Configuration
    
#     Returns:
#         Refined registration result
#     """
#     logging.info("\n" + "="*70)
#     logging.info("  LOCAL REFINEMENT (ICP)")
#     logging.info("="*70)
    
#     # Downsample
#     logging.info(f"🔧 Downsampling (voxel={config.LOCAL_VOXEL_SIZE})...")
#     source_down = source.voxel_down_sample(config.LOCAL_VOXEL_SIZE)
#     target_down = target.voxel_down_sample(config.LOCAL_VOXEL_SIZE)
#     logging.info(f"   Source: {len(source.points)} → {len(source_down.points)}")
#     logging.info(f"   Target: {len(target.points)} → {len(target_down.points)}")
    
#     # Ensure normals
#     if not source_down.has_normals():
#         source_down.estimate_normals(
#             search_param=o3d.geometry.KDTreeSearchParamHybrid(
#                 radius=config.LOCAL_VOXEL_SIZE * 5, max_nn=30
#             )
#         )
#     if not target_down.has_normals():
#         target_down.estimate_normals(
#             search_param=o3d.geometry.KDTreeSearchParamHybrid(
#                 radius=config.LOCAL_VOXEL_SIZE * 5, max_nn=30
#             )
#         )
    
#     # ICP
#     logging.info(f"🔧 Running point-to-plane ICP...")
#     logging.info(f"   Distance threshold: {config.LOCAL_ICP_DIST}")
#     logging.info(f"   Max iterations: {config.LOCAL_ICP_ITERATIONS}")
    
#     result = o3d.pipelines.registration.registration_icp(
#         source_down, target_down,
#         config.LOCAL_ICP_DIST,
#         initial_transform,
#         o3d.pipelines.registration.TransformationEstimationPointToPlane(),
#         criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
#             max_iteration=config.LOCAL_ICP_ITERATIONS,
#             relative_fitness=1e-6,
#             relative_rmse=1e-6
#         )
#     )
    
#     logging.info(f"\n✅ LOCAL REFINEMENT COMPLETE")
#     logging.info(f"   Fitness: {result.fitness:.4f}")
#     logging.info(f"   RMSE: {result.inlier_rmse:.6f}")
#     logging.info(f"   Correspondences: {len(result.correspondence_set)}")
    
#     return result


# # ============================================================================
# # ADAPTIVE REFINEMENT (from your second code)
# # ============================================================================

# @timeit
# def adaptive_refinement(source: o3d.geometry.PointCloud,
#                         target: o3d.geometry.PointCloud,
#                         initial_result: o3d.pipelines.registration.RegistrationResult,
#                         config: ICPConfig = ICPConfig()) -> o3d.pipelines.registration.RegistrationResult:
#     """
#     Adaptive iterative refinement with random perturbations
    
#     This method from your second code is excellent for escaping local minima!
    
#     Strategy:
#     - Start from current best transformation
#     - Apply small random perturbations (rotation + translation)
#     - Try ICP refinement
#     - Keep result if better
#     - Increase perturbation if stuck
    
#     Args:
#         source: Source point cloud
#         target: Target point cloud
#         initial_result: Initial registration result
#         config: Configuration
    
#     Returns:
#         Improved registration result
#     """
#     logging.info("\n" + "="*70)
#     logging.info("  ADAPTIVE REFINEMENT")
#     logging.info("="*70)
    
#     best_fitness = initial_result.fitness
#     best_rmse = initial_result.inlier_rmse
#     best_transformation = initial_result.transformation
    
#     iteration = 0
#     noise_translation = config.ADAPTIVE_NOISE_TRANSLATION_START
    
#     logging.info(f"🎯 Target: fitness > {config.ADAPTIVE_FITNESS_THRESHOLD}, RMSE < {config.ADAPTIVE_RMSE_THRESHOLD}")
#     logging.info(f"🔧 Starting from: fitness={best_fitness:.4f}, RMSE={best_rmse:.6f}")
    
#     while (iteration < config.ADAPTIVE_MAX_ITERATIONS and
#            (best_fitness < config.ADAPTIVE_FITNESS_THRESHOLD or
#             best_rmse > config.ADAPTIVE_RMSE_THRESHOLD)):
        
#         # Generate random perturbation
#         noise_rotation = o3d.geometry.get_rotation_matrix_from_xyz(
#             [np.random.uniform(-config.ADAPTIVE_NOISE_ROTATION_RANGE, 
#                               config.ADAPTIVE_NOISE_ROTATION_RANGE) for _ in range(3)]
#         )
#         noise_trans_vec = np.random.uniform(-noise_translation, noise_translation, 3)
        
#         noise_transform = np.eye(4)
#         noise_transform[:3, :3] = noise_rotation
#         noise_transform[:3, 3] = noise_trans_vec
        
#         current_transform = noise_transform @ best_transformation
        
#         # Try refinement
#         try:
#             result = o3d.pipelines.registration.registration_icp(
#                 source, target,
#                 config.LOCAL_ICP_DIST,
#                 current_transform,
#                 o3d.pipelines.registration.TransformationEstimationPointToPlane(),
#                 criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
#                     max_iteration=50,
#                     relative_fitness=1e-6,
#                     relative_rmse=1e-6
#                 )
#             )
            
#             # Check if improved
#             if result.fitness > 0 and result.inlier_rmse > 0:
#                 if (result.fitness > best_fitness or
#                     (result.fitness == best_fitness and result.inlier_rmse < best_rmse)):
                    
#                     improvement = result.fitness - best_fitness
#                     best_fitness = result.fitness
#                     best_rmse = result.inlier_rmse
#                     best_transformation = result.transformation
                    
#                     logging.info(f"   ✅ Iter {iteration+1}: fitness={best_fitness:.4f} (+{improvement:.4f}), RMSE={best_rmse:.6f}")
                    
#                     # Check if reached target
#                     if best_fitness >= config.ADAPTIVE_FITNESS_THRESHOLD and best_rmse <= config.ADAPTIVE_RMSE_THRESHOLD:
#                         logging.info(f"   🎉 Target reached!")
#                         break
#             else:
#                 # Increase noise if stuck
#                 noise_translation += 0.25
                
#         except Exception as e:
#             logging.debug(f"   ⚠️  Iter {iteration+1} error: {e}")
#             noise_translation += 0.1
        
#         iteration += 1
        
#         if iteration % 10 == 0:
#             logging.info(f"   📊 Progress: iter={iteration}, best_fitness={best_fitness:.4f}, noise={noise_translation:.2f}")
    
#     final_result = o3d.pipelines.registration.RegistrationResult()
#     final_result.fitness = best_fitness
#     final_result.inlier_rmse = best_rmse
#     final_result.transformation = best_transformation
    
#     logging.info(f"\n✅ ADAPTIVE REFINEMENT COMPLETE ({iteration} iterations)")
#     logging.info(f"   Final fitness: {best_fitness:.4f}")
#     logging.info(f"   Final RMSE: {best_rmse:.6f}")
    
#     return final_result


# # ============================================================================
# # MAIN PIPELINE
# # ============================================================================

# @timeit
# def run_icp_pipeline(object_ply_path: Path,
#                      scene_ply_path: Path,
#                      output_dir: Path,
#                      config: ICPConfig = ICPConfig()) -> Dict:
#     """
#     Complete ICP alignment pipeline
    
#     Pipeline stages:
#     1. Preprocess scene (remove plane, outliers, downsample)
#     2. Preprocess object (outliers, downsample, normals)
#     3. Estimate scale (multi-scale search)
#     4. Global registration (FPFH + RANSAC)
#     5. Local refinement (point-to-plane ICP)
#     6. Adaptive refinement (iterative improvement)
    
#     Args:
#         object_ply_path: Path to object CAD model
#         scene_ply_path: Path to reconstructed scene
#         output_dir: Directory to save results
#         config: Configuration object
    
#     Returns:
#         Dictionary with metrics and results
#     """
#     start_time = time.perf_counter()
    
#     output_dir.mkdir(parents=True, exist_ok=True)
    
#     logging.info("\n" + "="*70)
#     logging.info("  ICP ALIGNMENT PIPELINE")
#     logging.info("="*70)
#     logging.info(f"Object: {object_ply_path.name}")
#     logging.info(f"Scene:  {scene_ply_path.name}")
#     logging.info(f"Output: {output_dir}")
#     logging.info("="*70 + "\n")
    
#     # ========================================================================
#     # STAGE 1: PREPROCESSING
#     # ========================================================================
    
#     scene_pcd = preprocess_scene(
#         scene_ply_path,
#         config=config,
#         save_path=output_dir / "scene_preprocessed.ply"
#     )
    
#     object_pcd = preprocess_object(
#         object_ply_path,
#         config=config
#     )
    
#     # Visualize before alignment
#     if config.VISUALIZE_STEPS:
#         visualize_alignment(object_pcd, scene_pcd, title="Before Alignment")
    
#     # ========================================================================
#     # STAGE 2: SCALE ESTIMATION
#     # ========================================================================
    
#     scale, scale_transform = estimate_scale(object_pcd, scene_pcd, config)
    
#     # Apply scale
#     object_pcd_scaled = copy.deepcopy(object_pcd)
#     object_pcd_scaled.scale(scale, center=object_pcd_scaled.get_center())
#     object_pcd_scaled.transform(scale_transform)
    
#     if config.VISUALIZE_STEPS:
#         visualize_alignment(object_pcd_scaled, scene_pcd, title="After Scale + Center Alignment")
    
#     # ========================================================================
#     # STAGE 3: GLOBAL REGISTRATION
#     # ========================================================================
    
#     global_result = global_registration(object_pcd_scaled, scene_pcd, config)
    
#     if config.VISUALIZE_STEPS:
#         visualize_alignment(object_pcd_scaled, scene_pcd, 
#                           global_result.transformation,
#                           title="After Global Registration")
    
#     # ========================================================================
#     # STAGE 4: LOCAL REFINEMENT
#     # ========================================================================
    
#     local_result = local_refinement(
#         object_pcd_scaled, scene_pcd,
#         global_result.transformation,
#         config
#     )
    
#     if config.VISUALIZE_STEPS:
#         visualize_alignment(object_pcd_scaled, scene_pcd,
#                           local_result.transformation,
#                           title="After Local Refinement")
    
#     # ========================================================================
#     # STAGE 5: ADAPTIVE REFINEMENT
#     # ========================================================================
    
#     final_result = adaptive_refinement(
#         object_pcd_scaled, scene_pcd,
#         local_result,
#         config
#     )
    
#     # ========================================================================
#     # FINALIZE
#     # ========================================================================
    
#     # Combine all transformations
#     final_transformation = final_result.transformation
    
#     # Transform original object
#     object_aligned = copy.deepcopy(object_pcd)
#     #object_aligned.scale(scale, center=object_aligned.get_center())
#     object_aligned.transform(final_transformation)
    
#     # Visualize final result
#     if config.VISUALIZE_FINAL:
#         source_vis = copy.deepcopy(object_pcd).paint_uniform_color([1, 0.706, 0])  # Orange
#         target_vis = copy.deepcopy(scene_pcd).paint_uniform_color([0, 0.651, 0.929])  # Blue
#         aligned_vis = copy.deepcopy(object_aligned).paint_uniform_color([0, 1, 0])  # Green
        
#         logging.info("\n🎬 Final Visualization")
#         logging.info("   Orange = Original Object")
#         logging.info("   Blue = Scene")
#         logging.info("   Green = Aligned Object")
        
#         o3d.visualization.draw_geometries(
#             [source_vis, target_vis, aligned_vis],
#             window_name="Final Result",
#             width=1280,
#             height=720
#         )
    
#     # Save results
#     logging.info("\n💾 Saving results...")
    
#     np.save(output_dir / "transformation.npy", final_transformation)
#     np.save(output_dir / "scale.npy", np.array([scale]))
#     o3d.io.write_point_cloud(str(output_dir / "object_aligned.ply"), object_aligned)
    
#     metrics = {
#         'scale': float(scale),
#         'global_fitness': float(global_result.fitness),
#         'global_rmse': float(global_result.inlier_rmse),
#         'local_fitness': float(local_result.fitness),
#         'local_rmse': float(local_result.inlier_rmse),
#         'final_fitness': float(final_result.fitness),
#         'final_rmse': float(final_result.inlier_rmse),
#         'final_correspondences': len(final_result.correspondence_set),
#         'transformation': final_transformation.tolist(),
#         'elapsed_time': time.perf_counter() - start_time
#     }
    
#     with open(output_dir / "metrics.json", 'w') as f:
#         json.dump(metrics, f, indent=2)
    
#     logging.info(f"   ✓ transformation.npy")
#     logging.info(f"   ✓ scale.npy")
#     logging.info(f"   ✓ object_aligned.ply")
#     logging.info(f"   ✓ metrics.json")
    
#     # Final summary
#     elapsed = time.perf_counter() - start_time
    
#     logging.info("\n" + "="*70)
#     logging.info("  PIPELINE COMPLETE")
#     logging.info("="*70)
#     logging.info(f"⏱️  Total time: {elapsed:.2f}s")
#     logging.info(f"📏 Scale: {scale:.6f}")
#     logging.info(f"📊 Final fitness: {final_result.fitness:.4f}")
#     logging.info(f"📊 Final RMSE: {final_result.inlier_rmse:.6f}")
#     logging.info(f"📊 Correspondences: {len(final_result.correspondence_set)}")
    
#     if final_result.fitness >= 0.85:
#         logging.info(f"✅ EXCELLENT alignment!")
#     elif final_result.fitness >= 0.6:
#         logging.info(f"✓  GOOD alignment")
#     elif final_result.fitness >= 0.3:
#         logging.info(f"⚠️  MODERATE alignment - may need tuning")
#     else:
#         logging.info(f"❌ POOR alignment - check input data")
    
#     logging.info("="*70 + "\n")
    
#     return metrics


# # ============================================================================
# # EXAMPLE USAGE
# # ============================================================================

# if __name__ == "__main__":
#     """
#     Example usage for UR5e robot scanning pipeline
#     """
    
#     # Setup logging
#     setup_logging(level=logging.INFO)
    
#     # Paths (MODIFY THESE)
#     SCENE_DIR = Path("/home/AP_PathMatters/path_matters/datasets/yoda")
#     OBJECT_PLY = SCENE_DIR / "Baby_Yoda.ply"
#     SCENE_PLY = SCENE_DIR / "sparse/points.ply"
#     OUTPUT_DIR = SCENE_DIR / "icp_results_improved"
    
#     # Configuration
#     config = ICPConfig()
    
#     # Adjust for your scene (IMPORTANT!)
#     # If scene is very small/sparse, use smaller values
#     config.SCENE_DOWNSAMPLE_VOXEL = 0.002  # 2mm
#     config.OBJECT_DOWNSAMPLE_VOXEL = 0.002
#     config.PLANE_OFFSET = -0.02  # Keep more points
#     config.SCALE_CANDIDATES = [0.1, 0.15, 0.2, 0.25, 0.3]  # Adjust range
#     config.VISUALIZE_STEPS = True
#     config.DEBUG_MODE = True
    
#     # Run pipeline
#     metrics = run_icp_pipeline(
#         object_ply_path=OBJECT_PLY,
#         scene_ply_path=SCENE_PLY,
#         output_dir=OUTPUT_DIR,
#         config=config
#     )
    
#     print(f"\n✅ Pipeline complete! Results saved to: {OUTPUT_DIR}")
#     print(f"   Final fitness: {metrics['final_fitness']:.4f}")
#     print(f"   Final RMSE: {metrics['final_rmse']:.6f}")



################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################




# # Copyright (c) Meta Platforms, Inc. and affiliates.
# # All rights reserved.
# #
# # This source code is licensed under the license found in the
# # LICENSE file in the root directory of this source tree.
# """
# this code reconstruct and then use icp directly 

# Anleitung
# # 1. Just reconstruction (original behavior)
# python /path/to/z_demo_reconstruction_icp.py  --scene_dir /path/to/scene/

# # 2. Reconstruction + ICP with visualization
# python /path/to/z_demo_reconstruction_icp.py \
#     --scene_dir /path/to/scene/ \
#     --run_icp \
#     --object_ply /path/to/object.ply \
#     --visualize_icp \
#     --show_before

# # 3. Full pipeline with custom parameters
# python /path/to/z_demo_reconstruction_icp.py  \
#     --scene_dir /path/to/scene/ \
#     --conf_thres_value 3.0 \
#     --run_icp \
#     --object_ply object.ply \
#     --voxel_size_object 0.002 \
#     --voxel_size_scene 0.005 \
#     --max_correspondence_dist 0.02 \
#     --visualize_icp \
#     --show_before


# python /home/AP_PathMatters/vggt/z_demo_reconstruction_icp.py \
#     --scene_dir /home/AP_PathMatters/path_matters/datasets/yoda \
#     --run_icp \
#     --object_ply /home/AP_PathMatters/path_matters/datasets/yoda/Baby_Yoda.ply\
#     --visualize_reconstruction \
#     --visualize_preprocessing \
#     --voxel_size_object 0.005 \
#     --voxel_size_scene 0.005 \
#     --max_correspondence_dist 0.05 \
#     --debug \
#     --no_global \
#     --visualize_icp \
#     --show_before

# python /home/AP_PathMatters/vggt/z_demo_reconstruction_icp.py \
#     --scene_dir  /home/AP_PathMatters/path_matters/datasets/pix3d/img/misc/Bowl_white \
#     --run_icp \
#     --object_ply /home/AP_PathMatters/path_matters/datasets/pix3d/img/misc/Bowl_white/model.ply \
#     --visualize_icp \
#     --show_before
# """


# import random
# import numpy as np
# import glob
# import os
# import copy
# import torch
# import torch.nn.functional as F
# import json

# # Configure CUDA settings
# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = False

# import argparse
# from pathlib import Path
# import trimesh
# import pycolmap
# import open3d as o3d

# from vggt.models.vggt import VGGT
# from vggt.utils.load_fn import load_and_preprocess_images_square
# from vggt.utils.pose_enc import pose_encoding_to_extri_intri
# from vggt.utils.geometry import unproject_depth_map_to_point_map
# from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
# from vggt.dependency.track_predict import predict_tracks
# from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap, batch_np_matrix_to_pycolmap_wo_track


# def parse_args():
#     parser = argparse.ArgumentParser(description="VGGT Demo with ICP")
    
#     # Original VGGT args
#     parser.add_argument("--scene_dir", type=str, required=True, help="Directory containing the scene images")
#     parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
#     parser.add_argument("--use_ba", action="store_true", default=False, help="Use BA for reconstruction")
    
#     # BA parameters
#     parser.add_argument("--max_reproj_error", type=float, default=8.0, help="Maximum reprojection error")
#     parser.add_argument("--shared_camera", action="store_true", default=False, help="Use shared camera")
#     parser.add_argument("--camera_type", type=str, default="SIMPLE_PINHOLE", help="Camera type")
#     parser.add_argument("--vis_thresh", type=float, default=0.2, help="Visibility threshold for tracks")
#     parser.add_argument("--query_frame_num", type=int, default=8, help="Number of frames to query")
#     parser.add_argument("--max_query_pts", type=int, default=4096, help="Maximum number of query points")
#     parser.add_argument("--fine_tracking", action="store_true", default=True, help="Use fine tracking")
#     parser.add_argument("--conf_thres_value", type=float, default=2.0, help="Confidence threshold (wo BA)")
    
#     # ICP parameters - ALL INDEPENDENT NOW
#     parser.add_argument("--run_icp", action="store_true", help="Run ICP alignment")
#     parser.add_argument("--object_ply", type=str, default=None, help="Object PLY path")
    
#     # Voxel sizes (independent)
#     parser.add_argument("--voxel_size_object", type=float, default=0.005, help="Voxel size for object downsampling")
#     parser.add_argument("--voxel_size_scene", type=float, default=0.005, help="Voxel size for scene downsampling")
#     parser.add_argument("--voxel_size_global", type=float, default=0.02, help="Voxel size for global registration")
#     parser.add_argument("--voxel_size_local", type=float, default=0.005, help="Voxel size for local ICP refinement")
    
#     # Correspondence distances (independent)
#     parser.add_argument("--max_correspondence_dist", type=float, default=0.05, help="Max correspondence distance for ICP")
#     parser.add_argument("--global_correspondence_dist", type=float, default=0.03, help="Distance threshold for global RANSAC")
#     parser.add_argument("--local_correspondence_dist", type=float, default=0.01, help="Distance threshold for local ICP")
    
#     # Scale estimation
#     parser.add_argument("--scale_correspondence_dist", type=float, default=0.05, help="Correspondence distance for scale estimation")
    
#     # Other ICP parameters
#     parser.add_argument("--use_point_to_point", action="store_true", help="Use point-to-point ICP (default: point-to-plane)")
#     parser.add_argument("--no_scale", action="store_true", help="Disable scale estimation")
#     parser.add_argument("--scale_method", type=str, default="auto",
#                         choices=["auto", "ransac", "umeyama", "multi_scale", "bbox"],
#                         help="Scale estimation method")
#     parser.add_argument("--no_global", action="store_true", 
#                     help="Skip global registration (use direct ICP only)")
    
#     # Visualization
#     parser.add_argument("--visualize_icp", action="store_true", help="Visualize ICP")
#     parser.add_argument("--show_before", action="store_true", help="Show before/after")
#     parser.add_argument("--visualize_reconstruction", action="store_true", 
#                     help="Visualize reconstructed scene before preprocessing")
#     parser.add_argument("--visualize_preprocessing", action="store_true",
#                     help="Visualize each preprocessing step")
#     parser.add_argument("--debug", action="store_true", help="Enable detailed debugging output")
    
#     return parser.parse_args()

# def visualize_point_cloud(pcd, window_name="Point Cloud Visualization", width=1280, height=720):
#     """Visualize point cloud with Open3D"""
#     print(f"\n[VIS] Showing: {window_name}")
#     print(f"      Points: {len(pcd.points)}")
#     print(f"      Close window to continue...")
#     o3d.visualization.draw_geometries(
#         [pcd],
#         window_name=window_name,
#         width=width,
#         height=height
#     )

# def debug_point_clouds(source, target, name="Debug"):
#     """Print detailed statistics about point clouds"""
#     print(f"\n{'='*70}")
#     print(f"  DEBUG: {name}")
#     print(f"{'='*70}")
    
#     # Basic stats
#     print(f"Source points: {len(source.points)}")
#     print(f"Target points: {len(target.points)}")
    
#     # Bounding boxes
#     source_bbox = source.get_axis_aligned_bounding_box()
#     target_bbox = target.get_axis_aligned_bounding_box()
    
#     print(f"\nSource bounding box:")
#     print(f"  Min: {source_bbox.min_bound}")
#     print(f"  Max: {source_bbox.max_bound}")
#     print(f"  Extent: {source_bbox.get_extent()}")
#     print(f"  Center: {source.get_center()}")
    
#     print(f"\nTarget bounding box:")
#     print(f"  Min: {target_bbox.min_bound}")
#     print(f"  Max: {target_bbox.max_bound}")
#     print(f"  Extent: {target_bbox.get_extent()}")
#     print(f"  Center: {target.get_center()}")
    
#     # Distance between centers
#     distance = np.linalg.norm(source.get_center() - target.get_center())
#     print(f"\nDistance between centers: {distance:.6f}")
    
#     # Check overlap
#     source_min = source_bbox.min_bound
#     source_max = source_bbox.max_bound
#     target_min = target_bbox.min_bound
#     target_max = target_bbox.max_bound
    
#     overlap_x = min(source_max[0], target_max[0]) > max(source_min[0], target_min[0])
#     overlap_y = min(source_max[1], target_max[1]) > max(source_min[1], target_min[1])
#     overlap_z = min(source_max[2], target_max[2]) > max(source_min[2], target_min[2])
    
#     print(f"\nBounding box overlap:")
#     print(f"  X-axis: {'YES' if overlap_x else 'NO'}")
#     print(f"  Y-axis: {'YES' if overlap_y else 'NO'}")
#     print(f"  Z-axis: {'YES' if overlap_z else 'NO'}")
    
#     if not (overlap_x and overlap_y and overlap_z):
#         print(f"\n  ⚠️  WARNING: No 3D overlap detected!")
#         print(f"     This will likely cause ICP to fail.")
    
#     # Normals check
#     print(f"\nNormals:")
#     print(f"  Source has normals: {source.has_normals()}")
#     print(f"  Target has normals: {target.has_normals()}")
    
#     # Colors check
#     print(f"\nColors:")
#     print(f"  Source has colors: {source.has_colors()}")
#     print(f"  Target has colors: {target.has_colors()}")
    
#     print(f"{'='*70}\n")


# def visualize_alignment_side_by_side(source, target, transformation=None, window_name="Alignment Debug"):
#     """Visualize source and target with optional transformation"""
#     print(f"\n[VIS] {window_name}")
    
#     # Color point clouds
#     source_colored = copy.deepcopy(source).paint_uniform_color([1.0, 0.0, 0.0])  # Red
#     target_colored = copy.deepcopy(target).paint_uniform_color([0.0, 1.0, 0.0])  # Green
    
#     geometries = [source_colored, target_colored]
    
#     # Add coordinate frame at origin
#     coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
#     geometries.append(coord_frame)
    
#     # If transformation provided, also show transformed source
#     if transformation is not None:
#         source_transformed = copy.deepcopy(source)
#         source_transformed.transform(transformation)
#         source_transformed = source_transformed.paint_uniform_color([0.0, 0.0, 1.0])  # Blue
#         geometries.append(source_transformed)
#         print(f"      Red=Original Source, Green=Target, Blue=Transformed Source")
#     else:
#         print(f"      Red=Source, Green=Target")
    
#     print(f"      Close window to continue...")
    
#     o3d.visualization.draw_geometries(
#         geometries,
#         window_name=window_name,
#         width=1280,
#         height=720
#     )

# def align_centers_translation(source, target):
#     """
#     Create transformation to translate source center to target center
#     """
#     source_center = source.get_center()
#     target_center = target.get_center()
    
#     transformation = np.eye(4)
#     transformation[:3, 3] = target_center - source_center
    
#     print(f"\n[ALIGN] Center alignment:")
#     print(f"        Source center: {source_center}")
#     print(f"        Target center: {target_center}")
#     print(f"        Translation: {transformation[:3, 3]}")
    
#     return transformation

# def test_correspondence_distance(source, target, distances=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5]):
#     """Test ICP with different correspondence distances to find optimal"""
#     print(f"\n{'='*70}")
#     print(f"  TESTING CORRESPONDENCE DISTANCES")
#     print(f"{'='*70}")
    
#     results = []
    
#     for dist in distances:
#         result = o3d.pipelines.registration.registration_icp(
#             source=source,
#             target=target,
#             max_correspondence_distance=dist,
#             init=np.eye(4),
#             estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
#             criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
#         )
        
#         results.append({
#             'distance': dist,
#             'fitness': result.fitness,
#             'rmse': result.inlier_rmse,
#             'correspondences': len(result.correspondence_set)
#         })
        
#         print(f"  Distance {dist:.3f}: fitness={result.fitness:.4f}, rmse={result.inlier_rmse:.6f}, corr={len(result.correspondence_set)}")
    
#     # Find best
#     best = max(results, key=lambda x: x['fitness'])
#     print(f"\n  ✓ Best distance: {best['distance']:.3f} (fitness={best['fitness']:.4f})")
#     print(f"{'='*70}\n")
    
#     return best['distance']

# def run_VGGT(model, images, dtype, resolution=518):
#     """Run VGGT inference"""
#     assert len(images.shape) == 4
#     assert images.shape[1] == 3

#     images = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)

#     with torch.no_grad():
#         with torch.cuda.amp.autocast(dtype=dtype):
#             images = images[None]  # add batch dimension
#             aggregated_tokens_list, ps_idx = model.aggregator(images)

#         pose_enc = model.camera_head(aggregated_tokens_list)[-1]
#         extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
#         depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

#     extrinsic = extrinsic.squeeze(0).cpu().numpy()
#     intrinsic = intrinsic.squeeze(0).cpu().numpy()
#     depth_map = depth_map.squeeze(0).cpu().numpy()
#     depth_conf = depth_conf.squeeze(0).cpu().numpy()
    
#     return extrinsic, intrinsic, depth_map, depth_conf


# def preprocess_scene_point_cloud(
#     scene_ply_path: str,
#     output_path: str = None,
#     # Filtering options
#     remove_statistical_outliers: bool = True,
#     nb_neighbors: int = 20,
#     std_ratio: float = 2.0,
#     # Cropping options
#     crop_box: tuple = None,
#     # Color filtering
#     remove_color_range: list = None,
#     # Downsampling
#     voxel_size: float = None,
#     # Plane removal
#     remove_table_plane: bool = False,
#     plane_distance_threshold: float = 0.01,
#     plane_num_iterations: int = 1000,
#     remove_points_below: bool = True,
#     plane_offset: float = 0.02,
#     visualize_steps: bool = False  # NEW PARAMETER
# ) -> o3d.geometry.PointCloud:
#     """
#     Preprocess reconstructed scene point cloud before ICP
    
#     Args:
#         scene_ply_path: Path to scene point cloud
#         output_path: Where to save preprocessed cloud
#         remove_statistical_outliers: Remove statistical outliers
#         nb_neighbors: Number of neighbors for outlier removal
#         std_ratio: Standard deviation ratio for outlier removal
#         crop_box: Bounding box to crop ((min_x,y,z), (max_x,y,z))
#         remove_color_range: List of RGB ranges to remove
#         voxel_size: Voxel size for downsampling
#         remove_table_plane: Remove table/floor plane
#         plane_distance_threshold: Distance threshold for plane detection
#         plane_num_iterations: RANSAC iterations for plane fitting
#         remove_points_below: Remove points below plane (not just plane itself)
#         plane_offset: Distance to shift plane upward before removal (removes entire plane)
        
#     Returns:
#         Preprocessed point cloud
#     """

#     print(f"\n{'='*70}")
#     print(f"  PREPROCESSING SCENE POINT CLOUD")
#     print(f"{'='*70}")
    
#     # Load
#     pcd = o3d.io.read_point_cloud(scene_ply_path)
#     original_count = len(pcd.points)
#     print(f"[PREP] Loaded: {original_count} points")

#     if visualize_steps:
#         visualize_point_cloud(pcd, "Step 0: Original Point Cloud")  # ADD THIS

#     # 1. Remove statistical outliers
#     if remove_statistical_outliers:
#         print(f"[PREP] Removing statistical outliers...")
#         pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
#         print(f"       {original_count} → {len(pcd.points)} points")

#         if visualize_steps:
#             visualize_point_cloud(pcd, "Step 1: After Outlier Removal")  # ADD THIS

#     # 2. Crop to bounding box
#     if crop_box is not None:
#         print(f"[PREP] Cropping to bounding box...")
#         min_bound, max_bound = crop_box
#         bbox = o3d.geometry.AxisAlignedBoundingBox(
#             min_bound=np.array(min_bound),
#             max_bound=np.array(max_bound)
#         )
#         pcd = pcd.crop(bbox)
#         print(f"       → {len(pcd.points)} points")
    
#         if visualize_steps:
#             visualize_point_cloud(pcd, "Step 2: After Bounding Box Crop")  # ADD THIS

#     # 3. Remove by color
#     if remove_color_range is not None and pcd.has_colors():
#         print(f"[PREP] Filtering by color...")
#         points = np.asarray(pcd.points)
#         colors = np.asarray(pcd.colors) * 255
        
#         mask = np.ones(len(points), dtype=bool)
        
#         for color_min, color_max in remove_color_range:
#             color_min = np.array(color_min)
#             color_max = np.array(color_max)
#             in_range = np.all((colors >= color_min) & (colors <= color_max), axis=1)
#             mask &= ~in_range
        
#         pcd = pcd.select_by_index(np.where(mask)[0])
#         print(f"       → {len(pcd.points)} points")
        
#         if visualize_steps:
#             visualize_point_cloud(pcd, "Step 3: After Color Filtering")  # ADD THIS

#     # 4. Remove table/floor plane
#     if remove_table_plane:
#         print(f"[PREP] Removing table/floor plane...")
        
#         # Estimate normals
#         print(f"       Estimating normals...")
#         pcd.estimate_normals(
#             search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=10)
#         )
        
#         # Compute average normal (majority direction)
#         pcd_downsampled = copy.deepcopy(pcd).voxel_down_sample(voxel_size=10)
#         normals = np.asarray(pcd_downsampled.normals)
#         average_normal = np.mean(normals, axis=0)
#         average_normal /= np.linalg.norm(average_normal)
#         print(f"       Average normal: {average_normal}")
        
#         # Perform plane segmentation
#         print(f"       Running RANSAC plane detection...")
#         plane_model, inliers = pcd.segment_plane(
#             distance_threshold=plane_distance_threshold,
#             ransac_n=3,
#             num_iterations=plane_num_iterations
#         )
        
#         [a, b, c, d] = plane_model
#         print(f"       Plane equation: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")
#         print(f"       Inliers: {len(inliers)} points")
        
#         # Flip plane normal if needed (to ensure correct orientation)
#         plane_normal = np.array([a, b, c])
#         plane_normal /= np.linalg.norm(plane_normal)
        
#         dot_product = np.dot(plane_normal, average_normal)
#         if dot_product > 0:
#             print(f"       ⚠️  Flipping plane normal to match majority")
#             plane_model = [-a, -b, -c, -d]
#             [a, b, c, d] = plane_model
#             plane_normal = -plane_normal
        
#         if remove_points_below:
#             # Apply offset to shift cutting plane upward
#             # This ensures the entire physical plane is removed
#             print(f"       Applying plane offset: {plane_offset} units upward")
            
#             # Shift the plane by moving d
#             # Plane equation: ax + by + cz + d = 0
#             # To shift by offset along normal: d_new = d - offset * sqrt(a² + b² + c²)
#             plane_norm = np.sqrt(a**2 + b**2 + c**2)
#             d_offset = d - plane_offset * plane_norm
            
#             print(f"       Original d: {d:.3f}, Offset d: {d_offset:.3f}")
            
#             # Remove plane AND all points below the OFFSET plane
#             print(f"       Removing plane and points below offset plane...")
#             points = np.asarray(pcd.points)
#             colors = np.asarray(pcd.colors) if pcd.has_colors() else None
            
#             # Calculate signed distance to OFFSET plane
#             distances = (a * points[:, 0] + 
#                         b * points[:, 1] + 
#                         c * points[:, 2] + d_offset) / plane_norm
            
#             # Keep only points ABOVE the offset plane
#             above_plane_mask = distances <= 0
            
#             pcd_filtered = o3d.geometry.PointCloud()
#             pcd_filtered.points = o3d.utility.Vector3dVector(points[above_plane_mask])
            
#             if colors is not None:
#                 pcd_filtered.colors = o3d.utility.Vector3dVector(colors[above_plane_mask])
            
#             pcd = pcd_filtered
#             print(f"       → {len(pcd.points)} points (removed plane + below + offset)")

#             # ADD THIS: Re-estimate normals after filtering
#             print(f"       Re-estimating normals after filtering...")
#             pcd.estimate_normals(
#                 search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
#             )
#             print(f"       ✓ Normals re-estimated")

#             if visualize_steps:
#                 visualize_point_cloud(pcd, "Step 4: After Plane Removal")  # ADD THIS

#         else:
#             # Remove only the plane itself
#             print(f"       Removing plane only...")
#             pcd = pcd.select_by_index(inliers, invert=True)
#             print(f"       → {len(pcd.points)} points")
    
#     # 5. Downsample
#     if voxel_size is not None:
#         print(f"[PREP] Downsampling (voxel_size={voxel_size})...")
#         pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
#         print(f"       → {len(pcd.points)} points")
    
#         if visualize_steps:
#             visualize_point_cloud(pcd, "Step 5: After Downsampling")  # ADD THIS

#     # Save if requested
#     if output_path is not None:
#         output_path = Path(output_path)
#         output_path.parent.mkdir(parents=True, exist_ok=True)
#         o3d.io.write_point_cloud(str(output_path), pcd)
#         print(f"[PREP] ✓ Saved preprocessed cloud: {output_path}")
    
#     print(f"[PREP] ✓ Preprocessing complete: {original_count} → {len(pcd.points)} points")
    
#     return pcd


# def run_icp_alignment(
#     object_ply_path: str,
#     scene_pcd: o3d.geometry.PointCloud,
#     output_dir: str,
#     voxel_size_object: float,
#     voxel_size_scene: float,
#     max_correspondence_distance: float,
#     voxel_size_global: float,
#     voxel_size_local: float,
#     global_correspondence_dist: float,
#     local_correspondence_dist: float,
#     scale_correspondence_dist: float,
#     use_point_to_plane: bool = True,
#     estimate_scale: bool = True,
#     scale_method: str = "auto",
#     use_global_registration: bool = True,
#     visualize: bool = True,
#     show_before: bool = True,
#     debug: bool = False
# ):
#     """Run ICP alignment with optional global registration first"""

#     print(f"\n{'='*70}")
#     print(f"  ICP ALIGNMENT {'(DEBUG MODE)' if debug else ''}")
#     print(f"{'='*70}")
    
#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)
    
#     # Load object
#     print(f"[ICP] Loading object: {object_ply_path}")
#     object_pcd = o3d.io.read_point_cloud(object_ply_path)
#     print(f"      {len(object_pcd.points)} points")
    
#     # Check if file loaded properly
#     if len(object_pcd.points) == 0:
#         print(f"❌ ERROR: Object PLY has no points!")
#         return None
    
#     # DEBUG: Check initial point clouds
#     if debug:
#         debug_point_clouds(object_pcd, scene_pcd, "Initial Point Clouds (Before Scale)")
#         visualize_alignment_side_by_side(object_pcd, scene_pcd, window_name="DEBUG: Before Scale")
    
#     # Estimate scale
#     if estimate_scale:
#         print(f"\n[SCALE] Estimating scale...")
#         scale = estimate_scale_robust(
#             source=object_pcd,
#             target=scene_pcd,
#             method=scale_method,
#             correspondence_distance=scale_correspondence_dist  # INDEPENDENT PARAMETER
#         )
        
#         # Apply scale to object
#         print(f"\n[ICP] Applying scale {scale:.6f} to object...")
#         object_pcd_scaled = copy.deepcopy(object_pcd)
#         object_pcd_scaled.scale(scale, center=object_pcd_scaled.get_center())
        
#         # DEBUG: Check after scaling
#         if debug:
#             debug_point_clouds(object_pcd_scaled, scene_pcd, "After Scaling")
#             visualize_alignment_side_by_side(object_pcd_scaled, scene_pcd, window_name="DEBUG: After Scale")

#         # Initial alignment by translating centers
#         initial_transform = align_centers_translation(object_pcd_scaled, scene_pcd)
#         object_pcd_scaled.transform(initial_transform)
        
#         # DEBUG: Check after center alignment
#         if debug:
#             debug_point_clouds(object_pcd_scaled, scene_pcd, "After Center Alignment")
#             visualize_alignment_side_by_side(object_pcd_scaled, scene_pcd, window_name="DEBUG: After Center Alignment")
        
#         # Ensure scene has normals
#         if not scene_pcd.has_normals():
#             print(f"\n[PREP] Estimating normals for scene...")
#             scene_pcd.estimate_normals(
#                 search_param=o3d.geometry.KDTreeSearchParamHybrid(
#                     radius=0.05, max_nn=30
#                 )
#             )
#             print(f"       ✓ Scene normals estimated")

#     else:
#         scale = 1.0
#         object_pcd_scaled = object_pcd
#         initial_transform = np.eye(4)
    
#     # DEBUG: Test different correspondence distances
#     if debug:
#         print(f"\n[DEBUG] Testing correspondence distances...")
#         test_dist = test_correspondence_distance(
#             object_pcd_scaled.voxel_down_sample(voxel_size_object),
#             scene_pcd.voxel_down_sample(voxel_size_scene)
#         )
#         print(f"[DEBUG] Suggested distance: {test_dist:.6f}")
#         print(f"[DEBUG] Current max_correspondence_dist: {max_correspondence_distance:.6f}")
    
#     # Run registration
#     if use_global_registration:
#         # DEBUG: Before global registration
#         if debug:
#             source_down = object_pcd_scaled.voxel_down_sample(voxel_size_global)
#             target_down = scene_pcd.voxel_down_sample(voxel_size_global)
#             debug_point_clouds(source_down, target_down, "Before Global Registration (Downsampled)")
        
#         # Global + Local registration with INDEPENDENT parameters
#         result_global, result_local = run_global_then_local_registration(
#             source=object_pcd_scaled,
#             target=scene_pcd,
#             voxel_size_global=voxel_size_global,        # INDEPENDENT
#             voxel_size_local=voxel_size_local,          # INDEPENDENT
#             global_distance_threshold=global_correspondence_dist,  # INDEPENDENT
#             local_distance_threshold=local_correspondence_dist,    # INDEPENDENT
#             use_point_to_plane=use_point_to_plane,
#             debug=debug
#         )
        
#         transformation = result_local.transformation
#         result = result_local
        
#         # DEBUG: After global registration
#         if debug:
#             visualize_alignment_side_by_side(
#                 object_pcd_scaled, scene_pcd, 
#                 transformation=result_global.transformation,
#                 window_name="DEBUG: After Global Registration"
#             )
            
#             print(f"\n[DEBUG] Global transformation matrix:")
#             print(result_global.transformation)
#             print(f"\n[DEBUG] Local transformation matrix:")
#             print(result_local.transformation)
        
#     else:
#         # Direct ICP
#         print(f"\n[ICP] Running direct ICP (no global registration)...")
        
#         object_pcd_processed = object_pcd_scaled.voxel_down_sample(voxel_size=voxel_size_object)
#         scene_pcd_processed = scene_pcd.voxel_down_sample(voxel_size=voxel_size_scene)
        
#         # DEBUG: Before ICP
#         if debug:
#             debug_point_clouds(object_pcd_processed, scene_pcd_processed, "Before Direct ICP (Downsampled)")
        
#         # Estimate normals
#         if not object_pcd_processed.has_normals():
#             object_pcd_processed.estimate_normals(
#                 search_param=o3d.geometry.KDTreeSearchParamHybrid(
#                     radius=voxel_size_object * 5, max_nn=30
#                 )
#             )
#         if not scene_pcd_processed.has_normals():
#             scene_pcd_processed.estimate_normals(
#                 search_param=o3d.geometry.KDTreeSearchParamHybrid(
#                     radius=voxel_size_scene * 5, max_nn=30
#                 )
#             )
        
#         print(f"\n[ICP] Running ICP...")
#         print(f"      Voxel size object: {voxel_size_object}")
#         print(f"      Voxel size scene: {voxel_size_scene}")
#         print(f"      Max correspondence: {max_correspondence_distance}")
        
#         if use_point_to_plane:
#             estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPlane()
#         else:
#             estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        
#         result = o3d.pipelines.registration.registration_icp(
#             source=object_pcd_processed,
#             target=scene_pcd_processed,
#             max_correspondence_distance=max_correspondence_distance,
#             init=initial_transform,  # Use center alignment as init
#             estimation_method=estimation_method,
#             criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
#                 max_iteration=200,
#                 relative_fitness=1e-6,
#                 relative_rmse=1e-6
#             )
#         )
        
#         transformation = result.transformation
    
#     # Metrics
#     metrics = {
#         'fitness': float(result.fitness),
#         'inlier_rmse': float(result.inlier_rmse),
#         'num_correspondences': len(result.correspondence_set),
#         'scale': float(scale),
#         'transformation': transformation.tolist(),
#         'used_global_registration': use_global_registration
#     }
    
#     print(f"\n[ICP] ✓ Alignment complete")
#     print(f"      Scale:           {metrics['scale']:.6f}")
#     print(f"      Fitness:         {metrics['fitness']:.4f}")
#     print(f"      Inlier RMSE:     {metrics['inlier_rmse']:.6f}")
#     print(f"      Correspondences: {metrics['num_correspondences']}")
    
#     # DEBUG: Diagnosis
#     if debug:
#         print(f"\n{'='*70}")
#         print(f"  DIAGNOSIS")
#         print(f"{'='*70}")
        
#         if metrics['fitness'] < 0.1:
#             print(f"❌ Very low fitness ({metrics['fitness']:.4f})")
#             print(f"   Possible causes:")
#             print(f"   1. Point clouds don't overlap")
#             print(f"   2. Scale is incorrect")
#             print(f"   3. Correspondence distance too small/large")
#             print(f"   4. Global registration failed")
#         elif metrics['fitness'] < 0.3:
#             print(f"⚠️  Low fitness ({metrics['fitness']:.4f})")
#             print(f"   Alignment is partial")
#         elif metrics['fitness'] < 0.6:
#             print(f"✓ Moderate fitness ({metrics['fitness']:.4f})")
#             print(f"   Alignment is decent")
#         else:
#             print(f"✓✓ Good fitness ({metrics['fitness']:.4f})")
#             print(f"   Alignment looks good!")
        
#         if metrics['num_correspondences'] < 100:
#             print(f"\n⚠️  Very few correspondences ({metrics['num_correspondences']})")
#             print(f"   Try increasing max_correspondence_dist")
        
#         print(f"{'='*70}\n")
    
#     # Save results
#     print(f"\n[ICP] Saving results to {output_dir}")
    
#     np.save(output_dir / "transformation.npy", transformation)
#     np.save(output_dir / "scale.npy", np.array([scale]))
    
#     with open(output_dir / "icp_metrics.json", 'w') as f:
#         json.dump(metrics, f, indent=2)
    
#     # Transform original object with BOTH scale AND transformation
#     object_aligned = copy.deepcopy(object_pcd)
#     object_aligned.scale(scale, center=object_aligned.get_center())
#     object_aligned.transform(initial_transform @ transformation)  # Apply both transforms
#     o3d.io.write_point_cloud(str(output_dir / "object_aligned.ply"), object_aligned)
    
#     print(f"      ✓ transformation.npy")
#     print(f"      ✓ scale.npy")
#     print(f"      ✓ icp_metrics.json")
#     print(f"      ✓ object_aligned.ply")
    
#     # Visualization
#     if visualize:
#         print(f"\n[VIS] Preparing visualization...")
        
#         target_colored = copy.deepcopy(scene_pcd).paint_uniform_color([0.8, 0.8, 0.8])
#         source_colored = copy.deepcopy(object_pcd).paint_uniform_color([1.0, 0.0, 0.0])
        
#         source_aligned_colored = copy.deepcopy(object_pcd)
#         source_aligned_colored.scale(scale, center=source_aligned_colored.get_center())
#         source_aligned_colored.transform(initial_transform @ transformation)
#         source_aligned_colored = source_aligned_colored.paint_uniform_color([0.0, 1.0, 0.0])
        
#         if show_before:
#             print("[VIS] Showing BEFORE (Red=Object, Gray=Scene)")
#             o3d.visualization.draw_geometries(
#                 [source_colored, target_colored],
#                 window_name="Before Alignment",
#                 width=1280,
#                 height=720
#             )
        
#         print(f"[VIS] Showing AFTER (Green=Aligned, Gray=Scene)")
#         o3d.visualization.draw_geometries(
#             [source_aligned_colored, target_colored],
#             window_name="After Alignment",
#             width=1280,
#             height=720
#         )
    
#     return metrics

# def compute_fpfh_features(
#     pcd: o3d.geometry.PointCloud,
#     voxel_size: float,
#     radius_multiplier: float = 5.0
# ):
#     """
#     Compute FPFH features for point cloud
    
#     Args:
#         pcd: Point cloud
#         voxel_size: Voxel size for downsampling
#         radius_multiplier: Multiplier for search radius
        
#     Returns:
#         FPFH feature object
#     """
#     radius_normal = voxel_size * 2
#     radius_feature = voxel_size * radius_multiplier
    
#     # Estimate normals if not present
#     if not pcd.has_normals():
#         pcd.estimate_normals(
#             search_param=o3d.geometry.KDTreeSearchParamHybrid(
#                 radius=radius_normal, max_nn=30
#             )
#         )
    
#     # Compute FPFH features
#     fpfh = o3d.pipelines.registration.compute_fpfh_feature(
#         pcd,
#         o3d.geometry.KDTreeSearchParamHybrid(
#             radius=radius_feature, max_nn=100
#         )
#     )
    
#     return fpfh


# def execute_global_registration(
#     source: o3d.geometry.PointCloud,
#     target: o3d.geometry.PointCloud,
#     source_fpfh,
#     target_fpfh,
#     voxel_size: float,
#     distance_threshold: float = None,
#     edge_length_ratio: float = 0.9,
#     angle_threshold: float = np.deg2rad(30),
#     ransac_iterations: int = 100000,
#     ransac_confidence: float = 0.999
# ):
#     """
#     Execute global registration using RANSAC-based feature matching
    
#     Args:
#         source: Source point cloud
#         target: Target point cloud
#         source_fpfh: Source FPFH features
#         target_fpfh: Target FPFH features
#         voxel_size: Voxel size used for downsampling
#         distance_threshold: Max correspondence distance
#         edge_length_ratio: Edge length ratio for correspondence checker
#         angle_threshold: Normal angle threshold (radians)
#         ransac_iterations: Number of RANSAC iterations
#         ransac_confidence: RANSAC confidence
        
#     Returns:
#         Registration result
#     """
#     if distance_threshold is None:
#         distance_threshold = voxel_size * 1.5
    
#     print(f"\n[GLOBAL] Running RANSAC global registration...")
#     print(f"         Distance threshold: {distance_threshold:.6f}")
#     print(f"         RANSAC iterations: {ransac_iterations}")
#     print(f"         Edge length ratio: {edge_length_ratio}")
#     print(f"         Normal angle threshold: {np.rad2deg(angle_threshold):.1f}°")
    
#     result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
#         source, target,
#         source_fpfh, target_fpfh,
#         mutual_filter=True,
#         max_correspondence_distance=distance_threshold,
#         estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
#         ransac_n=3,
#         checkers=[
#             o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
#                 edge_length_ratio
#             ),
#             o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
#                 distance_threshold
#             ),
#             o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(
#                 angle_threshold
#             )
#         ],
#         criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
#             ransac_iterations,
#             ransac_confidence
#         )
#     )
    
#     print(f"[GLOBAL] ✓ Global registration complete")
#     print(f"         Fitness: {result.fitness:.4f}")
#     print(f"         Inlier RMSE: {result.inlier_rmse:.6f}")
#     print(f"         Correspondences: {len(result.correspondence_set)}")
    
#     return result


# def refine_registration(
#     source: o3d.geometry.PointCloud,
#     target: o3d.geometry.PointCloud,
#     initial_transformation: np.ndarray,
#     distance_threshold: float,
#     use_point_to_plane: bool = True,
#     max_iterations: int = 200
# ):
#     """
#     Refine registration using ICP
    
#     Args:
#         source: Source point cloud
#         target: Target point cloud
#         initial_transformation: Initial transformation from global registration
#         distance_threshold: Max correspondence distance
#         use_point_to_plane: Use point-to-plane ICP
#         max_iterations: Maximum ICP iterations
        
#     Returns:
#         Refined registration result
#     """
#     print(f"\n[REFINE] Running ICP refinement...")
#     print(f"         Distance threshold: {distance_threshold:.6f}")
#     print(f"         Method: {'Point-to-Plane' if use_point_to_plane else 'Point-to-Point'}")
#     print(f"         Max iterations: {max_iterations}")
    
#     # Ensure normals for point-to-plane
#     if use_point_to_plane:
#         if not source.has_normals():
#             source.estimate_normals(
#                 search_param=o3d.geometry.KDTreeSearchParamHybrid(
#                     radius=distance_threshold * 2, max_nn=30
#                 )
#             )
#         if not target.has_normals():
#             target.estimate_normals(
#                 search_param=o3d.geometry.KDTreeSearchParamHybrid(
#                     radius=distance_threshold * 2, max_nn=30
#                 )
#             )
        
#         estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPlane()
#     else:
#         estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    
#     result = o3d.pipelines.registration.registration_icp(
#         source, target,
#         distance_threshold,
#         initial_transformation,
#         estimation_method,
#         criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
#             max_iteration=max_iterations,
#             relative_fitness=1e-6,
#             relative_rmse=1e-6
#         )
#     )
    
#     print(f"[REFINE] ✓ ICP refinement complete")
#     print(f"         Fitness: {result.fitness:.4f}")
#     print(f"         Inlier RMSE: {result.inlier_rmse:.6f}")
#     print(f"         Correspondences: {len(result.correspondence_set)}")
    
#     return result


# def run_global_then_local_registration(
#     source: o3d.geometry.PointCloud,
#     target: o3d.geometry.PointCloud,
#     voxel_size_global: float,           # INDEPENDENT
#     voxel_size_local: float,            # INDEPENDENT
#     global_distance_threshold: float,   # INDEPENDENT
#     local_distance_threshold: float,    # INDEPENDENT
#     use_point_to_plane: bool = True,
#     ransac_iterations: int = 100000,
#     icp_iterations: int = 200,
#     debug: bool = False
# ):
#     """Complete registration pipeline: Global (RANSAC+FPFH) → Local (ICP)"""
    
#     print(f"\n{'='*70}")
#     print(f"  GLOBAL + LOCAL REGISTRATION PIPELINE")
#     print(f"{'='*70}")
    
#     # Step 1: Downsample for global registration
#     print(f"\n[PREP] Downsampling for global registration...")
#     print(f"       Voxel size: {voxel_size_global}")
    
#     source_down_global = source.voxel_down_sample(voxel_size_global)
#     target_down_global = target.voxel_down_sample(voxel_size_global)
    
#     print(f"       Source: {len(source.points)} → {len(source_down_global.points)} points")
#     print(f"       Target: {len(target.points)} → {len(target_down_global.points)} points")
    
#     # Step 2: Compute FPFH features
#     print(f"\n[PREP] Computing FPFH features...")
#     source_fpfh = compute_fpfh_features(source_down_global, voxel_size_global)
#     target_fpfh = compute_fpfh_features(target_down_global, voxel_size_global)
#     print(f"       ✓ Features computed")
    
#     # Step 3: Global registration (RANSAC)
#     result_global = execute_global_registration(
#         source=source_down_global,
#         target=target_down_global,
#         source_fpfh=source_fpfh,
#         target_fpfh=target_fpfh,
#         voxel_size=voxel_size_global,
#         distance_threshold=global_distance_threshold,  # INDEPENDENT
#         ransac_iterations=ransac_iterations
#     )
    
#     # Check if global registration succeeded
#     if debug and result_global.fitness < 0.1:
#         print(f"\n⚠️  [DEBUG] Global registration failed!")
#         print(f"   Fitness: {result_global.fitness:.4f}")
    
#     # Step 4: Downsample for local refinement
#     print(f"\n[PREP] Downsampling for local refinement...")
#     print(f"       Voxel size: {voxel_size_local}")
    
#     source_down_local = source.voxel_down_sample(voxel_size_local)
#     target_down_local = target.voxel_down_sample(voxel_size_local)
    
#     print(f"       Source: {len(source.points)} → {len(source_down_local.points)} points")
#     print(f"       Target: {len(target.points)} → {len(target_down_local.points)} points")
    
#     # Step 5: Local refinement (ICP)
#     result_local = refine_registration(
#         source=source_down_local,
#         target=target_down_local,
#         initial_transformation=result_global.transformation,
#         distance_threshold=local_distance_threshold,  # INDEPENDENT
#         use_point_to_plane=use_point_to_plane,
#         max_iterations=icp_iterations
#     )
    
#     print(f"\n{'='*70}")
#     print(f"  REGISTRATION COMPLETE")
#     print(f"{'='*70}")
#     print(f"Global → Local fitness: {result_global.fitness:.4f} → {result_local.fitness:.4f}")
#     print(f"Global → Local RMSE:    {result_global.inlier_rmse:.6f} → {result_local.inlier_rmse:.6f}")
#     print(f"{'='*70}")
    
#     return result_global, result_local

# def get_initial_alignment(source, target, method="center"):
#     """Get initial transformation for better ICP convergence"""
#     if method == "center":
#         # Simple center alignment
#         source_center = source.get_center()
#         target_center = target.get_center()
        
#         transformation = np.eye(4)
#         transformation[:3, 3] = target_center - source_center
        
#         print(f"[INIT] Center alignment: {transformation[:3, 3]}")
#         return transformation
    
#     elif method == "pca":
#         # PCA-based alignment
#         source_centered = copy.deepcopy(source)
#         target_centered = copy.deepcopy(target)
        
#         source_center = source.get_center()
#         target_center = target.get_center()
        
#         source_centered.translate(-source_center)
#         target_centered.translate(-target_center)
        
#         # Compute PCA
#         source_points = np.asarray(source_centered.points)
#         target_points = np.asarray(target_centered.points)
        
#         _, _, source_v = np.linalg.svd(source_points.T @ source_points)
#         _, _, target_v = np.linalg.svd(target_points.T @ target_points)
        
#         # Rotation to align principal axes
#         R = target_v.T @ source_v
        
#         transformation = np.eye(4)
#         transformation[:3, :3] = R
#         transformation[:3, 3] = target_center - R @ source_center
        
#         print(f"[INIT] PCA alignment computed")
#         return transformation
    
#     return np.eye(4)


# def estimate_scale_from_bbox(source, target):
#     """Fallback: estimate scale from bounding box diagonal"""
#     source_bbox = source.get_axis_aligned_bounding_box()
#     target_bbox = target.get_axis_aligned_bounding_box()
    
#     source_diagonal = np.linalg.norm(source_bbox.get_extent())
#     target_diagonal = np.linalg.norm(target_bbox.get_extent())
    
#     scale = target_diagonal / source_diagonal
#     print(f"[SCALE] Bounding box diagonal ratio: {scale:.6f}")
#     print(f"        Source extent: {source_bbox.get_extent()}")
#     print(f"        Target extent: {target_bbox.get_extent()}")
    
#     return scale


# def estimate_scale_ransac(
#     source: o3d.geometry.PointCloud,
#     target: o3d.geometry.PointCloud,
#     correspondence_distance: float = 0.1,
#     ransac_iterations: int = 1000,
#     confidence: float = 0.99
# ):
#     """
#     Estimate scale using RANSAC on feature correspondences
#     Based on: "Least-Squares Fitting of Two 3-D Point Sets" (Arun et al., 1987)
#     """
#     print(f"[SCALE] Estimating scale with RANSAC...")
    
#     # 1. Extract features (FPFH - Fast Point Feature Histograms)
#     print(f"[SCALE] Computing FPFH features...")
    
#     # Downsample for feature extraction
#     source_down = source.voxel_down_sample(voxel_size=correspondence_distance * 2)
#     target_down = target.voxel_down_sample(voxel_size=correspondence_distance * 2)
    
#     print(f"        Source: {len(source.points)} → {len(source_down.points)} points")
#     print(f"        Target: {len(target.points)} → {len(target_down.points)} points")
    
#     # Estimate normals
#     source_down.estimate_normals(
#         search_param=o3d.geometry.KDTreeSearchParamHybrid(
#             radius=correspondence_distance * 5, max_nn=30
#         )
#     )
#     target_down.estimate_normals(
#         search_param=o3d.geometry.KDTreeSearchParamHybrid(
#             radius=correspondence_distance * 5, max_nn=30
#         )
#     )
    
#     # Compute FPFH features
#     print(f"[SCALE] Computing features...")
#     source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
#         source_down,
#         o3d.geometry.KDTreeSearchParamHybrid(
#             radius=correspondence_distance * 5, max_nn=100
#         )
#     )
#     target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
#         target_down,
#         o3d.geometry.KDTreeSearchParamHybrid(
#             radius=correspondence_distance * 5, max_nn=100
#         )
#     )
    
#     # 2. RANSAC feature matching
#     print(f"[SCALE] Running RANSAC feature matching...")
#     result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
#         source_down, target_down,
#         source_fpfh, target_fpfh,
#         mutual_filter=True,
#         max_correspondence_distance=correspondence_distance,
#         estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
#         ransac_n=3,
#         checkers=[
#             o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
#             o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(correspondence_distance)
#         ],
#         criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(ransac_iterations, confidence)
#     )
    
#     # 3. Estimate scale from correspondences
#     correspondences = np.asarray(result.correspondence_set)
    
#     print(f"[SCALE] Found {len(correspondences)} correspondences (fitness: {result.fitness:.4f})")
    
#     if len(correspondences) < 10:
#         print(f"[SCALE] ⚠️  Too few correspondences, falling back to bbox method")
#         return estimate_scale_from_bbox(source, target)
    
#     source_points = np.asarray(source_down.points)[correspondences[:, 0]]
#     target_points = np.asarray(target_down.points)[correspondences[:, 1]]
    
#     # Calculate distances from centroid
#     source_center = source_points.mean(axis=0)
#     target_center = target_points.mean(axis=0)
    
#     source_distances = np.linalg.norm(source_points - source_center, axis=1)
#     target_distances = np.linalg.norm(target_points - target_center, axis=1)
    
#     # Filter out very small distances
#     valid_mask = (source_distances > 0.001) & (target_distances > 0.001)
    
#     if valid_mask.sum() < 10:
#         print(f"[SCALE] ⚠️  Too few valid distances, falling back to bbox method")
#         return estimate_scale_from_bbox(source, target)
    
#     # Robust scale estimation using median
#     scale_ratios = target_distances[valid_mask] / source_distances[valid_mask]
#     scale = np.median(scale_ratios)
    
#     # Additional validation: use MAD (Median Absolute Deviation) to filter outliers
#     mad = np.median(np.abs(scale_ratios - scale))
#     inlier_mask = np.abs(scale_ratios - scale) < 3 * mad
    
#     if inlier_mask.sum() > 10:
#         scale_refined = np.median(scale_ratios[inlier_mask])
#         print(f"[SCALE] ✓ Refined scale with {inlier_mask.sum()}/{len(scale_ratios)} inliers: {scale_refined:.6f}")
#         return scale_refined
#     else:
#         print(f"[SCALE] ✓ Scale from {len(scale_ratios)} correspondences: {scale:.6f}")
#         return scale


# def estimate_scale_umeyama(source_points, target_points):
#     """
#     Umeyama algorithm: Closed-form solution for similarity transformation
#     Returns: scale, rotation, translation
    
#     Reference: "Least-squares estimation of transformation parameters 
#                 between two point patterns" (Umeyama, 1991)
#     """
#     assert source_points.shape == target_points.shape
    
#     m, n = source_points.shape  # m = num_points, n = dimension (3)
    
#     # Center the point sets
#     source_mean = source_points.mean(axis=0)
#     target_mean = target_points.mean(axis=0)
    
#     source_centered = source_points - source_mean
#     target_centered = target_points - target_mean
    
#     # Compute variances
#     source_var = np.sum(source_centered ** 2) / m
    
#     # Covariance matrix
#     cov = (target_centered.T @ source_centered) / m
    
#     # SVD
#     U, D, Vt = np.linalg.svd(cov)
    
#     # Construct S matrix
#     S = np.eye(n)
#     if np.linalg.det(U) * np.linalg.det(Vt) < 0:
#         S[n-1, n-1] = -1
    
#     # Rotation
#     R = U @ S @ Vt
    
#     # Scale
#     scale = np.trace(np.diag(D) @ S) / source_var
    
#     # Translation
#     t = target_mean - scale * R @ source_mean
    
#     return scale, R, t


# def apply_umeyama_registration(
#     source: o3d.geometry.PointCloud,
#     target: o3d.geometry.PointCloud,
#     correspondence_distance: float = 0.1
# ):
#     """Apply Umeyama registration with scale"""
#     print(f"[SCALE] Running Umeyama algorithm...")
    
#     # Get initial correspondences via nearest neighbors
#     source_down = source.voxel_down_sample(voxel_size=correspondence_distance)
#     target_down = target.voxel_down_sample(voxel_size=correspondence_distance)
    
#     # Build KD-tree for target
#     target_tree = o3d.geometry.KDTreeFlann(target_down)
    
#     # Find correspondences
#     source_points = np.asarray(source_down.points)
#     target_points = np.asarray(target_down.points)
    
#     correspondences = []
#     for i, point in enumerate(source_points):
#         [_, idx, dist] = target_tree.search_knn_vector_3d(point, 1)
#         if dist[0] < correspondence_distance ** 2:
#             correspondences.append((i, idx[0]))
    
#     print(f"[SCALE] Found {len(correspondences)} nearest neighbor correspondences")
    
#     if len(correspondences) < 10:
#         print(f"[SCALE] ⚠️  Too few correspondences, falling back to bbox method")
#         return estimate_scale_from_bbox(source, target), np.eye(3), np.zeros(3)
    
#     correspondences = np.array(correspondences)
#     source_corr = source_points[correspondences[:, 0]]
#     target_corr = target_points[correspondences[:, 1]]
    
#     # Apply Umeyama
#     scale, R, t = estimate_scale_umeyama(source_corr, target_corr)
    
#     print(f"[SCALE] ✓ Umeyama scale: {scale:.6f}")
    
#     return scale, R, t


# def multi_scale_icp(
#     source: o3d.geometry.PointCloud,
#     target: o3d.geometry.PointCloud,
#     scales: list = None,
#     max_correspondence_distance: float = 0.05
# ):
#     """
#     Try multiple scales and pick the best one based on fitness
#     """
#     if scales is None:
#         # Auto-generate scale candidates around bbox estimate
#         bbox_scale = estimate_scale_from_bbox(source, target)
#         scales = [
#             bbox_scale * 0.5,
#             bbox_scale * 0.75,
#             bbox_scale * 1.0,
#             bbox_scale * 1.25,
#             bbox_scale * 1.5,
#             bbox_scale * 2.0
#         ]
    
#     print(f"[SCALE] Testing multiple scales...")
#     print(f"        Candidates: {[f'{s:.3f}' for s in scales]}")
    
#     best_fitness = -1
#     best_scale = 1.0
#     best_transformation = np.eye(4)
#     best_rmse = float('inf')
    
#     for scale in scales:
#         # Scale source
#         source_scaled = copy.deepcopy(source)
#         source_scaled.scale(scale, center=source_scaled.get_center())
        
#         # Run ICP
#         result = o3d.pipelines.registration.registration_icp(
#             source=source_scaled,
#             target=target,
#             max_correspondence_distance=max_correspondence_distance,
#             init=np.eye(4),
#             estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
#             criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
#         )
        
#         print(f"        Scale {scale:.3f}: fitness={result.fitness:.4f}, rmse={result.inlier_rmse:.6f}")
        
#         # Pick best based on fitness (or could use rmse)
#         if result.fitness > best_fitness:
#             best_fitness = result.fitness
#             best_scale = scale
#             best_transformation = result.transformation
#             best_rmse = result.inlier_rmse
    
#     print(f"[SCALE] ✓ Best scale: {best_scale:.6f} (fitness={best_fitness:.4f}, rmse={best_rmse:.6f})")
    
#     return best_scale, best_transformation


# def estimate_scale_robust(
#     source: o3d.geometry.PointCloud,
#     target: o3d.geometry.PointCloud,
#     method: str = "auto",  # "auto", "ransac", "umeyama", "multi_scale", "bbox"
#     correspondence_distance: float = None
# ):
#     """
#     Robust scale estimation with multiple methods
    
#     Args:
#         source: Object point cloud
#         target: Scene point cloud
#         method: Scale estimation method
#             - "auto": Try ransac, fallback to multi_scale if fails
#             - "ransac": RANSAC-based (most robust for noisy data)
#             - "umeyama": Closed-form solution (fast, needs good correspondences)
#             - "multi_scale": Brute force search (guaranteed result)
#             - "bbox": Bounding box ratio (fast approximation)
#         correspondence_distance: Max distance for correspondences (auto if None)
    
#     Returns:
#         scale: Estimated scale factor
#     """
#     print(f"\n{'='*70}")
#     print(f"  SCALE ESTIMATION ({method.upper()})")
#     print(f"{'='*70}")
    
#     # Auto-determine correspondence distance if not provided
#     if correspondence_distance is None:
#         target_bbox = target.get_axis_aligned_bounding_box()
#         correspondence_distance = np.linalg.norm(target_bbox.get_extent()) * 0.05
#         print(f"[SCALE] Auto correspondence distance: {correspondence_distance:.6f}")
    
#     if method == "auto":
#         # Try RANSAC first (most robust)
#         try:
#             scale = estimate_scale_ransac(
#                 source, target,
#                 correspondence_distance=correspondence_distance
#             )
#             # Sanity check: scale should be reasonable (0.01 to 100)
#             if 0.01 < scale < 100:
#                 return scale
#             else:
#                 print(f"[SCALE] ⚠️  RANSAC scale {scale:.6f} seems unreasonable, trying multi-scale")
#         except Exception as e:
#             print(f"[SCALE] ⚠️  RANSAC failed: {e}, trying multi-scale")
        
#         # Fallback to multi-scale
#         scale, _ = multi_scale_icp(
#             source, target,
#             max_correspondence_distance=correspondence_distance
#         )
#         return scale
    
#     elif method == "ransac":
#         return estimate_scale_ransac(
#             source, target,
#             correspondence_distance=correspondence_distance
#         )
    
#     elif method == "umeyama":
#         scale, _, _ = apply_umeyama_registration(
#             source, target,
#             correspondence_distance=correspondence_distance
#         )
#         return scale
    
#     elif method == "multi_scale":
#         scale, _ = multi_scale_icp(
#             source, target,
#             max_correspondence_distance=correspondence_distance
#         )
#         return scale
    
#     elif method == "bbox":
#         return estimate_scale_from_bbox(source, target)
    
#     else:
#         raise ValueError(f"Unknown method: {method}. Use 'auto', 'ransac', 'umeyama', 'multi_scale', or 'bbox'")


# def demo_fn(args):
#     """Main VGGT reconstruction with optional ICP"""
    
#     # Print configuration
#     print("Arguments:", vars(args))

#     # Set seed
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
#     random.seed(args.seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(args.seed)
#         torch.cuda.manual_seed_all(args.seed)
#     print(f"Setting seed as: {args.seed}")

#     # Set device and dtype
#     dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Using device: {device}")
#     print(f"Using dtype: {dtype}")

#     # Load VGGT model
#     model = VGGT()
#     _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
#     model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
#     model.eval()
#     model = model.to(device)
#     print(f"Model loaded")

#     # Get image paths
#     image_dir = os.path.join(args.scene_dir, "images")
#     print(image_dir)
#     image_path_list = glob.glob(os.path.join(image_dir, "*"))
#     if len(image_path_list) == 0:
#         raise ValueError(f"No images found in {image_dir}")
#     base_image_path_list = [os.path.basename(path) for path in image_path_list]

#     # Load images
#     vggt_fixed_resolution = 518
#     img_load_resolution = 1024

#     images, original_coords = load_and_preprocess_images_square(image_path_list, img_load_resolution)
#     images = images.to(device)
#     original_coords = original_coords.to(device)
#     print(f"Loaded {len(images)} images from {image_dir}")

#     # Run VGGT
#     extrinsic, intrinsic, depth_map, depth_conf = run_VGGT(model, images, dtype, vggt_fixed_resolution)
#     points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
    
#     # Bundle Adjustment or feedforward
#     if args.use_ba:
#         image_size = np.array(images.shape[-2:])
#         scale = img_load_resolution / vggt_fixed_resolution
#         shared_camera = args.shared_camera

#         with torch.cuda.amp.autocast(dtype=dtype):
#             pred_tracks, pred_vis_scores, pred_confs, points_3d, points_rgb = predict_tracks(
#                 images,
#                 conf=depth_conf,
#                 points_3d=points_3d,
#                 masks=None,
#                 max_query_pts=args.max_query_pts,
#                 query_frame_num=args.query_frame_num,
#                 keypoint_extractor="aliked+sp",
#                 fine_tracking=args.fine_tracking,
#             )
#             torch.cuda.empty_cache()

#         intrinsic[:, :2, :] *= scale
#         track_mask = pred_vis_scores > args.vis_thresh

#         reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
#             points_3d,
#             extrinsic,
#             intrinsic,
#             pred_tracks,
#             image_size,
#             masks=track_mask,
#             max_reproj_error=args.max_reproj_error,
#             shared_camera=shared_camera,
#             camera_type=args.camera_type,
#             points_rgb=points_rgb,
#         )

#         if reconstruction is None:
#             raise ValueError("No reconstruction can be built with BA")

#         ba_options = pycolmap.BundleAdjustmentOptions()
#         pycolmap.bundle_adjustment(reconstruction, ba_options)
#         reconstruction_resolution = img_load_resolution
        
#     else:
#         conf_thres_value = args.conf_thres_value
#         max_points_for_colmap = 100000
#         shared_camera = False
#         camera_type = "PINHOLE"

#         image_size = np.array([vggt_fixed_resolution, vggt_fixed_resolution])
#         num_frames, height, width, _ = points_3d.shape

#         points_rgb = F.interpolate(
#             images, size=(vggt_fixed_resolution, vggt_fixed_resolution), mode="bilinear", align_corners=False
#         )
#         points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8)
#         points_rgb = points_rgb.transpose(0, 2, 3, 1)

#         points_xyf = create_pixel_coordinate_grid(num_frames, height, width)

#         conf_mask = depth_conf >= conf_thres_value
#         conf_mask = randomly_limit_trues(conf_mask, max_points_for_colmap)

#         points_3d = points_3d[conf_mask]
#         points_xyf = points_xyf[conf_mask]
#         points_rgb = points_rgb[conf_mask]

#         print("Converting to COLMAP format")
#         reconstruction = batch_np_matrix_to_pycolmap_wo_track(
#             points_3d,
#             points_xyf,
#             points_rgb,
#             extrinsic,
#             intrinsic,
#             image_size,
#             shared_camera=shared_camera,
#             camera_type=camera_type,
#         )
#         reconstruction_resolution = vggt_fixed_resolution

#     # Rescale camera
#     reconstruction = rename_colmap_recons_and_rescale_camera(
#         reconstruction,
#         base_image_path_list,
#         original_coords.cpu().numpy(),
#         img_size=reconstruction_resolution,
#         shift_point2d_to_original_res=True,
#         shared_camera=shared_camera,
#     )

#     # Save reconstruction
#     print(f"Saving reconstruction to {args.scene_dir}/sparse")
#     sparse_reconstruction_dir = os.path.join(args.scene_dir, "sparse")
#     os.makedirs(sparse_reconstruction_dir, exist_ok=True)
#     reconstruction.write(sparse_reconstruction_dir)

#     # Save point cloud
#     scene_ply_path = os.path.join(args.scene_dir, "sparse/points.ply")
#     trimesh.PointCloud(points_3d, colors=points_rgb).export(scene_ply_path)
#     print(f"✓ Saved point cloud: {scene_ply_path}")

#     # ADD THIS: Visualize reconstructed scene
#     if args.run_icp and args.visualize_reconstruction:
#         scene_pcd_original = o3d.io.read_point_cloud(scene_ply_path)
#         visualize_point_cloud(scene_pcd_original, "Reconstructed Scene (Before Preprocessing)")

#     # ICP alignment if requested
#     if args.run_icp:
        
#         if args.object_ply is None:
#             print("❌ --object_ply required for ICP")
#             return True
        
#         scene_pcd = preprocess_scene_point_cloud(
#             scene_ply_path=scene_ply_path,
#             output_path=os.path.join(args.scene_dir, "sparse/points_preprocessed.ply"),
#             remove_statistical_outliers=True,
#             remove_table_plane=True,
#             plane_distance_threshold=0.01,
#             plane_num_iterations=1000,
#             remove_points_below=True,
#             plane_offset=-0.02,
#             visualize_steps=args.visualize_preprocessing  # ADD THIS
#         )
        
#         # Run ICP
#         icp_output_dir = os.path.join(args.scene_dir, "icp_results")
#         metrics = run_icp_alignment(
#             object_ply_path=args.object_ply,
#             scene_pcd=scene_pcd,
#             output_dir=icp_output_dir,
#             voxel_size_object=args.voxel_size_object,
#             voxel_size_scene=args.voxel_size_scene,
#             max_correspondence_distance=args.max_correspondence_dist,
#             voxel_size_global=args.voxel_size_global,
#             voxel_size_local=args.voxel_size_local,
#             global_correspondence_dist=args.global_correspondence_dist,
#             local_correspondence_dist=args.local_correspondence_dist,
#             scale_correspondence_dist=args.scale_correspondence_dist,
#             use_point_to_plane=not args.use_point_to_point,
#             estimate_scale=not args.no_scale,
#             scale_method=args.scale_method,
#             use_global_registration=not args.no_global,
#             visualize=args.visualize_icp,
#             show_before=args.show_before,
#             debug=args.debug
#         )

#         print(f"\n{'='*70}")
#         print(f"  ICP COMPLETE")
#         print(f"{'='*70}")
#         print(f"Scale:           {metrics['scale']:.6f}")  # NEW
#         print(f"Fitness:         {metrics['fitness']:.4f}")
#         print(f"Inlier RMSE:     {metrics['inlier_rmse']:.6f}")
#         print(f"Correspondences: {metrics['num_correspondences']}")

#     return True


# def rename_colmap_recons_and_rescale_camera(
#     reconstruction, image_paths, original_coords, img_size, shift_point2d_to_original_res=False, shared_camera=False
# ):
#     """Rename and rescale camera parameters"""
#     rescale_camera = True

#     for pyimageid in reconstruction.images:
#         pyimage = reconstruction.images[pyimageid]
#         pycamera = reconstruction.cameras[pyimage.camera_id]
#         pyimage.name = image_paths[pyimageid - 1]

#         if rescale_camera:
#             pred_params = copy.deepcopy(pycamera.params)
#             real_image_size = original_coords[pyimageid - 1, -2:]
#             resize_ratio = max(real_image_size) / img_size
#             pred_params = pred_params * resize_ratio
#             real_pp = real_image_size / 2
#             pred_params[-2:] = real_pp

#             pycamera.params = pred_params
#             pycamera.width = real_image_size[0]
#             pycamera.height = real_image_size[1]

#         if shift_point2d_to_original_res:
#             top_left = original_coords[pyimageid - 1, :2]
#             for point2D in pyimage.points2D:
#                 point2D.xy = (point2D.xy - top_left) * resize_ratio

#         if shared_camera:
#             rescale_camera = False

#     return reconstruction


# if __name__ == "__main__":
#     args = parse_args()
#     with torch.no_grad():
#         demo_fn(args)

