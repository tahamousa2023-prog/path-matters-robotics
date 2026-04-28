
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
this code reconstruct and then use icp directly 

Anleitung
#### new usage
# Basic usage
python pipeline.py \
    --scene_dir /path/to/scene \
    --object_ply /path/to/object.ply

# With custom parameters
python pipeline.py \
    --scene_dir /path/to/scene \
    --object_ply /path/to/object.ply \
    --scene_downsample 0.005 \
    --local_icp_dist 0.01 \
    --adaptive_fitness_threshold 0.98 \
    --visualize_final

# Skip reconstruction if already done
python pipeline.py \
    --scene_dir /path/to/scene \
    --object_ply /path/to/object.ply \
    --skip_reconstruction

# With visualization and debugging
python pipeline.py \
    --scene_dir /path/to/scene \
    --object_ply /path/to/object.ply \
    --visualize_reconstruction \
    --visualize_preprocessing \
    --visualize_steps \
    --debug

### my usage:
python /home/AP_PathMatters/vggt/z_demo_reconstruction_icp.py \
    --scene_dir /home/AP_PathMatters/path_matters/datasets/yoda \
    --object_ply /home/AP_PathMatters/path_matters/datasets/yoda/Baby_Yoda.ply\
    --visualize_reconstruction \
    --visualize_preprocessing \
    --visualize_steps \
    --debug 
"""

# ============================================================================
# PIPELINE OVERVIEW
# ============================================================================
#
#            +----------------------+
#            |        START         |
#            +----------+-----------+
#                       |
#                       v
#            +----------------------+
#            | 1) VGGT RECONSTRUCT |
#            |  - load images      |
#            |  - run VGGT        |
#            |  - save points.ply |
#            +----------+-----------+
#                       |
#                       v
#            +----------------------+
#            | 2) PREPROCESSING    |
#            +----------+-----------+
#                       |
#        +--------------+--------------+
#        |                             |
#        v                             v
# +--------------+              +--------------+
# | 2a) SCENE    |              | 2b) OBJECT   |
# |  - outliers  |              |  - downsample|
# |  - plane rmv |              |  - normals   |
# |  - downsample|              +------+-------+
# +------+-------+                     |
#        |                             |
#        +--------------+--------------+
#                       |
#                       v
#            +----------------------+
#            | 3) SCALE ESTIMATION |
#            |  - bbox diagonal    |
#            |  - scale & center   |
#            +----------+-----------+
#                       |
#                       v
#            +----------------------+
#            | 4) RANSAC + ICP     |
#            |  - FPFH features    |
#            |  - multi RANSAC     |
#            |  - local ICP refine |
#            +----------+-----------+
#                       |
#                       v
#            +----------------------+
#            | 5) ADAPTIVE REFINE  |
#            |  - random noise     |
#            |  - ICP loop         |
#            |  - improve fit/RMSE |
#            +----------+-----------+
#                       |
#                       v
#            +-------------------------------+
#            | 6) ADAPTIVE SCALE REFINE     |
#            |  - start [0.5x, 1.5x]        |
#            |  - test min/mid/max (RMSE)   |
#            |  - shrink range iteratively  |
#            |  - stop when ±5% range      |
#            +----------+--------------------+
#                       |
#                       v
#            +----------------------+
#            |   FINALIZE & SAVE   |
#            |  - combine transforms|
#            |  - apply scale      |
#            |  - visualize        |
#            |  - save .npy/.ply   |
#            +----------+-----------+
#                       |
#                       v
#            +----------------------+
#            |         END          |
#            +----------------------+


import numpy as np
import copy
import time
import json
import logging
import argparse
import random
import glob
import os
from pathlib import Path
from typing import Tuple, Optional, Dict

import open3d as o3d
import torch
import torch.nn.functional as F
import trimesh
import pycolmap

# VGGT imports
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap_wo_track

# Configure CUDA
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


# ============================================================================
# ARGUMENT PARSER
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="VGGT Reconstruction + ICP Alignment Pipeline")
    
    # ==== REQUIRED ====
    parser.add_argument("--scene_dir", type=str, required=True,
                       help="Directory containing scene images (images/ subfolder required)")
    parser.add_argument("--object_ply", type=str, required=True,
                       help="Path to object PLY file for alignment")
    
    # ==== VGGT RECONSTRUCTION ====
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--conf_thres_value", type=float, default=4.0,# Taha_Edit: I have edited this to remove the outlaiers points, 4 is the best
                       help="Confidence threshold for point filtering (default: 2.0)")
    parser.add_argument("--skip_reconstruction", action="store_true",
                       help="Skip reconstruction if sparse/points.ply already exists")
    
    # ==== SCENE PREPROCESSING ====
    parser.add_argument("--scene_downsample", type=float, default=0.001,  ######
                       help="Voxel size for scene downsampling (default: 0.02)")
    parser.add_argument("--scene_outlier_neighbors", type=int, default=15,
                       help="Neighbors for scene outlier removal (default: 5)")
    parser.add_argument("--scene_outlier_std", type=float, default=10.0,
                       help="Std ratio for scene outlier removal (default: 2.0)")
    parser.add_argument("--plane_threshold", type=float, default=0.015,
                       help="Distance threshold for plane detection (default: 0.0015)")
    parser.add_argument("--plane_offset", type=float, default=-0.015,
                       help="Plane removal offset (default: 0.005)")
    parser.add_argument("--no_plane_removal", action="store_true",
                       help="Disable plane removal")
    
    # ==== OBJECT PREPROCESSING ====
    parser.add_argument("--object_downsample", type=float, default=0.01,   #######
                       help="Voxel size for object downsampling (default: 0.05)")
    
    # ==== SCALE ESTIMATION ====
    parser.add_argument("--scale_method", type=str, default="multi_scale",
                       choices=["bbox", "multi_scale"],
                       help="Scale estimation method (default: bbox)")
    parser.add_argument("--no_scale", action="store_true",
                       help="Disable scale estimation")
    
    # ==== RANSAC INITIAL ALIGNMENT ====
    parser.add_argument("--ransac_tries", type=int, default=20,
                       help="Number of RANSAC attempts (default: 20)")
    parser.add_argument("--ransac_downsample", type=float, default=0.01,  ######
                       help="Voxel size for RANSAC (default: 0.01)")
    parser.add_argument("--ransac_max_dist", type=float, default=0.1, ######
                       help="Max correspondence distance for RANSAC (default: 0.05)")
    
    # ==== LOCAL ICP REFINEMENT ====
    parser.add_argument("--local_icp_dist", type=float, default=0.025, ###### this is what affect the fitness and rmse at the end too low and the fitness gets low too
                       help="Distance threshold for local ICP (default: 0.025)")
    parser.add_argument("--local_icp_iters", type=int, default=500,
                       help="Max iterations for local ICP (default: 500)")
    
    # ==== ADAPTIVE REFINEMENT ====
    parser.add_argument("--adaptive_iters", type=int, default=50,
                       help="Max adaptive refinement iterations (default: 50)")
    parser.add_argument("--adaptive_fitness_threshold", type=float, default=0.95,
                       help="Target fitness threshold (default: 0.95)")
    parser.add_argument("--adaptive_rmse_threshold", type=float, default=0.005,
                       help="Target RMSE threshold (default: 0.005)")
    parser.add_argument("--adaptive_rotation_noise", type=float, default=0.01,
                       help="Rotation noise range in radians (default: 0.001)")
    parser.add_argument("--adaptive_translation_noise", type=float, default=0.1,
                       help="Initial translation noise (default: 0.01)")
    
    # ==== VISUALIZATION ====
    parser.add_argument("--visualize_reconstruction", action="store_true",
                       help="Visualize raw reconstructed point cloud")
    parser.add_argument("--visualize_preprocessing", action="store_true",
                       help="Visualize each preprocessing step")
    parser.add_argument("--visualize_steps", action="store_true",
                       help="Visualize intermediate alignment steps")
    parser.add_argument("--visualize_final", action="store_true", default=True,
                       help="Visualize final result (default: True)")
    parser.add_argument("--no_visualize_final", dest="visualize_final", action="store_false",
                       help="Disable final visualization")
    
    # ==== DEBUG ====
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    
    return parser.parse_args()


# ============================================================================
# CONFIGURATION CLASS (FROM ARGS)
# ============================================================================

class PipelineConfig:
    """Configuration built from command-line arguments"""
    
    def __init__(self, args):
        # Scene preprocessing
        self.SCENE_DOWNSAMPLE_VOXEL = args.scene_downsample
        self.SCENE_OUTLIER_NEIGHBORS = args.scene_outlier_neighbors
        self.SCENE_OUTLIER_STD = args.scene_outlier_std
        self.PLANE_DISTANCE_THRESHOLD = args.plane_threshold
        self.PLANE_OFFSET = args.plane_offset
        self.REMOVE_PLANE = not args.no_plane_removal
        
        # Object preprocessing
        self.OBJECT_DOWNSAMPLE_VOXEL = args.object_downsample
        
        # Scale
        self.SCALE_METHOD = args.scale_method
        self.ESTIMATE_SCALE = not args.no_scale
        
        # RANSAC
        self.RANSAC_TRIES = args.ransac_tries
        self.RANSAC_DOWNSAMPLE = args.ransac_downsample
        self.RANSAC_MAX_DIST = args.ransac_max_dist
        
        # Local ICP
        self.LOCAL_ICP_DIST = args.local_icp_dist
        self.LOCAL_ICP_ITERATIONS = args.local_icp_iters
        
        # Adaptive refinement
        self.ADAPTIVE_MAX_ITERATIONS = args.adaptive_iters
        self.ADAPTIVE_FITNESS_THRESHOLD = args.adaptive_fitness_threshold
        self.ADAPTIVE_RMSE_THRESHOLD = args.adaptive_rmse_threshold
        self.ADAPTIVE_NOISE_ROTATION_RANGE = args.adaptive_rotation_noise
        self.ADAPTIVE_NOISE_TRANSLATION_START = args.adaptive_translation_noise
        
        # Visualization
        self.VISUALIZE_RECONSTRUCTION = args.visualize_reconstruction
        self.VISUALIZE_PREPROCESSING = args.visualize_preprocessing
        self.VISUALIZE_STEPS = args.visualize_steps
        self.VISUALIZE_FINAL = args.visualize_final
        
        # Debug
        self.DEBUG = args.debug


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def setup_logging(level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler()]
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


def visualize_pcd(pcd: o3d.geometry.PointCloud, title: str = "Point Cloud"):
    """Visualize single point cloud"""
    logging.info(f"🔍 Visualizing: {title} ({len(pcd.points)} points)")
    o3d.visualization.draw_geometries([pcd], window_name=title, width=1280, height=720)


# ============================================================================
# VGGT RECONSTRUCTION
# ============================================================================

@timeit
def run_vggt_reconstruction(scene_dir: Path, conf_threshold: float, seed: int) -> Path:
    """Run VGGT reconstruction and return path to point cloud"""
    
    logging.info("\n" + "="*70)
    logging.info("  VGGT RECONSTRUCTION")
    logging.info("="*70)
    
    # Set seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Device setup
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Device: {device}, dtype: {dtype}")
    
    # Load model
    logging.info("Loading VGGT model...")
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval().to(device)
    
    # Load images
    image_dir = scene_dir / "images"
    image_paths = sorted(glob.glob(str(image_dir / "*")))
    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")
    
    logging.info(f"Loading {len(image_paths)} images...")
    images, original_coords = load_and_preprocess_images_square(image_paths, 1024)
    images = images.to(device)
    original_coords = original_coords.to(device)
    
    # Run VGGT
    logging.info("Running VGGT inference...")
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images_batch = F.interpolate(images, size=(518, 518), mode="bilinear", align_corners=False)
            images_batch = images_batch[None]
            
            aggregated_tokens_list, ps_idx = model.aggregator(images_batch)
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images_batch.shape[-2:])
            depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images_batch, ps_idx)
    
    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    depth_map = depth_map.squeeze(0).cpu().numpy()
    depth_conf = depth_conf.squeeze(0).cpu().numpy()
    
    # Unproject to 3D
    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
    
    # Filter by confidence
    num_frames, height, width, _ = points_3d.shape
    points_rgb = F.interpolate(images, size=(518, 518), mode="bilinear", align_corners=False)
    points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)
    
    points_xyf = create_pixel_coordinate_grid(num_frames, height, width)
    conf_mask = depth_conf >= conf_threshold
    conf_mask = randomly_limit_trues(conf_mask, 100000)
    
    points_3d = points_3d[conf_mask]
    points_xyf = points_xyf[conf_mask]
    points_rgb = points_rgb[conf_mask]
    
    logging.info(f"Filtered points: {len(points_3d)} (conf >= {conf_threshold})")
    
    # Convert to COLMAP
    logging.info("Converting to COLMAP format...")
    image_size = np.array([518, 518])
    reconstruction = batch_np_matrix_to_pycolmap_wo_track(
        points_3d, points_xyf, points_rgb, extrinsic, intrinsic,
        image_size, shared_camera=False, camera_type="PINHOLE"
    )
    
    # Rescale cameras
    base_image_paths = [os.path.basename(p) for p in image_paths]
    reconstruction = rename_and_rescale_colmap(
        reconstruction, base_image_paths, original_coords.cpu().numpy(), 518
    )
    
    # Save reconstruction
    sparse_dir = scene_dir / "sparse"
    sparse_dir.mkdir(exist_ok=True)
    reconstruction.write(str(sparse_dir))
    
    # Save point cloud
    scene_ply = sparse_dir / "points.ply"
    trimesh.PointCloud(points_3d, colors=points_rgb).export(str(scene_ply))
    logging.info(f"✓ Saved: {scene_ply}")
    
    return scene_ply


def rename_and_rescale_colmap(reconstruction, image_paths, original_coords, img_size):
    """Rename and rescale COLMAP reconstruction"""
    for pyimageid in reconstruction.images:
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = image_paths[pyimageid - 1]
        
        pred_params = copy.deepcopy(pycamera.params)
        real_image_size = original_coords[pyimageid - 1, -2:]
        resize_ratio = max(real_image_size) / img_size
        pred_params = pred_params * resize_ratio
        pred_params[-2:] = real_image_size / 2
        
        pycamera.params = pred_params
        pycamera.width = int(real_image_size[0])
        pycamera.height = int(real_image_size[1])
    
    return reconstruction


# ============================================================================
# PREPROCESSING
# ============================================================================

@timeit
def preprocess_scene(pcd_path: Path, config: PipelineConfig, save_path: Optional[Path] = None):
    """Preprocess scene point cloud"""
    
    logging.info("\n" + "="*70)
    logging.info("  SCENE PREPROCESSING")
    logging.info("="*70)
    
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    original_count = len(pcd.points)
    logging.info(f"Loaded: {original_count} points")
    
    if config.VISUALIZE_PREPROCESSING:
        visualize_pcd(pcd, "Original Scene")
    
    # Outlier removal
    logging.info("Removing outliers...")
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=config.SCENE_OUTLIER_NEIGHBORS,
        std_ratio=config.SCENE_OUTLIER_STD
    )
    logging.info(f"  {original_count} → {len(pcd.points)} points")
    
    if config.VISUALIZE_PREPROCESSING:
        visualize_pcd(pcd, "After Outlier Removal")
    
    # Plane removal
    if config.REMOVE_PLANE:
        logging.info("Removing plane...")
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=30))
        
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=config.PLANE_DISTANCE_THRESHOLD,
            ransac_n=3,
            num_iterations=1000
        )
        
        a, b, c, d = plane_model
        logging.info(f"  Plane: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")
        
        # Remove points below plane
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None
        
        plane_norm = np.sqrt(a**2 + b**2 + c**2)
        d_offset = d - config.PLANE_OFFSET * plane_norm
        distances = (a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d_offset) / plane_norm
        
        above_mask = distances <= 0
        pcd_filtered = o3d.geometry.PointCloud()
        pcd_filtered.points = o3d.utility.Vector3dVector(points[above_mask])
        if colors is not None:
            pcd_filtered.colors = o3d.utility.Vector3dVector(colors[above_mask])
        
        pcd = pcd_filtered
        logging.info(f"  → {len(pcd.points)} points")
        
        # Re-estimate normals
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=30))
        
        if config.VISUALIZE_PREPROCESSING:
            visualize_pcd(pcd, "After Plane Removal")
    
    # Downsample
    logging.info(f"Downsampling (voxel={config.SCENE_DOWNSAMPLE_VOXEL})...")
    pcd = pcd.voxel_down_sample(voxel_size=config.SCENE_DOWNSAMPLE_VOXEL)
    logging.info(f"  → {len(pcd.points)} points")
    
    if config.VISUALIZE_PREPROCESSING:
        visualize_pcd(pcd, "Final Preprocessed Scene")
    
    if save_path:
        o3d.io.write_point_cloud(str(save_path), pcd)
        logging.info(f"✓ Saved: {save_path}")
    
    logging.info(f"✓ Complete: {original_count} → {len(pcd.points)} points")
    return pcd


@timeit
def preprocess_object(pcd_path: Path, config: PipelineConfig):
    """Preprocess object point cloud"""
    
    logging.info("\n" + "="*70)
    logging.info("  OBJECT PREPROCESSING")
    logging.info("="*70)
    
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    logging.info(f"Loaded: {len(pcd.points)} points")
    
    # Downsample
    logging.info(f"Downsampling (voxel={config.OBJECT_DOWNSAMPLE_VOXEL})...")
    pcd = pcd.voxel_down_sample(voxel_size=config.OBJECT_DOWNSAMPLE_VOXEL)
    logging.info(f"  → {len(pcd.points)} points")
    
    # Normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=1, max_nn=30
        )
    )
    
    return pcd


# ============================================================================
# SCALE ESTIMATION
# ============================================================================

def estimate_scale_bbox(source, target):
    """Estimate scale from bounding box"""
    source_bbox = source.get_axis_aligned_bounding_box()
    target_bbox = target.get_axis_aligned_bounding_box()
    
    source_diag = np.linalg.norm(source_bbox.get_extent())
    target_diag = np.linalg.norm(target_bbox.get_extent())
    
    scale = target_diag / source_diag
    logging.info(f"  BBox ratio: {scale:.6f}")
    return scale


@timeit
def estimate_scale(source, target, config):
    """Estimate scale"""
    logging.info("\n" + "="*70)
    logging.info("  SCALE ESTIMATION")
    logging.info("="*70)
    
    if config.SCALE_METHOD == "bbox":
        scale = estimate_scale_bbox(source, target)
    else:
        # Could add multi_scale here
        scale = estimate_scale_bbox(source, target)
    #scale = 0.15
    # Create initial transform (scale + center align)
    source_scaled = copy.deepcopy(source)
    source_scaled.scale(scale, center=source_scaled.get_center())
    translation = target.get_center() - source_scaled.get_center()
    init_transform = np.eye(4)
    init_transform[:3, 3] = translation
    
    logging.info(f"✓ Scale: {scale:.6f}")
    return scale, init_transform


# ============================================================================
# RANSAC INITIAL ALIGNMENT
# ============================================================================

@timeit
def initial_alignment_ransac(source, target, config):
    """RANSAC-based initial alignment"""
    
    logging.info("\n" + "="*70)
    logging.info(f"  RANSAC INITIAL ALIGNMENT ({config.RANSAC_TRIES} attempts)")
    logging.info("="*70)
    
    # Downsample
    source_down = source#.voxel_down_sample(config.RANSAC_DOWNSAMPLE)
    target_down = target.voxel_down_sample(config.RANSAC_DOWNSAMPLE)
    logging.info(f"Downsampled: {len(source.points)} → {len(source_down.points)} (source)")
    logging.info(f"             {len(target.points)} → {len(target_down.points)} (target)")
    
    # Normals
    source_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=100)
    )
    target_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=100)
    )
    
    # FPFH
    logging.info("Computing FPFH features...")
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down, o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=5)
    )
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=5)
    )
    
    # Multiple RANSAC attempts
    all_results = []
    best_result = None
    best_fitness = -1
    
    for i in range(config.RANSAC_TRIES):
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh,
            mutual_filter=True,
            max_correspondence_distance=config.RANSAC_MAX_DIST*0.1,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=3,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(config.RANSAC_MAX_DIST)
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
        )
        
        all_results.append({'attempt': i + 1, 'fitness': result.fitness, 'rmse': result.inlier_rmse})
        logging.info(f"  Try {i+1}/{config.RANSAC_TRIES}: fitness={result.fitness:.4f}, RMSE={result.inlier_rmse:.6f}")
        
        if result.fitness > best_fitness:
            best_fitness = result.fitness
            best_result = result
            logging.info(f"    ✅ NEW BEST!")
    
    logging.info(f"✓ Best fitness: {best_fitness:.4f}")
    return best_result, all_results


# ============================================================================
# ADAPTIVE REFINEMENT
# ============================================================================

@timeit
def adaptive_refinement(source, target, initial_result, config):
    """Adaptive refinement with random perturbations"""
    
    logging.info("\n" + "="*70)
    logging.info("  ADAPTIVE REFINEMENT")
    logging.info("="*70)
    
    best_fitness = initial_result.fitness
    best_rmse = initial_result.inlier_rmse
    best_transformation = initial_result.transformation
    
    iteration = 0
    noise_translation = config.ADAPTIVE_NOISE_TRANSLATION_START
    
    logging.info(f"Target: fitness > {config.ADAPTIVE_FITNESS_THRESHOLD}, RMSE < {config.ADAPTIVE_RMSE_THRESHOLD}")
    logging.info(f"Starting: fitness={best_fitness:.4f}, RMSE={best_rmse:.6f}")
    
    while (iteration < config.ADAPTIVE_MAX_ITERATIONS and
           (best_fitness < config.ADAPTIVE_FITNESS_THRESHOLD or best_rmse > config.ADAPTIVE_RMSE_THRESHOLD)):
        
        # Random perturbation
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
            
            if result.fitness > 0 and result.inlier_rmse > 0:
                if (result.fitness > best_fitness or
                    (result.fitness == best_fitness and result.inlier_rmse < best_rmse)):
                    
                    improvement = result.fitness - best_fitness
                    best_fitness = result.fitness
                    best_rmse = result.inlier_rmse
                    best_transformation = result.transformation
                    
                    logging.info(f"  ✅ Iter {iteration+1}: fitness={best_fitness:.4f} (+{improvement:.4f}), RMSE={best_rmse:.6f}")
                    
                    if best_fitness >= config.ADAPTIVE_FITNESS_THRESHOLD and best_rmse <= config.ADAPTIVE_RMSE_THRESHOLD:
                        logging.info("  🎉 Target reached!")
                        break
            else:
                noise_translation += 0.75
                
        except Exception as e:
            logging.debug(f"  ⚠️  Iter {iteration+1} error: {e}")
            noise_translation += 0.1
        
        iteration += 1
        
        if iteration % 10 == 0:
            logging.info(f"  Progress: iter={iteration}, best_fitness={best_fitness:.4f}")
    
    final_result = o3d.pipelines.registration.RegistrationResult()
    final_result.fitness = best_fitness
    final_result.inlier_rmse = best_rmse
    final_result.transformation = best_transformation
    
    logging.info(f"✓ Complete ({iteration} iterations)")
    logging.info(f"  Final fitness: {best_fitness:.4f}")
    logging.info(f"  Final RMSE: {best_rmse:.6f}")
    
    return final_result

@timeit
def refine_scale_by_fitness(object_pcd, scene_pcd, initial_scale, initial_result, config):
    """Adaptive scale refinement - starts broad, narrows down intelligently"""
    
    logging.info("\n" + "="*70)
    logging.info("  ADAPTIVE SCALE REFINEMENT")
    logging.info("="*70)
    
    best_rmse = initial_result.inlier_rmse
    best_scale = initial_scale
    best_result = initial_result
    all_tested = []
    
    scale_min = initial_scale * 0.5
    scale_max = initial_scale * 1.5
    iteration = 0
    max_iterations = 6
    
    logging.info(f"Initial scale: {initial_scale:.6f}")
    logging.info(f"Initial RMSE: {best_rmse:.6f}")
    logging.info(f"Starting range: [{scale_min:.6f}, {scale_max:.6f}]\n")
    
    while iteration < max_iterations:
        iteration += 1
        scale_range = scale_max - scale_min
        
        scales_to_test = [scale_min, (scale_min + scale_max) / 2, scale_max]
        logging.info(f"Iteration {iteration}: Testing [{scale_min:.6f}, {scale_max:.6f}]")
        
        results = []
        
        for i, scale in enumerate(scales_to_test):
            label = ["min", "mid", "max"][i]
            
            obj = copy.deepcopy(object_pcd)
            obj.scale(scale, center=obj.get_center())
            trans = np.eye(4)
            trans[:3, 3] = scene_pcd.get_center() - obj.get_center()
            obj.transform(trans)
            
            try:
                result = o3d.pipelines.registration.registration_icp(
                    obj, scene_pcd,
                    config.LOCAL_ICP_DIST,
                    initial_result.transformation,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
                )
                
                rmse = result.inlier_rmse
                
                best_marker = "✅ BEST" if rmse < best_rmse else ""
                logging.info(f"  {label:3s} ({scale:.6f}): rmse={rmse:.6f} {best_marker}")
                
                results.append({
                    'scale': scale,
                    'rmse': rmse,
                    'label': label,
                    'result': result
                })
                all_tested.append({
                    'iteration': iteration,
                    'scale': float(scale),
                    'rmse': float(rmse)
                })
                
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_scale = scale
                    best_result = result
                    
            except Exception as e:
                logging.warning(f"  {label:3s} ({scale:.6f}): FAILED")
                results.append({
                    'scale': scale,
                    'rmse': float('inf'),
                    'label': label,
                    'result': None
                })
        
        min_r = results[0]
        mid_r = results[1]
        max_r = results[2]
        
        # Check convergence
        if scale_range < initial_scale * 0.05:
            logging.info(f"  ✓ Converged! Range is ±{100*scale_range/(2*initial_scale):.1f}%\n")
            break
        
        # Adaptive decision logic
        logging.info(f"  Analysis:")
        
        # If middle is best (lowest RMSE)
        if mid_r['rmse'] <= min_r['rmse'] and mid_r['rmse'] <= max_r['rmse']:
            logging.info(f"    → Middle {mid_r['scale']:.6f} is best, zooming in")
            scale_min = (scale_min + mid_r['scale']) / 2
            scale_max = (scale_max + mid_r['scale']) / 2
        
        # If left is better than right
        elif min_r['rmse'] <= max_r['rmse']:
            logging.info(f"    → Left {min_r['scale']:.6f} is better, shifting left")
            scale_max = mid_r['scale']
            scale_min = min_r['scale'] * 0.9
        
        # If right is better than left
        elif max_r['rmse'] <= min_r['rmse']:
            logging.info(f"    → Right {max_r['scale']:.6f} is better, shifting right")
            scale_min = mid_r['scale']
            scale_max = max_r['scale'] * 1.1
        
        # If both boundaries are bad
        else:
            logging.info(f"    → Both edges bad, testing inner range")
            scale_min = min_r['scale'] * 1.1
            scale_max = max_r['scale'] * 0.9
        
        logging.info(f"    Next range: [{scale_min:.6f}, {scale_max:.6f}]\n")
    
    improvement = initial_result.inlier_rmse - best_rmse
    improvement_pct = 100 * improvement / initial_result.inlier_rmse if initial_result.inlier_rmse > 0 else 0
    
    logging.info("="*70)
    logging.info(f"✓ Best scale found: {best_scale:.6f}")
    logging.info(f"  RMSE: {initial_result.inlier_rmse:.6f} → {best_rmse:.6f} (-{improvement_pct:.1f}%)")
    logging.info(f"  Iterations: {iteration}, Total tests: {len(all_tested)}")
    logging.info("="*70)
    
    return best_scale, best_result, all_tested

# ============================================================================
# MAIN PIPELINE
# ============================================================================

@timeit
def run_pipeline(args):
    """Complete pipeline: Reconstruction → Preprocessing → ICP"""
    
    start_time = time.perf_counter()
    
    scene_dir = Path(args.scene_dir)
    output_dir = scene_dir / "icp_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build config
    config = PipelineConfig(args)
    
    logging.info("\n" + "="*70)
    logging.info("  COMPLETE PIPELINE")
    logging.info("="*70)
    logging.info(f"Scene: {scene_dir}")
    logging.info(f"Object: {args.object_ply}")
    logging.info(f"Output: {output_dir}")
    logging.info("="*70)
    
    # ========================================================================
    # STAGE 1: RECONSTRUCTION
    # ========================================================================
    
    scene_ply_path = scene_dir / "sparse" / "points.ply"
    
    if args.skip_reconstruction and scene_ply_path.exists():
        logging.info(f"\n✓ Using existing reconstruction: {scene_ply_path}")
    else:
        scene_ply_path = run_vggt_reconstruction(
            scene_dir=scene_dir,
            conf_threshold=args.conf_thres_value,
            seed=args.seed
        )
    
    if config.VISUALIZE_RECONSTRUCTION:
        pcd_raw = o3d.io.read_point_cloud(str(scene_ply_path))
        visualize_pcd(pcd_raw, "Reconstructed Scene (Raw)")
    
    # ========================================================================
    # STAGE 2: PREPROCESSING
    # ========================================================================
    
    scene_pcd = preprocess_scene(
        pcd_path=scene_ply_path,
        config=config,
        save_path=output_dir / "scene_preprocessed.ply"
    )
    
    object_pcd = preprocess_object(
        pcd_path=Path(args.object_ply),
        config=config
    )
    
    # ========================================================================
    # STAGE 3: SCALE ESTIMATION
    # ========================================================================
    
    if config.ESTIMATE_SCALE:
        scale, scale_transform = estimate_scale(object_pcd, scene_pcd, config)
        
        object_pcd_scaled = copy.deepcopy(object_pcd)
        object_pcd_scaled.scale(scale, center=object_pcd_scaled.get_center())
        object_pcd_scaled.transform(scale_transform)
    else:
        scale = 1.0
        scale_transform = np.eye(4)
        object_pcd_scaled = object_pcd
    
    # ========================================================================
    # STAGE 4: RANSAC INITIAL ALIGNMENT
    # ========================================================================
    
    local_result, all_attempts = initial_alignment_ransac(
        object_pcd_scaled, scene_pcd, config
    )
    
    # Refine with ICP
    logging.info("\n🔧 Refining RANSAC with ICP...")
    local_result = o3d.pipelines.registration.registration_icp(
        object_pcd_scaled, scene_pcd,
        config.LOCAL_ICP_DIST,
        local_result.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=config.LOCAL_ICP_ITERATIONS)
    )
    logging.info(f"  After ICP: fitness={local_result.fitness:.4f}, RMSE={local_result.inlier_rmse:.6f}")
    
    # ========================================================================
    # STAGE 5: ADAPTIVE REFINEMENT
    # ========================================================================
    final_result = adaptive_refinement(
        object_pcd_scaled, scene_pcd, local_result, config
    )
    
    # ========================================================================
    # STAGE 6: SCALE REFINEMENT (NEW!)
    # ========================================================================
    
    refined_scale, refined_result, scale_results = refine_scale_by_fitness(
        object_pcd, scene_pcd, scale, final_result, config
    )
    
    if refined_result.fitness > final_result.fitness:
        logging.info(f"✅ Using refined scale: {refined_scale:.6f}")
        scale = refined_scale
        final_result = refined_result
    else:
        logging.info(f"⚠️  Keeping original scale: {scale:.6f}")
    
    # ========================================================================
    # FINALIZE
    # ========================================================================
    
    final_transformation = np.dot(final_result.transformation, scale_transform)
    
    object_aligned = copy.deepcopy(object_pcd)
    object_aligned.scale(scale, center=object_aligned.get_center())
    object_aligned.transform(final_transformation)
    
    # Visualization
    if config.VISUALIZE_FINAL:
        target_vis = copy.deepcopy(scene_pcd).paint_uniform_color([1, 0, 0])
        aligned_vis = copy.deepcopy(object_aligned).paint_uniform_color([0, 1, 0])
        
        logging.info("\n🎬 Final Visualization")
        logging.info("  Red = Scene")
        logging.info("  Green = Aligned Object")
        
        o3d.visualization.draw_geometries(
            [target_vis, aligned_vis],
            window_name="Final Result",
            width=1280, height=720
        )
    
    # Save
    logging.info("\n💾 Saving results...")
    np.save(output_dir / "transformation.npy", final_transformation)
    np.save(output_dir / "scale.npy", np.array([scale]))
    o3d.io.write_point_cloud(str(output_dir / "object_aligned.ply"), object_aligned)
    
    metrics = {
        'scale': float(scale),
        'ransac_best_fitness': float(local_result.fitness),
        'ransac_best_rmse': float(local_result.inlier_rmse),
        'final_fitness': float(final_result.fitness),
        'final_rmse': float(final_result.inlier_rmse),
        'final_correspondences': len(final_result.correspondence_set),
        'transformation': final_transformation.tolist(),
        'elapsed_time': time.perf_counter() - start_time
    }
    
    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logging.info(f"  ✓ transformation.npy")
    logging.info(f"  ✓ scale.npy")
    logging.info(f"  ✓ object_aligned.ply")
    logging.info(f"  ✓ metrics.json")
    
    # Summary
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
        logging.info("✅ EXCELLENT alignment!")
    elif final_result.fitness >= 0.6:
        logging.info("✓  GOOD alignment")
    else:
        logging.info("⚠️  MODERATE alignment - may need tuning")
    
    logging.info("="*70)
    
    return metrics


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    args = parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(level=log_level)
    
    # Run pipeline
    with torch.no_grad():
        metrics = run_pipeline(args)
    
    print(f"\n✅ Pipeline complete!")
    print(f"   Final fitness: {metrics['final_fitness']:.4f}")
    print(f"   Final RMSE: {metrics['final_rmse']:.6f}")


