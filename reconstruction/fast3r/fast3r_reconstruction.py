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
python /home/AP_PathMatters/fast3r/fast3r_reconstruction.py \
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
#            | 1) FAST3R RECONSTRUCT|
#            |  - load images      |
#            |  - run Fast3R       |
#            |  - save points.ply  |
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
#            |  - stop when +/-5% range     |
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


"""
Fast3R Reconstruction + ICP Alignment Pipeline with Adaptive Scale Refinement

This pipeline reconstructs a 3D scene from images using Fast3R, then aligns a known
3D object to the reconstructed scene using ICP (Iterative Closest Point) with
adaptive scale refinement.

Main stages:
1. Fast3R Reconstruction - Convert images to 3D point cloud
2. Preprocessing - Clean up point clouds (remove noise, plane, downsample)
3. Scale Estimation - Estimate initial object scale
4. RANSAC Alignment - Find initial rough alignment
5. ICP Refinement - Refine alignment precisely
6. Adaptive Refinement - Fine-tune with small random perturbations
7. Scale Refinement - Find optimal scale by testing different values
"""

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

# Fast3R imports
from fast3r.models.fast3r import Fast3R
from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule
from fast3r.dust3r.utils.image import load_images
from fast3r.dust3r.inference_multiview import inference

# Configure CUDA
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


# ============================================================================
# DEFAULT PARAMETER VALUES
# ============================================================================

# FAST3R RECONSTRUCTION PARAMETERS
DEFAULT_SEED = 42
DEFAULT_CONFIDENCE_THRESHOLD = 0.15  # 0-1, higher = fewer points but higher quality
DEFAULT_IMAGE_SIZE = 512
DEFAULT_MODEL_NAME = "jedyang97/Fast3R_ViT_Large_512"

# SCENE PREPROCESSING PARAMETERS
DEFAULT_SCENE_DOWNSAMPLE_VOXEL = 0.001
DEFAULT_SCENE_OUTLIER_NEIGHBORS = 50
DEFAULT_SCENE_OUTLIER_STD_RATIO = 5.0
DEFAULT_PLANE_DISTANCE_THRESHOLD = 0.015
DEFAULT_PLANE_OFFSET = -0.015
DEFAULT_SCENE_NORMAL_RADIUS = 1.0
DEFAULT_SCENE_NORMAL_MAX_NN = 30

# OBJECT PREPROCESSING PARAMETERS
DEFAULT_OBJECT_DOWNSAMPLE_VOXEL = 0.01
DEFAULT_OBJECT_NORMAL_RADIUS = 1.0
DEFAULT_OBJECT_NORMAL_MAX_NN = 30

# SCALE ESTIMATION PARAMETERS
DEFAULT_SCALE_METHOD = "bbox"

# RANSAC ALIGNMENT PARAMETERS
DEFAULT_RANSAC_TRIES = 20
DEFAULT_RANSAC_DOWNSAMPLE_VOXEL = 0.01
DEFAULT_RANSAC_MAX_CORRESPONDENCE_DISTANCE = 0.1
DEFAULT_RANSAC_CORRESPONDENCE_CHECKER_DISTANCE = 0.1
DEFAULT_RANSAC_NORMAL_RADIUS = 5.0
DEFAULT_RANSAC_NORMAL_MAX_NN = 100
DEFAULT_RANSAC_FPFH_RADIUS = 5.0
DEFAULT_RANSAC_FPFH_MAX_NN = 5

# ICP REFINEMENT PARAMETERS
DEFAULT_ICP_MAX_CORRESPONDENCE_DISTANCE = 0.025
DEFAULT_ICP_MAX_ITERATIONS = 500
DEFAULT_ADAPTIVE_ICP_DISTANCE_MULTIPLIER = 0.5

# ADAPTIVE REFINEMENT PARAMETERS
DEFAULT_ADAPTIVE_MAX_ITERATIONS = 50
DEFAULT_ADAPTIVE_FITNESS_THRESHOLD = 0.95
DEFAULT_ADAPTIVE_RMSE_THRESHOLD = 0.005
DEFAULT_ADAPTIVE_ROTATION_NOISE_RANGE = 0.01
DEFAULT_ADAPTIVE_TRANSLATION_NOISE_START = 0.1

# SCALE REFINEMENT PARAMETERS
DEFAULT_SCALE_SEARCH_MIN_FACTOR = 0.5
DEFAULT_SCALE_SEARCH_MAX_FACTOR = 1.5
DEFAULT_SCALE_REFINEMENT_MAX_ITERATIONS = 6
DEFAULT_SCALE_CONVERGENCE_THRESHOLD = 0.05
DEFAULT_SCALE_ICP_MAX_ITERATIONS = 200


# ============================================================================
# ARGUMENT PARSER
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fast3R Reconstruction + ICP Alignment Pipeline with Scale Refinement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # REQUIRED
    parser.add_argument("--scene_dir", type=str, required=True,
                       help="Directory containing scene images (must have images/ subfolder)")
    parser.add_argument("--object_ply", type=str, required=True,
                       help="Path to reference object PLY file for alignment")
    
    # FAST3R RECONSTRUCTION
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--conf_thres_value", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD)
    parser.add_argument("--image_size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--skip_reconstruction", action="store_true")
    
    # SCENE PREPROCESSING
    parser.add_argument("--scene_downsample", type=float, default=DEFAULT_SCENE_DOWNSAMPLE_VOXEL)
    parser.add_argument("--scene_outlier_neighbors", type=int, default=DEFAULT_SCENE_OUTLIER_NEIGHBORS)
    parser.add_argument("--scene_outlier_std", type=float, default=DEFAULT_SCENE_OUTLIER_STD_RATIO)
    parser.add_argument("--plane_threshold", type=float, default=DEFAULT_PLANE_DISTANCE_THRESHOLD)
    parser.add_argument("--plane_offset", type=float, default=DEFAULT_PLANE_OFFSET)
    parser.add_argument("--no_plane_removal", action="store_true")
    
    # OBJECT PREPROCESSING
    parser.add_argument("--object_downsample", type=float, default=DEFAULT_OBJECT_DOWNSAMPLE_VOXEL)
    
    # SCALE ESTIMATION
    parser.add_argument("--scale_method", type=str, default=DEFAULT_SCALE_METHOD, choices=["bbox", "multi_scale"])
    parser.add_argument("--no_scale", action="store_true")
    
    # RANSAC INITIAL ALIGNMENT
    parser.add_argument("--ransac_tries", type=int, default=DEFAULT_RANSAC_TRIES)
    parser.add_argument("--ransac_downsample", type=float, default=DEFAULT_RANSAC_DOWNSAMPLE_VOXEL)
    parser.add_argument("--ransac_max_dist", type=float, default=DEFAULT_RANSAC_MAX_CORRESPONDENCE_DISTANCE)
    
    # LOCAL ICP REFINEMENT
    parser.add_argument("--local_icp_dist", type=float, default=DEFAULT_ICP_MAX_CORRESPONDENCE_DISTANCE)
    parser.add_argument("--local_icp_iters", type=int, default=DEFAULT_ICP_MAX_ITERATIONS)
    
    # ADAPTIVE REFINEMENT
    parser.add_argument("--adaptive_iters", type=int, default=DEFAULT_ADAPTIVE_MAX_ITERATIONS)
    parser.add_argument("--adaptive_fitness_threshold", type=float, default=DEFAULT_ADAPTIVE_FITNESS_THRESHOLD)
    parser.add_argument("--adaptive_rmse_threshold", type=float, default=DEFAULT_ADAPTIVE_RMSE_THRESHOLD)
    parser.add_argument("--adaptive_rotation_noise", type=float, default=DEFAULT_ADAPTIVE_ROTATION_NOISE_RANGE)
    parser.add_argument("--adaptive_translation_noise", type=float, default=DEFAULT_ADAPTIVE_TRANSLATION_NOISE_START)
    
    # VISUALIZATION
    parser.add_argument("--visualize_reconstruction", action="store_true")
    parser.add_argument("--visualize_preprocessing", action="store_true")
    parser.add_argument("--visualize_steps", action="store_true")
    parser.add_argument("--visualize_final", action="store_true", default=True)
    parser.add_argument("--no_visualize_final", dest="visualize_final", action="store_false")
    
    # DEBUG
    parser.add_argument("--debug", action="store_true")
    
    return parser.parse_args()


# ============================================================================
# CONFIGURATION CLASS
# ============================================================================

class PipelineConfig:
    def __init__(self, args):
        # Fast3R Configuration
        self.IMAGE_SIZE = args.image_size
        self.MODEL_NAME = args.model_name
        self.CONFIDENCE_THRESHOLD = args.conf_thres_value
        
        # Scene Preprocessing Configuration
        self.SCENE_DOWNSAMPLE_VOXEL = args.scene_downsample
        self.SCENE_OUTLIER_NEIGHBORS = args.scene_outlier_neighbors
        self.SCENE_OUTLIER_STD = args.scene_outlier_std
        self.PLANE_DISTANCE_THRESHOLD = args.plane_threshold
        self.PLANE_OFFSET = args.plane_offset
        self.REMOVE_PLANE = not args.no_plane_removal
        self.SCENE_NORMAL_RADIUS = DEFAULT_SCENE_NORMAL_RADIUS
        self.SCENE_NORMAL_MAX_NN = DEFAULT_SCENE_NORMAL_MAX_NN
        
        # Object Preprocessing Configuration
        self.OBJECT_DOWNSAMPLE_VOXEL = args.object_downsample
        self.OBJECT_NORMAL_RADIUS = DEFAULT_OBJECT_NORMAL_RADIUS
        self.OBJECT_NORMAL_MAX_NN = DEFAULT_OBJECT_NORMAL_MAX_NN
        
        # Scale Estimation Configuration
        self.SCALE_METHOD = args.scale_method
        self.ESTIMATE_SCALE = not args.no_scale
        
        # RANSAC Configuration
        self.RANSAC_TRIES = args.ransac_tries
        self.RANSAC_DOWNSAMPLE = args.ransac_downsample
        self.RANSAC_MAX_CORRESPONDENCE_DISTANCE = args.ransac_max_dist
        self.RANSAC_CORRESPONDENCE_CHECKER_DISTANCE = DEFAULT_RANSAC_CORRESPONDENCE_CHECKER_DISTANCE
        self.RANSAC_NORMAL_RADIUS = DEFAULT_RANSAC_NORMAL_RADIUS
        self.RANSAC_NORMAL_MAX_NN = DEFAULT_RANSAC_NORMAL_MAX_NN
        self.RANSAC_FPFH_RADIUS = DEFAULT_RANSAC_FPFH_RADIUS
        self.RANSAC_FPFH_MAX_NN = DEFAULT_RANSAC_FPFH_MAX_NN
        
        # ICP Configuration
        self.ICP_MAX_CORRESPONDENCE_DISTANCE = args.local_icp_dist
        self.ICP_MAX_ITERATIONS = args.local_icp_iters
        
        # Adaptive Refinement Configuration
        self.ADAPTIVE_MAX_ITERATIONS = args.adaptive_iters
        self.ADAPTIVE_FITNESS_THRESHOLD = args.adaptive_fitness_threshold
        self.ADAPTIVE_RMSE_THRESHOLD = args.adaptive_rmse_threshold
        self.ADAPTIVE_NOISE_ROTATION_RANGE = args.adaptive_rotation_noise
        self.ADAPTIVE_NOISE_TRANSLATION_START = args.adaptive_translation_noise
        self.ADAPTIVE_ICP_DISTANCE_MULTIPLIER = DEFAULT_ADAPTIVE_ICP_DISTANCE_MULTIPLIER
        
        # Scale Refinement Configuration
        self.SCALE_SEARCH_MIN_FACTOR = DEFAULT_SCALE_SEARCH_MIN_FACTOR
        self.SCALE_SEARCH_MAX_FACTOR = DEFAULT_SCALE_SEARCH_MAX_FACTOR
        self.SCALE_REFINEMENT_MAX_ITERATIONS = DEFAULT_SCALE_REFINEMENT_MAX_ITERATIONS
        self.SCALE_CONVERGENCE_THRESHOLD = DEFAULT_SCALE_CONVERGENCE_THRESHOLD
        self.SCALE_ICP_MAX_ITERATIONS = DEFAULT_SCALE_ICP_MAX_ITERATIONS
        
        # Visualization Configuration
        self.VISUALIZE_RECONSTRUCTION = args.visualize_reconstruction
        self.VISUALIZE_PREPROCESSING = args.visualize_preprocessing
        self.VISUALIZE_STEPS = args.visualize_steps
        self.VISUALIZE_FINAL = args.visualize_final
        
        # Debug Configuration
        self.DEBUG = args.debug


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler()]
    )


def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logging.info(f"⏱️  {func.__name__}: {elapsed:.3f}s")
        return result
    return wrapper


def visualize_pcd(pcd: o3d.geometry.PointCloud, title: str = "Point Cloud"):
    logging.info(f"🔍 Visualizing: {title} ({len(pcd.points)} points)")
    o3d.visualization.draw_geometries([pcd], window_name=title, width=1280, height=720)


# ============================================================================
# FAST3R RECONSTRUCTION
# ============================================================================

@timeit
def run_fast3r_reconstruction(scene_dir: Path, conf_threshold: float, seed: int,
                               image_size: int = 512, model_name: str = DEFAULT_MODEL_NAME) -> Path:
    """
    Run Fast3R neural reconstruction to convert images into a 3D point cloud.
    """
    
    logging.info("\n" + "="*70)
    logging.info("  FAST3R RECONSTRUCTION")
    logging.info("="*70)
    
    # Set random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Setup device and precision
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}, dtype: {dtype}")
    
    # Load Fast3R model
    logging.info(f"Loading Fast3R model: {model_name}")
    model = Fast3R.from_pretrained(model_name)
    model.eval().to(device)
    
    # Create lightning module wrapper
    lit_module = MultiViewDUSt3RLitModule.load_for_inference(model)
    lit_module.eval()
    
    # Load images
    image_dir = scene_dir / "images"
    image_paths = sorted(glob.glob(str(image_dir / "*")))
    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")
    
    logging.info(f"Loading {len(image_paths)} images...")
    images = load_images(image_paths, size=image_size, verbose=True)
    
    # Run Fast3R inference
    logging.info("Running Fast3R inference...")
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=dtype):
            output_dict, profiling_info = inference(
                images,
                model,
                device,
                dtype=dtype,
                verbose=True,
                profiling=True,
            )
    
    logging.info(f"Inference completed. Profiling: {profiling_info}")
    
    # Extract 3D points from predictions
    logging.info("Extracting 3D points from predictions...")
    all_points = []
    all_colors = []
    
    for view_idx, pred in enumerate(output_dict['preds']):
        # Get pointmap
        if 'pts3d_in_other_view' in pred:
            pts3d = pred['pts3d_in_other_view']
        elif 'pts3d' in pred:
            pts3d = pred['pts3d']
        else:
            logging.warning(f"View {view_idx}: No pointmap found, keys: {pred.keys()}")
            continue
        
        if isinstance(pts3d, torch.Tensor):
            pts3d = pts3d.cpu().numpy()
        if pts3d.ndim == 4:
            pts3d = pts3d[0]
        
        # Get confidence
        conf = None
        if 'conf' in pred:
            conf = pred['conf']
            if isinstance(conf, torch.Tensor):
                conf = conf.cpu().numpy()
            if conf.ndim == 3:
                conf = conf[0]
        
        # Get colors from original image
        img_data = images[view_idx]
        if isinstance(img_data, dict) and 'img' in img_data:
            img = img_data['img']
        else:
            img = img_data
        
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        if img.ndim == 4:
            img = img[0]
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        
        # Resize image to match pointmap
        H, W, _ = pts3d.shape
        if img.shape[:2] != (H, W):
            import cv2
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
        
        # Flatten
        pts_flat = pts3d.reshape(-1, 3)
        colors_flat = img.reshape(-1, 3)
        
        # Filter by confidence
        if conf is not None:
            conf_flat = conf.reshape(-1)
            if conf_flat.max() > 1.0:
                conf_flat = conf_flat / conf_flat.max()
            mask = conf_flat >= conf_threshold
            pts_flat = pts_flat[mask]
            colors_flat = colors_flat[mask]
        
        # Filter invalid points
        valid_mask = np.isfinite(pts_flat).all(axis=1) & (np.abs(pts_flat) < 100).all(axis=1)
        pts_flat = pts_flat[valid_mask]
        colors_flat = colors_flat[valid_mask]
        
        all_points.append(pts_flat)
        all_colors.append(colors_flat)
        logging.info(f"  View {view_idx}: {len(pts_flat)} valid points")
    
    if not all_points:
        raise ValueError("No valid 3D points extracted!")
    
    points_3d = np.concatenate(all_points, axis=0)
    points_rgb = np.concatenate(all_colors, axis=0)
    
    logging.info(f"Total filtered points: {len(points_3d)} (conf >= {conf_threshold})")
    
    # Optionally save camera poses
    try:
        logging.info("Estimating camera poses...")
        poses_c2w_batch, estimated_focals = MultiViewDUSt3RLitModule.estimate_camera_poses(
            output_dict['preds'],
            niter_PnP=100,
            focal_length_estimation_method='first_view_from_global_head'
        )
        camera_poses = poses_c2w_batch[0]
        
        sparse_dir = scene_dir / "fast3r_sparse"
        sparse_dir.mkdir(exist_ok=True)
        
        poses_dict = {
            'camera_poses': [pose.tolist() for pose in camera_poses],
            'estimated_focals': estimated_focals.tolist() if hasattr(estimated_focals, 'tolist') else list(estimated_focals),
            'image_paths': [os.path.basename(p) for p in image_paths]
        }
        with open(sparse_dir / "camera_poses.json", 'w') as f:
            json.dump(poses_dict, f, indent=2)
        logging.info(f"✓ Saved camera poses")
    except Exception as e:
        logging.warning(f"Could not estimate camera poses: {e}")
    
    # Save point cloud
    sparse_dir = scene_dir / "fast3r_sparse"
    sparse_dir.mkdir(exist_ok=True)
    scene_ply = sparse_dir / "points.ply"
    
    trimesh.PointCloud(points_3d, colors=points_rgb).export(str(scene_ply))
    logging.info(f"✓ Saved: {scene_ply}")
    
    return scene_ply


# ============================================================================
# PREPROCESSING
# ============================================================================

@timeit
def preprocess_scene(pcd_path: Path, config: PipelineConfig, save_path: Optional[Path] = None):
    logging.info("\n" + "="*70)
    logging.info("  SCENE PREPROCESSING")
    logging.info("="*70)
    
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    original_count = len(pcd.points)
    logging.info(f"Loaded: {original_count} points")
    
    if config.VISUALIZE_PREPROCESSING:
        visualize_pcd(pcd, "Original Scene")
    
    # Remove outliers
    logging.info("Removing outliers...")
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=config.SCENE_OUTLIER_NEIGHBORS,
        std_ratio=config.SCENE_OUTLIER_STD
    )
    logging.info(f"  {original_count} → {len(pcd.points)} points")
    
    if config.VISUALIZE_PREPROCESSING:
        visualize_pcd(pcd, "After Outlier Removal")
    
    # Remove plane
    if config.REMOVE_PLANE:
        logging.info("Removing plane...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=config.SCENE_NORMAL_RADIUS,
                max_nn=config.SCENE_NORMAL_MAX_NN
            )
        )
        
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=config.PLANE_DISTANCE_THRESHOLD,
            ransac_n=3,
            num_iterations=1000
        )
        
        a, b, c, d = plane_model
        logging.info(f"  Plane: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")
        
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
        logging.info(f"  → {len(pcd.points)} points remaining")
        
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=config.SCENE_NORMAL_RADIUS,
                max_nn=config.SCENE_NORMAL_MAX_NN
            )
        )
        
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
    logging.info("\n" + "="*70)
    logging.info("  OBJECT PREPROCESSING")
    logging.info("="*70)
    
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    original_count = len(pcd.points)
    logging.info(f"Loaded: {original_count} points")
    
    logging.info(f"Downsampling (voxel={config.OBJECT_DOWNSAMPLE_VOXEL})...")
    pcd = pcd.voxel_down_sample(voxel_size=config.OBJECT_DOWNSAMPLE_VOXEL)
    logging.info(f"  → {len(pcd.points)} points")
    
    logging.info("Estimating normals...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=config.OBJECT_NORMAL_RADIUS,
            max_nn=config.OBJECT_NORMAL_MAX_NN
        )
    )
    
    logging.info(f"✓ Complete: {original_count} → {len(pcd.points)} points")
    return pcd


# ============================================================================
# SCALE ESTIMATION
# ============================================================================

def estimate_scale_bbox(source, target):
    source_bbox = source.get_axis_aligned_bounding_box()
    target_bbox = target.get_axis_aligned_bounding_box()
    
    source_diag = np.linalg.norm(source_bbox.get_extent())
    target_diag = np.linalg.norm(target_bbox.get_extent())
    
    scale = target_diag / source_diag
    logging.info(f"  Source bbox diagonal: {source_diag:.6f}")
    logging.info(f"  Target bbox diagonal: {target_diag:.6f}")
    logging.info(f"  BBox ratio: {scale:.6f}")
    return scale


@timeit
def estimate_scale(source, target, config):
    logging.info("\n" + "="*70)
    logging.info("  SCALE ESTIMATION")
    logging.info("="*70)
    
    if config.SCALE_METHOD == "bbox":
        scale = estimate_scale_bbox(source, target)
    else:
        scale = estimate_scale_bbox(source, target)
    
    source_scaled = copy.deepcopy(source)
    source_scaled.scale(scale, center=source_scaled.get_center())
    translation = target.get_center() - source_scaled.get_center()
    init_transform = np.eye(4)
    init_transform[:3, 3] = translation
    
    logging.info(f"✓ Initial scale: {scale:.6f}")
    return scale, init_transform


# ============================================================================
# RANSAC INITIAL ALIGNMENT
# ============================================================================

@timeit
def initial_alignment_ransac(source, target, config):
    logging.info("\n" + "="*70)
    logging.info(f"  RANSAC INITIAL ALIGNMENT ({config.RANSAC_TRIES} attempts)")
    logging.info("="*70)
    
    source_down = source
    target_down = target.voxel_down_sample(config.RANSAC_DOWNSAMPLE)
    logging.info(f"Downsampled: source={len(source_down.points)}, target={len(target_down.points)}")
    
    logging.info("Estimating normals...")
    source_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=config.RANSAC_NORMAL_RADIUS, max_nn=config.RANSAC_NORMAL_MAX_NN
        )
    )
    target_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=config.RANSAC_NORMAL_RADIUS, max_nn=config.RANSAC_NORMAL_MAX_NN
        )
    )
    
    logging.info("Computing FPFH features...")
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=config.RANSAC_FPFH_RADIUS, max_nn=config.RANSAC_FPFH_MAX_NN)
    )
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=config.RANSAC_FPFH_RADIUS, max_nn=config.RANSAC_FPFH_MAX_NN)
    )
    
    logging.info("Running RANSAC...")
    all_results = []
    best_result = None
    best_fitness = -1
    
    for i in range(config.RANSAC_TRIES):
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh,
            mutual_filter=True,
            max_correspondence_distance=config.RANSAC_MAX_CORRESPONDENCE_DISTANCE * 0.1,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=3,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(config.RANSAC_CORRESPONDENCE_CHECKER_DISTANCE)
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
        )
        
        all_results.append({'attempt': i + 1, 'fitness': result.fitness, 'rmse': result.inlier_rmse})
        logging.info(f"  Try {i+1}/{config.RANSAC_TRIES}: fitness={result.fitness:.4f}, RMSE={result.inlier_rmse:.6f}")
        
        if result.fitness > best_fitness:
            best_fitness = result.fitness
            best_result = result
            logging.info(f"    ✅ NEW BEST!")
    
    logging.info(f"✓ Best: fitness={best_fitness:.4f}, RMSE={best_result.inlier_rmse:.6f}")
    return best_result, all_results


# ============================================================================
# ADAPTIVE REFINEMENT
# ============================================================================

@timeit
def adaptive_refinement(source, target, initial_result, config):
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
        
        noise_rotation = o3d.geometry.get_rotation_matrix_from_xyz(
            [np.random.uniform(-config.ADAPTIVE_NOISE_ROTATION_RANGE, config.ADAPTIVE_NOISE_ROTATION_RANGE) for _ in range(3)]
        )
        noise_trans_vec = np.random.uniform(-noise_translation, noise_translation, 3)
        
        noise_transform = np.eye(4)
        noise_transform[:3, :3] = noise_rotation
        noise_transform[:3, 3] = noise_trans_vec
        
        current_transform = noise_transform @ best_transformation
        
        try:
            result = o3d.pipelines.registration.registration_icp(
                source, target,
                config.ICP_MAX_CORRESPONDENCE_DISTANCE * config.ADAPTIVE_ICP_DISTANCE_MULTIPLIER,
                current_transform,
                o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )
            
            if result.fitness > 0 and result.inlier_rmse > 0:
                if result.fitness > best_fitness or (result.fitness == best_fitness and result.inlier_rmse < best_rmse):
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
    
    logging.info(f"✓ Complete ({iteration} iterations): fitness={best_fitness:.4f}, RMSE={best_rmse:.6f}")
    return final_result


@timeit
def refine_scale_by_fitness(object_pcd, scene_pcd, initial_scale, initial_result, config):
    logging.info("\n" + "="*70)
    logging.info("  ADAPTIVE SCALE REFINEMENT")
    logging.info("="*70)
    
    best_rmse = initial_result.inlier_rmse
    best_scale = initial_scale
    best_result = initial_result
    all_tested = []
    
    scale_min = initial_scale * config.SCALE_SEARCH_MIN_FACTOR
    scale_max = initial_scale * config.SCALE_SEARCH_MAX_FACTOR
    iteration = 0
    
    logging.info(f"Initial scale: {initial_scale:.6f}, RMSE: {best_rmse:.6f}")
    logging.info(f"Search range: [{scale_min:.6f}, {scale_max:.6f}]")
    
    while iteration < config.SCALE_REFINEMENT_MAX_ITERATIONS:
        iteration += 1
        scale_range = scale_max - scale_min
        
        scales_to_test = [scale_min, (scale_min + scale_max) / 2, scale_max]
        logging.info(f"Iteration {iteration}: range [{scale_min:.6f}, {scale_max:.6f}]")
        
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
                    config.ICP_MAX_CORRESPONDENCE_DISTANCE,
                    initial_result.transformation,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=config.SCALE_ICP_MAX_ITERATIONS)
                )
                
                rmse = result.inlier_rmse
                best_marker = "✅ BEST" if rmse < best_rmse else ""
                logging.info(f"  {label:3s} ({scale:.6f}): fitness={result.fitness:.4f}, rmse={rmse:.6f} {best_marker}")
                
                results.append({'scale': scale, 'rmse': rmse, 'fitness': result.fitness, 'result': result})
                all_tested.append({'iteration': iteration, 'scale': float(scale), 'rmse': float(rmse), 'fitness': float(result.fitness)})
                
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_scale = scale
                    best_result = result
            except Exception as e:
                logging.warning(f"  {label:3s} ({scale:.6f}): FAILED - {e}")
                results.append({'scale': scale, 'rmse': float('inf'), 'fitness': 0.0, 'result': None})
        
        min_r, mid_r, max_r = results[0], results[1], results[2]
        
        if scale_range < initial_scale * config.SCALE_CONVERGENCE_THRESHOLD:
            logging.info("  ✓ Converged!")
            break
        
        if mid_r['rmse'] <= min_r['rmse'] and mid_r['rmse'] <= max_r['rmse']:
            scale_min = (scale_min + mid_r['scale']) / 2
            scale_max = (scale_max + mid_r['scale']) / 2
        elif min_r['rmse'] <= max_r['rmse']:
            scale_max = mid_r['scale']
            scale_min = min_r['scale'] * 0.9
        else:
            scale_min = mid_r['scale']
            scale_max = max_r['scale'] * 1.1
    
    scale_change_pct = 100 * (best_scale - initial_scale) / initial_scale
    logging.info(f"✓ Best scale: {best_scale:.6f} ({scale_change_pct:+.1f}% from initial)")
    
    return best_scale, best_result, all_tested


# ============================================================================
# MAIN PIPELINE
# ============================================================================

@timeit
def run_pipeline(args):
    start_time = time.perf_counter()
    
    scene_dir = Path(args.scene_dir)
    output_dir = scene_dir / "fast3r_icp_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = PipelineConfig(args)
    
    logging.info("\n" + "="*70)
    logging.info("  COMPLETE PIPELINE")
    logging.info("="*70)
    logging.info(f"Scene: {scene_dir}")
    logging.info(f"Object: {args.object_ply}")
    logging.info(f"Output: {output_dir}")
    
    # STAGE 1: RECONSTRUCTION
    scene_ply_path = scene_dir / "fast3r_sparse" / "points.ply"
    
    if args.skip_reconstruction and scene_ply_path.exists():
        logging.info(f"\n✓ Using existing reconstruction: {scene_ply_path}")
    else:
        scene_ply_path = run_fast3r_reconstruction(
            scene_dir=scene_dir,
            conf_threshold=args.conf_thres_value,
            seed=args.seed,
            image_size=args.image_size,
            model_name=args.model_name
        )
    
    if config.VISUALIZE_RECONSTRUCTION:
        pcd_raw = o3d.io.read_point_cloud(str(scene_ply_path))
        visualize_pcd(pcd_raw, "Reconstructed Scene (Raw)")
    
    # STAGE 2: PREPROCESSING
    scene_pcd = preprocess_scene(scene_ply_path, config, output_dir / "scene_preprocessed.ply")
    object_pcd = preprocess_object(Path(args.object_ply), config)
    
    # STAGE 3: SCALE ESTIMATION
    if config.ESTIMATE_SCALE:
        scale, scale_transform = estimate_scale(object_pcd, scene_pcd, config)
        object_pcd_scaled = copy.deepcopy(object_pcd)
        object_pcd_scaled.scale(scale, center=object_pcd_scaled.get_center())
        object_pcd_scaled.transform(scale_transform)
    else:
        scale = 1.0
        scale_transform = np.eye(4)
        object_pcd_scaled = object_pcd
    
    # STAGE 4: RANSAC INITIAL ALIGNMENT
    local_result, all_attempts = initial_alignment_ransac(object_pcd_scaled, scene_pcd, config)
    
    logging.info("\n🔧 Refining RANSAC with ICP...")
    local_result = o3d.pipelines.registration.registration_icp(
        object_pcd_scaled, scene_pcd,
        config.ICP_MAX_CORRESPONDENCE_DISTANCE,
        local_result.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=config.ICP_MAX_ITERATIONS)
    )
    logging.info(f"  After ICP: fitness={local_result.fitness:.4f}, RMSE={local_result.inlier_rmse:.6f}")
    
    # STAGE 5: ADAPTIVE REFINEMENT
    final_result = adaptive_refinement(object_pcd_scaled, scene_pcd, local_result, config)
    
    # STAGE 6: SCALE REFINEMENT
    refined_scale, refined_result, scale_history = refine_scale_by_fitness(object_pcd, scene_pcd, scale, final_result, config)
    
    if refined_result.inlier_rmse < final_result.inlier_rmse:
        logging.info(f"✅ Using refined scale: {refined_scale:.6f}")
        scale = refined_scale
        final_result = refined_result
    else:
        logging.info(f"⚠️  Keeping original scale: {scale:.6f}")
    
    # FINALIZE
    final_transformation = np.dot(final_result.transformation, scale_transform)
    
    object_aligned = copy.deepcopy(object_pcd)
    object_aligned.scale(scale, center=object_aligned.get_center())
    object_aligned.transform(final_transformation)
    
    if config.VISUALIZE_FINAL:
        target_vis = copy.deepcopy(scene_pcd).paint_uniform_color([1, 0, 0])
        aligned_vis = copy.deepcopy(object_aligned).paint_uniform_color([0, 1, 0])
        logging.info("\n🎬 Final Visualization (Red=Scene, Green=Object)")
        o3d.visualization.draw_geometries([target_vis, aligned_vis], window_name="Final Alignment", width=1280, height=720)
    
    # Save results
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
        'scale_refinement_history': scale_history,
        'elapsed_time': time.perf_counter() - start_time
    }
    
    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logging.info(f"  ✓ transformation.npy")
    logging.info(f"  ✓ scale.npy")
    logging.info(f"  ✓ object_aligned.ply")
    logging.info(f"  ✓ metrics.json")
    
    elapsed = time.perf_counter() - start_time
    logging.info("\n" + "="*70)
    logging.info("  PIPELINE COMPLETE")
    logging.info("="*70)
    logging.info(f"⏱️  Total time: {elapsed:.2f}s")
    logging.info(f"📏 Final scale: {scale:.6f}")
    logging.info(f"📊 Final fitness: {final_result.fitness:.4f}")
    logging.info(f"📊 Final RMSE: {final_result.inlier_rmse:.6f}")
    
    if final_result.fitness >= 0.85:
        logging.info("✅ EXCELLENT alignment!")
    elif final_result.fitness >= 0.6:
        logging.info("✓  GOOD alignment")
    else:
        logging.info("⚠️  MODERATE alignment - may need parameter tuning")
    
    return metrics


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    args = parse_args()
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(level=log_level)
    
    with torch.no_grad():
        metrics = run_pipeline(args)
    
    print(f"\n✅ Pipeline complete!")
    print(f"   Final fitness: {metrics['final_fitness']:.4f}")
    print(f"   Final RMSE: {metrics['final_rmse']:.6f}")
    print(f"   Final scale: {metrics['scale']:.6f}")