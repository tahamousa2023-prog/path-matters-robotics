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
python /home/AP_PathMatters/vggt/z_demo_reconstruction_icp_21_12.py \
    --scene_dir /home/AP_PathMatters/path_matters/datasets/yoda \
    --object_ply /home/AP_PathMatters/path_matters/datasets/yoda/Baby_Yoda.ply\
    --visualize_reconstruction \
    --visualize_preprocessing \
    --visualize_steps \
    --debug

python /home/AP_PathMatters/vggt/z_demo_reconstruction_icp_21_12.py \
    --scene_dir /home/AP_PathMatters/path_matters/datasets/Reallife_Dataset_Haroun_Aziz/scenes-others_SUBSAMPLED/air_conditioner_control_camera3 \
    --object_ply /home/AP_PathMatters/path_matters/datasets/Reallife_Dataset_Haroun_Aziz/scenes-others_SUBSAMPLED/air_conditioner_control_camera3/textured.ply \
    --visualize_reconstruction \
    --visualize_preprocessing \
    --visualize_steps \
    --debug \
    --auto

### best results, auto-tuning, sometimes need several tries:
python /home/AP_PathMatters/vggt/z_demo_reconstruction_icp_21_12.py \
    --scene_dir /home/AP_PathMatters/path_matters/datasets/gso/adizero \
    --object_ply /home/AP_PathMatters/path_matters/datasets/gso/adizero/model.ply \
    --no_plane_removal \
    --auto \
    --visualize_final

### Last
python /home/AP_PathMatters/vggt/h_demo_reconstruction_icp_05_01.py \
  --scene_dir /home/AP_PathMatters/path_matters/datasets/Reallife_Dataset_Haroun_Aziz/scenes-others_SUBSAMPLED/baofengmojing_conctrol_camera3 \
  --object_ply /home/AP_PathMatters/path_matters/datasets/Reallife_Dataset_Haroun_Aziz/scenes-others_SUBSAMPLED/baofengmojing_conctrol_camera3/textured.ply \
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


# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
VGGT Reconstruction + ICP Alignment Pipeline with Adaptive Scale Refinement

This pipeline reconstructs a 3D scene from images using VGGT, then aligns a known
3D object to the reconstructed scene using ICP (Iterative Closest Point) with
adaptive scale refinement.

Main stages:
1. VGGT Reconstruction - Convert images to 3D point cloud
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
# DEFAULT PARAMETER VALUES
# ============================================================================
# All pipeline parameters are defined here for easy tuning.
# Modify these values to change pipeline behavior.
# ============================================================================

# ----------------------------------------------------------------------------
# VGGT RECONSTRUCTION PARAMETERS
# ----------------------------------------------------------------------------
DEFAULT_SEED = 42  # Random seed for reproducibility
DEFAULT_CONFIDENCE_THRESHOLD = 3  # Minimum confidence for keeping 3D points (higher = fewer points but higher quality)

# ----------------------------------------------------------------------------
# SCENE PREPROCESSING PARAMETERS
# ----------------------------------------------------------------------------
# These parameters control how the reconstructed scene is cleaned up

DEFAULT_SCENE_DOWNSAMPLE_VOXEL = 0.001  # Voxel size for downsampling scene (smaller = more detail but slower)
DEFAULT_SCENE_OUTLIER_NEIGHBORS = 50  # Number of neighbors to check for outlier removal (higher = stricter)
DEFAULT_SCENE_OUTLIER_STD_RATIO = 5.0  # Standard deviation threshold for outliers (lower = more aggressive removal)
DEFAULT_PLANE_DISTANCE_THRESHOLD = 0.015  # Max distance from plane to be considered part of it (meters)
DEFAULT_PLANE_OFFSET = -0.015  # How much to offset plane removal (negative = remove more)
DEFAULT_SCENE_NORMAL_RADIUS = 1.0  # Search radius for normal estimation in scene
DEFAULT_SCENE_NORMAL_MAX_NN = 30  # Max neighbors for normadl estimation in scene

# ----------------------------------------------------------------------------
# OBJECT PREPROCESSING PARAMETERS
# ----------------------------------------------------------------------------
# These parameters control how the reference object is cleaned up

DEFAULT_OBJECT_DOWNSAMPLE_VOXEL = 0.01  # Voxel size for downsampling object (smaller = more detail)
DEFAULT_OBJECT_NORMAL_RADIUS = 1.0  # Search radius for normal estimation in object
DEFAULT_OBJECT_NORMAL_MAX_NN = 30  # Max neighbors for normal estimation in object

# ----------------------------------------------------------------------------
# SCALE ESTIMATION PARAMETERS
# ----------------------------------------------------------------------------
DEFAULT_SCALE_METHOD = "bbox"  # Method for initial scale estimation: "bbox" or "multi_scale"

# ----------------------------------------------------------------------------
# RANSAC ALIGNMENT PARAMETERS
# ----------------------------------------------------------------------------
# RANSAC finds initial rough alignment between object and scene

DEFAULT_RANSAC_TRIES = 20  # Number of RANSAC attempts (more = better chance of good alignment)
DEFAULT_RANSAC_DOWNSAMPLE_VOXEL = 0.01  # Voxel size for downsampling before RANSAC (larger = faster but less accurate)
DEFAULT_RANSAC_MAX_CORRESPONDENCE_DISTANCE = 0.1  # Max distance between matched points in RANSAC (meters)
DEFAULT_RANSAC_CORRESPONDENCE_CHECKER_DISTANCE = 0.1  # Distance threshold for correspondence validation (meters)
DEFAULT_RANSAC_NORMAL_RADIUS = 5.0  # Search radius for computing normals for RANSAC
DEFAULT_RANSAC_NORMAL_MAX_NN = 100  # Max neighbors for computing normals for RANSAC
DEFAULT_RANSAC_FPFH_RADIUS = 5.0  # Search radius for computing FPFH features (larger = more context)
DEFAULT_RANSAC_FPFH_MAX_NN = 75  # Max neighbors for computing FPFH features

# ----------------------------------------------------------------------------
# ICP REFINEMENT PARAMETERS
# ----------------------------------------------------------------------------
# ICP refines the alignment found by RANSAC

DEFAULT_ICP_MAX_CORRESPONDENCE_DISTANCE = 0.025  # this is used at different places in the code , maz need to fine tune each one bz itself, Max distance for point matching in ICP (CRITICAL: affects fitness and RMSE)
DEFAULT_ICP_MAX_ITERATIONS = 500  # Maximum ICP iterations
DEFAULT_ADAPTIVE_ICP_DISTANCE_MULTIPLIER = 0.5  # Multiplier for ICP distance in adaptive refinement

# ----------------------------------------------------------------------------
# ADAPTIVE REFINEMENT PARAMETERS
# ----------------------------------------------------------------------------
# Adaptive refinement tries small random perturbations to escape local minima

DEFAULT_ADAPTIVE_MAX_ITERATIONS = 50  # Max number of adaptive refinement attempts
DEFAULT_ADAPTIVE_FITNESS_THRESHOLD = 0.95  # Target fitness to reach (0-1, higher = better alignment)
DEFAULT_ADAPTIVE_RMSE_THRESHOLD = 0.005  # Target RMSE to reach (meters, lower = better)
DEFAULT_ADAPTIVE_ROTATION_NOISE_RANGE = 0.01  # Random rotation noise range (radians)
DEFAULT_ADAPTIVE_TRANSLATION_NOISE_START = 0.1  # Initial translation noise (meters)

# ----------------------------------------------------------------------------
# SCALE REFINEMENT PARAMETERS
# ----------------------------------------------------------------------------
# Scale refinement searches for the best object scale

DEFAULT_SCALE_SEARCH_MIN_FACTOR = 0.85  # Minimum scale to test (0.5 = 50% of initial scale)
DEFAULT_SCALE_SEARCH_MAX_FACTOR = 1.15  # Maximum scale to test (1.5 = 150% of initial scale)
DEFAULT_SCALE_REFINEMENT_MAX_ITERATIONS = 6  # Max iterations for scale search
DEFAULT_SCALE_CONVERGENCE_THRESHOLD = 0.05  # Stop when search range is within ±5% of initial scale
DEFAULT_SCALE_ICP_MAX_ITERATIONS = 200  # ICP iterations when testing each scale


# ============================================================================
# ARGUMENT PARSER
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="VGGT Reconstruction + ICP Alignment Pipeline with Scale Refinement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with visualization
  python %(prog)s --scene_dir /path/to/scene --object_ply /path/to/object.ply --visualize_final
  
  # Skip reconstruction if already done
  python %(prog)s --scene_dir /path/to/scene --object_ply /path/to/object.ply --skip_reconstruction
  
  # Full debug mode with all visualizations
  python %(prog)s --scene_dir /path/to/scene --object_ply /path/to/object.ply \\
                  --visualize_reconstruction --visualize_preprocessing --visualize_steps --debug
        """
    )
    
    # ==== REQUIRED ====
    parser.add_argument("--scene_dir", type=str, required=True,
                       help="Directory containing scene images (must have images/ subfolder)")
    parser.add_argument("--object_ply", type=str, required=True,
                       help="Path to reference object PLY file for alignment")
    
    # ==== VGGT RECONSTRUCTION ====
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                       help=f"Random seed for reproducibility (default: {DEFAULT_SEED})")
    parser.add_argument("--conf_thres_value", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD,
                       help=f"Confidence threshold for 3D point filtering (default: {DEFAULT_CONFIDENCE_THRESHOLD})")
    parser.add_argument("--skip_reconstruction", action="store_true",
                       help="Skip reconstruction if sparse/points.ply already exists")
    
    # ==== SCENE PREPROCESSING ====
    parser.add_argument("--scene_downsample", type=float, default=DEFAULT_SCENE_DOWNSAMPLE_VOXEL,
                       help=f"Voxel size for scene downsampling in meters (default: {DEFAULT_SCENE_DOWNSAMPLE_VOXEL})")
    parser.add_argument("--scene_outlier_neighbors", type=int, default=DEFAULT_SCENE_OUTLIER_NEIGHBORS,
                       help=f"Number of neighbors for outlier removal (default: {DEFAULT_SCENE_OUTLIER_NEIGHBORS})")
    parser.add_argument("--scene_outlier_std", type=float, default=DEFAULT_SCENE_OUTLIER_STD_RATIO,
                       help=f"Std ratio for outlier removal (default: {DEFAULT_SCENE_OUTLIER_STD_RATIO})")
    parser.add_argument("--plane_threshold", type=float, default=DEFAULT_PLANE_DISTANCE_THRESHOLD,
                       help=f"Distance threshold for plane detection in meters (default: {DEFAULT_PLANE_DISTANCE_THRESHOLD})")
    parser.add_argument("--plane_offset", type=float, default=DEFAULT_PLANE_OFFSET,
                       help=f"Plane removal offset in meters (default: {DEFAULT_PLANE_OFFSET})")
    parser.add_argument("--no_plane_removal", action="store_true",
                       help="Disable automatic plane removal from scene")
    
    # ==== OBJECT PREPROCESSING ====
    parser.add_argument("--object_downsample", type=float, default=DEFAULT_OBJECT_DOWNSAMPLE_VOXEL,
                       help=f"Voxel size for object downsampling in meters (default: {DEFAULT_OBJECT_DOWNSAMPLE_VOXEL})")
    
    # ==== SCALE ESTIMATION ====
    parser.add_argument("--scale_method", type=str, default=DEFAULT_SCALE_METHOD,
                       choices=["bbox", "multi_scale"],
                       help=f"Scale estimation method (default: {DEFAULT_SCALE_METHOD})")
    parser.add_argument("--no_scale", action="store_true",
                       help="Disable automatic scale estimation (use scale=1.0)")
    
    # ==== RANSAC INITIAL ALIGNMENT ====
    parser.add_argument("--ransac_tries", type=int, default=DEFAULT_RANSAC_TRIES,
                       help=f"Number of RANSAC attempts (default: {DEFAULT_RANSAC_TRIES})")
    parser.add_argument("--ransac_downsample", type=float, default=DEFAULT_RANSAC_DOWNSAMPLE_VOXEL,
                       help=f"Voxel size for RANSAC downsampling in meters (default: {DEFAULT_RANSAC_DOWNSAMPLE_VOXEL})")
    parser.add_argument("--ransac_max_dist", type=float, default=DEFAULT_RANSAC_MAX_CORRESPONDENCE_DISTANCE,
                       help=f"Max correspondence distance for RANSAC in meters (default: {DEFAULT_RANSAC_MAX_CORRESPONDENCE_DISTANCE})")
    
    # ==== LOCAL ICP REFINEMENT ====
    parser.add_argument("--local_icp_dist", type=float, default=DEFAULT_ICP_MAX_CORRESPONDENCE_DISTANCE,
                       help=f"Distance threshold for ICP in meters (default: {DEFAULT_ICP_MAX_CORRESPONDENCE_DISTANCE})")
    parser.add_argument("--local_icp_iters", type=int, default=DEFAULT_ICP_MAX_ITERATIONS,
                       help=f"Max iterations for ICP (default: {DEFAULT_ICP_MAX_ITERATIONS})")
    
    # ==== ADAPTIVE REFINEMENT ====
    parser.add_argument("--adaptive_iters", type=int, default=DEFAULT_ADAPTIVE_MAX_ITERATIONS,
                       help=f"Max adaptive refinement iterations (default: {DEFAULT_ADAPTIVE_MAX_ITERATIONS})")
    parser.add_argument("--adaptive_fitness_threshold", type=float, default=DEFAULT_ADAPTIVE_FITNESS_THRESHOLD,
                       help=f"Target fitness threshold 0-1 (default: {DEFAULT_ADAPTIVE_FITNESS_THRESHOLD})")
    parser.add_argument("--adaptive_rmse_threshold", type=float, default=DEFAULT_ADAPTIVE_RMSE_THRESHOLD,
                       help=f"Target RMSE threshold in meters (default: {DEFAULT_ADAPTIVE_RMSE_THRESHOLD})")
    parser.add_argument("--adaptive_rotation_noise", type=float, default=DEFAULT_ADAPTIVE_ROTATION_NOISE_RANGE,
                       help=f"Rotation noise range in radians (default: {DEFAULT_ADAPTIVE_ROTATION_NOISE_RANGE})")
    parser.add_argument("--adaptive_translation_noise", type=float, default=DEFAULT_ADAPTIVE_TRANSLATION_NOISE_START,
                       help=f"Initial translation noise in meters (default: {DEFAULT_ADAPTIVE_TRANSLATION_NOISE_START})")
    
    # ==== VISUALIZATION ====
    parser.add_argument("--visualize_reconstruction", action="store_true",
                       help="Show raw reconstructed point cloud")
    parser.add_argument("--visualize_preprocessing", action="store_true",
                       help="Show each preprocessing step")
    parser.add_argument("--visualize_steps", action="store_true",
                       help="Show intermediate alignment steps")
    parser.add_argument("--visualize_final", action="store_true", default=True,
                       help="Show final alignment result (default: True)")
    parser.add_argument("--no_visualize_final", dest="visualize_final", action="store_false",
                       help="Disable final visualization")
    # ==== AUTO-TUNING ====
    parser.add_argument("--auto", action="store_true",
                       help="Automatically tune parameters based on object/scene size")
    # ==== DEBUG ====
    parser.add_argument("--debug", action="store_true",
                       help="Enable detailed debug logging")
    
    return parser.parse_args()


# ============================================================================
# CONFIGURATION CLASS (FROM ARGS)
# ============================================================================

class PipelineConfig:
    """
    Configuration container for all pipeline parameters.
    Built from command-line arguments with sensible defaults.
    """
    
    def __init__(self, args):
        # --------------------------------------------------------------------
        # Scene Preprocessing Configuration
        # --------------------------------------------------------------------
        self.SCENE_DOWNSAMPLE_VOXEL = args.scene_downsample
        self.SCENE_OUTLIER_NEIGHBORS = args.scene_outlier_neighbors
        self.SCENE_OUTLIER_STD = args.scene_outlier_std
        self.PLANE_DISTANCE_THRESHOLD = args.plane_threshold
        self.PLANE_OFFSET = args.plane_offset
        self.REMOVE_PLANE = not args.no_plane_removal
        self.SCENE_NORMAL_RADIUS = DEFAULT_SCENE_NORMAL_RADIUS
        self.SCENE_NORMAL_MAX_NN = DEFAULT_SCENE_NORMAL_MAX_NN
        
        # --------------------------------------------------------------------
        # Object Preprocessing Configuration
        # --------------------------------------------------------------------
        self.OBJECT_DOWNSAMPLE_VOXEL = args.object_downsample
        self.OBJECT_NORMAL_RADIUS = DEFAULT_OBJECT_NORMAL_RADIUS
        self.OBJECT_NORMAL_MAX_NN = DEFAULT_OBJECT_NORMAL_MAX_NN
        
        # --------------------------------------------------------------------
        # Scale Estimation Configuration
        # --------------------------------------------------------------------
        self.SCALE_METHOD = args.scale_method
        self.ESTIMATE_SCALE = not args.no_scale
        
        # --------------------------------------------------------------------
        # RANSAC Configuration
        # --------------------------------------------------------------------
        self.RANSAC_TRIES = args.ransac_tries
        self.RANSAC_DOWNSAMPLE = args.ransac_downsample
        self.RANSAC_MAX_CORRESPONDENCE_DISTANCE = args.ransac_max_dist
        self.RANSAC_CORRESPONDENCE_CHECKER_DISTANCE = DEFAULT_RANSAC_CORRESPONDENCE_CHECKER_DISTANCE
        self.RANSAC_NORMAL_RADIUS = DEFAULT_RANSAC_NORMAL_RADIUS
        self.RANSAC_NORMAL_MAX_NN = DEFAULT_RANSAC_NORMAL_MAX_NN
        self.RANSAC_FPFH_RADIUS = DEFAULT_RANSAC_FPFH_RADIUS
        self.RANSAC_FPFH_MAX_NN = DEFAULT_RANSAC_FPFH_MAX_NN
        
        # --------------------------------------------------------------------
        # ICP Configuration
        # --------------------------------------------------------------------
        self.ICP_MAX_CORRESPONDENCE_DISTANCE = args.local_icp_dist
        self.ICP_MAX_ITERATIONS = args.local_icp_iters
        
        # --------------------------------------------------------------------
        # Adaptive Refinement Configuration
        # --------------------------------------------------------------------
        self.ADAPTIVE_MAX_ITERATIONS = args.adaptive_iters
        self.ADAPTIVE_FITNESS_THRESHOLD = args.adaptive_fitness_threshold
        self.ADAPTIVE_RMSE_THRESHOLD = args.adaptive_rmse_threshold
        self.ADAPTIVE_NOISE_ROTATION_RANGE = args.adaptive_rotation_noise
        self.ADAPTIVE_NOISE_TRANSLATION_START = args.adaptive_translation_noise
        self.ADAPTIVE_ICP_DISTANCE_MULTIPLIER = DEFAULT_ADAPTIVE_ICP_DISTANCE_MULTIPLIER
        
        # --------------------------------------------------------------------
        # Scale Refinement Configuration
        # --------------------------------------------------------------------
        self.SCALE_SEARCH_MIN_FACTOR = DEFAULT_SCALE_SEARCH_MIN_FACTOR
        self.SCALE_SEARCH_MAX_FACTOR = DEFAULT_SCALE_SEARCH_MAX_FACTOR
        self.SCALE_REFINEMENT_MAX_ITERATIONS = DEFAULT_SCALE_REFINEMENT_MAX_ITERATIONS
        self.SCALE_CONVERGENCE_THRESHOLD = DEFAULT_SCALE_CONVERGENCE_THRESHOLD
        self.SCALE_ICP_MAX_ITERATIONS = DEFAULT_SCALE_ICP_MAX_ITERATIONS
        
        # --------------------------------------------------------------------
        # Visualization Configuration
        # --------------------------------------------------------------------
        self.VISUALIZE_RECONSTRUCTION = args.visualize_reconstruction
        self.VISUALIZE_PREPROCESSING = args.visualize_preprocessing
        self.VISUALIZE_STEPS = args.visualize_steps
        self.VISUALIZE_FINAL = args.visualize_final
        
        # --------------------------------------------------------------------
        # Debug Configuration
        # --------------------------------------------------------------------
        self.DEBUG = args.debug


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compute_adaptive_parameters(object_pcd, scene_pcd, config):
    """
    Automatically compute parameters based on point cloud sizes.
    This makes the pipeline work across different object scales.
    """
    # Get bounding box sizes
    obj_bbox = object_pcd.get_axis_aligned_bounding_box()
    scene_bbox = scene_pcd.get_axis_aligned_bounding_box()
    
    obj_size = np.linalg.norm(obj_bbox.get_extent())
    scene_size = np.linalg.norm(scene_bbox.get_extent())
    
    # Estimate the scale ratio
    scale_ratio = scene_size / obj_size
    
    logging.info("\n" + "="*70)
    logging.info("  AUTO-TUNING PARAMETERS")
    logging.info("="*70)
    logging.info(f"Object size (bbox diagonal): {obj_size:.6f}")
    logging.info(f"Scene size (bbox diagonal): {scene_size:.6f}")
    logging.info(f"Estimated scale ratio: {scale_ratio:.4f}")
    
    # Use the LARGER of object or scene for parameter scaling
    # (after scaling, object should be similar size to scene)
    # 1) Choose ONE voxel size that represents point spacing for registration
    # For small objects (like your remote), use object size, not scene size.
    voxel = max(obj_size * 0.02, 0.002)   # 2% of object bbox diagonal, minimum 2mm

    # 2) Use voxel to set downsampling consistently
    config.RANSAC_DOWNSAMPLE = voxel
    config.SCENE_DOWNSAMPLE_VOXEL = max(0.5 * voxel, 0.001)
    config.OBJECT_DOWNSAMPLE_VOXEL = max(0.5 * voxel, 0.001)

    # 3) Derive RANSAC / FPFH / Normal params from voxel
    config.RANSAC_NORMAL_RADIUS = 2.0 * voxel
    config.RANSAC_FPFH_RADIUS   = 5.0 * voxel
    config.RANSAC_MAX_CORRESPONDENCE_DISTANCE = 1.5 * voxel
    config.RANSAC_CORRESPONDENCE_CHECKER_DISTANCE = 1.5 * voxel
    config.RANSAC_NORMAL_MAX_NN = 50
    config.RANSAC_FPFH_MAX_NN   = 100
    config.RANSAC_TRIES = 80

    # 4) Derive ICP distance from voxel (usually tighter than RANSAC)
    config.ICP_MAX_CORRESPONDENCE_DISTANCE = 0.4 * voxel
    config.ICP_MAX_ITERATIONS = 800

    # 5) Adaptive refinement noise tied to voxel (small nudges)
    config.ADAPTIVE_NOISE_TRANSLATION_START = 2.0 * voxel
    config.ADAPTIVE_NOISE_ROTATION_RANGE = 0.05
    config.ADAPTIVE_MAX_ITERATIONS = 200
    config.ADAPTIVE_ICP_DISTANCE_MULTIPLIER = 1.0

    
    # Adaptive refinement - larger search space
    config.ADAPTIVE_NOISE_TRANSLATION_START = reference_size * 0.1  # 10% of size
    config.ADAPTIVE_NOISE_ROTATION_RANGE = 0.05  # ~3 degrees
    config.ADAPTIVE_MAX_ITERATIONS = 150
    config.ADAPTIVE_ICP_DISTANCE_MULTIPLIER = 1.0  # Don't reduce ICP distance
    
    # Relaxed thresholds for difficult objects
    config.ADAPTIVE_FITNESS_THRESHOLD = 0.70  # Lower target
    config.ADAPTIVE_RMSE_THRESHOLD = reference_size * 0.03  # 3% of size
    
    # Scale search - narrow range since bbox estimate should be close
    config.SCALE_SEARCH_MIN_FACTOR = 0.7  # Only 70% to 130%
    config.SCALE_SEARCH_MAX_FACTOR = 1.3
    
    logging.info(f"\nAuto-computed parameters:")
    logging.info(f"  ICP distance: {config.ICP_MAX_CORRESPONDENCE_DISTANCE:.6f}")
    logging.info(f"  RANSAC distance: {config.RANSAC_MAX_CORRESPONDENCE_DISTANCE:.6f}")
    logging.info(f"  RANSAC tries: {config.RANSAC_TRIES}")
    logging.info(f"  FPFH radius: {config.RANSAC_FPFH_RADIUS:.6f}")
    logging.info(f"  Scene downsample: {config.SCENE_DOWNSAMPLE_VOXEL:.6f}")
    logging.info(f"  Object downsample: {config.OBJECT_DOWNSAMPLE_VOXEL:.6f}")
    logging.info(f"  Translation noise: {config.ADAPTIVE_NOISE_TRANSLATION_START:.6f}")
    logging.info(f"  Rotation noise: {config.ADAPTIVE_NOISE_ROTATION_RANGE:.4f} rad")
    logging.info(f"  Scale search: [{config.SCALE_SEARCH_MIN_FACTOR:.1f}x, {config.SCALE_SEARCH_MAX_FACTOR:.1f}x]")
    logging.info("="*70)
    
    return config

def setup_logging(level=logging.INFO):
    """Setup logging configuration with timestamps and levels"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler()]
    )


def timeit(func):
    """Decorator to measure and log function execution time"""
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logging.info(f"⏱️  {func.__name__}: {elapsed:.3f}s")
        return result
    return wrapper


def visualize_pcd(pcd: o3d.geometry.PointCloud, title: str = "Point Cloud"):
    """
    Visualize a single point cloud in a window.
    
    Args:
        pcd: Open3D point cloud to visualize
        title: Window title
    """
    logging.info(f"🔍 Visualizing: {title} ({len(pcd.points)} points)")
    o3d.visualization.draw_geometries([pcd], window_name=title, width=1280, height=720)


# ============================================================================
# VGGT RECONSTRUCTION
# ============================================================================

@timeit
def run_vggt_reconstruction(scene_dir: Path, conf_threshold: float, seed: int) -> Path:
    """
    Run VGGT neural reconstruction to convert images into a 3D point cloud.
    
    This function:
    1. Loads images from the scene directory
    2. Runs the VGGT model to predict depth and camera poses
    3. Unprojects depth maps to 3D points
    4. Filters points by confidence
    5. Saves both COLMAP reconstruction and PLY point cloud
    
    Args:
        scene_dir: Directory containing images/ subfolder
        conf_threshold: Minimum confidence for keeping points (higher = fewer, better points)
        seed: Random seed for reproducibility
        
    Returns:
        Path to the saved point cloud PLY file
    """
    
    logging.info("\n" + "="*70)
    logging.info("  VGGT RECONSTRUCTION")
    logging.info("="*70)
    
    # Set random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Setup device and precision
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Device: {device}, dtype: {dtype}")
    
    # Load VGGT model
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
    
    # Run VGGT inference
    logging.info("Running VGGT inference...")
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images_batch = F.interpolate(images, size=(518, 518), mode="bilinear", align_corners=False)
            images_batch = images_batch[None]
            
            aggregated_tokens_list, ps_idx = model.aggregator(images_batch)
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images_batch.shape[-2:])
            depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images_batch, ps_idx)
    
    # Move to CPU and convert to numpy
    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    depth_map = depth_map.squeeze(0).cpu().numpy()
    depth_conf = depth_conf.squeeze(0).cpu().numpy()
    
    # Unproject depth maps to 3D points
    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
    
    # Filter by confidence and get colors
    num_frames, height, width, _ = points_3d.shape
    points_rgb = F.interpolate(images, size=(518, 518), mode="bilinear", align_corners=False)
    points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)
    
    points_xyf = create_pixel_coordinate_grid(num_frames, height, width)
    conf_mask = depth_conf >= conf_threshold
    conf_mask = randomly_limit_trues(conf_mask, 100000)  # Limit to 100k points
    
    points_3d = points_3d[conf_mask]
    points_xyf = points_xyf[conf_mask]
    points_rgb = points_rgb[conf_mask]
    
    logging.info(f"Filtered points: {len(points_3d)} (conf >= {conf_threshold})")
    
    # Convert to COLMAP format
    logging.info("Converting to COLMAP format...")
    image_size = np.array([518, 518])
    reconstruction = batch_np_matrix_to_pycolmap_wo_track(
        points_3d, points_xyf, points_rgb, extrinsic, intrinsic,
        image_size, shared_camera=False, camera_type="PINHOLE"
    )
    
    # Rescale cameras to original image sizes
    base_image_paths = [os.path.basename(p) for p in image_paths]
    reconstruction = rename_and_rescale_colmap(
        reconstruction, base_image_paths, original_coords.cpu().numpy(), 518
    )
    
    # Save COLMAP reconstruction
    sparse_dir = scene_dir / "sparse"
    sparse_dir.mkdir(exist_ok=True)
    reconstruction.write(str(sparse_dir))
    
    # Save point cloud as PLY
    scene_ply = sparse_dir / "points.ply"
    trimesh.PointCloud(points_3d, colors=points_rgb).export(str(scene_ply))
    logging.info(f"✓ Saved: {scene_ply}")
    
    return scene_ply


def rename_and_rescale_colmap(reconstruction, image_paths, original_coords, img_size):
    """
    Rescale COLMAP camera parameters to match original image sizes.
    
    Args:
        reconstruction: PyColmap reconstruction object
        image_paths: List of image filenames
        original_coords: Original image dimensions
        img_size: Size images were resized to for processing
        
    Returns:
        Updated reconstruction with correct camera parameters
    """
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
    """
    Preprocess scene point cloud to improve quality for alignment.
    
    Steps:
    1. Load point cloud
    2. Remove statistical outliers (noise)
    3. Detect and remove ground plane
    4. Downsample to reduce point count
    5. Estimate normals for later use
    
    Args:
        pcd_path: Path to raw scene point cloud
        config: Pipeline configuration
        save_path: Optional path to save preprocessed cloud
        
    Returns:
        Preprocessed Open3D point cloud
    """
    
    logging.info("\n" + "="*70)
    logging.info("  SCENE PREPROCESSING")
    logging.info("="*70)
    
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    original_count = len(pcd.points)
    logging.info(f"Loaded: {original_count} points")
    
    if config.VISUALIZE_PREPROCESSING:
        visualize_pcd(pcd, "Original Scene")
    
    # -------------------------------------------------------------------------
    # STEP 1: Remove outlier points (noise, floating artifacts)
    # -------------------------------------------------------------------------
    logging.info("Removing outliers...")
    logging.info(f"  Using {config.SCENE_OUTLIER_NEIGHBORS} neighbors, {config.SCENE_OUTLIER_STD} std_ratio")
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=config.SCENE_OUTLIER_NEIGHBORS,
        std_ratio=config.SCENE_OUTLIER_STD
    )
    logging.info(f"  {original_count} → {len(pcd.points)} points")
    
    if config.VISUALIZE_PREPROCESSING:
        visualize_pcd(pcd, "After Outlier Removal")
    
    # -------------------------------------------------------------------------
    # STEP 2: Remove ground plane (table, floor, etc.)
    # -------------------------------------------------------------------------
    if config.REMOVE_PLANE:
        logging.info("Removing plane...")
        
        # First estimate normals (needed for plane detection)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=config.SCENE_NORMAL_RADIUS,
                max_nn=config.SCENE_NORMAL_MAX_NN
            )
        )
        
        # Detect dominant plane using RANSAC
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=config.PLANE_DISTANCE_THRESHOLD,
            ransac_n=3,
            num_iterations=1000
        )
        
        a, b, c, d = plane_model
        logging.info(f"  Plane equation: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")
        
        # Remove points below plane (with offset)
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
        logging.info(f"  Removed plane points: → {len(pcd.points)} points remaining")
        
        # Re-estimate normals after plane removal
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=config.SCENE_NORMAL_RADIUS,
                max_nn=config.SCENE_NORMAL_MAX_NN
            )
        )
        
        if config.VISUALIZE_PREPROCESSING:
            visualize_pcd(pcd, "After Plane Removal")
    
    # -------------------------------------------------------------------------
    # STEP 3: Downsample to reduce computational cost
    # -------------------------------------------------------------------------
    logging.info(f"Downsampling (voxel size={config.SCENE_DOWNSAMPLE_VOXEL})...")
    pcd = pcd.voxel_down_sample(voxel_size=config.SCENE_DOWNSAMPLE_VOXEL)
    logging.info(f"  → {len(pcd.points)} points")
    
    if config.VISUALIZE_PREPROCESSING:
        visualize_pcd(pcd, "Final Preprocessed Scene")
    
    # Save if requested
    if save_path:
        o3d.io.write_point_cloud(str(save_path), pcd)
        logging.info(f"✓ Saved: {save_path}")
    
    logging.info(f"✓ Complete: {original_count} → {len(pcd.points)} points ({100*len(pcd.points)/original_count:.1f}% remaining)")

    # Ensure normals exist (needed for point-to-plane ICP)
    if not pcd.has_normals():
       pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=config.SCENE_NORMAL_RADIUS,
               max_nn=config.SCENE_NORMAL_MAX_NN
            )
        )

    return pcd


@timeit
def preprocess_object(pcd_path: Path, config: PipelineConfig):
    """
    Preprocess reference object point cloud.
    
    Steps:
    1. Load point cloud
    2. Downsample to reduce point count
    3. Estimate normals for ICP and feature computation
    
    Args:
        pcd_path: Path to object PLY file
        config: Pipeline configuration
        
    Returns:
        Preprocessed Open3D point cloud
    """
    
    logging.info("\n" + "="*70)
    logging.info("  OBJECT PREPROCESSING")
    logging.info("="*70)
    
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    original_count = len(pcd.points)
    logging.info(f"Loaded: {original_count} points")
    
    # Downsample
    logging.info(f"Downsampling (voxel size={config.OBJECT_DOWNSAMPLE_VOXEL})...")
    pcd = pcd.voxel_down_sample(voxel_size=config.OBJECT_DOWNSAMPLE_VOXEL)
    logging.info(f"  → {len(pcd.points)} points")
    
    # Estimate normals
    logging.info(f"Estimating normals (radius={config.OBJECT_NORMAL_RADIUS}, max_nn={config.OBJECT_NORMAL_MAX_NN})...")
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
    """
    Estimate scale by comparing bounding box diagonal lengths.
    
    This is a simple but effective method: we measure the diagonal of both
    bounding boxes and use their ratio as the scale estimate.
    
    Args:
        source: Object point cloud (to be scaled)
        target: Scene point cloud (reference)
        
    Returns:
        Estimated scale factor
    """
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
    """
    Estimate initial scale and create centered transformation.
    
    Args:
        source: Object point cloud
        target: Scene point cloud
        config: Pipeline configuration
        
    Returns:
        (scale, initial_transform): Scale factor and 4x4 transformation matrix
    """
    logging.info("\n" + "="*70)
    logging.info("  SCALE ESTIMATION")
    logging.info("="*70)
    
    # Compute scale
    if config.SCALE_METHOD == "bbox":
        scale = estimate_scale_bbox(source, target)
    else:
        # Future: could add other methods here
        scale = estimate_scale_bbox(source, target)
    
    # Create transformation that scales and centers object on scene
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

def try_multiple_rotations(source, target, config):
    """
    Try RANSAC with multiple initial rotations to avoid local minima.
    Tests 24 different orientations (90° increments around all axes).
    """
    logging.info("\n" + "="*70)
    logging.info("  MULTI-ROTATION RANSAC")
    logging.info("="*70)
    
    # Generate rotation matrices for 90° increments around each axis
    rotations = []
    # 45-degree increments for better coverage
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]
    for rx in angles:
        for ry in [0, np.pi/2, np.pi, 3*np.pi/2]:  # Keep Y at 90°
            for rz in [0, np.pi/2]:
                R = o3d.geometry.get_rotation_matrix_from_xyz([rx, ry, rz])
                rotations.append(R)
    
    # Remove duplicates (some combinations give same result)
    unique_rotations = []
    for R in rotations:
        is_duplicate = False
        for R_existing in unique_rotations:
            if np.allclose(R, R_existing, atol=0.01):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_rotations.append(R)
    
    logging.info(f"Testing {len(unique_rotations)} unique rotations...")
    
    best_result = None
    best_fitness = -1
    best_rotation = np.eye(3)
    
    for i, R in enumerate(unique_rotations):
        # Rotate the source point cloud
        source_rotated = copy.deepcopy(source)
        center = source_rotated.get_center()
        source_rotated.rotate(R, center=center)
        
        # Quick RANSAC with fewer tries
        try:
            # Downsample for speed
            source_down = source_rotated.voxel_down_sample(config.RANSAC_DOWNSAMPLE)
            target_down = target.voxel_down_sample(config.RANSAC_DOWNSAMPLE)
            
            # Compute features
            source_down.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=config.RANSAC_NORMAL_RADIUS, max_nn=config.RANSAC_NORMAL_MAX_NN))
            target_down.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=config.RANSAC_NORMAL_RADIUS, max_nn=config.RANSAC_NORMAL_MAX_NN))
            
            source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                source_down, o3d.geometry.KDTreeSearchParamHybrid(
                    radius=config.RANSAC_FPFH_RADIUS, max_nn=config.RANSAC_FPFH_MAX_NN))
            target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                target_down, o3d.geometry.KDTreeSearchParamHybrid(
                    radius=config.RANSAC_FPFH_RADIUS, max_nn=config.RANSAC_FPFH_MAX_NN))
            
            # Quick RANSAC
            result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source_down, target_down, source_fpfh, target_fpfh,
                mutual_filter=True,
                max_correspondence_distance=config.RANSAC_MAX_CORRESPONDENCE_DISTANCE,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                ransac_n=3,
                checkers=[
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                        config.RANSAC_CORRESPONDENCE_CHECKER_DISTANCE)
                ],
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 0.999)
            )
            
            # Refine with ICP
            result = o3d.pipelines.registration.registration_icp(
                source_rotated, target,
                config.ICP_MAX_CORRESPONDENCE_DISTANCE,
                result.transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
            )
            
            if result.fitness > best_fitness:
                best_fitness = result.fitness
                best_result = result
                best_rotation = R
                logging.info(f"  Rotation {i+1}/{len(unique_rotations)}: fitness={result.fitness:.4f} ✅ NEW BEST!")
            else:
                logging.debug(f"  Rotation {i+1}/{len(unique_rotations)}: fitness={result.fitness:.4f}")
                
        except Exception as e:
            logging.debug(f"  Rotation {i+1} failed: {e}")
            continue
    
    logging.info(f"✓ Best rotation found with fitness={best_fitness:.4f}")
    
    return best_result, best_rotation

@timeit
def initial_alignment_ransac(source, target, config):
    """
    Find initial rough alignment using RANSAC feature matching.
    
    RANSAC (RANdom SAmple Consensus) finds correspondences between object
    and scene by matching geometric features (FPFH), then estimates a
    transformation that aligns them.
    
    Steps:
    1. Downsample for speed
    2. Compute normals
    3. Compute FPFH features (Fast Point Feature Histograms)
    4. Run RANSAC multiple times, keep best result
    
    Args:
        source: Scaled object point cloud
        target: Scene point cloud
        config: Pipeline configuration
        
    Returns:
        (best_result, all_results): Best registration result and list of all attempts
    """
    
    logging.info("\n" + "="*70)
    logging.info(f"  RANSAC INITIAL ALIGNMENT ({config.RANSAC_TRIES} attempts)")
    logging.info("="*70)
    
    # -------------------------------------------------------------------------
    # STEP 1: Downsample for computational efficiency
    # -------------------------------------------------------------------------
    source_down = source  # Object already downsampled in preprocessing
    target_down = target.voxel_down_sample(config.RANSAC_DOWNSAMPLE)
    logging.info(f"Downsampled for RANSAC:")
    logging.info(f"  Source: {len(source.points)} → {len(source_down.points)} points")
    logging.info(f"  Target: {len(target.points)} → {len(target_down.points)} points")
    
    # -------------------------------------------------------------------------
    # STEP 2: Estimate normals (needed for FPFH features)
    # -------------------------------------------------------------------------
    logging.info(f"Estimating normals (radius={config.RANSAC_NORMAL_RADIUS}, max_nn={config.RANSAC_NORMAL_MAX_NN})...")
    source_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=config.RANSAC_NORMAL_RADIUS,
            max_nn=config.RANSAC_NORMAL_MAX_NN
        )
    )
    target_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=config.RANSAC_NORMAL_RADIUS,
            max_nn=config.RANSAC_NORMAL_MAX_NN
        )
    )
    
    # -------------------------------------------------------------------------
    # STEP 3: Compute FPFH features for matching
    # -------------------------------------------------------------------------
    logging.info(f"Computing FPFH features (radius={config.RANSAC_FPFH_RADIUS}, max_nn={config.RANSAC_FPFH_MAX_NN})...")
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=config.RANSAC_FPFH_RADIUS,
            max_nn=config.RANSAC_FPFH_MAX_NN
        )
    )
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=config.RANSAC_FPFH_RADIUS,
            max_nn=config.RANSAC_FPFH_MAX_NN
        )
    )
    
    # -------------------------------------------------------------------------
    # STEP 4: Run RANSAC multiple times and keep best result
    # -------------------------------------------------------------------------
    logging.info(f"Running RANSAC alignment ({config.RANSAC_TRIES} attempts)...")
    all_results = []
    best_result = None
    best_fitness = -1
    
    for i in range(config.RANSAC_TRIES):
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh,
            mutual_filter=True,
            max_correspondence_distance=config.RANSAC_MAX_CORRESPONDENCE_DISTANCE,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=3,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    config.RANSAC_CORRESPONDENCE_CHECKER_DISTANCE
                )
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
        )
        
        all_results.append({
            'attempt': i + 1,
            'fitness': result.fitness,
            'rmse': result.inlier_rmse
        })
        
        logging.info(f"  Try {i+1}/{config.RANSAC_TRIES}: fitness={result.fitness:.4f}, RMSE={result.inlier_rmse:.6f}")
        
        if result.fitness > best_fitness:
            best_fitness = result.fitness
            best_result = result
            logging.info(f"    ✅ NEW BEST!")
    
    logging.info(f"✓ Best RANSAC result: fitness={best_fitness:.4f}, RMSE={best_result.inlier_rmse:.6f}")
    return best_result, all_results


# ============================================================================
# ADAPTIVE REFINEMENT
# ============================================================================

@timeit
def adaptive_refinement(source, target, initial_result, config):
    """
    Refine alignment using random perturbations to escape local minima.
    
    This function tries small random rotations and translations around the
    current best transformation, using ICP to refine each perturbation.
    If a perturbation improves the result, it becomes the new best.
    
    Args:
        source: Scaled object point cloud
        target: Scene point cloud
        initial_result: Starting registration result
        config: Pipeline configuration
        
    Returns:
        Refined registration result
    """
    
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
    
    # Keep trying until we reach target or max iterations
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
        
        # Try ICP refinement with this perturbation
        try:
            result = o3d.pipelines.registration.registration_icp(
                source, target,
                config.ICP_MAX_CORRESPONDENCE_DISTANCE * config.ADAPTIVE_ICP_DISTANCE_MULTIPLIER,
                current_transform,
                o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )
            
            # Check if this perturbation improved the result
            if result.fitness > 0 and result.inlier_rmse > 0:
                if (result.fitness > best_fitness or
                    (result.fitness == best_fitness and result.inlier_rmse < best_rmse)):
                    
                    improvement = result.fitness - best_fitness
                    best_fitness = result.fitness
                    best_rmse = result.inlier_rmse
                    best_transformation = result.transformation
                    
                    logging.info(f"  ✅ Iter {iteration+1}: fitness={best_fitness:.4f} (+{improvement:.4f}), RMSE={best_rmse:.6f}")
                    
                    # Check if we reached target
                    if (best_fitness >= config.ADAPTIVE_FITNESS_THRESHOLD and
                        best_rmse <= config.ADAPTIVE_RMSE_THRESHOLD):
                        logging.info("  🎉 Target reached!")
                        break
            else:
                # Increase noise if refinement failed
                noise_translation += 0.75
                
        except Exception as e:
            logging.debug(f"  ⚠️  Iter {iteration+1} error: {e}")
            noise_translation += 0.1
        
        iteration += 1
        
        # Progress update every 10 iterations
        if iteration % 10 == 0:
            logging.info(f"  Progress: iter={iteration}, best_fitness={best_fitness:.4f}")
    
    # Create final result object
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
    """
    Adaptively search for the best object scale using binary search.
    
    This function tests different scales and uses ICP to evaluate each one.
    It starts with a wide range (50% to 150% of initial scale) and progressively
    narrows down to find the scale that gives the best RMSE.
    
    Strategy:
    - Test min, middle, and max of current range
    - Move toward the best one
    - Narrow the range progressively
    - Stop when range is very small (±5% of initial scale)
    
    Args:
        object_pcd: Original unscaled object point cloud
        scene_pcd: Scene point cloud
        initial_scale: Starting scale estimate
        initial_result: Current registration result
        config: Pipeline configuration
        
    Returns:
        (best_scale, best_result, all_tested): Best scale, its result, and history
    """
    
    logging.info("\n" + "="*70)
    logging.info("  ADAPTIVE SCALE REFINEMENT")
    logging.info("="*70)
    
    best_rmse = initial_result.inlier_rmse
    best_scale = initial_scale
    best_result = initial_result
    all_tested = []
    
    # Define initial search range
    scale_min = initial_scale * config.SCALE_SEARCH_MIN_FACTOR
    scale_max = initial_scale * config.SCALE_SEARCH_MAX_FACTOR
    iteration = 0
    
    logging.info(f"Initial scale: {initial_scale:.6f}")
    logging.info(f"Initial RMSE: {best_rmse:.6f}")
    logging.info(f"Search range: [{scale_min:.6f}, {scale_max:.6f}] "
                f"({config.SCALE_SEARCH_MIN_FACTOR*100:.0f}% to {config.SCALE_SEARCH_MAX_FACTOR*100:.0f}%)\n")
    
    # Iterative binary search
    while iteration < config.SCALE_REFINEMENT_MAX_ITERATIONS:
        iteration += 1
        scale_range = scale_max - scale_min
        
        # Test three scales: min, middle, max
        scales_to_test = [
            scale_min,
            (scale_min + scale_max) / 2,  # Middle
            scale_max
        ]
        
        logging.info(f"Iteration {iteration}: Testing range [{scale_min:.6f}, {scale_max:.6f}]")
        
        results = []
        
        for i, scale in enumerate(scales_to_test):
            label = ["min", "mid", "max"][i]
            
            # Scale object
            obj = copy.deepcopy(object_pcd)
            obj.scale(scale, center=obj.get_center())
            
            # Center on scene
            trans = np.eye(4)
            trans[:3, 3] = scene_pcd.get_center() - obj.get_center()
            obj.transform(trans)
            
            try:
                # Run ICP to evaluate this scale
                result = o3d.pipelines.registration.registration_icp(
                    obj, scene_pcd,
                    config.ICP_MAX_CORRESPONDENCE_DISTANCE,
                    initial_result.transformation,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                        max_iteration=config.SCALE_ICP_MAX_ITERATIONS
                    )
                )
                
                rmse = result.inlier_rmse
                fitness = result.fitness

                # ---- anti-tiny-scale guard ----
                # 1) reject scales that are too far from initial scale
                if scale < initial_scale * 0.7 or scale > initial_scale * 1.3:
                    rmse = float("inf")

                # 2) require a minimum fitness (avoid "small object matches a random patch")
                if fitness < 0.6:
                    rmse = float("inf")

                
                # FIX: Skip failed alignments (fitness=0 means no correspondences)
                if fitness == 0 or rmse == 0:
                    logging.info(f"  {label:3s} ({scale:.6f}): fitness={fitness:.4f}, "
                               f"rmse={rmse:.6f} ⚠️ FAILED")
                    results.append({
                        'scale': scale,
                        'rmse': float('inf'),
                        'fitness': 0.0,
                        'label': label,
                        'result': None
                    })
                    continue
                
                best_marker = "✅ BEST" if rmse < best_rmse else ""
                logging.info(f"  {label:3s} ({scale:.6f}): fitness={fitness:.4f}, "
                           f"rmse={rmse:.6f} {best_marker}")
                
                results.append({
                    'scale': scale,
                    'rmse': rmse,
                    'fitness': result.fitness,
                    'label': label,
                    'result': result
                })
                
                all_tested.append({
                    'iteration': iteration,
                    'scale': float(scale),
                    'rmse': float(rmse),
                    'fitness': float(result.fitness)
                })
                
                # Update best if this is better
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_scale = scale
                    best_result = result
                    
            except Exception as e:
                logging.warning(f"  {label:3s} ({scale:.6f}): FAILED - {e}")
                results.append({
                    'scale': scale,
                    'rmse': float('inf'),
                    'fitness': 0.0,
                    'label': label,
                    'result': None
                })
        
        min_r, mid_r, max_r = results[0], results[1], results[2]
        
        # Check convergence: stop if range is very small
        if scale_range < initial_scale * config.SCALE_CONVERGENCE_THRESHOLD:
            logging.info(f"  ✓ Converged! Range is ±{100*scale_range/(2*initial_scale):.1f}% of initial scale\n")
            break
        
        # Decide where to search next based on which scale performed best
        logging.info(f"  Analysis:")
        
        if mid_r['rmse'] <= min_r['rmse'] and mid_r['rmse'] <= max_r['rmse']:
            # Middle is best - zoom in around it
            logging.info(f"    → Middle {mid_r['scale']:.6f} is best, zooming in")
            scale_min = (scale_min + mid_r['scale']) / 2
            scale_max = (scale_max + mid_r['scale']) / 2
            
        elif min_r['rmse'] <= max_r['rmse']:
            # Left side is better - search lower
            logging.info(f"    → Left {min_r['scale']:.6f} is better, shifting left")
            scale_max = mid_r['scale']
            scale_min = min_r['scale'] * 0.9
            
        else:
            # Right side is better - search higher
            logging.info(f"    → Right {max_r['scale']:.6f} is better, shifting right")
            scale_min = mid_r['scale']
            scale_max = max_r['scale'] * 1.1
        
        logging.info(f"    Next range: [{scale_min:.6f}, {scale_max:.6f}]\n")
    
    # Calculate improvement
    improvement = initial_result.inlier_rmse - best_rmse
    improvement_pct = 100 * improvement / initial_result.inlier_rmse if initial_result.inlier_rmse > 0 else 0
    scale_change_pct = 100 * (best_scale - initial_scale) / initial_scale
    
    logging.info("="*70)
    logging.info(f"✓ Best scale found: {best_scale:.6f} ({scale_change_pct:+.1f}% from initial)")
    logging.info(f"  Fitness: {initial_result.fitness:.4f} → {best_result.fitness:.4f}")
    logging.info(f"  RMSE: {initial_result.inlier_rmse:.6f} → {best_rmse:.6f} (-{improvement_pct:.1f}%)")
    logging.info(f"  Iterations: {iteration}, Total tests: {len(all_tested)}")
    logging.info("="*70)
    
    return best_scale, best_result, all_tested


# ============================================================================
# MAIN PIPELINE
# ============================================================================

@timeit
def run_pipeline(args):
    """
    Complete reconstruction and alignment pipeline.
    
    Full pipeline stages:
    1. VGGT Reconstruction - Convert images to 3D point cloud
    2. Preprocessing - Clean scene and object point clouds
    3. Scale Estimation - Estimate initial object scale
    4. RANSAC Alignment - Find rough initial alignment
    5. ICP Refinement - Refine alignment precisely
    6. Adaptive Refinement - Escape local minima
    7. Scale Refinement - Find optimal scale
    8. Finalization - Apply final transformation and save results
    
    Args:
        args: Command-line arguments
        
    Returns:
        Dictionary of metrics and results
    """
    
    start_time = time.perf_counter()
    
    scene_dir = Path(args.scene_dir)
    output_dir = scene_dir / "icp_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build configuration from arguments
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
    
    # --- First pass preprocessing (needed so we can measure sizes for auto) ---
    scene_pcd = preprocess_scene(
        pcd_path=scene_ply_path,
        config=config,
        save_path=output_dir / "scene_preprocessed_pass1.ply"
    )

    object_pcd = preprocess_object(
        pcd_path=Path(args.object_ply),
        config=config
    )

    # --- AUTO-TUNING ---
    if args.auto:
        config = compute_adaptive_parameters(object_pcd, scene_pcd, config)

        # IMPORTANT: re-run preprocessing using the auto-tuned config
        scene_pcd = preprocess_scene(
        pcd_path=scene_ply_path,
        config=config,
        save_path=output_dir / "scene_preprocessed.ply"
        )
        object_pcd = preprocess_object(
        pcd_path=Path(args.object_ply),
        config=config
        )
    else:
    # keep the first pass results as the final ones
        o3d.io.write_point_cloud(str(output_dir / "scene_preprocessed.ply"), scene_pcd)

    
    # ========================================================================
    # ========================================================================
    # STAGE 3: SCALE ESTIMATION
    # ========================================================================
    
    if config.ESTIMATE_SCALE:
        scale, scale_transform = estimate_scale(object_pcd, scene_pcd, config)
        
        # Apply initial scale and centering
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
    
    # Try multiple rotations first to find best orientation
    if args.auto:
        multi_result, best_rotation = try_multiple_rotations(
            object_pcd_scaled, scene_pcd, config
        )
        
        # Apply the best rotation
        if multi_result is not None and multi_result.fitness > 0.3:
            object_pcd_scaled.rotate(best_rotation, center=object_pcd_scaled.get_center())
            local_result = multi_result
            logging.info(f"Using multi-rotation result: fitness={local_result.fitness:.4f}")
        else:
            # Fallback to standard RANSAC
            local_result, all_attempts = initial_alignment_ransac(
                object_pcd_scaled, scene_pcd, config
            )
    else:
        local_result, all_attempts = initial_alignment_ransac(
            object_pcd_scaled, scene_pcd, config
        )
    
    # Refine RANSAC result with ICP
    logging.info("\n🔧 Refining RANSAC with ICP...")
    local_result = o3d.pipelines.registration.registration_icp(
        object_pcd_scaled, scene_pcd,
        config.ICP_MAX_CORRESPONDENCE_DISTANCE,
        local_result.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=config.ICP_MAX_ITERATIONS
        )
    )

    logging.info(f"  After ICP: fitness={local_result.fitness:.4f}, RMSE={local_result.inlier_rmse:.6f}")
    
    # ========================================================================
    # STAGE 5: ADAPTIVE REFINEMENT
    # ========================================================================
    # Force more refinement after multi-rotation
    if args.auto:
        config.ADAPTIVE_FITNESS_THRESHOLD = 0.95
        config.ADAPTIVE_RMSE_THRESHOLD = config.ADAPTIVE_RMSE_THRESHOLD * 0.5
    final_result = adaptive_refinement(
        object_pcd_scaled, scene_pcd, local_result, config
    )
    
    # ========================================================================
    # STAGE 6: SCALE REFINEMENT
    # ========================================================================
    
    refined_scale, refined_result, scale_history = refine_scale_by_fitness(
        object_pcd, scene_pcd, scale, final_result, config
    )
    
    # Use refined scale if it's better
    if refined_result.inlier_rmse < final_result.inlier_rmse:
        logging.info(f"✅ Using refined scale: {refined_scale:.6f}")
        scale = refined_scale
        final_result = refined_result
    else:
        logging.info(f"⚠️  Keeping original scale: {scale:.6f}")
    
    # ========================================================================
    # FINALIZE
    # ========================================================================
    
    # Combine all transformations
    final_transformation = np.dot(final_result.transformation, scale_transform)
    
    # Create final aligned object
    object_aligned = copy.deepcopy(object_pcd)
    object_aligned.scale(scale, center=object_aligned.get_center())
    object_aligned.transform(final_transformation)
    
    # Visualization
    if config.VISUALIZE_FINAL:
        target_vis = copy.deepcopy(scene_pcd).paint_uniform_color([1, 0, 0])  # Red
        aligned_vis = copy.deepcopy(object_aligned).paint_uniform_color([0, 1, 0])  # Green
        
        logging.info("\n🎬 Final Visualization")
        logging.info("  Red = Scene Point Cloud")
        logging.info("  Green = Aligned Object")
        
        o3d.visualization.draw_geometries(
            [target_vis, aligned_vis],
            window_name="Final Alignment Result",
            width=1280, height=720
        )
    
    # Save results
    logging.info("\n💾 Saving results...")
    np.save(output_dir / "transformation.npy", final_transformation)
    np.save(output_dir / "scale.npy", np.array([scale]))
    o3d.io.write_point_cloud(str(output_dir / "object_aligned.ply"), object_aligned)
    
    # Save metrics
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
    
    logging.info(f"  ✓ transformation.npy (4x4 transformation matrix)")
    logging.info(f"  ✓ scale.npy (final scale factor)")
    logging.info(f"  ✓ object_aligned.ply (aligned object point cloud)")
    logging.info(f"  ✓ metrics.json (detailed metrics and history)")
    
    # Summary
    elapsed = time.perf_counter() - start_time
    
    logging.info("\n" + "="*70)
    logging.info("  PIPELINE COMPLETE")
    logging.info("="*70)
    logging.info(f"⏱️  Total time: {elapsed:.2f}s")
    logging.info(f"📏 Final scale: {scale:.6f}")
    logging.info(f"📊 Final fitness: {final_result.fitness:.4f}")
    logging.info(f"📊 Final RMSE: {final_result.inlier_rmse:.6f}")
    logging.info(f"📊 Correspondences: {len(final_result.correspondence_set)}")
    
    # Quality assessment
    if final_result.fitness >= 0.85:
        logging.info("✅ EXCELLENT alignment!")
    elif final_result.fitness >= 0.6:
        logging.info("✓  GOOD alignment")
    else:
        logging.info("⚠️  MODERATE alignment - may need parameter tuning")
    
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
    
    # Run complete pipeline
    with torch.no_grad():
        metrics = run_pipeline(args)
    
    print(f"\n✅ Pipeline complete!")
    print(f"   Final fitness: {metrics['final_fitness']:.4f}")
    print(f"   Final RMSE: {metrics['final_rmse']:.6f}")
    print(f"   Final scale: {metrics['scale']:.6f}")