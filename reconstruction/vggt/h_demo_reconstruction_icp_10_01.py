# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
VGGT Reconstruction + ICP Alignment Pipeline with Adaptive Scale Refinement
(Batch-enabled + parameter logging + screenshot outputs)

What was added / changed (high level):
- NEW: Batch mode over a root directory (runs pipeline for each scene folder)
- NEW: Save final alignment screenshot PNG (pictured results)
- NEW: Save run_config.json (all parameters actually used, including auto-tuned)
- FIX: RANSAC max_correspondence_distance was multiplied by 0.1 (too strict) -> removed
- IMPROVE: Plane removal now removes a band around the plane (stable for "object on table")
- IMPROVE: Two-stage ICP refinement (coarse -> fine)
- IMPROVE: Auto-tuning is voxel-driven (radii/distances derived from voxel sizes)

Main stages (unchanged structure):
1. VGGT Reconstruction
2. Preprocessing (scene + object)
3. Scale Estimation
4. RANSAC Alignment
5. ICP Refinement
6. Adaptive Refinement
7. Scale Refinement
8. Finalize & Save

Typical usage:
# Single scene
python h_demo_reconstruction_icp_05_01.py --scene_dir /path/to/scene --object_ply /path/to/object.ply --auto --save_screenshot

# Batch
python h_demo_reconstruction_icp_05_01.py --batch_root /path/to/root --object_ply_rel textured.ply --auto --skip_reconstruction --save_screenshot

# Terminal

#1
python /home/AP_PathMatters/vggt/h_demo_reconstruction_icp_10_01.py \
  --scene_dir /home/AP_PathMatters/path_matters/datasets/Reallife_Dataset_Haroun_Aziz/scenes-others_SUBSAMPLED/air_conditioner_control_camera3 \
  --object_ply /home/AP_PathMatters/path_matters/datasets/Reallife_Dataset_Haroun_Aziz/scenes-others_SUBSAMPLED/air_conditioner_control_camera3/textured.ply \
  --auto \
  --no_scale \
  --skip_reconstruction \
  --save_screenshot

#2
python /home/AP_PathMatters/vggt/h_demo_reconstruction_icp_10_01.py \
  --scene_dir /home/AP_PathMatters/path_matters/datasets/Reallife_Dataset_Haroun_Aziz/scenes-others_SUBSAMPLED/air_conditioner_control_camera3 \
  --object_ply /home/AP_PathMatters/path_matters/datasets/Reallife_Dataset_Haroun_Aziz/scenes-others_SUBSAMPLED/air_conditioner_control_camera3/textured.ply \
  --auto \
  --no_scale \
  --skip_reconstruction \
  --scene_outlier_std 2.0 \
  --plane_offset -0.05 \
  --save_screenshot
3#
source /home/AP_PathMatters/vggt/.venv/bin/activate
export PYTHONPATH="/home/AP_PathMatters/vggt:$PYTHONPATH"

python /home/AP_PathMatters/vggt/h_demo_reconstruction_icp_10_01.py \
  --scene_dir /home/AP_PathMatters/path_matters/datasets/Synthetic_datasets_Haroun_Aziz/Objaverse_named_OBJ/phonograph_record__gramophone__55adaf6069c048bfb4f7b5f47c98384d_2 \
  --object_ply /home/AP_PathMatters/path_matters/datasets/Synthetic_datasets_Haroun_Aziz/Objaverse_named_OBJ/phonograph_record__gramophone__55adaf6069c048bfb4f7b5f47c98384d_2/phonograph_record__gramophone__55adaf6069c048bfb4f7b5f47c98384d_2.ply \
  --auto \
  --no_scale \
  --visualize_reconstruction \
  --visualize_preprocessing \
  --visualize_steps \
  --visualize_final \
  --save_screenshot


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
import csv
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

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

# ----------------------------------------------------------------------------
# VGGT RECONSTRUCTION PARAMETERS
# ----------------------------------------------------------------------------
DEFAULT_SEED = 42
DEFAULT_CONFIDENCE_THRESHOLD = 3
DEFAULT_MAX_VGGT_POINTS = 100000  # NEW: make the point limit configurable

# ----------------------------------------------------------------------------
# SCENE PREPROCESSING PARAMETERS
# ----------------------------------------------------------------------------
DEFAULT_SCENE_DOWNSAMPLE_VOXEL = 0.001
DEFAULT_SCENE_OUTLIER_NEIGHBORS = 50
DEFAULT_SCENE_OUTLIER_STD_RATIO = 5.0
DEFAULT_PLANE_DISTANCE_THRESHOLD = 0.015
DEFAULT_PLANE_OFFSET = -0.015
DEFAULT_SCENE_NORMAL_RADIUS = 1.0
DEFAULT_SCENE_NORMAL_MAX_NN = 30

# ----------------------------------------------------------------------------
# OBJECT PREPROCESSING PARAMETERS
# ----------------------------------------------------------------------------
DEFAULT_OBJECT_DOWNSAMPLE_VOXEL = 0.01
DEFAULT_OBJECT_NORMAL_RADIUS = 1.0
DEFAULT_OBJECT_NORMAL_MAX_NN = 30

# ----------------------------------------------------------------------------
# SCALE ESTIMATION PARAMETERS
# ----------------------------------------------------------------------------
DEFAULT_SCALE_METHOD = "bbox"

# ----------------------------------------------------------------------------
# RANSAC ALIGNMENT PARAMETERS
# ----------------------------------------------------------------------------
DEFAULT_RANSAC_TRIES = 20
DEFAULT_RANSAC_DOWNSAMPLE_VOXEL = 0.01
DEFAULT_RANSAC_MAX_CORRESPONDENCE_DISTANCE = 0.1
DEFAULT_RANSAC_CORRESPONDENCE_CHECKER_DISTANCE = 0.1
DEFAULT_RANSAC_NORMAL_RADIUS = 5.0
DEFAULT_RANSAC_NORMAL_MAX_NN = 100
DEFAULT_RANSAC_FPFH_RADIUS = 5.0
DEFAULT_RANSAC_FPFH_MAX_NN = 5

# ----------------------------------------------------------------------------
# ICP REFINEMENT PARAMETERS
# ----------------------------------------------------------------------------
DEFAULT_ICP_MAX_CORRESPONDENCE_DISTANCE = 0.025  # interpreted as "FINE" distance
DEFAULT_ICP_MAX_ITERATIONS = 500
DEFAULT_ADAPTIVE_ICP_DISTANCE_MULTIPLIER = 0.5

# ----------------------------------------------------------------------------
# ADAPTIVE REFINEMENT PARAMETERS
# ----------------------------------------------------------------------------
DEFAULT_ADAPTIVE_MAX_ITERATIONS = 50
DEFAULT_ADAPTIVE_FITNESS_THRESHOLD = 0.95
DEFAULT_ADAPTIVE_RMSE_THRESHOLD = 0.005
DEFAULT_ADAPTIVE_ROTATION_NOISE_RANGE = 0.01
DEFAULT_ADAPTIVE_TRANSLATION_NOISE_START = 0.1

# ----------------------------------------------------------------------------
# SCALE REFINEMENT PARAMETERS
# ----------------------------------------------------------------------------
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
        description="VGGT Reconstruction + ICP Alignment Pipeline with Scale Refinement (Batch-enabled)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=r"""
Examples:
  # Single scene
  python %(prog)s --scene_dir /path/to/scene --object_ply /path/to/object.ply --auto --save_screenshot

  # Batch over a root containing many scenes
  python %(prog)s --batch_root /path/to/root --object_ply_rel textured.ply --auto --skip_reconstruction --save_screenshot
        """
    )

    # ------------------------------------------------------------------------
    # SINGLE-SCENE MODE (same as before)
    # ------------------------------------------------------------------------
    parser.add_argument("--scene_dir", type=str, default=None,
                        help="Directory containing scene images (must have images/ subfolder)")
    parser.add_argument("--object_ply", type=str, default=None,
                        help="Path to reference object PLY file for alignment")

    # ------------------------------------------------------------------------
    # BATCH MODE (NEW)
    # ------------------------------------------------------------------------
    parser.add_argument("--batch_root", type=str, default=None,
                        help="If set, run pipeline for each subfolder under this root")
    parser.add_argument("--batch_glob", type=str, default="*",
                        help="Glob for scene folders under batch_root (default: '*')")
    parser.add_argument("--object_ply_rel", type=str, default="textured.ply",
                        help="Relative object ply path inside each scene folder (default: textured.ply)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="In batch: skip scenes that already have icp_results/metrics.json")

    # ------------------------------------------------------------------------
    # VGGT RECONSTRUCTION
    # ------------------------------------------------------------------------
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Random seed (default: {DEFAULT_SEED})")
    parser.add_argument("--conf_thres_value", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD,
                        help=f"Confidence threshold for 3D point filtering (default: {DEFAULT_CONFIDENCE_THRESHOLD})")
    parser.add_argument("--max_vggt_points", type=int, default=DEFAULT_MAX_VGGT_POINTS,
                        help=f"Max 3D points kept after confidence filtering (default: {DEFAULT_MAX_VGGT_POINTS})")
    parser.add_argument("--skip_reconstruction", action="store_true",
                        help="Skip reconstruction if sparse/points.ply already exists")

    # ------------------------------------------------------------------------
    # SCENE PREPROCESSING
    # ------------------------------------------------------------------------
    parser.add_argument("--scene_downsample", type=float, default=DEFAULT_SCENE_DOWNSAMPLE_VOXEL,
                        help=f"Scene voxel downsample size in meters (default: {DEFAULT_SCENE_DOWNSAMPLE_VOXEL})")
    parser.add_argument("--scene_outlier_neighbors", type=int, default=DEFAULT_SCENE_OUTLIER_NEIGHBORS,
                        help=f"Neighbors for outlier removal (default: {DEFAULT_SCENE_OUTLIER_NEIGHBORS})")
    parser.add_argument("--scene_outlier_std", type=float, default=DEFAULT_SCENE_OUTLIER_STD_RATIO,
                        help=f"Std ratio for outlier removal (default: {DEFAULT_SCENE_OUTLIER_STD_RATIO})")
    parser.add_argument("--plane_threshold", type=float, default=DEFAULT_PLANE_DISTANCE_THRESHOLD,
                        help=f"Plane distance threshold in meters (default: {DEFAULT_PLANE_DISTANCE_THRESHOLD})")
    parser.add_argument("--plane_offset", type=float, default=DEFAULT_PLANE_OFFSET,
                        help=f"Plane removal offset in meters (negative = remove more band) (default: {DEFAULT_PLANE_OFFSET})")
    parser.add_argument("--no_plane_removal", action="store_true",
                        help="Disable plane removal from scene")
    parser.add_argument("--plane_mode", type=str, default="band", choices=["band"],
                        help="Plane removal strategy (default: band)")

    # ------------------------------------------------------------------------
    # OBJECT PREPROCESSING
    # ------------------------------------------------------------------------
    parser.add_argument("--object_downsample", type=float, default=DEFAULT_OBJECT_DOWNSAMPLE_VOXEL,
                        help=f"Object voxel downsample size in meters (default: {DEFAULT_OBJECT_DOWNSAMPLE_VOXEL})")

    # ------------------------------------------------------------------------
    # SCALE ESTIMATION
    # ------------------------------------------------------------------------
    parser.add_argument("--scale_method", type=str, default=DEFAULT_SCALE_METHOD,
                        choices=["bbox", "multi_scale"],
                        help=f"Scale estimation method (default: {DEFAULT_SCALE_METHOD})")
    parser.add_argument("--no_scale", action="store_true",
                        help="Disable automatic scale estimation (use scale=1.0)")

    # ------------------------------------------------------------------------
    # RANSAC INITIAL ALIGNMENT
    # ------------------------------------------------------------------------
    parser.add_argument("--ransac_tries", type=int, default=DEFAULT_RANSAC_TRIES,
                        help=f"RANSAC attempts (default: {DEFAULT_RANSAC_TRIES})")
    parser.add_argument("--ransac_downsample", type=float, default=DEFAULT_RANSAC_DOWNSAMPLE_VOXEL,
                        help=f"Voxel size for RANSAC downsampling (default: {DEFAULT_RANSAC_DOWNSAMPLE_VOXEL})")
    parser.add_argument("--ransac_max_dist", type=float, default=DEFAULT_RANSAC_MAX_CORRESPONDENCE_DISTANCE,
                        help=f"Max correspondence distance for RANSAC (default: {DEFAULT_RANSAC_MAX_CORRESPONDENCE_DISTANCE})")

    # ------------------------------------------------------------------------
    # ICP REFINEMENT (two-stage)
    # ------------------------------------------------------------------------
    parser.add_argument("--local_icp_dist", type=float, default=DEFAULT_ICP_MAX_CORRESPONDENCE_DISTANCE,
                        help=f"FINE ICP distance threshold (default: {DEFAULT_ICP_MAX_CORRESPONDENCE_DISTANCE})")
    parser.add_argument("--coarse_icp_dist", type=float, default=None,
                        help="COARSE ICP distance threshold (default: 3x local_icp_dist, or auto-tuned)")
    parser.add_argument("--local_icp_iters", type=int, default=DEFAULT_ICP_MAX_ITERATIONS,
                        help=f"ICP max iterations (default: {DEFAULT_ICP_MAX_ITERATIONS})")
    parser.add_argument("--coarse_icp_iters", type=int, default=200,
                        help="COARSE ICP iterations (default: 200)")

    # ------------------------------------------------------------------------
    # ADAPTIVE REFINEMENT
    # ------------------------------------------------------------------------
    parser.add_argument("--adaptive_iters", type=int, default=DEFAULT_ADAPTIVE_MAX_ITERATIONS,
                        help=f"Adaptive refinement iterations (default: {DEFAULT_ADAPTIVE_MAX_ITERATIONS})")
    parser.add_argument("--adaptive_fitness_threshold", type=float, default=DEFAULT_ADAPTIVE_FITNESS_THRESHOLD,
                        help=f"Fitness threshold (default: {DEFAULT_ADAPTIVE_FITNESS_THRESHOLD})")
    parser.add_argument("--adaptive_rmse_threshold", type=float, default=DEFAULT_ADAPTIVE_RMSE_THRESHOLD,
                        help=f"RMSE threshold (default: {DEFAULT_ADAPTIVE_RMSE_THRESHOLD})")
    parser.add_argument("--adaptive_rotation_noise", type=float, default=DEFAULT_ADAPTIVE_ROTATION_NOISE_RANGE,
                        help=f"Rotation noise range in radians (default: {DEFAULT_ADAPTIVE_ROTATION_NOISE_RANGE})")
    parser.add_argument("--adaptive_translation_noise", type=float, default=DEFAULT_ADAPTIVE_TRANSLATION_NOISE_START,
                        help=f"Translation noise start (default: {DEFAULT_ADAPTIVE_TRANSLATION_NOISE_START})")

    # ------------------------------------------------------------------------
    # VISUALIZATION (interactive + screenshot)
    # ------------------------------------------------------------------------
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

    # NEW: save screenshot of final alignment
    parser.add_argument("--save_screenshot", action="store_true",
                        help="Save final alignment screenshot as PNG into icp_results/")
    parser.add_argument("--screenshot_width", type=int, default=1280,
                        help="Screenshot width (default: 1280)")
    parser.add_argument("--screenshot_height", type=int, default=720,
                        help="Screenshot height (default: 720)")
    parser.add_argument("--headless", action="store_true",
                        help="Avoid opening interactive windows; still tries to save screenshot if requested")

    # ------------------------------------------------------------------------
    # AUTO-TUNING
    # ------------------------------------------------------------------------
    parser.add_argument("--auto", action="store_true",
                        help="Automatically tune parameters based on voxel sizes and bbox sizes")

    # ------------------------------------------------------------------------
    # DEBUG
    # ------------------------------------------------------------------------
    parser.add_argument("--debug", action="store_true",
                        help="Enable detailed debug logging")

    args = parser.parse_args()

    # Validate mode selection
    if args.batch_root is None:
        # single-scene mode must have scene_dir and object_ply
        if args.scene_dir is None or args.object_ply is None:
            parser.error("Single-scene mode requires --scene_dir and --object_ply (or use --batch_root).")

    return args


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
        # Scene preprocessing
        # --------------------------------------------------------------------
        self.SCENE_DOWNSAMPLE_VOXEL = args.scene_downsample
        self.SCENE_OUTLIER_NEIGHBORS = args.scene_outlier_neighbors
        self.SCENE_OUTLIER_STD = args.scene_outlier_std
        self.PLANE_DISTANCE_THRESHOLD = args.plane_threshold
        self.PLANE_OFFSET = args.plane_offset
        self.REMOVE_PLANE = not args.no_plane_removal
        self.PLANE_MODE = args.plane_mode

        self.SCENE_NORMAL_RADIUS = DEFAULT_SCENE_NORMAL_RADIUS
        self.SCENE_NORMAL_MAX_NN = DEFAULT_SCENE_NORMAL_MAX_NN

        # --------------------------------------------------------------------
        # Object preprocessing
        # --------------------------------------------------------------------
        self.OBJECT_DOWNSAMPLE_VOXEL = args.object_downsample
        self.OBJECT_NORMAL_RADIUS = DEFAULT_OBJECT_NORMAL_RADIUS
        self.OBJECT_NORMAL_MAX_NN = DEFAULT_OBJECT_NORMAL_MAX_NN

        # --------------------------------------------------------------------
        # Scale estimation
        # --------------------------------------------------------------------
        self.SCALE_METHOD = args.scale_method
        self.ESTIMATE_SCALE = not args.no_scale

        # --------------------------------------------------------------------
        # RANSAC
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
        # ICP (two-stage)
        # --------------------------------------------------------------------
        self.ICP_FINE_DISTANCE = args.local_icp_dist
        self.ICP_COARSE_DISTANCE = args.coarse_icp_dist if args.coarse_icp_dist is not None else (args.local_icp_dist * 3.0)
        self.ICP_MAX_ITERATIONS = args.local_icp_iters
        self.ICP_COARSE_ITERATIONS = args.coarse_icp_iters

        # --------------------------------------------------------------------
        # Adaptive refinement
        # --------------------------------------------------------------------
        self.ADAPTIVE_MAX_ITERATIONS = args.adaptive_iters
        self.ADAPTIVE_FITNESS_THRESHOLD = args.adaptive_fitness_threshold
        self.ADAPTIVE_RMSE_THRESHOLD = args.adaptive_rmse_threshold
        self.ADAPTIVE_NOISE_ROTATION_RANGE = args.adaptive_rotation_noise
        self.ADAPTIVE_NOISE_TRANSLATION_START = args.adaptive_translation_noise
        self.ADAPTIVE_ICP_DISTANCE_MULTIPLIER = DEFAULT_ADAPTIVE_ICP_DISTANCE_MULTIPLIER

        # --------------------------------------------------------------------
        # Scale refinement
        # --------------------------------------------------------------------
        self.SCALE_SEARCH_MIN_FACTOR = DEFAULT_SCALE_SEARCH_MIN_FACTOR
        self.SCALE_SEARCH_MAX_FACTOR = DEFAULT_SCALE_SEARCH_MAX_FACTOR
        self.SCALE_REFINEMENT_MAX_ITERATIONS = DEFAULT_SCALE_REFINEMENT_MAX_ITERATIONS
        self.SCALE_CONVERGENCE_THRESHOLD = DEFAULT_SCALE_CONVERGENCE_THRESHOLD
        self.SCALE_ICP_MAX_ITERATIONS = DEFAULT_SCALE_ICP_MAX_ITERATIONS

        # --------------------------------------------------------------------
        # Visualization
        # --------------------------------------------------------------------
        self.VISUALIZE_RECONSTRUCTION = args.visualize_reconstruction
        self.VISUALIZE_PREPROCESSING = args.visualize_preprocessing
        self.VISUALIZE_STEPS = args.visualize_steps
        self.VISUALIZE_FINAL = args.visualize_final
        self.SAVE_SCREENSHOT = args.save_screenshot
        self.SCREENSHOT_WIDTH = args.screenshot_width
        self.SCREENSHOT_HEIGHT = args.screenshot_height
        self.HEADLESS = args.headless

        # --------------------------------------------------------------------
        # VGGT
        # --------------------------------------------------------------------
        self.SEED = args.seed
        self.CONF_THRES_VALUE = args.conf_thres_value
        self.MAX_VGGT_POINTS = args.max_vggt_points

        # --------------------------------------------------------------------
        # Debug
        # --------------------------------------------------------------------
        self.DEBUG = args.debug

    def to_dict(self) -> Dict[str, Any]:
        # Simple JSON-serializable view for run_config.json
        return {
            "SCENE_DOWNSAMPLE_VOXEL": self.SCENE_DOWNSAMPLE_VOXEL,
            "SCENE_OUTLIER_NEIGHBORS": self.SCENE_OUTLIER_NEIGHBORS,
            "SCENE_OUTLIER_STD": self.SCENE_OUTLIER_STD,
            "PLANE_DISTANCE_THRESHOLD": self.PLANE_DISTANCE_THRESHOLD,
            "PLANE_OFFSET": self.PLANE_OFFSET,
            "REMOVE_PLANE": self.REMOVE_PLANE,
            "PLANE_MODE": self.PLANE_MODE,
            "SCENE_NORMAL_RADIUS": self.SCENE_NORMAL_RADIUS,
            "SCENE_NORMAL_MAX_NN": self.SCENE_NORMAL_MAX_NN,
            "OBJECT_DOWNSAMPLE_VOXEL": self.OBJECT_DOWNSAMPLE_VOXEL,
            "OBJECT_NORMAL_RADIUS": self.OBJECT_NORMAL_RADIUS,
            "OBJECT_NORMAL_MAX_NN": self.OBJECT_NORMAL_MAX_NN,
            "SCALE_METHOD": self.SCALE_METHOD,
            "ESTIMATE_SCALE": self.ESTIMATE_SCALE,
            "RANSAC_TRIES": self.RANSAC_TRIES,
            "RANSAC_DOWNSAMPLE": self.RANSAC_DOWNSAMPLE,
            "RANSAC_MAX_CORRESPONDENCE_DISTANCE": self.RANSAC_MAX_CORRESPONDENCE_DISTANCE,
            "RANSAC_CORRESPONDENCE_CHECKER_DISTANCE": self.RANSAC_CORRESPONDENCE_CHECKER_DISTANCE,
            "RANSAC_NORMAL_RADIUS": self.RANSAC_NORMAL_RADIUS,
            "RANSAC_NORMAL_MAX_NN": self.RANSAC_NORMAL_MAX_NN,
            "RANSAC_FPFH_RADIUS": self.RANSAC_FPFH_RADIUS,
            "RANSAC_FPFH_MAX_NN": self.RANSAC_FPFH_MAX_NN,
            "ICP_FINE_DISTANCE": self.ICP_FINE_DISTANCE,
            "ICP_COARSE_DISTANCE": self.ICP_COARSE_DISTANCE,
            "ICP_MAX_ITERATIONS": self.ICP_MAX_ITERATIONS,
            "ICP_COARSE_ITERATIONS": self.ICP_COARSE_ITERATIONS,
            "ADAPTIVE_MAX_ITERATIONS": self.ADAPTIVE_MAX_ITERATIONS,
            "ADAPTIVE_FITNESS_THRESHOLD": self.ADAPTIVE_FITNESS_THRESHOLD,
            "ADAPTIVE_RMSE_THRESHOLD": self.ADAPTIVE_RMSE_THRESHOLD,
            "ADAPTIVE_NOISE_ROTATION_RANGE": self.ADAPTIVE_NOISE_ROTATION_RANGE,
            "ADAPTIVE_NOISE_TRANSLATION_START": self.ADAPTIVE_NOISE_TRANSLATION_START,
            "ADAPTIVE_ICP_DISTANCE_MULTIPLIER": self.ADAPTIVE_ICP_DISTANCE_MULTIPLIER,
            "SCALE_SEARCH_MIN_FACTOR": self.SCALE_SEARCH_MIN_FACTOR,
            "SCALE_SEARCH_MAX_FACTOR": self.SCALE_SEARCH_MAX_FACTOR,
            "SCALE_REFINEMENT_MAX_ITERATIONS": self.SCALE_REFINEMENT_MAX_ITERATIONS,
            "SCALE_CONVERGENCE_THRESHOLD": self.SCALE_CONVERGENCE_THRESHOLD,
            "SCALE_ICP_MAX_ITERATIONS": self.SCALE_ICP_MAX_ITERATIONS,
            "VISUALIZE_RECONSTRUCTION": self.VISUALIZE_RECONSTRUCTION,
            "VISUALIZE_PREPROCESSING": self.VISUALIZE_PREPROCESSING,
            "VISUALIZE_STEPS": self.VISUALIZE_STEPS,
            "VISUALIZE_FINAL": self.VISUALIZE_FINAL,
            "SAVE_SCREENSHOT": self.SAVE_SCREENSHOT,
            "SCREENSHOT_WIDTH": self.SCREENSHOT_WIDTH,
            "SCREENSHOT_HEIGHT": self.SCREENSHOT_HEIGHT,
            "HEADLESS": self.HEADLESS,
            "SEED": self.SEED,
            "CONF_THRES_VALUE": self.CONF_THRES_VALUE,
            "MAX_VGGT_POINTS": self.MAX_VGGT_POINTS,
            "DEBUG": self.DEBUG,
        }


# ============================================================================
# LOGGING + TIMING
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


# ============================================================================
# VISUALIZATION HELPERS
# ============================================================================

def visualize_pcd(pcd: o3d.geometry.PointCloud, title: str = "Point Cloud"):
    logging.info(f"🔍 Visualizing: {title} ({len(pcd.points)} points)")
    o3d.visualization.draw_geometries([pcd], window_name=title, width=1280, height=720)

def try_save_alignment_screenshot(scene_pcd: o3d.geometry.PointCloud,
                                  object_pcd: o3d.geometry.PointCloud,
                                  out_path: Path,
                                  width: int,
                                  height: int) -> bool:
    """
    Save a screenshot of scene/object alignment.
    Works best when Open3D has access to a rendering backend.
    If it fails (common on headless without EGL), we warn and continue.
    """
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)

        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=width, height=height)

        scene_vis = copy.deepcopy(scene_pcd).paint_uniform_color([1, 0, 0])   # Red
        obj_vis = copy.deepcopy(object_pcd).paint_uniform_color([0, 1, 0])     # Green

        vis.add_geometry(scene_vis)
        vis.add_geometry(obj_vis)

        # Let renderer settle
        vis.poll_events()
        vis.update_renderer()

        vis.capture_screen_image(str(out_path), do_render=True)
        vis.destroy_window()

        logging.info(f"🖼️  Saved screenshot: {out_path}")
        return True
    except Exception as e:
        logging.warning(f"⚠️  Screenshot failed ({out_path.name}): {e}")
        return False


# ============================================================================
# AUTO-TUNING (UPDATED: voxel-driven)
# ============================================================================

def compute_auto_parameters_from_voxels(object_pcd: o3d.geometry.PointCloud,
                                       scene_pcd: o3d.geometry.PointCloud,
                                       config: PipelineConfig) -> PipelineConfig:
    """
    Auto-tune key parameters based on bbox size + chosen voxel sizes.

    Core idea:
    - Use voxel sizes as the "scale unit".
    - Derive normal radius, FPFH radius, correspondence distances from voxel size.

    This is much more stable across:
    - real vs synthetic scale differences
    - different point densities
    """
    obj_bbox = object_pcd.get_axis_aligned_bounding_box()
    scene_bbox = scene_pcd.get_axis_aligned_bounding_box()
    obj_diag = float(np.linalg.norm(obj_bbox.get_extent()))
    scene_diag = float(np.linalg.norm(scene_bbox.get_extent()))

    # Decide a "working voxel" for registration
    # Use RANSAC voxel downsample as the reference scale.
    # If user picked something too small/large, clamp for stability.
    v = float(config.RANSAC_DOWNSAMPLE)
    v = max(min(v, scene_diag / 50.0 if scene_diag > 0 else v), 1e-4)

    logging.info("\n" + "="*70)
    logging.info("  AUTO-TUNING (VOXEL-DRIVEN)")
    logging.info("="*70)
    logging.info(f"Scene bbox diagonal: {scene_diag:.6f}")
    logging.info(f"Object bbox diagonal: {obj_diag:.6f}")
    logging.info(f"Reference voxel (RANSAC_DOWNSAMPLE): {v:.6f}")

    # -----------------------------
    # RANSAC features (Open3D-style heuristics)
    # -----------------------------
    config.RANSAC_NORMAL_RADIUS = 2.0 * v
    config.RANSAC_FPFH_RADIUS = 5.0 * v
    config.RANSAC_NORMAL_MAX_NN = 50
    config.RANSAC_FPFH_MAX_NN = 30

    # Feature matching distance thresholds
    config.RANSAC_MAX_CORRESPONDENCE_DISTANCE = 1.5 * v
    config.RANSAC_CORRESPONDENCE_CHECKER_DISTANCE = 1.5 * v

    # More tries to reduce randomness
    config.RANSAC_TRIES = max(config.RANSAC_TRIES, 50)

    # -----------------------------
    # ICP distances (coarse -> fine)
    # -----------------------------
    # Fine uses local voxel scale
    config.ICP_FINE_DISTANCE = max(config.ICP_FINE_DISTANCE, 1.0 * v)
    config.ICP_COARSE_DISTANCE = max(config.ICP_COARSE_DISTANCE, 3.0 * config.ICP_FINE_DISTANCE)

    # -----------------------------
    # Preprocessing voxel sizes
    # -----------------------------
    # Keep some consistency: scene voxel slightly smaller than RANSAC voxel.
    config.SCENE_DOWNSAMPLE_VOXEL = min(config.SCENE_DOWNSAMPLE_VOXEL, 0.5 * v)
    config.OBJECT_DOWNSAMPLE_VOXEL = min(config.OBJECT_DOWNSAMPLE_VOXEL, 0.5 * v)

    # -----------------------------
    # Adaptive refinement noise / thresholds
    # -----------------------------
    config.ADAPTIVE_NOISE_TRANSLATION_START = max(config.ADAPTIVE_NOISE_TRANSLATION_START, 5.0 * v)
    config.ADAPTIVE_NOISE_ROTATION_RANGE = max(config.ADAPTIVE_NOISE_ROTATION_RANGE, 0.03)  # ~1.7 deg
    config.ADAPTIVE_MAX_ITERATIONS = max(config.ADAPTIVE_MAX_ITERATIONS, 100)

    # Relax thresholds slightly (you can still push them in experiments)
    config.ADAPTIVE_FITNESS_THRESHOLD = min(config.ADAPTIVE_FITNESS_THRESHOLD, 0.90)
    config.ADAPTIVE_RMSE_THRESHOLD = max(config.ADAPTIVE_RMSE_THRESHOLD, 2.0 * v)

    # Scale search narrower (bbox scale usually close)
    config.SCALE_SEARCH_MIN_FACTOR = max(config.SCALE_SEARCH_MIN_FACTOR, 0.7)
    config.SCALE_SEARCH_MAX_FACTOR = min(config.SCALE_SEARCH_MAX_FACTOR, 1.3)

    logging.info("Auto-computed parameters:")
    logging.info(f"  RANSAC_NORMAL_RADIUS: {config.RANSAC_NORMAL_RADIUS:.6f}")
    logging.info(f"  RANSAC_FPFH_RADIUS: {config.RANSAC_FPFH_RADIUS:.6f}")
    logging.info(f"  RANSAC_MAX_DIST: {config.RANSAC_MAX_CORRESPONDENCE_DISTANCE:.6f}")
    logging.info(f"  ICP_COARSE_DISTANCE: {config.ICP_COARSE_DISTANCE:.6f}")
    logging.info(f"  ICP_FINE_DISTANCE: {config.ICP_FINE_DISTANCE:.6f}")
    logging.info(f"  SCENE_DOWNSAMPLE_VOXEL: {config.SCENE_DOWNSAMPLE_VOXEL:.6f}")
    logging.info(f"  OBJECT_DOWNSAMPLE_VOXEL: {config.OBJECT_DOWNSAMPLE_VOXEL:.6f}")
    logging.info(f"  ADAPTIVE_TRANSLATION_START: {config.ADAPTIVE_NOISE_TRANSLATION_START:.6f}")
    logging.info(f"  ADAPTIVE_RMSE_THRESHOLD: {config.ADAPTIVE_RMSE_THRESHOLD:.6f}")
    logging.info("="*70)

    return config


# ============================================================================
# VGGT RECONSTRUCTION
# ============================================================================

@timeit
def run_vggt_reconstruction(scene_dir: Path, conf_threshold: float, seed: int, max_points: int) -> Path:
    logging.info("\n" + "="*70)
    logging.info("  VGGT RECONSTRUCTION")
    logging.info("="*70)

    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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

    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    depth_map = depth_map.squeeze(0).cpu().numpy()
    depth_conf = depth_conf.squeeze(0).cpu().numpy()

    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)

    # Filter by confidence and keep colors
    num_frames, height, width, _ = points_3d.shape
    points_rgb = F.interpolate(images, size=(518, 518), mode="bilinear", align_corners=False)
    points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)

    points_xyf = create_pixel_coordinate_grid(num_frames, height, width)
    conf_mask = depth_conf >= conf_threshold

    # NEW: configurable point limit
    conf_mask = randomly_limit_trues(conf_mask, int(max_points))

    points_3d = points_3d[conf_mask]
    points_xyf = points_xyf[conf_mask]
    points_rgb = points_rgb[conf_mask]

    logging.info(f"Filtered points: {len(points_3d)} (conf >= {conf_threshold}, max={max_points})")

    # Convert to COLMAP format
    logging.info("Converting to COLMAP format...")
    image_size = np.array([518, 518])
    reconstruction = batch_np_matrix_to_pycolmap_wo_track(
        points_3d, points_xyf, points_rgb, extrinsic, intrinsic,
        image_size, shared_camera=False, camera_type="PINHOLE"
    )

    base_image_paths = [os.path.basename(p) for p in image_paths]
    reconstruction = rename_and_rescale_colmap(
        reconstruction, base_image_paths, original_coords.cpu().numpy(), 518
    )

    sparse_dir = scene_dir / "sparse"
    sparse_dir.mkdir(exist_ok=True)
    reconstruction.write(str(sparse_dir))

    scene_ply = sparse_dir / "points.ply"
    trimesh.PointCloud(points_3d, colors=points_rgb).export(str(scene_ply))
    logging.info(f"✓ Saved: {scene_ply}")

    return scene_ply


def rename_and_rescale_colmap(reconstruction, image_paths, original_coords, img_size):
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
    logging.info("\n" + "="*70)
    logging.info("  SCENE PREPROCESSING")
    logging.info("="*70)

    pcd = o3d.io.read_point_cloud(str(pcd_path))
    original_count = len(pcd.points)
    logging.info(f"Loaded: {original_count} points")

    if config.VISUALIZE_PREPROCESSING and not config.HEADLESS:
        visualize_pcd(pcd, "Original Scene")

    # STEP 1: outlier removal
    logging.info("Removing outliers...")
    logging.info(f"  Using {config.SCENE_OUTLIER_NEIGHBORS} neighbors, {config.SCENE_OUTLIER_STD} std_ratio")
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=config.SCENE_OUTLIER_NEIGHBORS,
        std_ratio=config.SCENE_OUTLIER_STD
    )
    logging.info(f"  {original_count} → {len(pcd.points)} points")

    if config.VISUALIZE_PREPROCESSING and not config.HEADLESS:
        visualize_pcd(pcd, "After Outlier Removal")

    # STEP 2: plane removal (table)
    if config.REMOVE_PLANE:
        logging.info("Removing plane...")

        # Estimate normals for stable plane segmentation
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
        logging.info(f"  Plane equation: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")
        logging.info(f"  Inliers (plane points): {len(inliers)}")

        # CHANGE: "band" removal around plane using absolute distance
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None

        plane_norm = float(np.sqrt(a**2 + b**2 + c**2)) + 1e-12
        signed_dist = (a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / plane_norm

        # Effective band thickness:
        # - base threshold: PLANE_DISTANCE_THRESHOLD
        # - if PLANE_OFFSET is negative, remove more band (stable default for table)
        extra = max(0.0, -float(config.PLANE_OFFSET))
        band = float(config.PLANE_DISTANCE_THRESHOLD + extra)

        keep_mask = np.abs(signed_dist) > band

        pcd_filtered = o3d.geometry.PointCloud()
        pcd_filtered.points = o3d.utility.Vector3dVector(points[keep_mask])
        if colors is not None:
            pcd_filtered.colors = o3d.utility.Vector3dVector(colors[keep_mask])

        pcd = pcd_filtered
        logging.info(f"  Plane band removed (band={band:.6f}): → {len(pcd.points)} points remaining")

        # Re-estimate normals after plane removal
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=config.SCENE_NORMAL_RADIUS,
                max_nn=config.SCENE_NORMAL_MAX_NN
            )
        )

        if config.VISUALIZE_PREPROCESSING and not config.HEADLESS:
            visualize_pcd(pcd, "After Plane Removal (Band)")

    # STEP 3: downsample
    logging.info(f"Downsampling (voxel size={config.SCENE_DOWNSAMPLE_VOXEL})...")
    pcd = pcd.voxel_down_sample(voxel_size=config.SCENE_DOWNSAMPLE_VOXEL)
    logging.info(f"  → {len(pcd.points)} points")

    if config.VISUALIZE_PREPROCESSING and not config.HEADLESS:
        visualize_pcd(pcd, "Final Preprocessed Scene")

    if save_path:
        o3d.io.write_point_cloud(str(save_path), pcd)
        logging.info(f"✓ Saved: {save_path}")

    logging.info(f"✓ Complete: {original_count} → {len(pcd.points)} points ({100*len(pcd.points)/max(1, original_count):.1f}% remaining)")
    return pcd


@timeit
def preprocess_object(pcd_path: Path, config: PipelineConfig):
    logging.info("\n" + "="*70)
    logging.info("  OBJECT PREPROCESSING")
    logging.info("="*70)

    pcd = o3d.io.read_point_cloud(str(pcd_path))
    original_count = len(pcd.points)
    logging.info(f"Loaded: {original_count} points")

    logging.info(f"Downsampling (voxel size={config.OBJECT_DOWNSAMPLE_VOXEL})...")
    pcd = pcd.voxel_down_sample(voxel_size=config.OBJECT_DOWNSAMPLE_VOXEL)
    logging.info(f"  → {len(pcd.points)} points")

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
    source_bbox = source.get_axis_aligned_bounding_box()
    target_bbox = target.get_axis_aligned_bounding_box()
    source_diag = float(np.linalg.norm(source_bbox.get_extent()))
    target_diag = float(np.linalg.norm(target_bbox.get_extent()))
    scale = target_diag / max(1e-12, source_diag)
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

    source_down = source.voxel_down_sample(config.RANSAC_DOWNSAMPLE)
    target_down = target.voxel_down_sample(config.RANSAC_DOWNSAMPLE)


    logging.info(f"Downsampled for RANSAC:")
    logging.info(f"  Source: {len(source.points)} → {len(source_down.points)} points")
    logging.info(f"  Target: {len(target.points)} → {len(target_down.points)} points")

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

    logging.info(f"Running RANSAC alignment ({config.RANSAC_TRIES} attempts)...")
    all_results = []
    best_result = None
    best_fitness = -1.0

    for i in range(config.RANSAC_TRIES):
        # FIX: removed "* 0.1" bug here
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh,
            mutual_filter=False,
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
            "attempt": i + 1,
            "fitness": float(result.fitness),
            "rmse": float(result.inlier_rmse)
        })

        logging.info(f"  Try {i+1}/{config.RANSAC_TRIES}: fitness={result.fitness:.4f}, RMSE={result.inlier_rmse:.6f}")

        if result.fitness > best_fitness:
            best_fitness = float(result.fitness)
            best_result = result
            logging.info("    ✅ NEW BEST!")

    if best_result is None:
        raise RuntimeError("RANSAC failed completely (no best_result).")

    logging.info(f"✓ Best RANSAC result: fitness={best_result.fitness:.4f}, RMSE={best_result.inlier_rmse:.6f}")
    return best_result, all_results


# ============================================================================
# ADAPTIVE REFINEMENT
# ============================================================================

@timeit
def adaptive_refinement(source, target, initial_result, config):
    logging.info("\n" + "="*70)
    logging.info("  ADAPTIVE REFINEMENT")
    logging.info("="*70)

    best_fitness = float(initial_result.fitness)
    best_rmse = float(initial_result.inlier_rmse)
    best_transformation = initial_result.transformation

    iteration = 0
    noise_translation = float(config.ADAPTIVE_NOISE_TRANSLATION_START)

    logging.info(f"Target: fitness > {config.ADAPTIVE_FITNESS_THRESHOLD}, RMSE < {config.ADAPTIVE_RMSE_THRESHOLD}")
    logging.info(f"Starting: fitness={best_fitness:.4f}, RMSE={best_rmse:.6f}")

    while (iteration < config.ADAPTIVE_MAX_ITERATIONS and
           (best_fitness < config.ADAPTIVE_FITNESS_THRESHOLD or
            best_rmse > config.ADAPTIVE_RMSE_THRESHOLD)):

        noise_rotation = o3d.geometry.get_rotation_matrix_from_xyz(
            [np.random.uniform(-config.ADAPTIVE_NOISE_ROTATION_RANGE,
                               config.ADAPTIVE_NOISE_ROTATION_RANGE) for _ in range(3)]
        )
        noise_trans_vec = np.random.uniform(-noise_translation, noise_translation, 3)

        noise_transform = np.eye(4)
        noise_transform[:3, :3] = noise_rotation
        noise_transform[:3, 3] = noise_trans_vec

        current_transform = noise_transform @ best_transformation

        try:
            result = o3d.pipelines.registration.registration_icp(
                source, target,
                config.ICP_FINE_DISTANCE * config.ADAPTIVE_ICP_DISTANCE_MULTIPLIER,
                current_transform,
                o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )

            if result.fitness > 0 and result.inlier_rmse > 0:
                improved = (result.fitness > best_fitness) or (
                    result.fitness == best_fitness and result.inlier_rmse < best_rmse
                )
                if improved:
                    improvement = float(result.fitness - best_fitness)
                    best_fitness = float(result.fitness)
                    best_rmse = float(result.inlier_rmse)
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


# ============================================================================
# SCALE REFINEMENT
# ============================================================================

@timeit
def refine_scale_by_fitness(object_pcd, scene_pcd, initial_scale, initial_result, config):
    logging.info("\n" + "="*70)
    logging.info("  ADAPTIVE SCALE REFINEMENT")
    logging.info("="*70)

    best_fitness = float(initial_result.fitness)
    best_rmse = float(initial_result.inlier_rmse)
    best_scale = float(initial_scale)
    best_result = initial_result
    all_tested = []

    scale_min = float(initial_scale) * float(config.SCALE_SEARCH_MIN_FACTOR)
    scale_max = float(initial_scale) * float(config.SCALE_SEARCH_MAX_FACTOR)
    iteration = 0

    logging.info(f"Initial scale: {initial_scale:.6f}")
    logging.info(f"Initial RMSE: {best_rmse:.6f}")
    logging.info(f"Search range: [{scale_min:.6f}, {scale_max:.6f}]")

    while iteration < config.SCALE_REFINEMENT_MAX_ITERATIONS:
        iteration += 1
        scale_range = scale_max - scale_min

        scales_to_test = [
            scale_min,
            (scale_min + scale_max) / 2.0,
            scale_max
        ]

        logging.info(f"Iteration {iteration}: Testing range [{scale_min:.6f}, {scale_max:.6f}]")

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
                    config.ICP_FINE_DISTANCE,
                    initial_result.transformation,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                        max_iteration=config.SCALE_ICP_MAX_ITERATIONS
                    )
                )

                rmse = float(result.inlier_rmse)
                fitness = float(result.fitness)

                # If ICP found nothing, treat as invalid
                if fitness == 0 or rmse == 0:
                    logging.info(f"  {label:3s} ({scale:.6f}): fitness={fitness:.4f}, rmse={rmse:.6f} ⚠️ FAILED")
                    results.append({"scale": scale, "rmse": float('inf'), "fitness": 0.0, "label": label, "result": None})
                    continue

                # ---- PUT THIS HERE: ignore tiny-overlap solutions ----
                min_fitness = max(0.05, 0.5 * best_fitness)  # tweakable
                rmse_eff = rmse if fitness >= min_fitness else float("inf")

                improved = (fitness > best_fitness + 1e-4) or (
                    abs(fitness - best_fitness) <= 1e-4 and rmse < best_rmse
                )

                best_marker = "✅ BEST" if improved else ""
                low_overlap_marker = " ⚠️ low overlap, ignored" if rmse_eff == float("inf") else ""

                logging.info(
                    f"  {label:3s} ({scale:.6f}): fitness={fitness:.4f}, rmse={rmse:.6f}{low_overlap_marker} {best_marker}"
                )

                # Store rmse_eff so the min/mid/max range logic won't zoom into bad (low-fitness) scales
                results.append({"scale": scale, "rmse": rmse_eff, "fitness": fitness, "label": label, "result": result})
                all_tested.append({"iteration": iteration, "scale": float(scale), "rmse": rmse, "fitness": fitness})

                # Update best using fitness-first
                if improved:
                    best_fitness = fitness
                    best_rmse = rmse
                    best_scale = float(scale)
                    best_result = result


            except Exception as e:
                logging.warning(f"  {label:3s} ({scale:.6f}): FAILED - {e}")
                results.append({"scale": scale, "rmse": float("inf"), "fitness": 0.0, "label": label, "result": None})

        min_r, mid_r, max_r = results[0], results[1], results[2]

        if scale_range < float(initial_scale) * float(config.SCALE_CONVERGENCE_THRESHOLD):
            logging.info("  ✓ Converged (range small enough)")
            break

        logging.info("  Analysis:")
        if mid_r["rmse"] <= min_r["rmse"] and mid_r["rmse"] <= max_r["rmse"]:
            logging.info(f"    → Middle {mid_r['scale']:.6f} is best, zooming in")
            scale_min = (scale_min + mid_r["scale"]) / 2.0
            scale_max = (scale_max + mid_r["scale"]) / 2.0
        elif min_r["rmse"] <= max_r["rmse"]:
            logging.info(f"    → Left {min_r['scale']:.6f} is better, shifting left")
            scale_max = mid_r["scale"]
            scale_min = min_r["scale"] * 0.9
        else:
            logging.info(f"    → Right {max_r['scale']:.6f} is better, shifting right")
            scale_min = mid_r["scale"]
            scale_max = max_r["scale"] * 1.1

        logging.info(f"    Next range: [{scale_min:.6f}, {scale_max:.6f}]")

    improvement = float(initial_result.inlier_rmse) - best_rmse
    improvement_pct = 100.0 * improvement / float(initial_result.inlier_rmse) if float(initial_result.inlier_rmse) > 0 else 0.0
    scale_change_pct = 100.0 * (best_scale - float(initial_scale)) / float(initial_scale) if float(initial_scale) != 0 else 0.0

    logging.info("="*70)
    logging.info(f"✓ Best scale found: {best_scale:.6f} ({scale_change_pct:+.1f}% from initial)")
    logging.info(f"  Fitness: {float(initial_result.fitness):.4f} → {float(best_result.fitness):.4f}")
    logging.info(f"  RMSE: {float(initial_result.inlier_rmse):.6f} → {best_rmse:.6f} (-{improvement_pct:.1f}%)")
    logging.info("="*70)

    return best_scale, best_result, all_tested


# ============================================================================
# METRIC HELPERS
# ============================================================================

def evaluate_registration_metrics(source: o3d.geometry.PointCloud,
                                  target: o3d.geometry.PointCloud,
                                  max_corr_dist: float,
                                  transformation: np.ndarray) -> Dict[str, float]:
    """
    Stable way to compute fitness/RMSE + correspondence count for reporting,
    even when our intermediate RegistrationResult is manually created.
    """
    ev = o3d.pipelines.registration.evaluate_registration(source, target, max_corr_dist, transformation)
    # ev has: fitness, inlier_rmse, correspondence_set
    return {
        "fitness": float(ev.fitness),
        "rmse": float(ev.inlier_rmse),
        "correspondences": int(len(ev.correspondence_set))
    }


# ============================================================================
# MAIN PIPELINE (SINGLE SCENE)
# ============================================================================

def extract_uniform_scale(T: np.ndarray) -> float:
    sx = float(np.linalg.norm(T[:3, 0]))
    sy = float(np.linalg.norm(T[:3, 1]))
    sz = float(np.linalg.norm(T[:3, 2]))
    return (sx + sy + sz) / 3.0


@timeit
def run_pipeline(args) -> Dict[str, Any]:
    start_time = time.perf_counter()

    scene_dir = Path(args.scene_dir)
    output_dir = scene_dir / "icp_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = PipelineConfig(args)

    logging.info("\n" + "="*70)
    logging.info("  COMPLETE PIPELINE")
    logging.info("="*70)
    logging.info(f"Scene:  {scene_dir}")
    logging.info(f"Object: {args.object_ply}")
    logging.info(f"Output: {output_dir}")
    logging.info("="*70)

    # Save initial config snapshot (will be overwritten after auto-tuning below)
    with open(output_dir / "run_config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    # ------------------------------------------------------------------------
    # STAGE 1: RECONSTRUCTION
    # ------------------------------------------------------------------------
    scene_ply_path = scene_dir / "sparse" / "points.ply"

    if args.skip_reconstruction and scene_ply_path.exists():
        logging.info(f"\n✓ Using existing reconstruction: {scene_ply_path}")
    else:
        scene_ply_path = run_vggt_reconstruction(
            scene_dir=scene_dir,
            conf_threshold=config.CONF_THRES_VALUE,
            seed=config.SEED,
            max_points=config.MAX_VGGT_POINTS
        )

    if config.VISUALIZE_RECONSTRUCTION and not config.HEADLESS:
        pcd_raw = o3d.io.read_point_cloud(str(scene_ply_path))
        visualize_pcd(pcd_raw, "Reconstructed Scene (Raw)")

    # ------------------------------------------------------------------------
    # STAGE 2: PREPROCESSING
    # ------------------------------------------------------------------------
    scene_pcd = preprocess_scene(
        pcd_path=scene_ply_path,
        config=config,
        save_path=output_dir / "scene_preprocessed.ply"
    )

    object_pcd = preprocess_object(
        pcd_path=Path(args.object_ply),
        config=config
    )

    # ------------------------------------------------------------------------
    # AUTO-TUNING (UPDATED)
    # ------------------------------------------------------------------------
    if args.auto:
        config = compute_auto_parameters_from_voxels(object_pcd, scene_pcd, config)

        # Save updated config snapshot (important for thesis/evaluation)
        with open(output_dir / "run_config.json", "w") as f:
            json.dump(config.to_dict(), f, indent=2)

    # ------------------------------------------------------------------------
    # STAGE 3: SCALE ESTIMATION
    # ------------------------------------------------------------------------
    if config.ESTIMATE_SCALE:
        scale, scale_transform = estimate_scale(object_pcd, scene_pcd, config)

        object_pcd_scaled = copy.deepcopy(object_pcd)
        object_pcd_scaled.scale(scale, center=object_pcd_scaled.get_center())
        object_pcd_scaled.transform(scale_transform)
    else:
        scale = 1.0
        scale_transform = np.eye(4)
        object_pcd_scaled = object_pcd

    # ------------------------------------------------------------------------
    # STAGE 4: RANSAC INITIAL ALIGNMENT
    # ------------------------------------------------------------------------
    local_result, all_attempts = initial_alignment_ransac(object_pcd_scaled, scene_pcd, config)


    # ------------------------------------------------------------------------
    # STAGE 5: ICP REFINEMENT (COARSE -> FINE)
    # ------------------------------------------------------------------------
    est = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=False)

    logging.info("\n🔧 Refining RANSAC with COARSE ICP...")
    local_result = o3d.pipelines.registration.registration_icp(
        object_pcd_scaled, scene_pcd,
        config.ICP_COARSE_DISTANCE,
        local_result.transformation,
        est,
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=config.ICP_COARSE_ITERATIONS
        )
    )
    logging.info(f"  After COARSE ICP: fitness={local_result.fitness:.4f}, RMSE={local_result.inlier_rmse:.6f}")

    logging.info("\n🔧 Refining with FINE ICP...")
    local_result = o3d.pipelines.registration.registration_icp(
        object_pcd_scaled, scene_pcd,
        config.ICP_FINE_DISTANCE,
        local_result.transformation,
        est,
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=config.ICP_MAX_ITERATIONS
            )
        )
    logging.info(f"  After FINE ICP: fitness={local_result.fitness:.4f}, RMSE={local_result.inlier_rmse:.6f}")
    # --- scale clamp (prevents shrink-to-patch when with_scaling=True) ---
    icp_scale = extract_uniform_scale(local_result.transformation)
    logging.info(f"  ICP estimated scale factor: {icp_scale:.4f}")

    if icp_scale < 0.5 or icp_scale > 2.0:
        logging.warning("⚠️ ICP scale out of bounds → resetting scale to 1.0")

        # IMPORTANT: Open3D may return a read-only numpy view. Copy to make it writable.
        T = np.array(local_result.transformation, copy=True)  # writable copy
        T[:3, :3] /= icp_scale                                # remove uniform scale from linear part
        local_result.transformation = T                       # assign back

# --- fitness gate (reject tiny-overlap collapse) ---
# NOTE: fitness = (# inlier correspondences / # target points) :contentReference[oaicite:1]{index=1}
    if local_result.fitness < 0.05:
        raise RuntimeError(f"ICP collapsed to local patch (fitness={local_result.fitness:.3f}). Rejecting.")




    # ------------------------------------------------------------------------
    # STAGE 6: ADAPTIVE REFINEMENT
    # ------------------------------------------------------------------------
    final_result = adaptive_refinement(object_pcd_scaled, scene_pcd, local_result, config)

    # ------------------------------------------------------------------------
    # STAGE 7: SCALE REFINEMENT
    # ------------------------------------------------------------------------
    scale_history = []
    if config.ESTIMATE_SCALE:
        refined_scale, refined_result, scale_history = refine_scale_by_fitness(
        object_pcd, scene_pcd, scale, final_result, config
        )

        if float(refined_result.inlier_rmse) < float(final_result.inlier_rmse):
            logging.info(f"✅ Using refined scale: {refined_scale:.6f}")
            scale = refined_scale
            final_result = refined_result
        else:
            logging.info(f"⚠️  Keeping original scale: {scale:.6f}")
    else:
        logging.info("Skipping scale refinement because --no_scale was set.")

    # ------------------------------------------------------------------------
    # FINALIZE
    # ------------------------------------------------------------------------
    final_transformation = np.dot(final_result.transformation, scale_transform)

    object_aligned = copy.deepcopy(object_pcd)
    object_aligned.scale(scale, center=object_aligned.get_center())
    object_aligned.transform(final_transformation)

    # Interactive visualization (optional)
    if config.VISUALIZE_FINAL and not config.HEADLESS:
        target_vis = copy.deepcopy(scene_pcd).paint_uniform_color([1, 0, 0])
        aligned_vis = copy.deepcopy(object_aligned).paint_uniform_color([0, 1, 0])

        logging.info("\n🎬 Final Visualization")
        logging.info("  Red = Scene Point Cloud")
        logging.info("  Green = Aligned Object")

        o3d.visualization.draw_geometries(
            [target_vis, aligned_vis],
            window_name="Final Alignment Result",
            width=1280, height=720
        )

    # NEW: save screenshot even in headless mode (best-effort)
    if config.SAVE_SCREENSHOT:
        try_save_alignment_screenshot(
            scene_pcd=scene_pcd,
            object_pcd=object_aligned,
            out_path=output_dir / "final_alignment.png",
            width=config.SCREENSHOT_WIDTH,
            height=config.SCREENSHOT_HEIGHT
        )

    # Save results
    logging.info("\n💾 Saving results...")
    np.save(output_dir / "transformation.npy", final_transformation)
    np.save(output_dir / "scale.npy", np.array([scale], dtype=np.float64))
    o3d.io.write_point_cloud(str(output_dir / "object_aligned.ply"), object_aligned)

    # Robust evaluation metrics (doesn't depend on correspondence_set existing)
    eval_m = evaluate_registration_metrics(object_aligned, scene_pcd, config.ICP_FINE_DISTANCE, np.eye(4))

    metrics = {
        "scene_dir": str(scene_dir),
        "object_ply": str(args.object_ply),
        "scale": float(scale),
        "ransac_best_fitness": float(local_result.fitness),
        "ransac_best_rmse": float(local_result.inlier_rmse),
        "final_fitness": float(final_result.fitness),
        "final_rmse": float(final_result.inlier_rmse),
        "final_correspondences_eval": int(eval_m["correspondences"]),
        "transformation": final_transformation.tolist(),
        "scale_refinement_history": scale_history,
        "ransac_attempts": all_attempts,
        "elapsed_time": float(time.perf_counter() - start_time),
        "config": config.to_dict(),  # NEW: store actually-used config in metrics.json
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logging.info("  ✓ transformation.npy")
    logging.info("  ✓ scale.npy")
    logging.info("  ✓ object_aligned.ply")
    logging.info("  ✓ metrics.json")
    logging.info("  ✓ run_config.json")
    if config.SAVE_SCREENSHOT:
        logging.info("  ✓ final_alignment.png")

    # Summary
    elapsed = float(time.perf_counter() - start_time)
    logging.info("\n" + "="*70)
    logging.info("  PIPELINE COMPLETE")
    logging.info("="*70)
    logging.info(f"⏱️  Total time: {elapsed:.2f}s")
    logging.info(f"📏 Final scale: {float(scale):.6f}")
    logging.info(f"📊 Final fitness: {float(final_result.fitness):.4f}")
    logging.info(f"📊 Final RMSE: {float(final_result.inlier_rmse):.6f}")

    if float(final_result.fitness) >= 0.85:
        logging.info("✅ EXCELLENT alignment!")
    elif float(final_result.fitness) >= 0.6:
        logging.info("✓  GOOD alignment")
    else:
        logging.info("⚠️  MODERATE alignment - may need parameter tuning")
        logging.info("="*70)
    return metrics


# ============================================================================
# BATCH RUNNER (NEW)
# ============================================================================

def run_batch(args) -> Dict[str, Any]:
    batch_root = Path(args.batch_root)
    scene_dirs = sorted([p for p in batch_root.glob(args.batch_glob) if p.is_dir()])

    logging.info("\n" + "="*70)
    logging.info("  BATCH MODE")
    logging.info("="*70)
    logging.info(f"Batch root: {batch_root}")
    logging.info(f"Scenes found: {len(scene_dirs)} (glob='{args.batch_glob}')")
    logging.info(f"Object ply rel: {args.object_ply_rel}")
    logging.info("="*70)

    all_metrics: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []

    # In batch, default to headless behavior (no pop-up windows)
    args.headless = True
    args.visualize_reconstruction = False
    args.visualize_preprocessing = False
    args.visualize_steps = False
    args.visualize_final = False

    for i, sd in enumerate(scene_dirs, start=1):
        obj_ply = sd / args.object_ply_rel
        images_dir = sd / "images"

        logging.info("\n" + "-"*70)
        logging.info(f"[{i}/{len(scene_dirs)}] Scene: {sd.name}")
        logging.info("-"*70)

        if not images_dir.exists():
            logging.warning(f"Skipping (no images/): {sd}")
            continue
        if not obj_ply.exists():
            logging.warning(f"Skipping (missing object ply): {obj_ply}")
            continue

        metrics_path = sd / "icp_results" / "metrics.json"
        if args.skip_existing and metrics_path.exists():
            logging.info(f"Skipping (existing metrics.json): {sd.name}")
            continue

        # Build per-scene args (reuse same argparse namespace)
        scene_args = copy.deepcopy(args)
        scene_args.scene_dir = str(sd)
        scene_args.object_ply = str(obj_ply)

        try:
            m = run_pipeline(scene_args)
            all_metrics.append(m)
        except Exception as e:
            logging.exception(f"FAILED scene {sd.name}: {e}")
            failures.append({"scene": sd.name, "error": str(e)})

    # Save batch summary
    summary_json = batch_root / "batch_summary.json"
    summary_csv = batch_root / "batch_summary.csv"

    summary = {
        "batch_root": str(batch_root),
        "num_scenes_found": len(scene_dirs),
        "num_scenes_run": len(all_metrics),
        "num_failures": len(failures),
        "failures": failures,
        "results": all_metrics,
    }

    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    # Write compact CSV (easy for thesis tables)
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scene",
            "final_fitness",
            "final_rmse",
            "scale",
            "ransac_best_fitness",
            "ransac_best_rmse",
            "elapsed_time",
            "icp_fine_dist",
            "icp_coarse_dist",
            "scene_voxel",
            "object_voxel",
            "ransac_voxel",
            "ransac_max_dist",
            "plane_threshold",
            "plane_offset",
        ])
        for m in all_metrics:
            cfg = m.get("config", {})
            scene_name = Path(m.get("scene_dir", "")).name
            writer.writerow([
                scene_name,
                m.get("final_fitness", ""),
                m.get("final_rmse", ""),
                m.get("scale", ""),
                m.get("ransac_best_fitness", ""),
                m.get("ransac_best_rmse", ""),
                m.get("elapsed_time", ""),
                cfg.get("ICP_FINE_DISTANCE", ""),
                cfg.get("ICP_COARSE_DISTANCE", ""),
                cfg.get("SCENE_DOWNSAMPLE_VOXEL", ""),
                cfg.get("OBJECT_DOWNSAMPLE_VOXEL", ""),
                cfg.get("RANSAC_DOWNSAMPLE", ""),
                cfg.get("RANSAC_MAX_CORRESPONDENCE_DISTANCE", ""),
                cfg.get("PLANE_DISTANCE_THRESHOLD", ""),
                cfg.get("PLANE_OFFSET", ""),
            ])

    logging.info("\n" + "="*70)
    logging.info("  BATCH COMPLETE")
    logging.info("="*70)
    logging.info(f"✅ batch_summary.json: {summary_json}")
    logging.info(f"✅ batch_summary.csv : {summary_csv}")
    logging.info(f"Runs: {len(all_metrics)}, Failures: {len(failures)}")
    logging.info("="*70)

    return summary


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    args = parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(level=log_level)

    with torch.no_grad():
        if args.batch_root is not None:
            run_batch(args)
        else:
            metrics = run_pipeline(args)

            print("\n✅ Pipeline complete!")
            print(f"   Final fitness: {metrics['final_fitness']:.4f}")
            print(f"   Final RMSE: {metrics['final_rmse']:.6f}")
            print(f"   Final scale: {metrics['scale']:.6f}")




