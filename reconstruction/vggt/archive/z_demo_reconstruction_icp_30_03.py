# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
VGGT Reconstruction + ICP Alignment Pipeline with Adaptive Scale Refinement
+ YOLO-based 2D Segmentation -> 3D Projection Crop

CHANGELOG vs z_demo_reconstruction_icp_21_12.py:
  - Added YOLO-based object crop stage (stage 2b):
      * Runs YOLO11 instance segmentation on N evenly-spaced images
      * Projects each 3D scene point into camera frames using VGGT extrinsics
      * Keeps only points that fall inside a detection mask in >= min_votes cameras
      * Result: scene point cloud cropped to the object BEFORE scale estimation
      * Plane removal automatically disabled when YOLO crop is active
  - New CLI args:
      --yolo_crop               Enable YOLO crop stage (off by default)
      --yolo_model              Ultralytics model name (default: yolo11x-seg.pt, auto-download)
      --yolo_num_images         How many images to run YOLO on (default: 5)
      --yolo_conf               Detection confidence threshold (default: 0.25)
      --yolo_classes            Optional COCO class IDs to filter by (e.g. --yolo_classes 39 41)
                                If not set, the largest detection per frame is used
      --yolo_min_votes          Min cameras a point must be inside a mask (default: 1)
      --yolo_visualize          Show YOLO detections + masks + cropped cloud

Install dependency:
    pip install ultralytics

Usage examples:
    # Basic — YOLO picks the largest object automatically
    python pipeline_yolo.py \\
        --scene_dir /path/to/scene \\
        --object_ply /path/to/object.ply \\
        --yolo_crop \\
        --visualize_final

    # Filter to specific COCO class (e.g. 39=bottle, 41=cup, 67=cell phone)
    python pipeline_yolo.py \\
        --scene_dir /path/to/scene \\
        --object_ply /path/to/object.ply \\
        --yolo_crop \\
        --yolo_classes 41 \\
        --yolo_visualize \\
        --visualize_final

    # Lower confidence + more images for difficult scenes
    python pipeline_yolo.py \\
        --scene_dir /path/to/scene \\
        --object_ply /path/to/object.ply \\
        --yolo_crop \\
        --yolo_conf 0.1 \\
        --yolo_num_images 10 \\
        --yolo_visualize \\
        --debug

    # Skip reconstruction + YOLO crop + auto-tuning
    python pipeline_yolo.py \\
        --scene_dir /path/to/scene \\
        --object_ply /path/to/object.ply \\
        --skip_reconstruction \\
        --yolo_crop \\
        --auto \\
        --visualize_final
"""

# ============================================================================
# PIPELINE OVERVIEW (updated)
# ============================================================================
#
#            +----------------------+
#            |        START         |
#            +----------+-----------+
#                       |
#                       v
#            +----------------------+
#            | 1) VGGT RECONSTRUCT  |
#            |  - load images       |
#            |  - run VGGT          |
#            |  - save points.ply   |
#            +----------+-----------+
#                       |
#                       v
#            +----------------------+
#            | 2a) SCENE PREPROCESS |
#            |  - outlier removal   |
#            |  - plane removal     |
#            |  - downsample        |
#            |  - normals           |
#            +----------+-----------+
#                       |
#                       v
#            +------------------------------+   <-- NEW STAGE (optional)
#            | 2b) SAM OBJECT CROP          |
#            |  - run SAM on N images       |
#            |  - get 2D object masks       |
#            |  - project 3D pts -> 2D      |
#            |  - vote: keep pts inside     |
#            |    mask in >= K cameras      |
#            +----------+-------------------+
#                       |
#                       v
#            +----------------------+
#            | 2c) OBJECT PREPROCESS|
#            |  - downsample        |
#            |  - normals           |
#            +----------+-----------+
#                       |
#                       v
#            +----------------------+
#            | 3) SCALE ESTIMATION  |
#            |  - bbox diagonal     |  <-- now much more reliable after crop
#            |  - scale & center    |
#            +----------+-----------+
#                       |
#                    [... rest unchanged ...]


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
from typing import Tuple, Optional, Dict, List

import open3d as o3d
import torch
import torch.nn.functional as F
import trimesh
import pycolmap

# VGGT imports
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid
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
DEFAULT_CONFIDENCE_THRESHOLD = 10.0   # normalised 0-100 percentage threshold
                                       # 5 = keep top 95% most confident points (dense)
                                       # 50 = keep top 50% (moderate)
                                       # 80 = keep top 20% (sparse but clean)
DEFAULT_CONF_LOW_THRESHOLD = 50.0      # kept for CLI backwards compatibility, no longer used
DEFAULT_CONF_LOW_RADIUS = 0.05         # kept for CLI backwards compatibility, no longer used
DEFAULT_CONF_LOW_MIN_NEIGHBORS = 5     # kept for CLI backwards compatibility, no longer used

# ----------------------------------------------------------------------------
# SCENE PREPROCESSING PARAMETERS
# ----------------------------------------------------------------------------
DEFAULT_SCENE_DOWNSAMPLE_VOXEL = 0.005   # 0.5mm voxel — was 1mm, halved to keep more detail
DEFAULT_SCENE_OUTLIER_NEIGHBORS = 20       # was 50 — 50 is too strict for small cropped clouds
DEFAULT_SCENE_OUTLIER_STD_RATIO = 3.0      # was 5.0 — slightly more aggressive std filter to
                                            # compensate for the looser neighbour count
DEFAULT_PLANE_DISTANCE_THRESHOLD = 0.015
DEFAULT_PLANE_OFFSET = -0.015
DEFAULT_SCENE_NORMAL_RADIUS = 1.0
DEFAULT_SCENE_NORMAL_MAX_NN = 30

# ----------------------------------------------------------------------------
# YOLO CROP PARAMETERS
# ----------------------------------------------------------------------------
DEFAULT_YOLO_MODEL = "yolo11x-seg.pt"  # any ultralytics seg model; downloaded automatically
                                        # options: yolo11n-seg, yolo11s-seg, yolo11m-seg,
                                        #          yolo11l-seg, yolo11x-seg  (n=nano … x=xlarge)
DEFAULT_YOLO_NUM_IMAGES = 9            # more frames = more votes = min_votes=2 is actually meaningful
DEFAULT_YOLO_CONF = 0.25               # detection confidence threshold
DEFAULT_YOLO_MIN_VOTES = 1             # 1 = keep point if ANY camera sees it as object
                                        # safe default — self-occluded surfaces only visible from 1 cam
                                        # raise to 2 only if you have many images (>=8) and clean masks
DEFAULT_YOLO_CLUSTER_CLEANUP = True    # after vote filtering, run DBSCAN and keep only the
                                        # largest cluster — removes residual surface fragments
DEFAULT_YOLO_CENTER_WEIGHT = 0.6       # weight of center-proximity in detection scoring (0-1)
                                        # 0.0 = pick purely by area (old behaviour)
                                        # 1.0 = pick purely by center proximity
                                        # 0.6 = recommended: center-biased but area still matters

# ----------------------------------------------------------------------------
# OBJECT PREPROCESSING PARAMETERS
# ----------------------------------------------------------------------------
DEFAULT_OBJECT_DOWNSAMPLE_VOXEL = 0.002   # was 0.01 (1cm) — 2mm keeps far more surface detail
                                            # especially important for small objects
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
DEFAULT_RANSAC_DOWNSAMPLE_VOXEL = 0.03    # was 0.01 — 3mm keeps enough points for FPFH features
DEFAULT_RANSAC_MAX_CORRESPONDENCE_DISTANCE = 0.1
DEFAULT_RANSAC_CORRESPONDENCE_CHECKER_DISTANCE = 0.1
DEFAULT_RANSAC_NORMAL_RADIUS = 5.0
DEFAULT_RANSAC_NORMAL_MAX_NN = 100
DEFAULT_RANSAC_FPFH_RADIUS = 5.0
DEFAULT_RANSAC_FPFH_MAX_NN = 5

# ----------------------------------------------------------------------------
# ICP REFINEMENT PARAMETERS
# ----------------------------------------------------------------------------
DEFAULT_ICP_MAX_CORRESPONDENCE_DISTANCE = 0.025
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
        description="VGGT Reconstruction + ICP + SAM Object Crop Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ==== REQUIRED ====
    parser.add_argument("--scene_dir", type=str, required=True)
    parser.add_argument("--object_ply", type=str, required=True)

    # ==== VGGT RECONSTRUCTION ====
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--conf_thres_value", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD,
                        help=f"High confidence threshold — points above this are always kept "
                             f"(default: {DEFAULT_CONFIDENCE_THRESHOLD})")
    parser.add_argument("--conf_low_threshold", type=float, default=DEFAULT_CONF_LOW_THRESHOLD,
                        help=f"Low confidence threshold — points in [low, high) are kept only if "
                             f"they have enough high-conf neighbours (default: {DEFAULT_CONF_LOW_THRESHOLD})")
    parser.add_argument("--conf_low_radius", type=float, default=DEFAULT_CONF_LOW_RADIUS,
                        help=f"KD-tree radius for low-conf neighbour check (default: {DEFAULT_CONF_LOW_RADIUS})")
    parser.add_argument("--conf_low_min_neighbors", type=int, default=DEFAULT_CONF_LOW_MIN_NEIGHBORS,
                        help=f"Min high-conf neighbours to keep a low-conf point "
                             f"(default: {DEFAULT_CONF_LOW_MIN_NEIGHBORS})")
    parser.add_argument("--skip_reconstruction", action="store_true")

    # ==== SCENE PREPROCESSING ====
    parser.add_argument("--scene_downsample", type=float, default=DEFAULT_SCENE_DOWNSAMPLE_VOXEL)
    parser.add_argument("--scene_outlier_neighbors", type=int, default=DEFAULT_SCENE_OUTLIER_NEIGHBORS)
    parser.add_argument("--scene_outlier_std", type=float, default=DEFAULT_SCENE_OUTLIER_STD_RATIO)
    parser.add_argument("--plane_threshold", type=float, default=DEFAULT_PLANE_DISTANCE_THRESHOLD)
    parser.add_argument("--plane_offset", type=float, default=DEFAULT_PLANE_OFFSET)
    parser.add_argument("--no_plane_removal", action="store_true")

    # ==== CROP MODE (choose one, default = classic) ====
    crop_group = parser.add_mutually_exclusive_group()
    crop_group.add_argument("--classic_crop", action="store_true", default=False,
                        help="Classic background removal: statistical outlier removal + "
                             "RANSAC plane (table) removal + voxel downsample. "
                             "This is the default when neither flag is passed.")
    crop_group.add_argument("--yolo_crop", action="store_true", default=False,
                        help="YOLO-based 2D segmentation → 3D projection crop. "
                             "Runs YOLO11 on N images, votes per 3D point, keeps only the object. "
                             "When active, plane removal inside preprocess_scene is disabled "
                             "— YOLO handles all background removal.")

    # ==== YOLO CROP OPTIONS (only used when --yolo_crop is set) ====
    parser.add_argument("--yolo_model", type=str, default=DEFAULT_YOLO_MODEL,
                        help=f"Ultralytics segmentation model name or path. "
                             f"Downloaded automatically if not found locally. "
                             f"(default: {DEFAULT_YOLO_MODEL})")
    parser.add_argument("--yolo_num_images", type=int, default=DEFAULT_YOLO_NUM_IMAGES,
                        help=f"Number of images to run YOLO on (default: {DEFAULT_YOLO_NUM_IMAGES})")
    parser.add_argument("--yolo_conf", type=float, default=DEFAULT_YOLO_CONF,
                        help=f"YOLO detection confidence threshold (default: {DEFAULT_YOLO_CONF})")
    parser.add_argument("--yolo_classes", type=int, nargs="+", default=None,
                        help="Optional COCO class IDs to keep (e.g. --yolo_classes 39 41 for bottle+cup). "
                             "If not set, the largest detection in each frame is used.")
    parser.add_argument("--yolo_min_votes", type=int, default=DEFAULT_YOLO_MIN_VOTES,
                        help=f"Keep a 3D point if it falls inside a detection mask in at least "
                             f"this many cameras. 1=full object, 2=recommended (removes surface bleed). "
                             f"(default: {DEFAULT_YOLO_MIN_VOTES})")
    parser.add_argument("--yolo_no_cluster_cleanup", action="store_true",
                        help="Disable post-crop DBSCAN cluster cleanup. "
                             "By default the largest cluster is kept to remove residual surface fragments.")
    parser.add_argument("--yolo_center_weight", type=float, default=DEFAULT_YOLO_CENTER_WEIGHT,
                        help=f"Weight of center-proximity when scoring YOLO detections (0-1). "
                             f"0=pick by area only, 1=pick by center only, "
                             f"0.6=recommended (default: {DEFAULT_YOLO_CENTER_WEIGHT})")
    parser.add_argument("--yolo_visualize", action="store_true",
                        help="Visualize YOLO masks and resulting cropped cloud")

    # ==== OBJECT PREPROCESSING ====
    parser.add_argument("--object_downsample", type=float, default=DEFAULT_OBJECT_DOWNSAMPLE_VOXEL)

    # ==== SCALE ESTIMATION ====
    parser.add_argument("--scale_method", type=str, default=DEFAULT_SCALE_METHOD,
                        choices=["bbox", "multi_scale"])
    parser.add_argument("--no_scale", action="store_true")

    # ==== RANSAC ====
    parser.add_argument("--ransac_tries", type=int, default=DEFAULT_RANSAC_TRIES)
    parser.add_argument("--ransac_downsample", type=float, default=DEFAULT_RANSAC_DOWNSAMPLE_VOXEL)
    parser.add_argument("--ransac_max_dist", type=float, default=DEFAULT_RANSAC_MAX_CORRESPONDENCE_DISTANCE)

    # ==== ICP ====
    parser.add_argument("--local_icp_dist", type=float, default=DEFAULT_ICP_MAX_CORRESPONDENCE_DISTANCE)
    parser.add_argument("--local_icp_iters", type=int, default=DEFAULT_ICP_MAX_ITERATIONS)

    # ==== ADAPTIVE REFINEMENT ====
    parser.add_argument("--adaptive_iters", type=int, default=DEFAULT_ADAPTIVE_MAX_ITERATIONS)
    parser.add_argument("--adaptive_fitness_threshold", type=float, default=DEFAULT_ADAPTIVE_FITNESS_THRESHOLD)
    parser.add_argument("--adaptive_rmse_threshold", type=float, default=DEFAULT_ADAPTIVE_RMSE_THRESHOLD)
    parser.add_argument("--adaptive_rotation_noise", type=float, default=DEFAULT_ADAPTIVE_ROTATION_NOISE_RANGE)
    parser.add_argument("--adaptive_translation_noise", type=float, default=DEFAULT_ADAPTIVE_TRANSLATION_NOISE_START)

    # ==== VISUALIZATION ====
    parser.add_argument("--visualize_reconstruction", action="store_true")
    parser.add_argument("--visualize_preprocessing", action="store_true")
    parser.add_argument("--visualize_steps", action="store_true")
    parser.add_argument("--visualize_final", action="store_true", default=True)
    parser.add_argument("--no_visualize_final", dest="visualize_final", action="store_false")

    # ==== AUTO-TUNING ====
    parser.add_argument("--auto", action="store_true")

    # ==== DEBUG ====
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


# ============================================================================
# CONFIGURATION CLASS
# ============================================================================

class PipelineConfig:
    def __init__(self, args):
        # Scene Preprocessing
        self.SCENE_DOWNSAMPLE_VOXEL = args.scene_downsample
        self.SCENE_OUTLIER_NEIGHBORS = args.scene_outlier_neighbors
        self.SCENE_OUTLIER_STD = args.scene_outlier_std
        self.PLANE_DISTANCE_THRESHOLD = args.plane_threshold
        self.PLANE_OFFSET = args.plane_offset
        self.REMOVE_PLANE = not args.no_plane_removal
        self.SCENE_NORMAL_RADIUS = DEFAULT_SCENE_NORMAL_RADIUS
        self.SCENE_NORMAL_MAX_NN = DEFAULT_SCENE_NORMAL_MAX_NN

        # VGGT two-pass confidence filtering
        self.CONF_HIGH = args.conf_thres_value
        self.CONF_LOW  = args.conf_low_threshold
        self.CONF_LOW_RADIUS        = args.conf_low_radius
        self.CONF_LOW_MIN_NEIGHBORS = args.conf_low_min_neighbors

        # --------------------------------------------------------------------
        # Crop mode:
        #   Neither flag   → classic_crop (default behaviour)
        #   --classic_crop → classic_crop explicitly
        #   --yolo_crop    → yolo_crop
        # --------------------------------------------------------------------
        self.YOLO_CROP    = args.yolo_crop
        self.CLASSIC_CROP = args.classic_crop or (not args.yolo_crop)  # default when neither set

        # YOLO Crop
        self.YOLO_MODEL = args.yolo_model
        self.YOLO_NUM_IMAGES = args.yolo_num_images
        self.YOLO_CONF = args.yolo_conf
        self.YOLO_CLASSES = args.yolo_classes
        self.YOLO_MIN_VOTES = args.yolo_min_votes
        self.YOLO_CLUSTER_CLEANUP = not args.yolo_no_cluster_cleanup
        self.YOLO_CENTER_WEIGHT = args.yolo_center_weight
        self.YOLO_VISUALIZE = args.yolo_visualize

        # When YOLO crop is active, disable plane removal inside preprocess_scene —
        # YOLO handles all background. Classic crop keeps REMOVE_PLANE as set by user.
        if self.YOLO_CROP:
            self.REMOVE_PLANE = False
            logging.debug("YOLO crop active → plane removal inside preprocess_scene disabled")

        # Object Preprocessing
        self.OBJECT_DOWNSAMPLE_VOXEL = args.object_downsample
        self.OBJECT_NORMAL_RADIUS = DEFAULT_OBJECT_NORMAL_RADIUS
        self.OBJECT_NORMAL_MAX_NN = DEFAULT_OBJECT_NORMAL_MAX_NN

        # Scale Estimation
        self.SCALE_METHOD = args.scale_method
        self.ESTIMATE_SCALE = not args.no_scale

        # RANSAC
        self.RANSAC_TRIES = args.ransac_tries
        self.RANSAC_DOWNSAMPLE = args.ransac_downsample
        self.RANSAC_MAX_CORRESPONDENCE_DISTANCE = args.ransac_max_dist
        self.RANSAC_CORRESPONDENCE_CHECKER_DISTANCE = DEFAULT_RANSAC_CORRESPONDENCE_CHECKER_DISTANCE
        self.RANSAC_NORMAL_RADIUS = DEFAULT_RANSAC_NORMAL_RADIUS
        self.RANSAC_NORMAL_MAX_NN = DEFAULT_RANSAC_NORMAL_MAX_NN
        self.RANSAC_FPFH_RADIUS = DEFAULT_RANSAC_FPFH_RADIUS
        self.RANSAC_FPFH_MAX_NN = DEFAULT_RANSAC_FPFH_MAX_NN

        # ICP
        self.ICP_MAX_CORRESPONDENCE_DISTANCE = args.local_icp_dist
        self.ICP_MAX_ITERATIONS = args.local_icp_iters

        # Adaptive Refinement
        self.ADAPTIVE_MAX_ITERATIONS = args.adaptive_iters
        self.ADAPTIVE_FITNESS_THRESHOLD = args.adaptive_fitness_threshold
        self.ADAPTIVE_RMSE_THRESHOLD = args.adaptive_rmse_threshold
        self.ADAPTIVE_NOISE_ROTATION_RANGE = args.adaptive_rotation_noise
        self.ADAPTIVE_NOISE_TRANSLATION_START = args.adaptive_translation_noise
        self.ADAPTIVE_ICP_DISTANCE_MULTIPLIER = DEFAULT_ADAPTIVE_ICP_DISTANCE_MULTIPLIER

        # Scale Refinement
        self.SCALE_SEARCH_MIN_FACTOR = DEFAULT_SCALE_SEARCH_MIN_FACTOR
        self.SCALE_SEARCH_MAX_FACTOR = DEFAULT_SCALE_SEARCH_MAX_FACTOR
        self.SCALE_REFINEMENT_MAX_ITERATIONS = DEFAULT_SCALE_REFINEMENT_MAX_ITERATIONS
        self.SCALE_CONVERGENCE_THRESHOLD = DEFAULT_SCALE_CONVERGENCE_THRESHOLD
        self.SCALE_ICP_MAX_ITERATIONS = DEFAULT_SCALE_ICP_MAX_ITERATIONS

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


def compute_adaptive_parameters(object_pcd, scene_pcd, config):
    """Automatically compute parameters based on point cloud sizes."""
    obj_bbox = object_pcd.get_axis_aligned_bounding_box()
    scene_bbox = scene_pcd.get_axis_aligned_bounding_box()
    obj_size = np.linalg.norm(obj_bbox.get_extent())
    scene_size = np.linalg.norm(scene_bbox.get_extent())
    scale_ratio = scene_size / obj_size

    logging.info("\n" + "="*70)
    logging.info("  AUTO-TUNING PARAMETERS")
    logging.info("="*70)
    logging.info(f"Object size (bbox diagonal): {obj_size:.6f}")
    logging.info(f"Scene size (bbox diagonal): {scene_size:.6f}")
    logging.info(f"Estimated scale ratio: {scale_ratio:.4f}")

    reference_size = scene_size
    config.ICP_MAX_CORRESPONDENCE_DISTANCE = reference_size * 0.05
    config.RANSAC_MAX_CORRESPONDENCE_DISTANCE = reference_size * 0.15
    config.RANSAC_CORRESPONDENCE_CHECKER_DISTANCE = reference_size * 0.15
    config.RANSAC_TRIES = 50
    config.RANSAC_NORMAL_RADIUS = reference_size * 0.15
    config.RANSAC_FPFH_RADIUS = reference_size * 0.15
    config.RANSAC_NORMAL_MAX_NN = 50
    config.RANSAC_FPFH_MAX_NN = 30
    config.SCENE_DOWNSAMPLE_VOXEL = max(reference_size * 0.005, 0.001)
    config.OBJECT_DOWNSAMPLE_VOXEL = max(obj_size * 0.01, 0.001)
    config.RANSAC_DOWNSAMPLE = max(reference_size * 0.015, 0.005)
    config.ADAPTIVE_NOISE_TRANSLATION_START = reference_size * 0.1
    config.ADAPTIVE_NOISE_ROTATION_RANGE = 0.05
    config.ADAPTIVE_MAX_ITERATIONS = 150
    config.ADAPTIVE_ICP_DISTANCE_MULTIPLIER = 1.0
    config.ADAPTIVE_FITNESS_THRESHOLD = 0.70
    config.ADAPTIVE_RMSE_THRESHOLD = reference_size * 0.03
    config.SCALE_SEARCH_MIN_FACTOR = 0.7
    config.SCALE_SEARCH_MAX_FACTOR = 1.3

    logging.info(f"\nAuto-computed parameters (summarised):")
    logging.info(f"  ICP distance:      {config.ICP_MAX_CORRESPONDENCE_DISTANCE:.6f}")
    logging.info(f"  RANSAC distance:   {config.RANSAC_MAX_CORRESPONDENCE_DISTANCE:.6f}")
    logging.info(f"  Scene downsample:  {config.SCENE_DOWNSAMPLE_VOXEL:.6f}")
    logging.info("="*70)

    return config


# ============================================================================
# VGGT RECONSTRUCTION
# ============================================================================

@timeit
def run_vggt_reconstruction(scene_dir: Path, conf_threshold: float, seed: int,
                             conf_low: float = None,
                             conf_low_radius: float = DEFAULT_CONF_LOW_RADIUS,
                             conf_low_min_neighbors: int = DEFAULT_CONF_LOW_MIN_NEIGHBORS) -> Path:
    """
    Run VGGT neural reconstruction to convert images into a 3D point cloud.

    Follows the official VGGT demo approach closely:
      - Uses load_and_preprocess_images (preserves aspect ratio, no square padding)
      - Calls model(images) as a unified forward pass (not manual head chaining)
      - Confidence threshold is a percentage 0-100 applied to the normalised
        per-point confidence map (matching the demo's slider behaviour)
      - Uses world_points directly from the pointmap branch output (higher
        quality than depth-unprojected points for close-up objects)

    conf_low and two-pass filtering parameters are kept in the signature for
    backwards compatibility but are no longer used.
    """
    logging.info("\n" + "="*70)
    logging.info("  VGGT RECONSTRUCTION")
    logging.info("="*70)

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Device: {device}, dtype: {dtype}")

    logging.info("Loading VGGT model...")
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval().to(device)

    image_dir = scene_dir / "images"
    image_paths = sorted(glob.glob(str(image_dir / "*")))
    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")

    logging.info(f"Loading {len(image_paths)} images...")
    # Use load_and_preprocess_images (aspect-ratio preserving), NOT the _square
    # variant. Square padding distorts image geometry and degrades VGGT quality.
    images = load_and_preprocess_images(image_paths).to(device)
    logging.info(f"Preprocessed images shape: {images.shape}")

    # ------------------------------------------------------------------
    # Run the full unified forward pass — model(images) as the demo does.
    # Do NOT manually chain aggregator → camera_head → depth_head.
    # The unified pass handles internal shapes and ordering correctly.
    # ------------------------------------------------------------------
    logging.info("Running VGGT inference (full forward pass)...")
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)

    # Convert pose encoding to extrinsic/intrinsic matrices
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # Move everything to CPU numpy (remove batch dimension)
    for key in list(predictions.keys()):
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)

    # Unproject depth map → world points (kept for compatibility)
    depth_map = predictions["depth"]               # (S, H, W, 1)
    world_points_from_depth = unproject_depth_map_to_point_map(
        depth_map, predictions["extrinsic"], predictions["intrinsic"])
    predictions["world_points_from_depth"] = world_points_from_depth

    # ------------------------------------------------------------------
    # Confidence filtering — percentage-based normalised approach.
    # Normalises world_points_conf to 0-100 and keeps points >= conf_threshold.
    # conf_threshold=5 is a good default (keeps top 95% most confident).
    # ------------------------------------------------------------------
    world_points = predictions["world_points"]
    world_points_conf = predictions["world_points_conf"]

    # Guard: if world_points is (S, 3, H, W) transpose to (S, H, W, 3)
    if world_points.ndim == 4 and world_points.shape[1] == 3:
        world_points = world_points.transpose(0, 2, 3, 1)

    if world_points_conf.ndim == 4:
        world_points_conf = world_points_conf.squeeze(-1)

    # Get colours from images — model stores as (S, 3, H, W), need (S, H, W, 3)
    images_np = predictions["images"]
    if images_np.ndim == 4 and images_np.shape[1] == 3:
        images_np = images_np.transpose(0, 2, 3, 1)
    if images_np.max() <= 1.0:
        images_np = (images_np * 255).astype(np.uint8)

    # Normalise confidence to 0-100 range then apply threshold
    conf_min = world_points_conf.min()
    conf_max = world_points_conf.max()
    if conf_max > conf_min:
        conf_normalised = (world_points_conf - conf_min) / (conf_max - conf_min) * 100.0
    else:
        conf_normalised = np.zeros_like(world_points_conf)

    conf_mask = conf_normalised >= conf_threshold   # (S, H, W) bool

    points_3d  = world_points[conf_mask]            # (N, 3)
    points_rgb = images_np[conf_mask]               # (N, 3)  uint8

    logging.info(f"Points after conf >= {conf_threshold}%: {len(points_3d)}")

    extrinsic_np = predictions["extrinsic"]    # (S, 4, 4)
    intrinsic_np = predictions["intrinsic"]    # (S, 3, 3)

    S, H, W, _ = world_points.shape
    points_xyf = create_pixel_coordinate_grid(S, H, W)
    points_xyf = points_xyf[conf_mask]

    logging.info("Converting to COLMAP format...")
    image_size = np.array([H, W])
    reconstruction = batch_np_matrix_to_pycolmap_wo_track(
        points_3d, points_xyf, points_rgb, extrinsic_np, intrinsic_np,
        image_size, shared_camera=False, camera_type="PINHOLE"
    )

    base_image_paths = [os.path.basename(p) for p in image_paths]

    # Rescale intrinsics — images were NOT padded to square so the
    # intrinsic already matches the actual image resolution.
    # We still run rename_and_rescale_colmap to set correct image names.
    # Determine actual image size from first loaded image for rescaling.
    from PIL import Image as _PILImg
    _actual_w, _actual_h = _PILImg.open(image_paths[0]).size
    _proc_size = max(H, W)   # processing resolution used by VGGT

    # Build a fake original_coords array for rename_and_rescale_colmap
    # (it expects shape (N, ..., h, w) with the last two values being dims)
    original_coords_fake = np.zeros((len(image_paths), 4), dtype=float)
    original_coords_fake[:, -2] = _actual_w
    original_coords_fake[:, -1] = _actual_h
    reconstruction = rename_and_rescale_colmap(
        reconstruction, base_image_paths, original_coords_fake, _proc_size
    )

    sparse_dir = scene_dir / "sparse"
    sparse_dir.mkdir(exist_ok=True)
    reconstruction.write(str(sparse_dir))

    scene_ply = sparse_dir / "points.ply"
    trimesh.PointCloud(points_3d, colors=points_rgb).export(str(scene_ply))
    logging.info(f"✓ Saved: {scene_ply}")

    # ------------------------------------------------------------------
    # Save camera data for the YOLO crop stage at original image resolution.
    # ------------------------------------------------------------------
    num_cams = extrinsic_np.shape[0]
    intrinsics_rescaled = np.zeros((num_cams, 3, 3), dtype=np.float64)

    for pyimageid in reconstruction.images:
        pyimage  = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        cam_idx  = pyimageid - 1
        fx, fy, cx, cy = pycamera.params
        K = np.array([[fx,  0, cx],
                      [ 0, fy, cy],
                      [ 0,  0,  1]], dtype=np.float64)
        intrinsics_rescaled[cam_idx] = K

    np.save(str(sparse_dir / "extrinsics.npy"), extrinsic_np)
    np.save(str(sparse_dir / "intrinsics.npy"), intrinsics_rescaled)
    logging.info(f"✓ Saved camera data (original image resolution)")

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
    """Preprocess scene point cloud (outlier removal, plane removal, downsample, normals)."""
    logging.info("\n" + "="*70)
    logging.info("  SCENE PREPROCESSING")
    logging.info("="*70)

    pcd = o3d.io.read_point_cloud(str(pcd_path))
    original_count = len(pcd.points)
    logging.info(f"Loaded: {original_count} points")

    if config.VISUALIZE_PREPROCESSING:
        visualize_pcd(pcd, "Original Scene")

    logging.info("Removing outliers...")
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=config.SCENE_OUTLIER_NEIGHBORS,
        std_ratio=config.SCENE_OUTLIER_STD
    )
    logging.info(f"  {original_count} → {len(pcd.points)} points")

    if len(pcd.points) == 0:
        logging.warning("⚠️  Outlier removal deleted all points — "
                        "try raising --scene_outlier_std or lowering --scene_outlier_neighbors. "
                        "Returning original unfiltered cloud.")
        pcd = o3d.io.read_point_cloud(str(pcd_path))

    if config.VISUALIZE_PREPROCESSING:
        visualize_pcd(pcd, "After Outlier Removal")

    if config.REMOVE_PLANE:
        logging.info("Removing plane...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=config.SCENE_NORMAL_RADIUS, max_nn=config.SCENE_NORMAL_MAX_NN))
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=config.PLANE_DISTANCE_THRESHOLD,
            ransac_n=3, num_iterations=1000)
        a, b, c, d = plane_model
        logging.info(f"  Plane: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None
        plane_norm = np.sqrt(a**2 + b**2 + c**2)
        d_offset = d - config.PLANE_OFFSET * plane_norm
        distances = (a*points[:,0] + b*points[:,1] + c*points[:,2] + d_offset) / plane_norm
        above_mask = distances <= 0
        pcd_filtered = o3d.geometry.PointCloud()
        pcd_filtered.points = o3d.utility.Vector3dVector(points[above_mask])
        if colors is not None:
            pcd_filtered.colors = o3d.utility.Vector3dVector(colors[above_mask])
        pcd = pcd_filtered
        logging.info(f"  → {len(pcd.points)} points remaining")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=config.SCENE_NORMAL_RADIUS, max_nn=config.SCENE_NORMAL_MAX_NN))
        if config.VISUALIZE_PREPROCESSING:
            visualize_pcd(pcd, "After Plane Removal")

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
    """Preprocess reference object point cloud (downsample, normals)."""
    logging.info("\n" + "="*70)
    logging.info("  OBJECT PREPROCESSING")
    logging.info("="*70)

    pcd = o3d.io.read_point_cloud(str(pcd_path))
    original_count = len(pcd.points)
    logging.info(f"Loaded: {original_count} points")

    pcd = pcd.voxel_down_sample(voxel_size=config.OBJECT_DOWNSAMPLE_VOXEL)
    logging.info(f"  → {len(pcd.points)} points")

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=config.OBJECT_NORMAL_RADIUS, max_nn=config.OBJECT_NORMAL_MAX_NN))

    logging.info(f"✓ Complete: {original_count} → {len(pcd.points)} points")
    return pcd


# ============================================================================
# YOLO 2D → 3D OBJECT CROP
# ============================================================================
#
# How it works:
#   1. Load a YOLO segmentation model (ultralytics).  Model weights are
#      downloaded automatically on first use — no manual checkpoint needed.
#   2. Run inference on N evenly-spaced images from the scene.
#   3. For each image, pick the "best" detection mask:
#        - If --yolo_classes is given, keep only detections of those classes
#          and union their masks.
#        - Otherwise pick the single largest detection by mask area (most
#          likely the foreground object in a robot tabletop scene).
#   4. Project every 3D scene point into each camera frame using the VGGT
#      extrinsics/intrinsics saved during reconstruction.
#   5. Vote: a point is kept if it fell inside a detection mask in at least
#      --yolo_min_votes cameras (default 1 = very permissive, keeps full object).
#
# Advantages over SAM automatic mode:
#   - YOLO is a *detector*, not just a segmentor — it has semantic class labels,
#     so it knows what an "object" looks like vs background.
#   - Instance masks from YOLO are already whole-object; no fragmentation issue.
#   - Model weights download automatically, no manual checkpoint management.
#   - Much faster than SAM automatic mask generator.
# ============================================================================

def _load_yolo_model(model_name: str):
    """
    Load an Ultralytics YOLO segmentation model.
    Weights are downloaded automatically if not found locally.

    Args:
        model_name: Model filename or path, e.g. 'yolo11x-seg.pt'.
                    Any model from https://docs.ultralytics.com/models/ works.
    Returns:
        Loaded YOLO model object.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "ultralytics is not installed. "
            "Install with: pip install ultralytics"
        )
    logging.info(f"Loading YOLO model: {model_name} (auto-download if needed)...")
    model = YOLO(model_name)
    logging.info(f"✓ YOLO model loaded: {model_name}")
    return model


def _get_yolo_mask(
    model,
    image_rgb: np.ndarray,
    conf: float,
    target_classes: Optional[List[int]],
    center_weight: float = DEFAULT_YOLO_CENTER_WEIGHT,
    visualize: bool = False,
    image_label: str = "",
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Run YOLO segmentation on one image and return a binary object mask
    AND the corresponding bounding box in pixel coordinates.

    Detection selection (when no target_classes given):
        score = (1 - center_weight) * norm_area
              + center_weight       * (1 - norm_center_dist)

        norm_area         = area / max_area  across all detections
        norm_center_dist  = dist_to_image_center / max_dist

        center_weight=0   → pick by area only (old behaviour)
        center_weight=1   → pick purely by center proximity
        center_weight=0.6 → recommended: center-biased, area still matters

    The object is almost always near the image center in robot tabletop
    scenes. Using center proximity as the primary criterion prevents YOLO
    from picking a large background element (chair, wall, table edge) over
    the smaller but centrally-placed object of interest.

    Returns:
        (mask, bbox_xyxy) or (None, None) if no valid detection.
    """
    H, W = image_rgb.shape[:2]

    results = model(image_rgb, conf=conf, verbose=False)

    if not results or results[0].masks is None:
        logging.warning(f"  YOLO {image_label}: no detections")
        return None, None

    result      = results[0]
    masks_tensor = result.masks.data
    classes      = result.boxes.cls.cpu().numpy().astype(int)
    confs        = result.boxes.conf.cpu().numpy()
    boxes_xyxy   = result.boxes.xyxy.cpu().numpy()
    num_det      = len(classes)

    logging.info(f"  YOLO {image_label}: {num_det} detections, "
                 f"classes={classes.tolist()}, confs={confs.round(2).tolist()}")

    import torch as _torch
    masks_full = _torch.nn.functional.interpolate(
        masks_tensor.unsqueeze(0).float(),
        size=(H, W),
        mode="bilinear",
        align_corners=False
    ).squeeze(0).cpu().numpy() > 0.5

    # COCO classes that are structural surfaces / furniture — never the object of interest.
    # Full list: https://docs.ultralytics.com/datasets/detect/coco/#categories
    BACKGROUND_CLASSES = {
        56,   # chair
        57,   # couch / sofa
        58,   # potted plant  (borderline, but usually not the target object)
        59,   # bed
        60,   # dining table
        61,   # toilet
        62,   # tv / monitor  (large flat surface)
        63,   # laptop        (flat surface when closed)
        72,   # refrigerator
        73,   # book
        74,   # clock
        75,   # vase          (remove if your object IS a vase)
        77,   # teddy bear    (remove if your object IS a teddy bear)
        # structural scene elements
         0,   # person        (should never be the reconstruction target)
        13,   # bench
        14,   # bird          (unlikely reconstruction target in lab)
    }

    if target_classes is not None:
        selected = [i for i, c in enumerate(classes) if c in target_classes]
        if not selected:
            logging.warning(f"  YOLO {image_label}: no detections for classes {target_classes}")
            return None, None
        final_mask = np.zeros((H, W), dtype=bool)
        for i in selected:
            final_mask |= masks_full[i]
        sel_boxes = boxes_xyxy[selected]
        bbox = np.array([sel_boxes[:, 0].min(), sel_boxes[:, 1].min(),
                         sel_boxes[:, 2].max(), sel_boxes[:, 3].max()])
        logging.info(f"  YOLO {image_label}: unioned {len(selected)} class-filtered masks")
    else:
        # Filter out background/furniture classes before scoring
        candidate_indices = [i for i, c in enumerate(classes)
                             if c not in BACKGROUND_CLASSES]

        if not candidate_indices:
            # All detections are background classes — log and bail
            detected_names = classes.tolist()
            logging.warning(f"  YOLO {image_label}: all {num_det} detections are "
                            f"background classes {detected_names} — skipping frame. "
                            f"If your object belongs to one of these classes, use "
                            f"--yolo_classes <id> to force it.")
            return None, None

        n_filtered = num_det - len(candidate_indices)
        if n_filtered > 0:
            logging.info(f"  YOLO {image_label}: filtered out {n_filtered} background "
                         f"detection(s), {len(candidate_indices)} candidates remaining")

        # Restrict all arrays to candidates only
        cand_classes   = classes[candidate_indices]
        cand_boxes     = boxes_xyxy[candidate_indices]
        cand_masks     = masks_full[candidate_indices]
        cand_confs     = confs[candidate_indices]
        n_cand         = len(candidate_indices)

        # --- combined center-proximity + area score ---
        img_cx, img_cy = W / 2.0, H / 2.0

        bx_cx = (cand_boxes[:, 0] + cand_boxes[:, 2]) / 2.0
        bx_cy = (cand_boxes[:, 1] + cand_boxes[:, 3]) / 2.0
        center_dists = np.sqrt((bx_cx - img_cx)**2 + (bx_cy - img_cy)**2)

        areas = cand_masks.reshape(n_cand, -1).sum(axis=1).astype(float)

        max_area = areas.max()
        max_dist = center_dists.max()
        norm_area = areas / max_area if max_area > 0 else np.ones(n_cand)
        norm_dist = center_dists / max_dist if max_dist > 0 else np.zeros(n_cand)

        scores   = (1.0 - center_weight) * norm_area + center_weight * (1.0 - norm_dist)
        best_idx = int(np.argmax(scores))

        final_mask = cand_masks[best_idx]
        bbox       = cand_boxes[best_idx]
        logging.info(f"  YOLO {image_label}: picked det {candidate_indices[best_idx]} "
                     f"(class={cand_classes[best_idx]}, conf={cand_confs[best_idx]:.2f}, "
                     f"area={int(areas[best_idx])}px, "
                     f"center_dist={center_dists[best_idx]:.1f}px, "
                     f"score={scores[best_idx]:.3f})")

    if visualize:
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            annotated_rgb = result.plot()[:, :, ::-1]
            axes[0].imshow(annotated_rgb)
            axes[0].set_title(f"YOLO detections — {image_label}")
            axes[0].axis('off')
            overlay = image_rgb.copy()
            overlay[final_mask] = (
                overlay[final_mask] * 0.35 + np.array([0, 220, 60]) * 0.65
            ).astype(np.uint8)
            x1, y1, x2, y2 = bbox.astype(int)
            import cv2 as _cv2
            _cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 80, 0), 3)
            # Mark image center
            _cv2.drawMarker(overlay, (int(W/2), int(H/2)),
                            (255, 0, 0), _cv2.MARKER_CROSS, 30, 3)
            coverage = 100.0 * final_mask.mean()
            axes[1].imshow(overlay)
            axes[1].set_title(f"Mask + bbox ({coverage:.1f}% coverage) — red cross = image center")
            axes[1].axis('off')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logging.warning(f"  Could not display YOLO visualisation: {e}")

    return final_mask, bbox


def _build_projection_matrix(extrinsic_4x4: np.ndarray, intrinsic_3x3: np.ndarray) -> np.ndarray:
    """
    Build a 3x4 projection matrix  P = K @ [R | t]  from VGGT camera data.

    VGGT extrinsics are world-to-camera 4x4 matrices:
        X_cam = R @ X_world + t
    intrinsic_3x3 is the standard pinhole K matrix.
    """
    R  = extrinsic_4x4[:3, :3]
    t  = extrinsic_4x4[:3, 3]
    Rt = np.hstack([R, t.reshape(3, 1)])  # (3, 4)
    P  = intrinsic_3x3 @ Rt               # (3, 4)
    return P


def _project_points_to_image(
    points_world: np.ndarray,   # (N, 3)
    P: np.ndarray,              # (3, 4)
    img_w: int,
    img_h: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3-D world points into 2-D pixel coordinates.

    Returns:
        px    (N, 2)  float pixel coords (col=x, row=y)
        valid (N,)    bool — True if point is in front of camera AND inside frame
    """
    N    = points_world.shape[0]
    ones = np.ones((N, 1), dtype=np.float64)
    pts_h = np.hstack([points_world, ones])   # (N, 4)
    proj  = P @ pts_h.T                        # (3, N)

    z      = proj[2]
    in_front = z > 0
    z_safe   = np.where(in_front, z, 1.0)

    px_col = proj[0] / z_safe
    px_row = proj[1] / z_safe

    in_bounds = (
        (px_col >= 0) & (px_col < img_w) &
        (px_row >= 0) & (px_row < img_h)
    )
    valid = in_front & in_bounds
    px = np.stack([px_col, px_row], axis=1)
    return px, valid


def _frustum_bbox_crop(
    points_3d: np.ndarray,      # (N, 3)
    all_bboxes: list,           # list of (bbox_xyxy, extr_4x4, intr_3x3, img_w, img_h)
    expansion: float = 0.10,    # expand bbox by this fraction on each side
) -> np.ndarray:
    """
    Keep only 3D points that project inside the 2D detection bounding box
    in at least ONE camera.

    This is a fast coarse filter applied BEFORE the pixel-level mask vote.
    A point that is completely outside the bbox in every camera is definitely
    not the object — remove it immediately.

    The bbox is expanded slightly (default 10%) to avoid clipping object edges.

    Args:
        points_3d:   (N, 3) scene point array.
        all_bboxes:  List of tuples (bbox_xyxy, extrinsic, intrinsic, img_w, img_h)
                     collected across all selected cameras.
        expansion:   Fractional expansion of each bbox side.

    Returns:
        Boolean keep mask of shape (N,).
    """
    N = points_3d.shape[0]
    keep = np.zeros(N, dtype=bool)

    for bbox_xyxy, extr, intr, img_w, img_h in all_bboxes:
        x1, y1, x2, y2 = bbox_xyxy
        # Expand bbox
        bw, bh = x2 - x1, y2 - y1
        x1 = max(0,     x1 - expansion * bw)
        y1 = max(0,     y1 - expansion * bh)
        x2 = min(img_w, x2 + expansion * bw)
        y2 = min(img_h, y2 + expansion * bh)

        P = _build_projection_matrix(extr, intr)
        px, valid = _project_points_to_image(points_3d, P, img_w, img_h)

        vis_idx = np.where(valid)[0]
        px_vis  = px[vis_idx]
        inside_bbox = (
            (px_vis[:, 0] >= x1) & (px_vis[:, 0] <= x2) &
            (px_vis[:, 1] >= y1) & (px_vis[:, 1] <= y2)
        )
        keep[vis_idx[inside_bbox]] = True

    return keep


def _remove_table_plane(
    pcd: o3d.geometry.PointCloud,
    distance_threshold: float,
    plane_offset: float,
) -> o3d.geometry.PointCloud:
    """
    Detect the dominant plane (table) in the point cloud and remove all
    points at or below it.

    Uses the same RANSAC plane detection logic as the original preprocess_scene,
    but applied to the already vote-filtered + bbox-cropped cloud.  At this
    stage the cloud is much smaller, so the plane that RANSAC finds is much
    more likely to be the actual table rather than a wall or ceiling.

    Args:
        pcd:                Point cloud to filter.
        distance_threshold: RANSAC inlier distance (metres).
        plane_offset:       Offset from plane for removal (negative = remove more).

    Returns:
        Point cloud with table and below removed.
    """
    if len(pcd.points) < 10:
        return pcd

    # Estimate normals if not present (needed for robust plane detection)
    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))

    plane_model, _ = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=3,
        num_iterations=1000
    )
    a, b, c, d = plane_model
    logging.info(f"  Table plane: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")

    points = np.asarray(pcd.points)
    plane_norm = np.sqrt(a**2 + b**2 + c**2)
    d_offset   = d - plane_offset * plane_norm
    distances  = (a*points[:,0] + b*points[:,1] + c*points[:,2] + d_offset) / plane_norm
    above_mask = distances <= 0

    result = o3d.geometry.PointCloud()
    result.points = o3d.utility.Vector3dVector(points[above_mask])
    if pcd.has_colors():
        result.colors = o3d.utility.Vector3dVector(
            np.asarray(pcd.colors)[above_mask])
    if pcd.has_normals():
        result.normals = o3d.utility.Vector3dVector(
            np.asarray(pcd.normals)[above_mask])

    logging.info(f"  Table removal: {len(pcd.points)} → {len(result.points)} points")
    return result


def _keep_largest_cluster(
    pcd: o3d.geometry.PointCloud,
    eps_fraction: float = 0.05,
    min_points: int = 20,
) -> o3d.geometry.PointCloud:
    """
    Run DBSCAN on a point cloud and return only the largest cluster.

    This removes residual surface/background fragments that survived the
    vote filter because they happened to lie inside the YOLO mask boundary
    in one or two cameras.

    eps (the DBSCAN neighbourhood radius) is set automatically as a fraction
    of the point cloud's bounding box diagonal, so it scales with the scene.

    Args:
        pcd:           Input point cloud (post vote-filter).
        eps_fraction:  DBSCAN eps = eps_fraction * bbox_diagonal. Default 0.05.
        min_points:    Minimum cluster size to be considered a real cluster.

    Returns:
        Point cloud containing only the largest cluster's points.
    """
    bbox_diag = np.linalg.norm(
        np.asarray(pcd.get_axis_aligned_bounding_box().get_extent()))
    eps = max(eps_fraction * bbox_diag, 1e-4)

    labels = np.array(pcd.cluster_dbscan(
        eps=eps, min_points=min_points, print_progress=False))

    if labels.max() < 0:
        # No clusters found at all — return as-is rather than empty cloud
        logging.warning("  DBSCAN found no clusters — returning unfiltered cloud")
        return pcd

    # Count points per cluster (label -1 = noise, skip it)
    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    largest_label = unique[np.argmax(counts)]
    largest_size  = counts.max()
    total_clusters = len(unique)

    logging.info(f"  DBSCAN: {total_clusters} clusters found, "
                 f"largest has {largest_size} pts (label={largest_label})")

    keep = labels == largest_label
    result = pcd.select_by_index(np.where(keep)[0])
    logging.info(f"  Cluster cleanup: {len(pcd.points)} → {len(result.points)} points")
    return result


@timeit
def yolo_crop_scene(
    scene_pcd: o3d.geometry.PointCloud,
    scene_dir: Path,
    config: PipelineConfig,
    save_path: Optional[Path] = None,
) -> o3d.geometry.PointCloud:
    """
    Crop the reconstructed scene point cloud to the object using YOLO
    instance segmentation, applying four progressive filtering stages:

      Stage A — Frustum / bbox crop:
          Project all 3D points into each camera and discard any point that
          falls outside the YOLO detection bounding box in EVERY camera.
          Fast coarse filter — removes ~90% of background immediately.

      Stage B — Pixel-level mask vote:
          For the survivors, count how many cameras place the point inside
          the YOLO segmentation mask. Keep points with >= yolo_min_votes.

      Stage C — Table plane removal:
          Detect the dominant plane (table surface) in the filtered cloud
          and remove everything at or below it.

      Stage D — DBSCAN largest-cluster:
          Remove any remaining disconnected fragments by keeping only the
          largest spatial cluster.

    Args:
        scene_pcd:  Scene point cloud (outlier-removed + downsampled).
        scene_dir:  Scene directory containing images/ and sparse/.
        config:     Pipeline configuration.
        save_path:  Optional path to save the cropped cloud.

    Returns:
        Cropped point cloud containing only the object.
    """
    logging.info("\n" + "="*70)
    logging.info("  YOLO 2D → 3D OBJECT CROP  (4-stage)")
    logging.info("="*70)
    logging.info(f"  Model:      {config.YOLO_MODEL}")
    logging.info(f"  Conf:       {config.YOLO_CONF}")
    logging.info(f"  Classes:    {config.YOLO_CLASSES or 'auto (largest detection)'}")
    logging.info(f"  Min votes:  {config.YOLO_MIN_VOTES}")

    if len(scene_pcd.points) == 0:
        logging.warning("⚠️  yolo_crop_scene received an empty point cloud — "
                        "returning as-is. Check that reconstruction and "
                        "preprocess_scene completed successfully.")
        return scene_pcd

    # ------------------------------------------------------------------
    # 1. Load camera data
    # ------------------------------------------------------------------
    sparse_dir = scene_dir / "sparse"
    extr_path  = sparse_dir / "extrinsics.npy"
    intr_path  = sparse_dir / "intrinsics.npy"

    if not extr_path.exists() or not intr_path.exists():
        raise FileNotFoundError(
            f"Camera data not found in {sparse_dir}. "
            "Re-run reconstruction (without --skip_reconstruction) to regenerate "
            "extrinsics.npy and intrinsics.npy."
        )

    all_extrinsics = np.load(str(extr_path))   # (F, 4, 4)
    all_intrinsics = np.load(str(intr_path))   # (F, 3, 3)
    num_frames = all_extrinsics.shape[0]
    logging.info(f"Loaded camera data: {num_frames} frames")

    # ------------------------------------------------------------------
    # 2. Select images
    # ------------------------------------------------------------------
    num_to_use = min(config.YOLO_NUM_IMAGES, num_frames)
    selected   = np.linspace(0, num_frames - 1, num_to_use, dtype=int).tolist()
    logging.info(f"Using {num_to_use} frames: {selected}")

    image_paths = sorted(glob.glob(str(scene_dir / "images" / "*")))
    if not image_paths:
        raise ValueError(f"No images found in {scene_dir / 'images'}")

    # ------------------------------------------------------------------
    # 3. Load YOLO model
    # ------------------------------------------------------------------
    yolo = _load_yolo_model(config.YOLO_MODEL)

    # ------------------------------------------------------------------
    # 4. Run YOLO on selected frames — collect masks, bboxes, camera data
    # ------------------------------------------------------------------
    points_3d = np.asarray(scene_pcd.points)  # (N, 3)
    N = points_3d.shape[0]
    logging.info(f"Scene points: {N}")

    votes_inside  = np.zeros(N, dtype=np.int32)
    votes_visible = np.zeros(N, dtype=np.int32)
    bbox_data     = []   # [(cam_idx, bbox_xyxy, extr, intr, img_w, img_h), ...]

    from PIL import Image as PILImage

    for cam_idx in selected:
        extr = all_extrinsics[cam_idx]
        intr = all_intrinsics[cam_idx]

        try:
            img = np.array(PILImage.open(image_paths[cam_idx]).convert("RGB"))
        except Exception as e:
            logging.warning(f"  Frame {cam_idx}: could not load — {e}")
            continue

        img_h, img_w = img.shape[:2]

        mask, bbox = _get_yolo_mask(
            yolo, img,
            conf=config.YOLO_CONF,
            target_classes=config.YOLO_CLASSES,
            center_weight=config.YOLO_CENTER_WEIGHT,
            visualize=config.YOLO_VISUALIZE,
            image_label=f"frame {cam_idx}",
        )
        if mask is None:
            logging.warning(f"  Frame {cam_idx}: no detection, skipping")
            continue

        # Store cam_idx alongside bbox info for vote re-run after plane removal
        bbox_data.append((cam_idx, bbox, extr, intr, img_w, img_h))

        # Accumulate mask votes
        P = _build_projection_matrix(extr, intr)
        px, valid = _project_points_to_image(points_3d, P, img_w, img_h)

        vis_idx = np.where(valid)[0]
        votes_visible[vis_idx] += 1

        px_int  = px[vis_idx].astype(int)
        row_idx = np.clip(px_int[:, 1], 0, img_h - 1)
        col_idx = np.clip(px_int[:, 0], 0, img_w - 1)
        inside  = mask[row_idx, col_idx]
        votes_inside[vis_idx[inside]] += 1

    if not bbox_data:
        logging.warning("⚠️  No YOLO detections across any frame — returning original cloud.")
        return scene_pcd

    # ------------------------------------------------------------------
    # STAGE A: Table plane removal
    #   Run on the full raw cloud FIRST, before any bbox or mask filtering.
    #   At this stage the cloud contains everything — table, object, background.
    #   RANSAC is most likely to find the dominant flat surface (table) here
    #   because it is by far the largest planar region in a tabletop scene.
    #   Removing it now means all subsequent stages operate on a cleaner cloud
    #   and the bbox/mask filters don't accidentally keep table points that
    #   happen to lie inside the detection region.
    # ------------------------------------------------------------------
    logging.info(f"\n[Stage A] Table plane removal (on full cloud, {len(scene_pcd.points)} pts)...")
    cropped = _remove_table_plane(
        scene_pcd,
        distance_threshold=config.PLANE_DISTANCE_THRESHOLD,
        plane_offset=config.PLANE_OFFSET,
    )
    if len(cropped.points) == 0:
        logging.warning("⚠️  Plane removal deleted everything — skipping plane step")
        cropped = scene_pcd

    # Refresh points_3d after plane removal — all subsequent stages use this
    points_3d = np.asarray(cropped.points)
    N_after_plane = len(points_3d)

    if N_after_plane == 0:
        logging.warning("⚠️  No points left after plane removal — returning original cloud.")
        return scene_pcd

    # ------------------------------------------------------------------
    # STAGE B: Frustum / bbox crop
    # ------------------------------------------------------------------
    logging.info(f"\n[Stage B] Frustum bbox crop ({len(bbox_data)} cameras)...")
    bbox_data_for_frustum = [(b, e, i, w, h) for _, b, e, i, w, h in bbox_data]
    bbox_keep = _frustum_bbox_crop(points_3d, bbox_data_for_frustum, expansion=0.10)
    n_after_bbox = int(bbox_keep.sum())
    logging.info(f"  {N_after_plane} → {n_after_bbox} points "
                 f"({100*n_after_bbox/N_after_plane:.1f}% kept)")

    # ------------------------------------------------------------------
    # STAGE C: Pixel-level mask vote
    #   Among bbox survivors, keep points seen inside the mask >= min_votes.
    #   Points never visible from any camera are kept conservatively.
    #   Re-runs projection on the plane-removed points_3d array.
    # ------------------------------------------------------------------
    votes_inside_new  = np.zeros(N_after_plane, dtype=np.int32)
    votes_visible_new = np.zeros(N_after_plane, dtype=np.int32)

    from PIL import Image as _PILVote

    for cam_idx_v, _, extr_v, intr_v, img_w_v, img_h_v in bbox_data:
        try:
            img_v = np.array(_PILVote.open(image_paths[cam_idx_v]).convert("RGB"))
        except Exception:
            continue
        mask_v, _ = _get_yolo_mask(
            yolo, img_v,
            conf=config.YOLO_CONF,
            target_classes=config.YOLO_CLASSES,
            center_weight=config.YOLO_CENTER_WEIGHT,
            visualize=False,
            image_label=f"vote-rerun frame {cam_idx_v}",
        )
        if mask_v is None:
            continue
        P_v = _build_projection_matrix(extr_v, intr_v)
        px_v, valid_v = _project_points_to_image(points_3d, P_v, img_w_v, img_h_v)
        vis_idx_v = np.where(valid_v)[0]
        votes_visible_new[vis_idx_v] += 1
        px_int_v = px_v[vis_idx_v].astype(int)
        row_v = np.clip(px_int_v[:, 1], 0, img_h_v - 1)
        col_v = np.clip(px_int_v[:, 0], 0, img_w_v - 1)
        votes_inside_new[vis_idx_v[mask_v[row_v, col_v]]] += 1

    logging.info(f"\n[Stage C] Mask vote filter (min_votes={config.YOLO_MIN_VOTES})...")
    visible_anywhere = votes_visible_new > 0
    mask_keep = (~visible_anywhere) | (votes_inside_new >= config.YOLO_MIN_VOTES)
    combined_keep = bbox_keep & mask_keep
    n_after_mask = int(combined_keep.sum())
    logging.info(f"  {n_after_bbox} → {n_after_mask} points "
                 f"({100*n_after_mask/N_after_plane:.1f}% of post-plane cloud)")

    if n_after_mask == 0:
        logging.warning("⚠️  All points removed after mask vote — "
                        "try --yolo_min_votes 1 or --yolo_conf 0.1")
        return scene_pcd

    result = cropped.select_by_index(np.where(combined_keep)[0])

    # ------------------------------------------------------------------
    # STAGE D: DBSCAN largest-cluster cleanup
    # ------------------------------------------------------------------
    if config.YOLO_CLUSTER_CLEANUP:
        logging.info(f"\n[Stage D] DBSCAN cluster cleanup...")
        result = _keep_largest_cluster(result)

    # ------------------------------------------------------------------
    # STAGE E: Final statistical outlier removal
    #   Last pass to remove any stray points that survived all previous
    #   stages. Uses the same parameters as scene preprocessing.
    # ------------------------------------------------------------------
    logging.info(f"\n[Stage E] Final outlier removal...")
    n_before_outlier = len(result.points)
    result, _ = result.remove_statistical_outlier(
        nb_neighbors=config.SCENE_OUTLIER_NEIGHBORS,
        std_ratio=config.SCENE_OUTLIER_STD
    )
    logging.info(f"  {n_before_outlier} → {len(result.points)} points")

    # ------------------------------------------------------------------
    # STAGE F: Final cluster cleanup
    #   After outlier removal, run DBSCAN one more time to drop any
    #   remaining disconnected fragments that are messing up the bbox
    #   diagonal used for scale estimation.
    # ------------------------------------------------------------------
    logging.info(f"\n[Stage F] Final cluster cleanup (DBSCAN)...")
    result = _keep_largest_cluster(result)

    if config.YOLO_VISUALIZE:
        visualize_pcd(result, "YOLO Cropped Scene")

    if save_path:
        o3d.io.write_point_cloud(str(save_path), result)
        logging.info(f"✓ Saved: {save_path}")

    logging.info(f"\n✓ YOLO crop complete: {N} → {len(result.points)} points "
                 f"({100*len(result.points)/N:.1f}% of original)")
    return result


# ============================================================================
# SCALE ESTIMATION  —  robust multi-estimator consensus
# ============================================================================
#
# Three independent estimators, each rotation-invariant:
#
#   1. BBox diagonal ratio
#      Simple but sensitive to outliers and orientation for elongated objects.
#      Kept as one vote, not the sole answer.
#
#   2. PCA axis lengths ratio
#      Decomposes each cloud into its three principal axes (sorted longest →
#      shortest) and averages the per-axis scale ratios. Rotation-invariant
#      because PCA axes are sorted by variance regardless of orientation.
#      More robust than bbox for elongated or asymmetric objects.
#
#   3. Volume ratio (convex hull)
#      Scale = cube_root(target_volume / source_volume). Completely
#      rotation-invariant and insensitive to individual outlier points.
#      The cube root correctly converts volumetric ratio to linear scale.
#
# Final scale = median of all three estimates. Using the median means one
# bad estimator (e.g. volume fails on a non-convex object) is simply
# outvoted — the result is still correct.
#
# Agreement check: if the three estimates spread more than 30% relative to
# their median, a warning is logged. This is a useful signal that the crop
# may still be imperfect or the object has unusual geometry.
# ============================================================================

def _estimate_scale_bbox(source: o3d.geometry.PointCloud,
                         target: o3d.geometry.PointCloud) -> float:
    """Scale from bounding box diagonal ratio."""
    src_diag = np.linalg.norm(
        source.get_axis_aligned_bounding_box().get_extent())
    tgt_diag = np.linalg.norm(
        target.get_axis_aligned_bounding_box().get_extent())
    scale = tgt_diag / src_diag if src_diag > 0 else 1.0
    logging.info(f"  [bbox]   src_diag={src_diag:.6f}  tgt_diag={tgt_diag:.6f}"
                 f"  → scale={scale:.6f}")
    return scale


def _estimate_scale_pca(source: o3d.geometry.PointCloud,
                        target: o3d.geometry.PointCloud) -> float:
    """
    Scale from PCA principal axis lengths.

    For each cloud: run PCA on the point positions, project all points onto
    each eigenvector, and take the span (max - min) as the axis length.
    Sort axis lengths descending so they correspond regardless of orientation.
    Scale is the mean ratio of matching axis lengths.
    """
    def _axis_lengths(pcd: o3d.geometry.PointCloud) -> np.ndarray:
        pts = np.asarray(pcd.points)
        pts_c = pts - pts.mean(axis=0)
        _, _, Vt = np.linalg.svd(pts_c, full_matrices=False)  # Vt: (3,3)
        projected = pts_c @ Vt.T                               # (N, 3)
        spans = projected.max(axis=0) - projected.min(axis=0)
        return np.sort(spans)[::-1]   # descending: [longest, medium, shortest]

    src_axes = _axis_lengths(source)
    tgt_axes = _axis_lengths(target)

    # Avoid division by zero for degenerate axes
    ratios = []
    for s, t in zip(src_axes, tgt_axes):
        if s > 1e-8:
            ratios.append(t / s)

    if not ratios:
        logging.warning("  [pca]    degenerate cloud — returning 1.0")
        return 1.0

    scale = float(np.mean(ratios))
    logging.info(f"  [pca]    src_axes={src_axes.round(6)}  tgt_axes={tgt_axes.round(6)}"
                 f"  ratios={[round(r,4) for r in ratios]}  → scale={scale:.6f}")
    return scale


def _estimate_scale_volume(source: o3d.geometry.PointCloud,
                           target: o3d.geometry.PointCloud) -> Optional[float]:
    """
    Scale from convex hull volume ratio: scale = cbrt(V_target / V_source).

    Returns None if either hull computation fails (e.g. degenerate/planar cloud).
    """
    try:
        _, src_vol = source.compute_convex_hull()
        _, tgt_vol = target.compute_convex_hull()
        if src_vol <= 0 or tgt_vol <= 0:
            logging.warning("  [volume] zero or negative hull volume — skipping")
            return None
        scale = float((tgt_vol / src_vol) ** (1.0 / 3.0))
        logging.info(f"  [volume] src_vol={src_vol:.6e}  tgt_vol={tgt_vol:.6e}"
                     f"  → scale={scale:.6f}")
        return scale
    except Exception as e:
        logging.warning(f"  [volume] hull failed: {e} — skipping")
        return None


@timeit
def estimate_scale(source: o3d.geometry.PointCloud,
                   target: o3d.geometry.PointCloud,
                   config) -> Tuple[float, np.ndarray]:
    """
    Robust scale estimation using multi-estimator consensus (median).

    Runs bbox diagonal, PCA axis lengths, and convex hull volume estimators
    independently, then takes their median. Logs each individual estimate
    and warns if they disagree by more than 30%.

    Returns:
        (scale, init_transform): scale factor and 4x4 centering transform.
    """
    logging.info("\n" + "="*70)
    logging.info("  SCALE ESTIMATION  (multi-estimator consensus)")
    logging.info("="*70)

    estimates = {}

    # --- estimator 1: bbox diagonal ---
    try:
        estimates['bbox'] = _estimate_scale_bbox(source, target)
    except Exception as e:
        logging.warning(f"  bbox estimator failed: {e}")

    # --- estimator 2: PCA axis lengths ---
    try:
        estimates['pca'] = _estimate_scale_pca(source, target)
    except Exception as e:
        logging.warning(f"  pca estimator failed: {e}")

    # --- estimator 3: convex hull volume ---
    vol = _estimate_scale_volume(source, target)
    if vol is not None:
        estimates['volume'] = vol

    if not estimates:
        logging.warning("  All estimators failed — defaulting scale to 1.0")
        scale = 1.0
    else:
        values = list(estimates.values())
        scale = float(np.median(values))

        # Agreement check
        spread = (max(values) - min(values)) / scale if scale > 0 else 0
        if spread > 0.30:
            logging.warning(
                f"  ⚠️  Estimators disagree (spread={100*spread:.1f}% > 30%). "
                f"Individual values: {estimates}. "
                f"Crop quality may be poor or object has unusual geometry.")
        else:
            logging.info(f"  ✓ Estimators agree (spread={100*spread:.1f}%)")

    logging.info(f"  Individual estimates: {estimates}")
    logging.info(f"  → Consensus scale (median): {scale:.6f}")

    # Build centering transform: scale source, then translate to target centroid
    source_scaled = copy.deepcopy(source)
    source_scaled.scale(scale, center=source_scaled.get_center())
    translation = target.get_center() - source_scaled.get_center()
    init_transform = np.eye(4)
    init_transform[:3, 3] = translation

    logging.info(f"✓ Final scale: {scale:.6f}")
    return scale, init_transform


# ============================================================================
# RANSAC INITIAL ALIGNMENT
# ============================================================================

def try_multiple_rotations(source, target, config):
    logging.info("\n" + "="*70)
    logging.info("  MULTI-ROTATION RANSAC")
    logging.info("="*70)

    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]
    rotations = []
    for rx in angles:
        for ry in [0, np.pi/2, np.pi, 3*np.pi/2]:
            for rz in [0, np.pi/2]:
                R = o3d.geometry.get_rotation_matrix_from_xyz([rx, ry, rz])
                rotations.append(R)

    unique_rotations = []
    for R in rotations:
        if not any(np.allclose(R, R_e, atol=0.01) for R_e in unique_rotations):
            unique_rotations.append(R)

    logging.info(f"Testing {len(unique_rotations)} unique rotations...")

    best_result = None
    best_fitness = -1
    best_rotation = np.eye(3)

    for i, R in enumerate(unique_rotations):
        source_rotated = copy.deepcopy(source)
        center = source_rotated.get_center()
        source_rotated.rotate(R, center=center)
        try:
            source_down = source_rotated.voxel_down_sample(config.RANSAC_DOWNSAMPLE)
            target_down = target.voxel_down_sample(config.RANSAC_DOWNSAMPLE)
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
                logging.info(f"  Rotation {i+1}/{len(unique_rotations)}: fitness={result.fitness:.4f} ✅")
            else:
                logging.debug(f"  Rotation {i+1}: fitness={result.fitness:.4f}")
        except Exception as e:
            logging.debug(f"  Rotation {i+1} failed: {e}")
            continue

    logging.info(f"✓ Best rotation fitness={best_fitness:.4f}")
    return best_result, best_rotation


@timeit
def initial_alignment_ransac(source, target, config):
    logging.info("\n" + "="*70)
    logging.info(f"  RANSAC INITIAL ALIGNMENT ({config.RANSAC_TRIES} attempts)")
    logging.info("="*70)

    source_down = source
    target_down = target.voxel_down_sample(config.RANSAC_DOWNSAMPLE)
    logging.info(f"Source: {len(source.points)} | Target: {len(target_down.points)} pts")

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
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    config.RANSAC_CORRESPONDENCE_CHECKER_DISTANCE)
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
        )
        all_results.append({'attempt': i+1, 'fitness': result.fitness, 'rmse': result.inlier_rmse})
        logging.info(f"  Try {i+1}/{config.RANSAC_TRIES}: fitness={result.fitness:.4f}, RMSE={result.inlier_rmse:.6f}")
        if result.fitness > best_fitness:
            best_fitness = result.fitness
            best_result = result
            logging.info(f"    ✅ NEW BEST!")

    logging.info(f"✓ Best RANSAC: fitness={best_fitness:.4f}, RMSE={best_result.inlier_rmse:.6f}")
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
                config.ICP_MAX_CORRESPONDENCE_DISTANCE * config.ADAPTIVE_ICP_DISTANCE_MULTIPLIER,
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
                    logging.info(f"  ✅ Iter {iteration+1}: fitness={best_fitness:.4f} (+{improvement:.4f}), "
                                 f"RMSE={best_rmse:.6f}")
                    if (best_fitness >= config.ADAPTIVE_FITNESS_THRESHOLD and
                        best_rmse <= config.ADAPTIVE_RMSE_THRESHOLD):
                        logging.info("  🎉 Target reached!")
                        break
            else:
                noise_translation += 0.75
        except Exception as e:
            logging.debug(f"  Iter {iteration+1} error: {e}")
            noise_translation += 0.1

        iteration += 1
        if iteration % 10 == 0:
            logging.info(f"  Progress: iter={iteration}, best_fitness={best_fitness:.4f}")

    final_result = o3d.pipelines.registration.RegistrationResult()
    final_result.fitness = best_fitness
    final_result.inlier_rmse = best_rmse
    final_result.transformation = best_transformation
    logging.info(f"✓ Complete ({iteration} iters) | fitness={best_fitness:.4f}, RMSE={best_rmse:.6f}")
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
                    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                        max_iteration=config.SCALE_ICP_MAX_ITERATIONS)
                )
                rmse = result.inlier_rmse
                fitness = result.fitness
                if fitness == 0 or rmse == 0:
                    logging.info(f"  {label} ({scale:.6f}): FAILED")
                    results.append({'scale': scale, 'rmse': float('inf'), 'fitness': 0.0,
                                    'label': label, 'result': None})
                    continue
                best_marker = "✅" if rmse < best_rmse else ""
                logging.info(f"  {label} ({scale:.6f}): fitness={fitness:.4f}, rmse={rmse:.6f} {best_marker}")
                results.append({'scale': scale, 'rmse': rmse, 'fitness': fitness,
                                'label': label, 'result': result})
                all_tested.append({'iteration': iteration, 'scale': float(scale),
                                   'rmse': float(rmse), 'fitness': float(fitness)})
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_scale = scale
                    best_result = result
            except Exception as e:
                logging.warning(f"  {label} FAILED: {e}")
                results.append({'scale': scale, 'rmse': float('inf'), 'fitness': 0.0,
                                'label': label, 'result': None})

        min_r, mid_r, max_r = results[0], results[1], results[2]
        if scale_range < initial_scale * config.SCALE_CONVERGENCE_THRESHOLD:
            logging.info(f"  ✓ Converged (range ±{100*scale_range/(2*initial_scale):.1f}%)")
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

    improvement_pct = 100 * (initial_result.inlier_rmse - best_rmse) / initial_result.inlier_rmse \
                      if initial_result.inlier_rmse > 0 else 0
    logging.info(f"✓ Best scale: {best_scale:.6f} | RMSE improvement: {improvement_pct:.1f}%")
    return best_scale, best_result, all_tested


# ============================================================================
# MAIN PIPELINE
# ============================================================================

@timeit
def run_pipeline(args):
    start_time = time.perf_counter()
    scene_dir = Path(args.scene_dir)
    output_dir = scene_dir / "icp_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    config = PipelineConfig(args)

    logging.info("\n" + "="*70)
    logging.info("  COMPLETE PIPELINE")
    logging.info("="*70)
    logging.info(f"Scene:  {scene_dir}")
    logging.info(f"Object: {Path(args.object_ply)}")
    logging.info(f"Output: {output_dir}")
    logging.info(f"Crop mode: {'YOLO' if config.YOLO_CROP else 'CLASSIC (plane removal + outliers)'}")
    logging.info("="*70)

    # ========================================================================
    # STAGE 1: RECONSTRUCTION
    # ========================================================================
    scene_ply_path = scene_dir / "sparse" / "points.ply"
    intr_path = scene_dir / "sparse" / "intrinsics.npy"

    if args.skip_reconstruction and scene_ply_path.exists():
        # When YOLO crop is active, check that the saved intrinsics.npy is at
        # the correct (original image) resolution, not the stale 518x518 version.
        # The stale version is a (F, 3, 3) array where K[0,0] (fx) is small
        # (~280 for 518px), while the rescaled version has fx matching the real image.
        if config.YOLO_CROP and intr_path.exists():
            intr_check = np.load(str(intr_path))
            image_paths_check = sorted(glob.glob(
                str(scene_dir / "images" / "*")))
            if image_paths_check:
                from PIL import Image as _PILCheck
                real_w = _PILCheck.open(image_paths_check[0]).size[0]
                saved_fx = float(intr_check[0, 0, 0])
                # If saved fx is suspiciously small (< half of real image width)
                # it's the old 518-resolution version — must re-run reconstruction
                if saved_fx < real_w * 0.3:
                    logging.warning(
                        f"⚠️  Stale intrinsics.npy detected (fx={saved_fx:.1f} but "
                        f"image width={real_w}px). The saved file was generated before "
                        f"the resolution-fix. Re-running reconstruction to regenerate "
                        f"correct intrinsics.")
                    scene_ply_path = run_vggt_reconstruction(
                        scene_dir=scene_dir,
                        conf_threshold=args.conf_thres_value,
                        seed=args.seed,
                        conf_low=args.conf_low_threshold,
                        conf_low_radius=args.conf_low_radius,
                        conf_low_min_neighbors=args.conf_low_min_neighbors,
                    )
                else:
                    logging.info(f"\n✓ Using existing reconstruction: {scene_ply_path} "
                                 f"(intrinsics look correct, fx={saved_fx:.1f})")
            else:
                logging.info(f"\n✓ Using existing reconstruction: {scene_ply_path}")
        else:
            logging.info(f"\n✓ Using existing reconstruction: {scene_ply_path}")
    else:
        scene_ply_path = run_vggt_reconstruction(
            scene_dir=scene_dir,
            conf_threshold=args.conf_thres_value,
            seed=args.seed,
            conf_low=args.conf_low_threshold,
            conf_low_radius=args.conf_low_radius,
            conf_low_min_neighbors=args.conf_low_min_neighbors,
        )

    if config.VISUALIZE_RECONSTRUCTION:
        visualize_pcd(o3d.io.read_point_cloud(str(scene_ply_path)), "Reconstructed Scene (Raw)")

    # ========================================================================
    # STAGE 2a: SCENE PREPROCESSING
    # ========================================================================
    scene_pcd = preprocess_scene(
        pcd_path=scene_ply_path,
        config=config,
        save_path=output_dir / "scene_preprocessed.ply"
    )

    # ========================================================================
    # STAGE 2b: OBJECT CROP — classic or YOLO
    #
    #   Classic (default): preprocess_scene already ran outlier removal +
    #     plane removal + downsample above. Nothing extra needed here.
    #
    #   YOLO: run the 6-stage YOLO 2D→3D projection crop on the
    #     preprocessed cloud, then downsample the result.
    # ========================================================================
    if config.YOLO_CROP:
        logging.info("\n[Crop mode: YOLO]")
        scene_pcd = yolo_crop_scene(
            scene_pcd=scene_pcd,
            scene_dir=scene_dir,
            config=config,
            save_path=output_dir / "scene_yolo_cropped.ply"
        )
        if config.VISUALIZE_PREPROCESSING:
            visualize_pcd(scene_pcd, "After YOLO Crop")
    else:
        logging.info("\n[Crop mode: CLASSIC — outlier removal + plane removal already applied]")

    # ========================================================================
    # STAGE 2c: OBJECT PREPROCESSING
    # ========================================================================
    object_pcd = preprocess_object(
        pcd_path=Path(args.object_ply) if Path(args.object_ply).is_absolute()
                 else scene_dir / args.object_ply,
        config=config
    )

    # ========================================================================
    # AUTO-TUNING (uses the potentially-cropped scene_pcd for better params)
    # ========================================================================
    if args.auto:
        config = compute_adaptive_parameters(object_pcd, scene_pcd, config)

    # ========================================================================
    # STAGE 3: SCALE ESTIMATION  (now uses cropped scene → much more reliable)
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
    if args.auto:
        multi_result, best_rotation = try_multiple_rotations(object_pcd_scaled, scene_pcd, config)
        if multi_result is not None and multi_result.fitness > 0.3:
            object_pcd_scaled.rotate(best_rotation, center=object_pcd_scaled.get_center())
            local_result = multi_result
        else:
            local_result, _ = initial_alignment_ransac(object_pcd_scaled, scene_pcd, config)
    else:
        local_result, _ = initial_alignment_ransac(object_pcd_scaled, scene_pcd, config)

    logging.info("\n🔧 Refining RANSAC with ICP...")
    local_result = o3d.pipelines.registration.registration_icp(
        object_pcd_scaled, scene_pcd,
        config.ICP_MAX_CORRESPONDENCE_DISTANCE,
        local_result.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=config.ICP_MAX_ITERATIONS)
    )
    logging.info(f"  After ICP: fitness={local_result.fitness:.4f}, RMSE={local_result.inlier_rmse:.6f}")

    # ========================================================================
    # STAGE 5: ADAPTIVE REFINEMENT
    # ========================================================================
    if args.auto:
        config.ADAPTIVE_FITNESS_THRESHOLD = 0.95
        config.ADAPTIVE_RMSE_THRESHOLD = config.ADAPTIVE_RMSE_THRESHOLD * 0.5

    final_result = adaptive_refinement(object_pcd_scaled, scene_pcd, local_result, config)

    # ========================================================================
    # STAGE 6: SCALE REFINEMENT  — disabled
    # ========================================================================
    # refine_scale_by_fitness() used ICP RMSE as a quality signal for scale,
    # but RMSE depends on the current transformation, making scale and alignment
    # quality inseparably coupled. The robust multi-estimator consensus in
    # estimate_scale() (median of bbox / PCA / volume) is more reliable and
    # decoupled from alignment quality. Scale refinement is therefore skipped.
    # The function is kept in the codebase for reference / future use.
    scale_history = []   # kept for metrics compatibility

    # ========================================================================
    # FINALIZE
    # ========================================================================
    final_transformation = np.dot(final_result.transformation, scale_transform)
    object_aligned = copy.deepcopy(object_pcd)
    object_aligned.scale(scale, center=object_aligned.get_center())
    object_aligned.transform(final_transformation)

    if config.VISUALIZE_FINAL:
        target_vis = copy.deepcopy(scene_pcd).paint_uniform_color([1, 0, 0])
        aligned_vis = copy.deepcopy(object_aligned).paint_uniform_color([0, 1, 0])
        logging.info("\n🎬 Final Visualization — Red=Scene  Green=Object")
        o3d.visualization.draw_geometries(
            [target_vis, aligned_vis],
            window_name="Final Alignment Result",
            width=1280, height=720
        )

    logging.info("\n💾 Saving results...")
    np.save(output_dir / "transformation.npy", final_transformation)
    np.save(output_dir / "scale.npy", np.array([scale]))
    o3d.io.write_point_cloud(str(output_dir / "object_aligned.ply"), object_aligned)

    metrics = {
        'scale': float(scale),
        'crop_mode': 'yolo' if config.YOLO_CROP else 'classic',
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

    elapsed = time.perf_counter() - start_time
    logging.info("\n" + "="*70)
    logging.info("  PIPELINE COMPLETE")
    logging.info("="*70)
    logging.info(f"⏱️  Total time:      {elapsed:.2f}s")
    logging.info(f"📏 Final scale:     {scale:.6f}")
    logging.info(f"📊 Final fitness:   {final_result.fitness:.4f}")
    logging.info(f"📊 Final RMSE:      {final_result.inlier_rmse:.6f}")
    logging.info(f"📊 Correspondences: {len(final_result.correspondence_set)}")
    if final_result.fitness >= 0.85:
        logging.info("✅ EXCELLENT alignment!")
    elif final_result.fitness >= 0.6:
        logging.info("✓  GOOD alignment")
    else:
        logging.info("⚠️  MODERATE alignment — may need parameter tuning")
    logging.info("="*70)
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
    print(f"   Final RMSE:    {metrics['final_rmse']:.6f}")
    print(f"   Final scale:   {metrics['scale']:.6f}")