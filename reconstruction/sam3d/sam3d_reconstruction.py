# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
SAM 3D Objects Reconstruction + ICP Alignment Pipeline

This pipeline reconstructs a 3D object from a SINGLE IMAGE using SAM 3D Objects,
then aligns a known 3D CAD model to the reconstructed object using ICP.

MASK OPTIONS:
- Manual mask: --mask_path /path/to/mask.png
- SAM masks directory: --mask_dir /path/to/masks/ --mask_index 0
- Auto-generate (largest): --auto_mask largest --sam_checkpoint /path/to/sam.pth
- Auto-generate (interactive): --auto_mask select --sam_checkpoint /path/to/sam.pth  
- Auto-generate (by index): --auto_mask all --mask_index N --sam_checkpoint /path/to/sam.pth
- Full image mask: --full_image_mask

Anleitung / Usage:
#### With manual mask
python sam3d_icp_pipeline.py \
    --image_path /path/to/image.png \
    --mask_path /path/to/mask.png \
    --object_ply /path/to/cad_model.ply

#### With automatic mask generation (largest object)
python sam3d_icp_pipeline.py \
    --image_path /path/to/image.png \
    --auto_mask largest \
    --sam_checkpoint /path/to/sam_vit_h_4b8939.pth \
    --object_ply /path/to/cad_model.ply

#### With interactive mask selection (shows all masks, you pick one)
python sam3d_icp_pipeline.py \
    --image_path /path/to/image.png \
    --auto_mask select \
    --sam_checkpoint /path/to/sam_vit_h_4b8939.pth \
    --object_ply /path/to/cad_model.ply

#### With full image as mask (single object scenes)
python sam3d_icp_pipeline.py \
    --image_path /path/to/image.png \
    --full_image_mask \
    --object_ply /path/to/cad_model.ply

### Example usage:
python sam3d_reconstruction.py \
    --image_path /home/AP_PathMatters/path_matters/datasets/yoda/images/view_00.png \
    --auto_mask largest \
    --sam_checkpoint /home/AP_PathMatters/Downloads/sam_vit_h_4b8939.pth \
    --object_ply /home/AP_PathMatters/path_matters/datasets/yoda/Baby_Yoda.ply \
    --output_dir /home/AP_PathMatters/path_matters/datasets/yoda/sam3d_results \
    --visualize_reconstruction \
    --visualize_final \
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
#            | 1) SAM3D RECONSTRUCT |
#            |  - load image+mask   |
#            |  - run SAM 3D Objects|
#            |  - export splat.ply  |
#            +----------+-----------+
#                       |
#                       v
#            +----------------------+
#            | 2) PREPROCESSING     |
#            +----------+-----------+
#                       |
#        +--------------+--------------+
#        |                             |
#        v                             v
# +--------------+              +--------------+
# | 2a) SCENE    |              | 2b) OBJECT   |
# |  - outliers  |              |  - downsample|
# |  - downsample|              |  - normals   |
# +------+-------+              +------+-------+
#        |                             |
#        +--------------+--------------+
#                       |
#                       v
#            +----------------------+
#            | 3) SCALE ESTIMATION  |
#            |  - bbox diagonal     |
#            |  - scale & center    |
#            +----------+-----------+
#                       |
#                       v
#            +----------------------+
#            | 4) RANSAC + ICP      |
#            |  - FPFH features     |
#            |  - multi RANSAC      |
#            |  - local ICP refine  |
#            +----------+-----------+
#                       |
#                       v
#            +----------------------+
#            | 5) ADAPTIVE REFINE   |
#            |  - random noise      |
#            |  - ICP loop          |
#            |  - improve fit/RMSE  |
#            +----------+-----------+
#                       |
#                       v
#            +-------------------------------+
#            | 6) ADAPTIVE SCALE REFINE      |
#            |  - start [0.5x, 1.5x]         |
#            |  - test min/mid/max (RMSE)    |
#            |  - shrink range iteratively   |
#            |  - stop when +/-5% range      |
#            +----------+--------------------+
#                       |
#                       v
#            +----------------------+
#            |   FINALIZE & SAVE    |
#            |  - combine transforms|
#            |  - apply scale       |
#            |  - visualize         |
#            |  - save .npy/.ply    |
#            +----------+-----------+
#                       |
#                       v
#            +----------------------+
#            |         END          |
#            +----------------------+


"""
SAM 3D Objects Reconstruction + ICP Alignment Pipeline with Adaptive Scale Refinement

This pipeline reconstructs a 3D object from a single image using SAM 3D Objects,
then aligns a known 3D CAD model to the reconstructed object using ICP.

Main stages:
1. SAM 3D Objects Reconstruction - Convert single image + mask to 3D Gaussian Splat
2. Preprocessing - Clean up point clouds (remove noise, downsample)
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
import sys
from pathlib import Path
from typing import Tuple, Optional, Dict, List

import open3d as o3d
import torch
import trimesh
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Configure CUDA
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


# ============================================================================
# DEFAULT PARAMETER VALUES
# ============================================================================

# SAM 3D OBJECTS RECONSTRUCTION PARAMETERS
DEFAULT_SEED = 42
DEFAULT_CONFIG_TAG = "hf"  # HuggingFace checkpoint tag
DEFAULT_COMPILE_MODEL = False  # Whether to compile model for faster inference

# SCENE PREPROCESSING PARAMETERS (for reconstructed object)
DEFAULT_SCENE_DOWNSAMPLE_VOXEL = 0.005
DEFAULT_SCENE_OUTLIER_NEIGHBORS = 30
DEFAULT_SCENE_OUTLIER_STD_RATIO = 3.0
DEFAULT_SCENE_NORMAL_RADIUS = 0.1
DEFAULT_SCENE_NORMAL_MAX_NN = 30

# OBJECT PREPROCESSING PARAMETERS (for CAD model)
DEFAULT_OBJECT_DOWNSAMPLE_VOXEL = 0.01
DEFAULT_OBJECT_NORMAL_RADIUS = 0.1
DEFAULT_OBJECT_NORMAL_MAX_NN = 30

# SCALE ESTIMATION PARAMETERS
DEFAULT_SCALE_METHOD = "bbox"

# RANSAC ALIGNMENT PARAMETERS
DEFAULT_RANSAC_TRIES = 20
DEFAULT_RANSAC_DOWNSAMPLE_VOXEL = 0.01
DEFAULT_RANSAC_MAX_CORRESPONDENCE_DISTANCE = 0.1
DEFAULT_RANSAC_CORRESPONDENCE_CHECKER_DISTANCE = 0.1
DEFAULT_RANSAC_NORMAL_RADIUS = 0.5
DEFAULT_RANSAC_NORMAL_MAX_NN = 100
DEFAULT_RANSAC_FPFH_RADIUS = 0.5
DEFAULT_RANSAC_FPFH_MAX_NN = 50

# ICP REFINEMENT PARAMETERS
DEFAULT_ICP_MAX_CORRESPONDENCE_DISTANCE = 0.05
DEFAULT_ICP_MAX_ITERATIONS = 500
DEFAULT_ADAPTIVE_ICP_DISTANCE_MULTIPLIER = 0.5

# ADAPTIVE REFINEMENT PARAMETERS
DEFAULT_ADAPTIVE_MAX_ITERATIONS = 50
DEFAULT_ADAPTIVE_FITNESS_THRESHOLD = 0.90
DEFAULT_ADAPTIVE_RMSE_THRESHOLD = 0.01
DEFAULT_ADAPTIVE_ROTATION_NOISE_RANGE = 0.05
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
        description="SAM 3D Objects Reconstruction + ICP Alignment Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # REQUIRED - Image
    parser.add_argument("--image_path", type=str, required=True,
                       help="Path to input image")
    parser.add_argument("--object_ply", type=str, required=True,
                       help="Path to reference CAD model PLY file for alignment")
    
    # Mask options (manual or automatic)
    mask_group = parser.add_mutually_exclusive_group(required=False)
    mask_group.add_argument("--mask_path", type=str,
                           help="Direct path to mask image (binary mask)")
    mask_group.add_argument("--mask_dir", type=str,
                           help="Directory containing SAM masks (use with --mask_index)")
    mask_group.add_argument("--auto_mask", type=str, choices=["largest", "all", "select"],
                           help="Auto-generate masks: 'largest'=use largest, 'all'=show all and pick, 'select'=interactive")
    mask_group.add_argument("--full_image_mask", action="store_true",
                           help="Use entire image as mask (for single object scenes)")
    parser.add_argument("--mask_index", type=int, default=0,
                       help="Index of mask to use from mask_dir or auto-generated masks (default: 0)")
    
    # SAM model for auto mask generation
    parser.add_argument("--sam_checkpoint", type=str, default=None,
                       help="Path to SAM checkpoint (e.g., sam_vit_h_4b8939.pth)")
    parser.add_argument("--sam_model_type", type=str, default="vit_h",
                       choices=["vit_h", "vit_l", "vit_b"],
                       help="SAM model type (default: vit_h)")
    parser.add_argument("--min_mask_area", type=int, default=1000,
                       help="Minimum mask area in pixels for auto mask (default: 1000)")
    parser.add_argument("--show_all_masks", action="store_true",
                       help="Visualize all generated masks before selecting")
    parser.add_argument("--invert_mask", action="store_true",
                       help="Invert the mask (swap foreground/background). Use when object is black and background is white.")
    
    # Output
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (default: same as image directory)")
    
    # SAM 3D OBJECTS RECONSTRUCTION
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--config_tag", type=str, default=DEFAULT_CONFIG_TAG,
                       help=f"Checkpoint tag (default: {DEFAULT_CONFIG_TAG})")
    parser.add_argument("--sam3d_path", type=str, default=None,
                       help="Path to SAM 3D Objects repository (if not in PYTHONPATH)")
    parser.add_argument("--compile_model", action="store_true",
                       help="Compile model for faster inference")
    parser.add_argument("--skip_reconstruction", action="store_true",
                       help="Skip reconstruction if splat.ply already exists")
    
    # SCENE PREPROCESSING (reconstructed object)
    parser.add_argument("--scene_downsample", type=float, default=DEFAULT_SCENE_DOWNSAMPLE_VOXEL)
    parser.add_argument("--scene_outlier_neighbors", type=int, default=DEFAULT_SCENE_OUTLIER_NEIGHBORS)
    parser.add_argument("--scene_outlier_std", type=float, default=DEFAULT_SCENE_OUTLIER_STD_RATIO)
    
    # OBJECT PREPROCESSING (CAD model)
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
        # SAM 3D Configuration
        self.CONFIG_TAG = args.config_tag
        self.COMPILE_MODEL = args.compile_model
        self.SEED = args.seed
        
        # Scene Preprocessing Configuration (reconstructed object)
        self.SCENE_DOWNSAMPLE_VOXEL = args.scene_downsample
        self.SCENE_OUTLIER_NEIGHBORS = args.scene_outlier_neighbors
        self.SCENE_OUTLIER_STD = args.scene_outlier_std
        self.SCENE_NORMAL_RADIUS = DEFAULT_SCENE_NORMAL_RADIUS
        self.SCENE_NORMAL_MAX_NN = DEFAULT_SCENE_NORMAL_MAX_NN
        
        # Object Preprocessing Configuration (CAD model)
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
# AUTOMATIC MASK GENERATION
# ============================================================================

def show_mask(mask, ax, color=None, alpha=0.5):
    """Display a mask on matplotlib axis."""
    if color is None:
        color = np.array([30/255, 144/255, 255/255, alpha])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_masks_on_image(image, masks, scores=None, title="Masks", save_path=None):
    """Display all masks overlaid on image with different colors."""
    plt.figure(figsize=(16, 10))
    plt.imshow(image)
    
    # Generate distinct colors for each mask
    colors = plt.cm.tab20(np.linspace(0, 1, len(masks)))
    
    for idx, mask in enumerate(masks):
        color = colors[idx]
        color_with_alpha = np.array([color[0], color[1], color[2], 0.5])
        show_mask(mask, plt.gca(), color=color_with_alpha)
        
        # Add label with index and score
        # Find mask center for label placement
        y_indices, x_indices = np.where(mask)
        if len(y_indices) > 0:
            center_y, center_x = np.mean(y_indices), np.mean(x_indices)
            score_text = f" ({scores[idx]:.2f})" if scores is not None else ""
            plt.text(center_x, center_y, f"{idx}{score_text}", 
                    fontsize=12, color='white', fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor=color[:3], alpha=0.8))
    
    plt.title(f"{title} - {len(masks)} masks found")
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        logging.info(f"✓ Saved masks visualization: {save_path}")
    
    plt.show()
    plt.close()


def show_single_mask(image, mask, title="Selected Mask", save_path=None):
    """Display a single mask on the image."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Mask only
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title(f"Mask (area: {mask.sum()} pixels)")
    axes[1].axis('off')
    
    # Image with mask overlay
    axes[2].imshow(image)
    show_mask(mask, axes[2], color=np.array([0, 1, 0, 0.5]))
    axes[2].set_title("Image + Mask Overlay")
    axes[2].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        logging.info(f"✓ Saved mask visualization: {save_path}")
    
    plt.show()
    plt.close()


@timeit
def generate_masks_with_sam(image_path: Path, sam_checkpoint: str, model_type: str = "vit_h",
                             min_area: int = 1000, device: str = "cuda") -> Tuple[List[np.ndarray], List[float]]:
    """
    Generate masks using Segment Anything Model (SAM).
    
    Args:
        image_path: Path to input image
        sam_checkpoint: Path to SAM checkpoint file
        model_type: SAM model type (vit_h, vit_l, vit_b)
        min_area: Minimum mask area in pixels
        device: Device to run on
        
    Returns:
        Tuple of (masks list, scores list) sorted by area (largest first)
    """
    logging.info("\n" + "="*70)
    logging.info("  AUTOMATIC MASK GENERATION (SAM)")
    logging.info("="*70)
    
    # Import SAM
    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    except ImportError:
        logging.error("segment_anything not installed!")
        logging.error("Install with: pip install segment-anything")
        logging.error("Download checkpoint from: https://github.com/facebookresearch/segment-anything#model-checkpoints")
        raise
    
    # Load image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    logging.info(f"Loaded image: {image_path} ({image.shape})")
    
    # Load SAM model
    logging.info(f"Loading SAM model: {model_type} from {sam_checkpoint}")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    
    # Generate masks
    logging.info("Generating masks (this may take a moment)...")
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=min_area,
    )
    
    masks_data = mask_generator.generate(image)
    logging.info(f"Generated {len(masks_data)} masks")
    
    # Filter by minimum area and sort by area (largest first)
    masks_data = [m for m in masks_data if m['area'] >= min_area]
    masks_data = sorted(masks_data, key=lambda x: x['area'], reverse=True)
    
    logging.info(f"After filtering (min_area={min_area}): {len(masks_data)} masks")
    
    # Extract masks and scores
    masks = [m['segmentation'].astype(np.uint8) for m in masks_data]
    scores = [m['predicted_iou'] for m in masks_data]
    areas = [m['area'] for m in masks_data]
    
    # Log mask info
    for i, (score, area) in enumerate(zip(scores, areas)):
        logging.info(f"  Mask {i}: area={area:,} pixels, score={score:.3f}")
    
    return masks, scores, image


def create_full_image_mask(image_path: Path) -> np.ndarray:
    """Create a mask covering the entire image."""
    image = cv2.imread(str(image_path))
    h, w = image.shape[:2]
    mask = np.ones((h, w), dtype=np.uint8)
    logging.info(f"Created full image mask: {w}x{h}")
    return mask


def select_mask_interactive(masks: List[np.ndarray], scores: List[float], image: np.ndarray) -> int:
    """Let user select a mask interactively."""
    print("\n" + "="*50)
    print("MASK SELECTION")
    print("="*50)
    print(f"Found {len(masks)} masks:")
    for i, (mask, score) in enumerate(zip(masks, scores)):
        area = mask.sum()
        print(f"  [{i}] Area: {area:,} pixels, Score: {score:.3f}")
    print("\nEnter mask index to use (or 'q' to quit): ", end="")
    
    while True:
        try:
            user_input = input().strip()
            if user_input.lower() == 'q':
                raise KeyboardInterrupt("User cancelled mask selection")
            idx = int(user_input)
            if 0 <= idx < len(masks):
                return idx
            else:
                print(f"Invalid index. Enter 0-{len(masks)-1}: ", end="")
        except ValueError:
            print("Invalid input. Enter a number: ", end="")


@timeit  
def get_mask(args, output_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get mask based on arguments - either from file, auto-generated, or full image.
    
    Returns:
        Tuple of (mask, original_image)
    """
    image_path = Path(args.image_path)
    
    # Load original image for visualization
    original_image = cv2.imread(str(image_path))
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Option 1: Direct mask path
    if args.mask_path:
        logging.info(f"Loading mask from: {args.mask_path}")
        mask = np.array(Image.open(args.mask_path).convert('L'))
        if mask.max() > 1:
            mask = (mask > 127).astype(np.uint8)
        return mask, original_image
    
    # Option 2: Mask directory with index
    if args.mask_dir:
        logging.info(f"Loading mask from directory: {args.mask_dir}")
        # Try to use SAM 3D's load_single_mask if available
        try:
            sys.path.insert(0, "notebook")
            from inference import load_single_mask
            mask = load_single_mask(args.mask_dir, index=args.mask_index)
        except:
            # Fallback: load PNG files from directory
            mask_files = sorted(glob.glob(os.path.join(args.mask_dir, "*.png")))
            if args.mask_index < len(mask_files):
                mask = np.array(Image.open(mask_files[args.mask_index]).convert('L'))
                if mask.max() > 1:
                    mask = (mask > 127).astype(np.uint8)
            else:
                raise ValueError(f"Mask index {args.mask_index} out of range (found {len(mask_files)} masks)")
        return mask, original_image
    
    # Option 3: Full image mask
    if args.full_image_mask:
        logging.info("Using full image as mask")
        mask = create_full_image_mask(image_path)
        
        # Show the mask
        show_single_mask(original_image, mask, "Full Image Mask", 
                        save_path=output_dir / "mask_selected.png")
        return mask, original_image
    
    # Option 4: Auto-generate masks with SAM
    if args.auto_mask:
        if not args.sam_checkpoint:
            # Try to find SAM checkpoint automatically
            possible_paths = [
                "sam_vit_h_4b8939.pth",
                "checkpoints/sam_vit_h_4b8939.pth",
                os.path.expanduser("~/sam_vit_h_4b8939.pth"),
                "/home/AP_PathMatters/sam_vit_h_4b8939.pth",
            ]
            for p in possible_paths:
                if os.path.exists(p):
                    args.sam_checkpoint = p
                    break
            
            if not args.sam_checkpoint:
                raise ValueError(
                    "SAM checkpoint not found! Please specify --sam_checkpoint path.\n"
                    "Download from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
                )
        
        # Generate masks
        device = "cuda" if torch.cuda.is_available() else "cpu"
        masks, scores, image_rgb = generate_masks_with_sam(
            image_path, args.sam_checkpoint, args.sam_model_type,
            args.min_mask_area, device
        )
        
        if len(masks) == 0:
            raise ValueError("No masks generated! Try lowering --min_mask_area")
        
        # Save all masks visualization
        masks_viz_path = output_dir / "masks_all.png"
        show_masks_on_image(image_rgb, masks, scores, 
                          title=f"Auto-generated Masks ({len(masks)} found)",
                          save_path=masks_viz_path)
        
        # Select mask based on mode
        if args.auto_mask == "largest":
            # Use largest mask (already sorted by area)
            selected_idx = 0
            logging.info(f"Auto-selecting largest mask (index 0)")
            
        elif args.auto_mask == "select":
            # Interactive selection
            selected_idx = select_mask_interactive(masks, scores, image_rgb)
            logging.info(f"User selected mask index: {selected_idx}")
            
        elif args.auto_mask == "all":
            # Show all and let user pick via --mask_index
            selected_idx = args.mask_index
            if selected_idx >= len(masks):
                logging.warning(f"Mask index {selected_idx} out of range, using 0")
                selected_idx = 0
            logging.info(f"Using mask index: {selected_idx}")
        
        mask = masks[selected_idx]
        
        # Save individual masks to directory
        masks_dir = output_dir / "masks"
        masks_dir.mkdir(exist_ok=True)
        for i, m in enumerate(masks):
            mask_path = masks_dir / f"mask_{i:03d}.png"
            Image.fromarray((m * 255).astype(np.uint8)).save(mask_path)
        logging.info(f"✓ Saved {len(masks)} individual masks to {masks_dir}")
        
        # Show selected mask
        show_single_mask(image_rgb, mask, 
                        f"Selected Mask (index {selected_idx}, score {scores[selected_idx]:.3f})",
                        save_path=output_dir / "mask_selected.png")
        
        return mask, image_rgb
    
    # No mask option specified - show help
    raise ValueError(
        "No mask specified! Use one of:\n"
        "  --mask_path /path/to/mask.png\n"
        "  --mask_dir /path/to/masks/ --mask_index 0\n"
        "  --auto_mask largest (auto-generate and use largest)\n"
        "  --auto_mask select (auto-generate and pick interactively)\n"
        "  --auto_mask all --mask_index N (auto-generate and use index N)\n"
        "  --full_image_mask (use entire image as mask)"
    )


def apply_mask_inversion(mask: np.ndarray, invert: bool, output_dir: Path, original_image: np.ndarray) -> np.ndarray:
    """
    Apply mask inversion if needed. 
    SAM 3D expects: white (1) = object, black (0) = background
    
    Args:
        mask: Binary mask
        invert: Whether to invert
        output_dir: For saving visualization
        original_image: For visualization
        
    Returns:
        Potentially inverted mask
    """
    # Log mask statistics before inversion
    total_pixels = mask.size
    object_pixels = mask.sum()
    object_percent = 100 * object_pixels / total_pixels
    
    logging.info(f"  Mask stats (before): {object_pixels:,} white pixels ({object_percent:.1f}%)")
    
    # Warn if mask looks inverted (object should typically be smaller than background)
    if object_percent > 70 and not invert:
        logging.warning(f"⚠️  Mask is {object_percent:.1f}% white - this might be inverted!")
        logging.warning("   Consider adding --invert_mask flag if reconstruction looks wrong")
    
    if invert:
        logging.info("🔄 Inverting mask (swapping foreground/background)")
        mask = 1 - mask
        
        # Recalculate stats
        object_pixels = mask.sum()
        object_percent = 100 * object_pixels / total_pixels
        logging.info(f"  Mask stats (after): {object_pixels:,} white pixels ({object_percent:.1f}%)")
        
        # Show the inverted mask
        show_single_mask(original_image, mask, "Inverted Mask (white=object)", 
                        save_path=output_dir / "mask_inverted.png")
    
    return mask


# ============================================================================
# SAM 3D OBJECTS RECONSTRUCTION
# ============================================================================

@timeit
def run_sam3d_reconstruction(image_path: Path, mask: np.ndarray, output_dir: Path,
                              config: PipelineConfig, sam3d_path: Optional[str] = None) -> Path:
    """
    Run SAM 3D Objects reconstruction to convert single image + mask into 3D Gaussian Splat.
    
    Args:
        image_path: Path to input image
        mask: Binary mask numpy array
        output_dir: Output directory for results
        config: Pipeline configuration
        sam3d_path: Optional path to SAM 3D Objects repository
        
    Returns:
        Path to the saved point cloud PLY file
    """
    
    logging.info("\n" + "="*70)
    logging.info("  SAM 3D OBJECTS RECONSTRUCTION")
    logging.info("="*70)
    
    # Set random seeds
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    random.seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)
    
    # Add SAM 3D Objects to path if specified
    if sam3d_path:
        sys.path.insert(0, sam3d_path)
        sys.path.insert(0, os.path.join(sam3d_path, "notebook"))
    
    # Import SAM 3D Objects modules
    try:
        from inference import Inference, load_image
    except ImportError as e:
        logging.error(f"Failed to import SAM 3D Objects modules: {e}")
        logging.error("Please ensure SAM 3D Objects is installed and in PYTHONPATH")
        logging.error("Or specify --sam3d_path /path/to/sam-3d-objects")
        raise
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")
    
    # Load model
    logging.info(f"Loading SAM 3D Objects model (tag: {config.CONFIG_TAG})...")
    config_path = f"checkpoints/{config.CONFIG_TAG}/pipeline.yaml"
    inference = Inference(config_path, compile=config.COMPILE_MODEL)
    
    # Load image
    logging.info(f"Loading image: {image_path}")
    image = load_image(str(image_path))
    
    logging.info(f"  Image shape: {np.array(image).shape if hasattr(image, '__array__') else 'PIL Image'}")
    logging.info(f"  Mask shape: {mask.shape}, unique values: {np.unique(mask)}")
    
    # Run SAM 3D Objects inference
    logging.info("Running SAM 3D Objects inference...")
    with torch.no_grad():
        output = inference(image, mask, seed=config.SEED)
    
    logging.info("Inference completed!")
    
    # Create output directory
    sam3d_output_dir = output_dir / "sam3d_sparse"
    sam3d_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save Gaussian Splat as PLY
    splat_ply = sam3d_output_dir / "splat.ply"
    output["gs"].save_ply(str(splat_ply))
    logging.info(f"✓ Saved Gaussian Splat: {splat_ply}")
    
    # Also convert to standard point cloud for ICP
    # The Gaussian Splat PLY contains positions that we can use
    points_ply = sam3d_output_dir / "points.ply"
    
    # Read the splat PLY and extract just the positions
    try:
        splat_pcd = o3d.io.read_point_cloud(str(splat_ply))
        if len(splat_pcd.points) > 0:
            o3d.io.write_point_cloud(str(points_ply), splat_pcd)
            logging.info(f"✓ Saved point cloud: {points_ply} ({len(splat_pcd.points)} points)")
        else:
            # If direct read doesn't work, try trimesh
            mesh = trimesh.load(str(splat_ply))
            if hasattr(mesh, 'vertices'):
                points = np.array(mesh.vertices)
                colors = np.array(mesh.colors)[:, :3] / 255.0 if hasattr(mesh, 'colors') else None
                
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                if colors is not None:
                    pcd.colors = o3d.utility.Vector3dVector(colors)
                
                o3d.io.write_point_cloud(str(points_ply), pcd)
                logging.info(f"✓ Saved point cloud: {points_ply} ({len(points)} points)")
    except Exception as e:
        logging.warning(f"Could not convert splat to point cloud: {e}")
        points_ply = splat_ply
    
    # Save metadata
    metadata = {
        'image_path': str(image_path),
        'mask_shape': list(mask.shape),
        'mask_area_pixels': int(mask.sum()),
        'seed': config.SEED,
        'config_tag': config.CONFIG_TAG
    }
    with open(sam3d_output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return points_ply


# ============================================================================
# PREPROCESSING
# ============================================================================

@timeit
def preprocess_scene(pcd_path: Path, config: PipelineConfig, save_path: Optional[Path] = None):
    """Preprocess reconstructed object point cloud."""
    
    logging.info("\n" + "="*70)
    logging.info("  SCENE PREPROCESSING (SAM 3D Reconstruction)")
    logging.info("="*70)
    
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    original_count = len(pcd.points)
    logging.info(f"Loaded: {original_count} points")
    
    if original_count == 0:
        raise ValueError("No points in reconstructed point cloud!")
    
    if config.VISUALIZE_PREPROCESSING:
        visualize_pcd(pcd, "Original Reconstruction")
    
    # Remove outliers
    logging.info("Removing outliers...")
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=config.SCENE_OUTLIER_NEIGHBORS,
        std_ratio=config.SCENE_OUTLIER_STD
    )
    logging.info(f"  {original_count} → {len(pcd.points)} points")
    
    if config.VISUALIZE_PREPROCESSING:
        visualize_pcd(pcd, "After Outlier Removal")
    
    # Downsample
    logging.info(f"Downsampling (voxel={config.SCENE_DOWNSAMPLE_VOXEL})...")
    pcd = pcd.voxel_down_sample(voxel_size=config.SCENE_DOWNSAMPLE_VOXEL)
    logging.info(f"  → {len(pcd.points)} points")
    
    # Estimate normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=config.SCENE_NORMAL_RADIUS,
            max_nn=config.SCENE_NORMAL_MAX_NN
        )
    )
    
    if config.VISUALIZE_PREPROCESSING:
        visualize_pcd(pcd, "Final Preprocessed Reconstruction")
    
    if save_path:
        o3d.io.write_point_cloud(str(save_path), pcd)
        logging.info(f"✓ Saved: {save_path}")
    
    logging.info(f"✓ Complete: {original_count} → {len(pcd.points)} points")
    return pcd


@timeit
def preprocess_object(pcd_path: Path, config: PipelineConfig):
    """Preprocess reference CAD model point cloud."""
    
    logging.info("\n" + "="*70)
    logging.info("  OBJECT PREPROCESSING (CAD Model)")
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
    
    image_path = Path(args.image_path)
    object_ply_path = Path(args.object_ply)
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = image_path.parent / "sam3d_icp_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = PipelineConfig(args)
    
    logging.info("\n" + "="*70)
    logging.info("  SAM 3D OBJECTS + ICP COMPLETE PIPELINE")
    logging.info("="*70)
    logging.info(f"Image: {image_path}")
    logging.info(f"CAD Model: {object_ply_path}")
    logging.info(f"Output: {output_dir}")
    
    # STAGE 0: GET OR GENERATE MASK
    mask, original_image = get_mask(args, output_dir)
    
    # Apply mask inversion if needed
    mask = apply_mask_inversion(mask, args.invert_mask, output_dir, original_image)
    
    logging.info(f"Mask ready: shape={mask.shape}, area={mask.sum()} pixels")
    
    # STAGE 1: RECONSTRUCTION
    scene_ply_path = output_dir / "sam3d_sparse" / "points.ply"
    
    if args.skip_reconstruction and scene_ply_path.exists():
        logging.info(f"\n✓ Using existing reconstruction: {scene_ply_path}")
    else:
        scene_ply_path = run_sam3d_reconstruction(
            image_path=image_path,
            mask=mask,
            output_dir=output_dir,
            config=config,
            sam3d_path=args.sam3d_path
        )
    
    if config.VISUALIZE_RECONSTRUCTION:
        pcd_raw = o3d.io.read_point_cloud(str(scene_ply_path))
        visualize_pcd(pcd_raw, "SAM 3D Reconstruction (Raw)")
    
    # STAGE 2: PREPROCESSING
    scene_pcd = preprocess_scene(scene_ply_path, config, output_dir / "scene_preprocessed.ply")
    object_pcd = preprocess_object(object_ply_path, config)
    
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
        logging.info("\n🎬 Final Visualization (Red=Reconstruction, Green=CAD Model)")
        o3d.visualization.draw_geometries([target_vis, aligned_vis], window_name="Final Alignment", width=1280, height=720)
    
    # Save results
    logging.info("\n💾 Saving results...")
    np.save(output_dir / "transformation.npy", final_transformation)
    np.save(output_dir / "scale.npy", np.array([scale]))
    o3d.io.write_point_cloud(str(output_dir / "object_aligned.ply"), object_aligned)
    
    metrics = {
        'image_path': str(image_path),
        'mask_mode': args.auto_mask if args.auto_mask else ('full_image' if args.full_image_mask else 'manual'),
        'mask_area_pixels': int(mask.sum()),
        'object_ply': str(object_ply_path),
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