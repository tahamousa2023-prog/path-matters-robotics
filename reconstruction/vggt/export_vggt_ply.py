#!/usr/bin/env python3
"""
Export a VGGT reconstruction directly to a colored PLY.

This script:
- loads images from a scene folder
- runs VGGT using the official demo flow
- unprojects depth maps to world-space points
- filters by confidence percentile
- writes a colored point cloud to --out_ply

Expected input:
  <scene>/images/

Example:
  conda run -n vggt python3 /home/AP_PathMatters/vggt/export_vggt_ply.py \
    --image_dir /path/to/scene/images \
    --out_ply /path/to/scene/recon_generated/vggt/points.ply \
    --max_images 120 \
    --stride 1
"""

from __future__ import annotations

import argparse
import glob
import os
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map


VGGT_MODEL_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
VGGT_FIXED_RESOLUTION = 518
IMAGE_LOAD_RESOLUTION = 1024


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir", required=True)
    p.add_argument("--out_ply", required=True)
    p.add_argument("--max_images", type=int, default=120)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max_points", type=int, default=800000)
    p.add_argument(
        "--conf_percentile",
        type=float,
        default=70.0,
        help="Keep points whose confidence is >= this percentile among finite points",
    )
    return p.parse_args()


def choose_dtype_and_device() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        major_cc = torch.cuda.get_device_capability()[0]
        dtype = torch.bfloat16 if major_cc >= 8 else torch.float16
        return "cuda", dtype
    return "cpu", torch.float32


def load_image_paths(image_dir: str, stride: int, max_images: int) -> list[str]:
    paths = sorted(glob.glob(os.path.join(image_dir, "*")))
    if stride > 1:
        paths = paths[::stride]
    if max_images > 0:
        paths = paths[:max_images]
    return paths


def main() -> None:
    args = parse_args()

    image_paths = load_image_paths(args.image_dir, args.stride, args.max_images)
    if len(image_paths) < 2:
        raise RuntimeError(f"Need at least 2 images in {args.image_dir}, got {len(image_paths)}")

    device, dtype = choose_dtype_and_device()
    print(f"[VGGT] device={device}, dtype={dtype}, images={len(image_paths)}", flush=True)

    # Load model exactly in the repo style.
    model = VGGT()
    state_dict = torch.hub.load_state_dict_from_url(VGGT_MODEL_URL, map_location=device)
    model.load_state_dict(state_dict)
    model.eval().to(device)

    # Load images in larger square resolution, then run VGGT at 518 like the official demo.
    images, _original_coords = load_and_preprocess_images_square(image_paths, IMAGE_LOAD_RESOLUTION)
    images = images.to(device)  # (S, 3, H, W)

    if images.ndim != 4 or images.shape[1] != 3:
        raise RuntimeError(f"Unexpected image tensor shape: {tuple(images.shape)}")

    images_vggt = F.interpolate(
        images,
        size=(VGGT_FIXED_RESOLUTION, VGGT_FIXED_RESOLUTION),
        mode="bilinear",
        align_corners=False,
    )
    images_batch = images_vggt[None]  # (1, S, 3, 518, 518)

    amp_ctx = torch.cuda.amp.autocast(dtype=dtype) if device == "cuda" else nullcontext()

    with torch.no_grad():
        with amp_ctx:
            aggregated_tokens_list, ps_idx = model.aggregator(images_batch)

            # Official camera/depth prediction flow
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                pose_enc, images_batch.shape[-2:]
            )
            depth_map, depth_conf = model.depth_head(
                aggregated_tokens_list, images_batch, ps_idx
            )

    # Remove only the batch dimension, matching the official demo.
    extrinsic = extrinsic.squeeze(0).detach().cpu().numpy()   # (S, 3, 4)
    intrinsic = intrinsic.squeeze(0).detach().cpu().numpy()   # (S, 3, 3)
    depth_map = depth_map.squeeze(0).detach().cpu().numpy()   # (S, H, W, 1) or (S, H, W)
    depth_conf = depth_conf.squeeze(0).detach().cpu().numpy() # (S, H, W) or compatible

    print(
        f"[VGGT] extrinsic={extrinsic.shape}, intrinsic={intrinsic.shape}, "
        f"depth_map={depth_map.shape}, depth_conf={depth_conf.shape}",
        flush=True,
    )

    # This utility expects depth of shape (S,H,W,1) or (S,H,W).
    world = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)  # (S, H, W, 3)

    # Colors: use the exact resized images fed into VGGT for spatial consistency.
    rgb = (images_vggt.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    rgb = rgb.transpose(0, 2, 3, 1)  # (S, H, W, 3)

    pts = world.reshape(-1, 3)
    col = rgb.reshape(-1, 3)

    # Confidence flattening: support either (S,H,W) or (S,H,W,1).
    conf = depth_conf
    if conf.ndim == 4 and conf.shape[-1] == 1:
        conf = conf[..., 0]
    conf = conf.reshape(-1)

    finite_mask = np.isfinite(pts).all(axis=1)
    if not np.any(finite_mask):
        raise RuntimeError("No finite points after unprojection")

    conf_thr = np.percentile(conf[finite_mask], args.conf_percentile)
    keep = finite_mask & (conf >= conf_thr)

    pts = pts[keep]
    col = col[keep]

    if len(pts) == 0:
        raise RuntimeError("No points left after confidence filtering")

    if args.max_points > 0 and len(pts) > args.max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pts), size=args.max_points, replace=False)
        pts = pts[idx]
        col = col[idx]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(col.astype(np.float64) / 255.0)

    out_ply = Path(args.out_ply)
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    ok = o3d.io.write_point_cloud(str(out_ply), pcd)
    if not ok:
        raise RuntimeError(f"Failed to write point cloud to {out_ply}")

    print(f"[VGGT] wrote {out_ply}  points={len(pts)}", flush=True)


if __name__ == "__main__":
    main()