#!/usr/bin/env python3
"""Run BUFFER-X on an arbitrary source/target mesh-or-cloud pair.

This script is meant to be executed inside the BUFFER-X conda environment.
It exposes BUFFER-X as a practical CLI for arbitrary .ply/.obj inputs instead
of benchmark datasets only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


def _add_repo_to_path(bufferx_root: Path) -> None:
    root = str(bufferx_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    os.chdir(root)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BUFFER-X pairwise wrapper for arbitrary source/target clouds")
    p.add_argument("--bufferx-root", required=True, help="Path to the BUFFER-X repo root")
    p.add_argument("--src", required=True, help="Source .ply/.obj file")
    p.add_argument("--tgt", required=True, help="Target .ply/.obj file")
    p.add_argument("--out-transform", required=True, help="Output 4x4 transform txt")
    p.add_argument("--out-json", default=None, help="Optional JSON summary output")
    p.add_argument("--out-aligned-src", default=None, help="Optional aligned source .ply output")
    p.add_argument("--experiment-id", default="threedmatch", help="Snapshot experiment id under snapshot/<id>/")
    p.add_argument("--device", default="cuda", help="cuda or cpu")
    p.add_argument("--sample-points", type=int, default=120000, help="Mesh sampling budget when loading meshes")
    p.add_argument("--max-points", type=int, default=30000, help="Maximum downsampled points passed to BUFFER-X")
    p.add_argument("--pose-refine", action="store_true", help="Enable BUFFER-X post-refinement if supported")
    return p.parse_args()


def _load_as_point_cloud(path: Path, sample_points: int):
    import open3d as o3d

    path = Path(path)
    pcd = None
    mesh = None

    try:
        pcd = o3d.io.read_point_cloud(str(path))
    except Exception:
        pcd = None

    if pcd is not None and len(pcd.points) > 0:
        pts = np.asarray(pcd.points)
        mask = np.isfinite(pts).all(axis=1)
        if not np.all(mask):
            clean = o3d.geometry.PointCloud()
            clean.points = o3d.utility.Vector3dVector(pts[mask])
            if pcd.has_colors():
                clean.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[mask])
            pcd = clean
        return pcd, "point_cloud"

    try:
        mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=True)
    except Exception:
        mesh = None

    if mesh is None or len(mesh.vertices) == 0:
        raise RuntimeError(f"Could not load any geometry from: {path}")

    if len(mesh.triangles) > 0:
        n = max(sample_points, min(sample_points, max(len(mesh.vertices), 5000)))
        pcd = mesh.sample_points_uniformly(number_of_points=n)
        return pcd, "mesh_sampled"

    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    if mesh.has_vertex_colors():
        pcd.colors = mesh.vertex_colors
    return pcd, "mesh_vertices"


def _random_subsample(points: np.ndarray, max_points: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    idx = np.random.choice(len(points), size=max_points, replace=False)
    return points[idx]


def _build_cfg(bufferx_root: Path, experiment_id: str, pose_refine: bool):
    from config.threedmatch_config import make_cfg as make_tm_cfg

    cfg = make_tm_cfg(bufferx_root)
    cfg.stage = "test"
    cfg.test.experiment_id = experiment_id
    cfg.test.pose_refine = bool(pose_refine)
    return cfg


def _load_model(cfg, experiment_id: str, device: str):
    import torch
    import torch.nn as nn
    from models.BUFFERX import BufferX

    model = BufferX(cfg)

    for stage in cfg.train.all_stage:
        model_path = Path("snapshot") / experiment_id / stage / "best.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing snapshot: {model_path}")
        state_dict = torch.load(str(model_path), map_location=device)
        new_dict = {k: v for k, v in state_dict.items() if stage in k}
        model_dict = model.state_dict()
        model_dict.update(new_dict)
        model.load_state_dict(model_dict, strict=False)

    model = model.to(device)
    # Keep parity with upstream testing code as much as possible.
    if device.startswith("cuda") and torch.cuda.is_available():
        model = nn.DataParallel(model, device_ids=[0])
    model.eval()
    return model


def _to_numpy_pose(result: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    info: Dict[str, Any] = {}
    pose = result
    if isinstance(result, tuple):
        pose = result[0]
        if len(result) > 1:
            info["raw_extra_outputs"] = len(result) - 1
        if len(result) > 1 and isinstance(result[1], (list, tuple)):
            info["times"] = [float(x) for x in result[1]]
        if len(result) > 2:
            for key, idx in [
                ("num_inliers", 2),
                ("num_mutual_inliers", 3),
                ("num_inlier_ind", 4),
                ("scales_used", 5),
            ]:
                if len(result) > idx:
                    try:
                        info[key] = int(result[idx])
                    except Exception:
                        info[key] = str(result[idx])

    if hasattr(pose, "detach"):
        pose = pose.detach().cpu().numpy()
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (4, 4):
        raise RuntimeError(f"Unexpected pose shape from BUFFER-X: {pose.shape}")
    return pose, info


def main() -> int:
    args = _parse_args()
    bufferx_root = Path(args.bufferx_root).expanduser().resolve()
    _add_repo_to_path(bufferx_root)

    import torch
    import open3d as o3d
    from utils.tools import sphericity_based_voxel_analysis

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA but torch.cuda.is_available() is False")

    src_pcd, src_mode = _load_as_point_cloud(Path(args.src), args.sample_points)
    tgt_pcd, tgt_mode = _load_as_point_cloud(Path(args.tgt), args.sample_points)

    voxel_size, sphericity, is_aligned_to_global_z = sphericity_based_voxel_analysis(src_pcd, tgt_pcd)
    src_ds = src_pcd.voxel_down_sample(voxel_size)
    tgt_ds = tgt_pcd.voxel_down_sample(voxel_size)

    src_pts = np.asarray(src_ds.points, dtype=np.float32)
    tgt_pts = np.asarray(tgt_ds.points, dtype=np.float32)
    if len(src_pts) == 0 or len(tgt_pts) == 0:
        raise RuntimeError("Downsampling produced an empty cloud; manual preprocessing is likely still needed")

    src_pts = _random_subsample(src_pts, args.max_points)
    tgt_pts = _random_subsample(tgt_pts, args.max_points)

    cfg = _build_cfg(bufferx_root, args.experiment_id, args.pose_refine)
    model = _load_model(cfg, args.experiment_id, args.device)

    sample = {
        "src_fds_pcd": torch.from_numpy(src_pts).to(args.device),
        "tgt_fds_pcd": torch.from_numpy(tgt_pts).to(args.device),
        "is_aligned_to_global_z": bool(is_aligned_to_global_z),
        # Harmless metadata in case local repo prints ids.
        "src_id": str(args.src),
        "tgt_id": str(args.tgt),
    }

    with torch.no_grad():
        result = model(sample)
    pose, extra = _to_numpy_pose(result)

    out_tf = Path(args.out_transform)
    out_tf.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_tf, pose, fmt="%.10f")

    if args.out_aligned_src:
        aligned = o3d.geometry.PointCloud(src_pcd)
        aligned.transform(pose)
        out_aligned = Path(args.out_aligned_src)
        out_aligned.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(out_aligned), aligned)

    summary = {
        "bufferx_root": str(bufferx_root),
        "experiment_id": args.experiment_id,
        "src": str(Path(args.src).resolve()),
        "tgt": str(Path(args.tgt).resolve()),
        "src_load_mode": src_mode,
        "tgt_load_mode": tgt_mode,
        "src_points_after_voxel": int(len(src_pts)),
        "tgt_points_after_voxel": int(len(tgt_pts)),
        "voxel_size": float(voxel_size),
        "sphericity": float(sphericity),
        "is_aligned_to_global_z": bool(is_aligned_to_global_z),
        "transform_path": str(out_tf.resolve()),
        **extra,
    }

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
