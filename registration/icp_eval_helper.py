#!/usr/bin/env python3
"""ICP refinement + evaluation helper.

Run this inside an environment that has Open3D.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import open3d as o3d


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ICP refinement and evaluation helper")
    p.add_argument("--src", required=True, help="Source .ply/.obj file")
    p.add_argument("--tgt", required=True, help="Target .ply/.obj file")
    p.add_argument("--init-transform", required=True, help="Initial 4x4 transform txt")
    p.add_argument("--out-transform", required=True, help="Output ICP 4x4 transform txt")
    p.add_argument("--out-json", required=True, help="Output metrics JSON")
    p.add_argument("--out-aligned-src", required=True, help="Output aligned source .ply")
    p.add_argument("--sample-points", type=int, default=120000, help="Mesh sampling budget")
    p.add_argument("--downsample-voxel", type=float, default=0.0, help="ICP voxel size; 0 means auto")
    p.add_argument("--max-correspondence", type=float, default=0.0, help="ICP max correspondence; 0 means auto")
    p.add_argument(
        "--icp-mode",
        choices=["point_to_plane", "point_to_point"],
        default="point_to_plane",
        help="ICP variant",
    )
    return p.parse_args()


def load_geometry(path: Path, sample_points: int):
    pcd = o3d.io.read_point_cloud(str(path))
    if len(pcd.points) > 0:
        return pcd, None, "point_cloud"

    mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=True)
    if len(mesh.vertices) == 0:
        raise RuntimeError(f"Could not load any geometry from: {path}")

    if len(mesh.triangles) > 0:
        pcd = mesh.sample_points_uniformly(number_of_points=max(sample_points, 5000))
        return pcd, mesh, "mesh_sampled"

    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    return pcd, mesh, "mesh_vertices"


def estimate_auto_scales(src: o3d.geometry.PointCloud, tgt: o3d.geometry.PointCloud) -> Tuple[float, float]:
    a = np.asarray(src.get_axis_aligned_bounding_box().get_extent())
    b = np.asarray(tgt.get_axis_aligned_bounding_box().get_extent())
    diag = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), 1e-6)
    voxel = max(diag / 150.0, 1e-4)
    max_corr = max(4.0 * voxel, 1e-3)
    return voxel, max_corr


def estimate_normals_if_needed(pcd: o3d.geometry.PointCloud, radius: float) -> None:
    if len(pcd.points) == 0:
        return
    if pcd.has_normals():
        return
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=max(radius, 1e-4), max_nn=50)
    )
    pcd.normalize_normals()


def c2c_stats(src_aligned: o3d.geometry.PointCloud, tgt: o3d.geometry.PointCloud) -> Dict[str, float]:
    d = np.asarray(src_aligned.compute_point_cloud_distance(tgt), dtype=np.float64)
    if d.size == 0:
        raise RuntimeError("Point-to-point distance computation returned no samples")
    return summarize_distances(d, mode="c2c")


def c2m_stats(src_aligned: o3d.geometry.PointCloud, tgt_mesh: o3d.geometry.TriangleMesh) -> Dict[str, float]:
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(tgt_mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh_t)
    pts = o3d.core.Tensor(np.asarray(src_aligned.points), dtype=o3d.core.Dtype.Float32)
    d = scene.compute_distance(pts).numpy().astype(np.float64)
    if d.size == 0:
        raise RuntimeError("Cloud-to-mesh distance computation returned no samples")
    return summarize_distances(d, mode="c2m")


def summarize_distances(d: np.ndarray, mode: str) -> Dict[str, float]:
    return {
        "mode": mode,
        "count": int(d.size),
        "mean": float(np.mean(d)),
        "median": float(np.median(d)),
        "rmse": float(np.sqrt(np.mean(np.square(d)))),
        "p90": float(np.percentile(d, 90)),
        "p95": float(np.percentile(d, 95)),
        "max": float(np.max(d)),
    }


def main() -> int:
    args = parse_args()

    src_pcd, _, src_mode = load_geometry(Path(args.src), args.sample_points)
    tgt_pcd, tgt_mesh, tgt_mode = load_geometry(Path(args.tgt), args.sample_points)

    init = np.loadtxt(args.init_transform, dtype=np.float64)
    if init.shape != (4, 4):
        raise RuntimeError(f"Unexpected transform shape: {init.shape}")

    auto_voxel, auto_max_corr = estimate_auto_scales(src_pcd, tgt_pcd)
    voxel = args.downsample_voxel if args.downsample_voxel > 0 else auto_voxel
    max_corr = args.max_correspondence if args.max_correspondence > 0 else auto_max_corr

    src_ds = src_pcd.voxel_down_sample(voxel) if voxel > 0 else src_pcd
    tgt_ds = tgt_pcd.voxel_down_sample(voxel) if voxel > 0 else tgt_pcd

    if len(src_ds.points) == 0 or len(tgt_ds.points) == 0:
        raise RuntimeError("Downsampled source or target is empty")

    if args.icp_mode == "point_to_plane":
        estimate_normals_if_needed(src_ds, radius=2.0 * voxel)
        estimate_normals_if_needed(tgt_ds, radius=2.0 * voxel)
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()

    reg = o3d.pipelines.registration.registration_icp(
        src_ds,
        tgt_ds,
        max_corr,
        init,
        estimation,
    )
    tf = reg.transformation

    aligned = o3d.geometry.PointCloud(src_pcd)
    aligned.transform(tf)

    out_tf = Path(args.out_transform)
    out_tf.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_tf, tf, fmt="%.10f")

    out_aligned = Path(args.out_aligned_src)
    out_aligned.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out_aligned), aligned)

    if tgt_mesh is not None and len(tgt_mesh.triangles) > 0:
        eval_stats = c2m_stats(aligned, tgt_mesh)
    else:
        eval_stats = c2c_stats(aligned, tgt_pcd)

    summary = {
        "src": str(Path(args.src).resolve()),
        "tgt": str(Path(args.tgt).resolve()),
        "src_load_mode": src_mode,
        "tgt_load_mode": tgt_mode,
        "voxel": float(voxel),
        "max_correspondence": float(max_corr),
        "icp_mode": args.icp_mode,
        "icp_fitness": float(reg.fitness),
        "icp_inlier_rmse": float(reg.inlier_rmse),
        "transform_path": str(out_tf.resolve()),
        **eval_stats,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())