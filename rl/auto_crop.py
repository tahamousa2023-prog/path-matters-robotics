"""
experiments/taha/auto_crop.py

Automatic background removal for a reconstructed point cloud.

Four stages:
    1. Statistical outlier removal   → removes floating noise
    2. RANSAC multi-plane removal    → removes floor AND table surface
    3. DBSCAN clustering             → segments remaining points
    4. Smart cluster selection       → picks most compact cluster = object
                                       with automatic fallback ranges

Key fixes vs original:
    - Original kept LARGEST cluster = background blob
    - This version filters by size fraction, then picks DENSEST cluster
    - Automatic fallback widens search range if nothing found in user range

Usage:
    conda activate bufferx_o3d

    # Default:
    python experiments/taha/auto_crop.py \
        --input  /path/to/points.ply \
        --output /path/to/points_cleaned.ply

    # Diagnose only — print all clusters, save nothing:
    python experiments/taha/auto_crop.py \
        --input  /path/to/points.ply \
        --output /path/to/points_cleaned.ply \
        --diagnose

    # Tune parameters:
    python experiments/taha/auto_crop.py \
        --input  /path/to/points.ply \
        --output /path/to/points_cleaned.ply \
        --num-planes     2      \
        --plane-distance 0.015  \
        --cluster-eps    0.05   \
        --min-fraction   0.05   \
        --max-fraction   0.15
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d


# ══════════════════════════════════════════════════════════════════════════════
# CORE FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def auto_remove_background(
    pcd_path: str,
    output_path: str,
    nb_neighbors: int      = 20,
    std_ratio: float       = 2.0,
    num_planes: int        = 2,
    plane_distance: float  = 0.015,
    plane_min_inliers: int = 500,
    cluster_eps: float     = 0.025,
    cluster_min_pts: int   = 50,
    min_fraction: float    = 0.001,
    max_fraction: float    = 0.20,
    diagnose: bool         = False,
    verbose: bool          = True,
) -> o3d.geometry.PointCloud | None:
    """
    Remove background from a reconstructed point cloud.

    Parameters
    ----------
    num_planes        : number of dominant planes to remove (2 = floor + table)
    min_fraction      : ignore clusters smaller than this fraction
    max_fraction      : ignore clusters larger than this fraction (background)
    diagnose          : print cluster info and exit without saving
    """

    def log(msg):
        if verbose:
            print(f"  [auto_crop] {msg}")

    # ── Stage 1: Load ─────────────────────────────────────────────────────
    pcd = o3d.io.read_point_cloud(pcd_path)
    if len(pcd.points) == 0:
        print(f"ERROR: Empty point cloud at {pcd_path}")
        sys.exit(1)
    log(f"Loaded:                  {len(pcd.points):>8,} points")

    # ── Stage 1: Statistical outlier removal ──────────────────────────────
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
    )
    log(f"After outlier removal:   {len(pcd.points):>8,} points")

    if len(pcd.points) == 0:
        print("ERROR: All points removed by outlier filter. "
              "Try increasing --std-ratio.")
        sys.exit(1)

    # ── Stage 2: Multi-plane removal (floor + table) ──────────────────────
    for plane_idx in range(num_planes):
        if len(pcd.points) < plane_min_inliers:
            log("  Stopping plane removal — too few points left.")
            break

        _, inliers = pcd.segment_plane(
            distance_threshold=plane_distance,
            ransac_n=3,
            num_iterations=2000,
        )

        if len(inliers) < plane_min_inliers:
            log(f"  Plane {plane_idx+1}: only {len(inliers)} inliers "
                f"(< {plane_min_inliers}), stopping.")
            break

        pcd = pcd.select_by_index(inliers, invert=True)
        log(f"After plane {plane_idx+1} removal:  {len(pcd.points):>8,} points "
            f"  (removed {len(inliers):,} plane inliers)")

    if len(pcd.points) == 0:
        print("ERROR: All points removed by plane segmentation.")
        sys.exit(1)

    pts_after_planes = len(pcd.points)

    # ── Stage 3: DBSCAN clustering ────────────────────────────────────────
    log(f"Running DBSCAN  (eps={cluster_eps}, min_pts={cluster_min_pts}) ...")
    labels = np.array(
        pcd.cluster_dbscan(
            eps=cluster_eps,
            min_points=cluster_min_pts,
            print_progress=False,
        )
    )

    noise_count = (labels == -1).sum()
    valid_mask  = labels >= 0
    valid_pts   = int(valid_mask.sum())

    if labels.max() < 0:
        log("WARNING: DBSCAN found no clusters. "
            "Try increasing --cluster-eps or decreasing --cluster-min-pts.")
        if diagnose:
            return None
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(output_path, pcd)
        return pcd

    cluster_sizes = np.bincount(labels[valid_mask])
    num_clusters  = len(cluster_sizes)

    # ── Diagnose mode: print all clusters, exit without saving ────────────
    if diagnose:
        print(f"\n  {'─'*55}")
        print(f"  CLUSTER DIAGNOSTIC")
        print(f"  {'─'*55}")
        print(f"  Total points after planes : {pts_after_planes:,}")
        print(f"  Noise points (label=-1)   : {noise_count:,}")
        print(f"  Valid clustered points    : {valid_pts:,}")
        print(f"  Number of clusters        : {num_clusters}")
        print(f"  {'─'*55}")
        print(f"  {'Cluster':>8}  {'Points':>10}  {'% of valid':>12}  "
              f"{'Action':>28}")
        print(f"  {'─'*55}")

        for cid, size in sorted(enumerate(cluster_sizes),
                                key=lambda x: x[1], reverse=True):
            frac = size / valid_pts
            if frac > max_fraction:
                action = "SKIP — too large (background)"
            elif frac < min_fraction:
                action = "SKIP — too small (noise)"
            else:
                action = "KEEP — in range"
            print(f"  {cid:>8}  {size:>10,}  {frac*100:>11.2f}%  {action:>28}")

        print(f"  {'─'*55}")
        print(f"\n  Tip: set --max-fraction below the background cluster fraction")
        print(f"       and --min-fraction below the object cluster fraction.\n")
        return None

    # ── Stage 4: Size filtering with automatic fallback ───────────────────
    def try_filter(mn: float, mx: float) -> list:
        return [
            (cid, size)
            for cid, size in enumerate(cluster_sizes)
            if mn <= (size / valid_pts) <= mx
        ]

    # Try progressively wider ranges until something is found
    fallback_ranges = [
        (min_fraction, max_fraction),   # user range:      e.g. 5%–15%
        (0.01,         max_fraction),   # widen lower:     1%–15%
        (0.001,        max_fraction),   # wider lower:     0.1%–15%
        (0.001,        0.40),           # much wider:      0.1%–40%
        (0.001,        0.90),           # last resort:     anything < 90%
    ]

    valid_clusters = []
    used_range     = None
    for mn, mx in fallback_ranges:
        valid_clusters = try_filter(mn, mx)
        if valid_clusters:
            used_range = (mn, mx)
            if (mn, mx) != (min_fraction, max_fraction):
                log(f"WARNING: Used fallback range "
                    f"[{mn*100:.1f}%–{mx*100:.1f}%] "
                    f"— original range [{min_fraction*100:.1f}%–"
                    f"{max_fraction*100:.1f}%] found nothing")
            break

    log(f"Clusters in range "
        f"[{used_range[0]*100:.1f}%–{used_range[1]*100:.1f}%]: "
        f"{len(valid_clusters)} / {num_clusters}")

    if len(valid_clusters) == 0:
        print("ERROR: No clusters survived even with maximum fallback range.")
        print("  Run with --diagnose to inspect cluster structure.")
        sys.exit(1)

    # Pick the most compact cluster (highest density = points / bbox volume)
    # Dense = small bounding box relative to point count = the object, not background
    best_cluster = None
    best_density = -1.0

    for cid, size in valid_clusters:
        cluster_pts = pcd.select_by_index(np.where(labels == cid)[0])
        bbox        = cluster_pts.get_axis_aligned_bounding_box()
        extent      = bbox.get_extent()
        volume      = float(extent[0] * extent[1] * extent[2]) + 1e-9
        density     = size / volume

        log(f"  Cluster {cid}: {size:,} pts  "
            f"bbox={extent[0]:.3f}×{extent[1]:.3f}×{extent[2]:.3f}m  "
            f"density={density:.1f} pts/m³")

        if density > best_density:
            best_density = density
            best_cluster = cid

    pcd = pcd.select_by_index(np.where(labels == best_cluster)[0])
    log(f"Selected cluster {best_cluster} "
        f"({len(pcd.points):,} pts, density={best_density:.1f} pts/m³)")

    # ── Save ──────────────────────────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(output_path, pcd)
    log(f"Saved to: {output_path}")

    return pcd


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Automatic point cloud background removal"
    )
    parser.add_argument("--input",             required=True,
                        help="Input .ply file")
    parser.add_argument("--output",            required=True,
                        help="Output .ply file")
    parser.add_argument("--nb-neighbors",      type=int,   default=20)
    parser.add_argument("--std-ratio",         type=float, default=2.0)
    parser.add_argument("--num-planes",        type=int,   default=2)
    parser.add_argument("--plane-distance",    type=float, default=0.015)
    parser.add_argument("--plane-min-inliers", type=int,   default=500)
    parser.add_argument("--cluster-eps",       type=float, default=0.025)
    parser.add_argument("--cluster-min-pts",   type=int,   default=50)
    parser.add_argument("--min-fraction",      type=float, default=0.001)
    parser.add_argument("--max-fraction",      type=float, default=0.20)
    parser.add_argument("--diagnose",          action="store_true",
                        help="Print cluster sizes and exit, save nothing")
    parser.add_argument("--quiet",             action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    auto_remove_background(
        pcd_path          = args.input,
        output_path       = args.output,
        nb_neighbors      = args.nb_neighbors,
        std_ratio         = args.std_ratio,
        num_planes        = args.num_planes,
        plane_distance    = args.plane_distance,
        plane_min_inliers = args.plane_min_inliers,
        cluster_eps       = args.cluster_eps,
        cluster_min_pts   = args.cluster_min_pts,
        min_fraction      = args.min_fraction,
        max_fraction      = args.max_fraction,
        diagnose          = args.diagnose,
        verbose           = not args.quiet,
    )