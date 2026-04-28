#!/usr/bin/env python3
"""Manual CloudCompare/Open3D + BUFFER-X + ICP/Eval pipeline orchestrator.

This script stages files for manual cleanup/alignment, then continues with
BUFFER-X and automated ICP/eval.

Key behavior:
- GT is never manually edited.
- Manual outputs are cached under: <run>/<scene>/manual/
- Optional Open3D backend can:
    1) crop/clean the recon
    2) reuse a previously saved cropped recon
    3) pre-scale the recon toward the GT using bounding boxes
    4) manually align recon to GT via picked correspondences
    5) save recon_manual.ply + manual_init_transform.txt
- Visualization can save per-stage overlays and optionally open final/failure views.
- Path transparency:
    - writes run_manifest.json at run root
    - writes scene_status.json with resolved paths for every scene
    - writes batch_summary.csv with those resolved paths
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np


DEFAULT_RECON_CANDIDATES = ["textured.ply", "textured.obj", "sparse/points.ply"]
DEFAULT_GT_CANDIDATES = ["object.ply", "object.obj", "textured.ply", "textured.obj", "points.ply"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CloudCompare / Open3D / BUFFER-X / ICP / evaluation pipeline")
    p.add_argument("--recon-root", required=True, help="Root folder with per-scene reconstruction folders")
    p.add_argument("--gt-root", required=True, help="Root folder with per-scene GT folders")
    p.add_argument("--output-base", default=".", help="Base output folder")
    p.add_argument("--run-name", default=None, help="Optional run folder name")
    p.add_argument("--scene-names", nargs="*", default=None, help="Optional subset of scene names")
    p.add_argument("--config", default=None, help="Optional JSON config file")

    p.add_argument("--bufferx-root", required=True, help="BUFFER-X repo root")
    p.add_argument("--bufferx-env", default="bufferx", help="Conda env for BUFFER-X + Open3D helpers")
    p.add_argument("--experiment-id", default="threedmatch", help="BUFFER-X snapshot id")

    p.add_argument("--recon-candidates", nargs="*", default=DEFAULT_RECON_CANDIDATES)
    p.add_argument("--gt-candidates", nargs="*", default=DEFAULT_GT_CANDIDATES)

    p.add_argument(
        "--manual-mode",
        choices=["off", "prefer", "require", "prepare"],
        default="prefer",
        help=(
            "off=never use manual files; "
            "prefer=use manual if present; "
            "require=make/open manual files if missing; "
            "prepare=stage and create manual files if missing"
        ),
    )
    p.add_argument(
        "--manual-backend",
        choices=["cloudcompare", "open3d"],
        default="cloudcompare",
        help="Manual cleanup/alignment backend",
    )
    p.add_argument("--launch-cloudcompare", action="store_true", help="Launch CloudCompare GUI when manual files are missing")
    p.add_argument("--cloudcompare-cmd", default=None, help="Optional CloudCompare executable path for GUI launch")

    # Open3D manual backend options
    p.add_argument("--open3d-crop", action="store_true", help="Run Open3D crop/cleanup step on recon before manual registration")
    p.add_argument(
        "--open3d-reuse-cropped-recon",
        action="store_true",
        help="Reuse manual/recon_cropped.ply when --open3d-crop is not requested",
    )
    p.add_argument("--open3d-sample-points", type=int, default=120000, help="Sampling budget for mesh -> point cloud during Open3D manual mode")
    p.add_argument("--open3d-with-scaling", action="store_true", help="Allow similarity transform (scale) in Open3D manual init")
    p.add_argument(
        "--open3d-pre-scale",
        choices=["off", "aabb_diag", "aabb_max_extent"],
        default="off",
        help=(
            "Pre-scale source before manual point picking. "
            "aabb_diag = match AABB diagonal, "
            "aabb_max_extent = match largest AABB extent, "
            "off = disable"
        ),
    )
    p.add_argument(
        "--open3d-preview-after-manual",
        action="store_true",
        help="Preview aligned recon + GT after manual Open3D registration",
    )

    # Visualization
    p.add_argument("--save-viz", action="store_true", help="Save Open3D screenshots for important stages")
    p.add_argument("--show-final-viz", action="store_true", help="Open an interactive final overlay window for each successful scene")
    p.add_argument("--show-failed-viz", action="store_true", help="Open an interactive overlay window when BUFFER-X or ICP fails")

    p.add_argument("--sample-points", type=int, default=120000)
    p.add_argument("--max-points", type=int, default=30000)
    p.add_argument("--pose-refine", action="store_true")
    p.add_argument("--icp-downsample-voxel", type=float, default=0.0)
    p.add_argument("--icp-max-correspondence", type=float, default=0.0)
    return p.parse_args()


def merge_config(args: argparse.Namespace) -> argparse.Namespace:
    if not args.config:
        return args
    cfg = json.loads(Path(args.config).read_text())
    for key, value in cfg.items():
        if hasattr(args, key) and getattr(args, key) in (None, [], False):
            setattr(args, key, value)
    return args


def list_scene_names(recon_root: Path, gt_root: Path, requested: Optional[List[str]]) -> List[str]:
    recon_names = {p.name for p in recon_root.iterdir() if p.is_dir()}
    gt_names = {p.name for p in gt_root.iterdir() if p.is_dir()}
    names = sorted(recon_names & gt_names)
    if requested:
        requested_set = set(requested)
        names = [n for n in names if n in requested_set]
    return names


def first_match(scene_dir: Path, candidates: Iterable[str]) -> Optional[Path]:
    for rel in candidates:
        p = scene_dir / rel
        if p.exists():
            return p
    meshes = sorted([p for p in scene_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".ply", ".obj"}])
    if not meshes:
        return None
    keywords = ["object", "textured", "points"]
    for kw in keywords:
        for p in meshes:
            if kw in p.name.lower() or kw in str(p.parent).lower():
                return p
    return meshes[0]


def ensure_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)


def maybe_launch_cloudcompare(cloudcompare_cmd: Optional[str], src_path: Path, tgt_path: Path) -> None:
    if not cloudcompare_cmd:
        return
    try:
        subprocess.Popen([cloudcompare_cmd, str(src_path), str(tgt_path)])
    except Exception as exc:
        print(f"[WARN] Failed to launch CloudCompare: {exc}", file=sys.stderr)


def run_subprocess(cmd: List[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    env = os.environ.copy()

    # prevent foreign env libraries from leaking into subprocesses
    for key in [
        "PYTHONPATH",
        "LD_LIBRARY_PATH",
        "QT_PLUGIN_PATH",
        "QML2_IMPORT_PATH",
        "QT_QPA_PLATFORM_PLUGIN_PATH",
    ]:
        env.pop(key, None)

    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        env=env,
    )


def write_scene_status(scene_dir: Path, payload: Dict) -> None:
    (scene_dir / "scene_status.json").write_text(json.dumps(payload, indent=2))


def make_scene_base_status(
    *,
    scene_name: str,
    recon_root: Path,
    gt_root: Path,
    run_dir: Path,
    scene_out: Path,
    raw_dir: Path,
    manual_dir: Path,
    bx_dir: Path,
    icp_dir: Path,
    viz_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, str]:
    recon_scene = recon_root / scene_name
    gt_scene = gt_root / scene_name
    return {
        "scene": scene_name,
        "run_dir": str(run_dir),
        "scene_out_dir": str(scene_out),
        "recon_root": str(recon_root),
        "gt_root": str(gt_root),
        "recon_scene_dir": str(recon_scene),
        "gt_scene_dir": str(gt_scene),
        "images_dir_expected": str(recon_scene / "images"),
        "raw_dir": str(raw_dir),
        "manual_dir": str(manual_dir),
        "bufferx_dir": str(bx_dir),
        "icp_dir": str(icp_dir),
        "viz_dir": str(viz_dir),
        "manual_mode": str(args.manual_mode),
        "manual_backend": str(args.manual_backend),
        "bufferx_env": str(args.bufferx_env),
        "bufferx_root": str(Path(args.bufferx_root).expanduser().resolve()),
        "experiment_id": str(args.experiment_id),
        "recon_candidates": json.dumps(list(args.recon_candidates)),
        "gt_candidates": json.dumps(list(args.gt_candidates)),
    }


def write_run_manifest(run_dir: Path, args: argparse.Namespace, scenes: List[str]) -> None:
    payload = {
        "run_dir": str(run_dir),
        "recon_root": str(Path(args.recon_root).expanduser().resolve()),
        "gt_root": str(Path(args.gt_root).expanduser().resolve()),
        "output_base": str(Path(args.output_base).expanduser().resolve()),
        "run_name": str(args.run_name) if args.run_name else "",
        "scene_names_selected": scenes,
        "recon_candidates": list(args.recon_candidates),
        "gt_candidates": list(args.gt_candidates),
        "manual_mode": args.manual_mode,
        "manual_backend": args.manual_backend,
        "bufferx_root": str(Path(args.bufferx_root).expanduser().resolve()),
        "bufferx_env": args.bufferx_env,
        "experiment_id": args.experiment_id,
        "path_contract": {
            "recon_input_search": "<recon-root>/<scene>/" + " ; ".join(args.recon_candidates),
            "gt_input_search": "<gt-root>/<scene>/" + " ; ".join(args.gt_candidates),
            "run_outputs": "<output-base>/<run-name>/<scene>/{raw,manual,bufferx,icp,viz}/",
            "scene_status": "<output-base>/<run-name>/<scene>/scene_status.json",
            "batch_summary": "<output-base>/<run-name>/batch_summary.csv",
        },
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(payload, indent=2))


# ----------------------------
# Open3D helpers
# ----------------------------

def _import_open3d():
    import open3d as o3d
    return o3d


def load_geometry_for_open3d(path: Path, sample_points: int):
    o3d = _import_open3d()

    pcd = o3d.io.read_point_cloud(str(path))
    if len(pcd.points) > 0:
        return pcd, "point_cloud"

    mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=True)
    if len(mesh.vertices) == 0:
        raise RuntimeError(f"Could not load any geometry from: {path}")

    if len(mesh.triangles) > 0:
        pcd = mesh.sample_points_uniformly(number_of_points=max(sample_points, 5000))
        return pcd, "mesh_sampled"

    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    if mesh.has_vertex_colors():
        pcd.colors = mesh.vertex_colors
    return pcd, "mesh_vertices"


def estimate_bbox_scale_ratio(source, target, mode: str) -> float:
    if mode == "off":
        return 1.0

    src_box = source.get_axis_aligned_bounding_box()
    tgt_box = target.get_axis_aligned_bounding_box()

    src_extent = np.asarray(src_box.get_extent(), dtype=np.float64)
    tgt_extent = np.asarray(tgt_box.get_extent(), dtype=np.float64)

    if mode == "aabb_diag":
        src_measure = float(np.linalg.norm(src_extent))
        tgt_measure = float(np.linalg.norm(tgt_extent))
    elif mode == "aabb_max_extent":
        src_measure = float(np.max(src_extent))
        tgt_measure = float(np.max(tgt_extent))
    else:
        raise ValueError(f"Unknown pre-scale mode: {mode}")

    if src_measure <= 1e-12 or tgt_measure <= 1e-12:
        return 1.0

    return tgt_measure / src_measure


def maybe_pre_scale_source_to_target(source, target, mode: str):
    o3d = _import_open3d()

    if mode == "off":
        return source, 1.0

    scale_ratio = estimate_bbox_scale_ratio(source, target, mode)
    scaled = o3d.geometry.PointCloud(source)
    scaled.scale(scale_ratio, center=scaled.get_center())

    print(f"[OPEN3D] Pre-scale mode: {mode}")
    print(f"[OPEN3D] Pre-scale factor applied to source: {scale_ratio:.6f}")

    return scaled, scale_ratio


def make_overlay_geometries(src_pcd, tgt_pcd):
    src = copy.deepcopy(src_pcd)
    tgt = copy.deepcopy(tgt_pcd)
    src.paint_uniform_color([1.0, 0.706, 0.0])   # yellow
    tgt.paint_uniform_color([0.0, 0.651, 0.929]) # cyan
    return src, tgt


def preview_registration(source, target, transform: np.ndarray, title: str) -> None:
    o3d = _import_open3d()
    src = o3d.geometry.PointCloud(source)
    tgt = o3d.geometry.PointCloud(target)
    src.paint_uniform_color([1.0, 0.706, 0.0])
    tgt.paint_uniform_color([0.0, 0.651, 0.929])
    src.transform(transform)
    o3d.visualization.draw_geometries([src, tgt], window_name=title, width=1600, height=900)


def crop_source_interactively(source):
    o3d = _import_open3d()
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="Open3D crop recon", width=1600, height=900)
    vis.add_geometry(source)

    print("\n[Open3D crop]")
    print("  Y twice : align geometry")
    print("  K       : selection mode")
    print("  drag    : rectangle selection")
    print("  Ctrl+LMB: polygon selection")
    print("  C       : crop selected geometry")
    print("  F       : freeview mode")
    print("  Q       : quit and continue\n")

    vis.run()
    cropped = vis.get_cropped_geometry()
    vis.destroy_window()

    if cropped is None:
        return source
    if isinstance(cropped, o3d.geometry.PointCloud) and len(cropped.points) > 0:
        return cropped
    return source


def pick_points(pcd, window_name: str) -> List[int]:
    o3d = _import_open3d()
    print("")
    print(f"[Open3D pick] {window_name}")
    print("1) Pick at least three correspondences using [shift + left click]")
    print("2) Undo with [shift + right click]")
    print("3) Press 'Q' to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=window_name, width=1600, height=900)
    vis.add_geometry(pcd)
    vis.run()
    picked = vis.get_picked_points()
    vis.destroy_window()
    print("")
    return picked


def save_overlay_screenshot(
    src_path: Path,
    tgt_path: Path,
    out_png: Path,
    sample_points: int,
    view_status_path: Optional[Path] = None,
    width: int = 1600,
    height: int = 900,
) -> None:
    o3d = _import_open3d()

    src_pcd, _ = load_geometry_for_open3d(src_path, sample_points)
    tgt_pcd, _ = load_geometry_for_open3d(tgt_path, sample_points)
    src_vis, tgt_vis = make_overlay_geometries(src_pcd, tgt_pcd)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=str(out_png.name), width=width, height=height, visible=False)
    vis.add_geometry(src_vis)
    vis.add_geometry(tgt_vis)

    if view_status_path is not None and view_status_path.exists():
        vis.set_view_status(view_status_path.read_text())
    else:
        vis.reset_view_point(True)

    vis.poll_events()
    vis.update_renderer()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    vis.capture_screen_image(str(out_png), do_render=True)

    if view_status_path is not None and not view_status_path.exists():
        view_status_path.write_text(vis.get_view_status())

    vis.destroy_window()


def show_overlay_interactive(
    src_path: Path,
    tgt_path: Path,
    sample_points: int,
    title: str,
) -> None:
    o3d = _import_open3d()
    src_pcd, _ = load_geometry_for_open3d(src_path, sample_points)
    tgt_pcd, _ = load_geometry_for_open3d(tgt_path, sample_points)
    src_vis, tgt_vis = make_overlay_geometries(src_pcd, tgt_pcd)
    o3d.visualization.draw_geometries(
        [src_vis, tgt_vis],
        window_name=title,
        width=1600,
        height=900,
    )


def run_open3d_manual_prepare_recon(
    raw_recon: Path,
    raw_gt: Path,
    cropped_recon_out: Path,
    manual_recon_out: Path,
    manual_transform_out: Path,
    manual_meta_out: Path,
    sample_points: int,
    do_crop: bool,
    reuse_cropped_recon: bool,
    allow_scaling: bool,
    preview_after_manual: bool,
    pre_scale_mode: str,
) -> Dict[str, object]:
    o3d = _import_open3d()

    source, source_mode = load_geometry_for_open3d(raw_recon, sample_points)
    target, target_mode = load_geometry_for_open3d(raw_gt, sample_points)

    if len(source.points) == 0:
        raise RuntimeError(f"No points loaded from recon: {raw_recon}")
    if len(target.points) == 0:
        raise RuntimeError(f"No points loaded from GT: {raw_gt}")

    crop_mode_used = "none"

    if do_crop:
        source = crop_source_interactively(source)
        cropped_recon_out.parent.mkdir(parents=True, exist_ok=True)
        ok = o3d.io.write_point_cloud(str(cropped_recon_out), source)
        if not ok:
            raise RuntimeError(f"Failed to write cropped recon to {cropped_recon_out}")
        crop_mode_used = "interactive_saved"
        print(f"[OPEN3D] Saved cropped recon: {cropped_recon_out}")

    elif reuse_cropped_recon:
        if cropped_recon_out.exists():
            source, source_mode = load_geometry_for_open3d(cropped_recon_out, sample_points)
            crop_mode_used = "reused_saved_crop"
            print(f"[OPEN3D] Reusing cropped recon: {cropped_recon_out}")
        else:
            print(f"[WARN] Requested --open3d-reuse-cropped-recon but file does not exist: {cropped_recon_out}")
            crop_mode_used = "reuse_requested_missing_file"

    source, pre_scale_factor = maybe_pre_scale_source_to_target(
        source, target, pre_scale_mode
    )

    picked_id_source = pick_points(source, f"Pick SOURCE correspondences: {raw_recon.name}")
    picked_id_target = pick_points(target, f"Pick TARGET correspondences: {raw_gt.name}")

    print(f"Picked source points: {len(picked_id_source)} -> {picked_id_source}")
    print(f"Picked target points: {len(picked_id_target)} -> {picked_id_target}")

    if len(picked_id_source) < 3 or len(picked_id_target) < 3:
        raise RuntimeError("Need at least 3 correspondences in source and target")
    if len(picked_id_source) != len(picked_id_target):
        raise RuntimeError("Source and target must have the same number of picked correspondences")

    corr = np.zeros((len(picked_id_source), 2), dtype=np.int32)
    corr[:, 0] = np.asarray(picked_id_source, dtype=np.int32)
    corr[:, 1] = np.asarray(picked_id_target, dtype=np.int32)

    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint(
        with_scaling=allow_scaling
    )
    trans_init = p2p.compute_transformation(
        source, target, o3d.utility.Vector2iVector(corr)
    )

    aligned = o3d.geometry.PointCloud(source)
    aligned.transform(trans_init)

    manual_recon_out.parent.mkdir(parents=True, exist_ok=True)
    ok = o3d.io.write_point_cloud(str(manual_recon_out), aligned)
    if not ok:
        raise RuntimeError(f"Failed to write manual recon to {manual_recon_out}")

    np.savetxt(manual_transform_out, trans_init, fmt="%.10f")

    meta = {
        "backend": "open3d",
        "raw_recon": str(raw_recon),
        "raw_gt": str(raw_gt),
        "cropped_recon": str(cropped_recon_out),
        "manual_recon": str(manual_recon_out),
        "manual_transform": str(manual_transform_out),
        "source_load_mode": source_mode,
        "target_load_mode": target_mode,
        "crop_mode_used": crop_mode_used,
        "cropped": bool(do_crop),
        "reuse_cropped_recon": bool(reuse_cropped_recon),
        "pre_scale_mode": str(pre_scale_mode),
        "pre_scale_factor": float(pre_scale_factor),
        "with_scaling": bool(allow_scaling),
        "num_correspondences": int(len(picked_id_source)),
        "picked_source_ids": [int(x) for x in picked_id_source],
        "picked_target_ids": [int(x) for x in picked_id_target],
    }
    manual_meta_out.write_text(json.dumps(meta, indent=2))

    if preview_after_manual:
        preview_registration(source, target, trans_init, title="Open3D manual alignment preview")

    return meta


# ----------------------------
# Main
# ----------------------------

def main() -> int:
    args = merge_config(parse_args())

    recon_root = Path(args.recon_root).expanduser().resolve()
    gt_root = Path(args.gt_root).expanduser().resolve()
    output_base = Path(args.output_base).expanduser().resolve()

    run_name = args.run_name or f"cc_bufferx_icp_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_base / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    here = Path(__file__).resolve().parent
    bufferx_wrapper = here / "bufferx_pair_wrapper.py"
    icp_helper = here / "icp_eval_helper.py"

    scenes = list_scene_names(recon_root, gt_root, args.scene_names)
    write_run_manifest(run_dir, args, scenes)

    print(f"Found {len(scenes)} scene(s): {', '.join(scenes) if scenes else '<none>'}")
    print(f"Run dir: {run_dir}")
    print(f"Recon root: {recon_root}")
    print(f"GT root: {gt_root}")
    print(f"Recon candidates: {args.recon_candidates}")
    print(f"GT candidates: {args.gt_candidates}")

    summary_rows: List[Dict[str, str]] = []

    for idx, scene_name in enumerate(scenes, start=1):
        print(f"\n=== [{idx}/{len(scenes)}] {scene_name} ===")
        scene_out = run_dir / scene_name
        raw_dir = scene_out / "raw"
        manual_dir = scene_out / "manual"
        bx_dir = scene_out / "bufferx"
        icp_dir = scene_out / "icp"
        viz_dir = scene_out / "viz"

        for d in (scene_out, raw_dir, manual_dir, bx_dir, icp_dir, viz_dir):
            d.mkdir(parents=True, exist_ok=True)

        view_status_path = viz_dir / "camera_view.json"

        base_status = make_scene_base_status(
            scene_name=scene_name,
            recon_root=recon_root,
            gt_root=gt_root,
            run_dir=run_dir,
            scene_out=scene_out,
            raw_dir=raw_dir,
            manual_dir=manual_dir,
            bx_dir=bx_dir,
            icp_dir=icp_dir,
            viz_dir=viz_dir,
            args=args,
        )

        recon_scene = recon_root / scene_name
        gt_scene = gt_root / scene_name
        recon_src = first_match(recon_scene, args.recon_candidates)
        gt_src = first_match(gt_scene, args.gt_candidates)

        if recon_src is None or gt_src is None:
            status = {
                **base_status,
                "status": "missing_input",
                "recon_src_used": str(recon_src) if recon_src else "",
                "gt_src_used": str(gt_src) if gt_src else "",
            }
            write_scene_status(scene_out, status)
            summary_rows.append(status)
            print("[SKIP] Missing recon or GT input")
            continue

        raw_recon = raw_dir / f"recon_input{recon_src.suffix.lower()}"
        raw_gt = raw_dir / f"gt_input{gt_src.suffix.lower()}"
        ensure_copy(recon_src, raw_recon)
        ensure_copy(gt_src, raw_gt)

        if args.save_viz:
            try:
                save_overlay_screenshot(
                    raw_recon,
                    raw_gt,
                    viz_dir / "01_raw_overlay.png",
                    sample_points=args.sample_points,
                    view_status_path=view_status_path,
                )
            except Exception as exc:
                print(f"[WARN] Raw visualization failed: {exc}")

        manual_cropped_recon = manual_dir / "recon_cropped.ply"
        manual_recon = manual_dir / "recon_manual.ply"
        manual_init_tf = manual_dir / "manual_init_transform.txt"
        manual_meta = manual_dir / "recon_manual.meta.json"
        manual_present = manual_recon.exists()

        # Manual preparation
        if args.manual_mode in {"prepare", "require"} and not manual_present:
            if args.manual_backend == "open3d":
                try:
                    print("[OPEN3D] Starting manual cleanup/alignment")
                    run_open3d_manual_prepare_recon(
                        raw_recon=raw_recon,
                        raw_gt=raw_gt,
                        cropped_recon_out=manual_cropped_recon,
                        manual_recon_out=manual_recon,
                        manual_transform_out=manual_init_tf,
                        manual_meta_out=manual_meta,
                        sample_points=args.open3d_sample_points,
                        do_crop=args.open3d_crop,
                        reuse_cropped_recon=args.open3d_reuse_cropped_recon,
                        allow_scaling=args.open3d_with_scaling,
                        preview_after_manual=args.open3d_preview_after_manual,
                        pre_scale_mode=args.open3d_pre_scale,
                    )
                    manual_present = manual_recon.exists()
                    if not manual_present:
                        raise RuntimeError("Open3D manual step completed but recon_manual.ply was not created")

                    if args.save_viz:
                        try:
                            save_overlay_screenshot(
                                manual_recon,
                                raw_gt,
                                viz_dir / "02_manual_overlay.png",
                                sample_points=args.sample_points,
                                view_status_path=view_status_path,
                            )
                        except Exception as exc:
                            print(f"[WARN] Manual visualization failed: {exc}")

                    print("[OPEN3D] Manual recon saved")
                except Exception as exc:
                    status = {
                        **base_status,
                        "status": "manual_open3d_failed",
                        "recon_src_used": str(recon_src),
                        "gt_src_used": str(gt_src),
                        "raw_recon": str(raw_recon),
                        "raw_gt": str(raw_gt),
                        "manual_cropped_recon_path": str(manual_cropped_recon) if manual_cropped_recon.exists() else "",
                        "manual_recon_path": str(manual_recon),
                        "manual_init_transform_path": str(manual_init_tf),
                        "stderr": str(exc),
                    }
                    write_scene_status(scene_out, status)
                    summary_rows.append(status)
                    print(f"[FAIL] Open3D manual stage failed: {exc}")
                    continue
            else:
                status = {
                    **base_status,
                    "status": "manual_needed",
                    "recon_src_used": str(recon_src),
                    "gt_src_used": str(gt_src),
                    "raw_recon": str(raw_recon),
                    "raw_gt": str(raw_gt),
                    "manual_cropped_recon_path": str(manual_cropped_recon) if manual_cropped_recon.exists() else "",
                    "manual_recon_path": str(manual_recon),
                    "manual_init_transform_path": str(manual_init_tf),
                }
                write_scene_status(scene_out, status)
                summary_rows.append(status)
                print("[STOP] Manual CloudCompare cleanup/alignment still needed")
                print(f"  Open recon: {raw_recon}")
                print(f"  Open GT   : {raw_gt}")
                print(f"  Save aligned/cleaned recon as: {manual_recon}")
                if args.launch_cloudcompare:
                    maybe_launch_cloudcompare(args.cloudcompare_cmd, raw_recon, raw_gt)
                continue

        if args.manual_mode == "off":
            work_recon, work_gt = raw_recon, raw_gt
        elif args.manual_mode == "prefer":
            work_recon, work_gt = (manual_recon, raw_gt) if manual_present else (raw_recon, raw_gt)
        else:
            work_recon, work_gt = manual_recon, raw_gt

        if args.save_viz and manual_present:
            manual_png = viz_dir / "02_manual_overlay.png"
            if not manual_png.exists():
                try:
                    save_overlay_screenshot(
                        work_recon,
                        raw_gt,
                        manual_png,
                        sample_points=args.sample_points,
                        view_status_path=view_status_path,
                    )
                except Exception as exc:
                    print(f"[WARN] Existing manual visualization failed: {exc}")

        # BUFFER-X stage
        init_tf = bx_dir / "init_transform.txt"
        init_json = bx_dir / "init_summary.json"
        init_aligned = bx_dir / "aligned_source_init.ply"
        bx_cmd = [
            "conda", "run", "-n", args.bufferx_env,
            "python3", str(bufferx_wrapper),
            "--bufferx-root", str(Path(args.bufferx_root).expanduser().resolve()),
            "--src", str(work_recon),
            "--tgt", str(work_gt),
            "--out-transform", str(init_tf),
            "--out-json", str(init_json),
            "--out-aligned-src", str(init_aligned),
            "--experiment-id", args.experiment_id,
            "--sample-points", str(args.sample_points),
            "--max-points", str(args.max_points),
        ]
        if args.pose_refine:
            bx_cmd.append("--pose-refine")

        bx_res = run_subprocess(bx_cmd, cwd=here)
        (bx_dir / "bufferx_stdout.log").write_text(bx_res.stdout)
        (bx_dir / "bufferx_stderr.log").write_text(bx_res.stderr)
        if bx_res.returncode != 0:
            status = {
                **base_status,
                "status": "bufferx_failed",
                "recon_src_used": str(recon_src),
                "gt_src_used": str(gt_src),
                "raw_recon": str(raw_recon),
                "raw_gt": str(raw_gt),
                "manual_cropped_recon_path": str(manual_cropped_recon) if manual_cropped_recon.exists() else "",
                "manual_recon_path": str(manual_recon) if manual_present else "",
                "manual_init_transform_path": str(manual_init_tf) if manual_init_tf.exists() else "",
                "work_recon": str(work_recon),
                "work_gt": str(work_gt),
                "bufferx_init_transform_path": str(init_tf),
                "bufferx_init_summary_path": str(init_json),
                "bufferx_aligned_source_path": str(init_aligned),
                "stderr": bx_res.stderr[-2000:],
            }
            write_scene_status(scene_out, status)
            summary_rows.append(status)
            print("[FAIL] BUFFER-X stage failed")

            if args.show_failed_viz:
                try:
                    show_overlay_interactive(
                        work_recon,
                        work_gt,
                        sample_points=args.sample_points,
                        title=f"{scene_name} :: BUFFER-X failed",
                    )
                except Exception as exc:
                    print(f"[WARN] Failed-scene visualization failed: {exc}")
            continue

        if args.save_viz and init_aligned.exists():
            try:
                save_overlay_screenshot(
                    init_aligned,
                    work_gt,
                    viz_dir / "03_bufferx_overlay.png",
                    sample_points=args.sample_points,
                    view_status_path=view_status_path,
                )
            except Exception as exc:
                print(f"[WARN] BUFFER-X visualization failed: {exc}")

        # ICP/Eval stage
        icp_tf = icp_dir / "icp_transform.txt"
        icp_json = icp_dir / "icp_summary.json"
        icp_aligned = icp_dir / "aligned_source_icp.ply"
        icp_cmd = [
            "conda", "run", "-n", args.bufferx_env,
            "python3", str(icp_helper),
            "--src", str(work_recon),
            "--tgt", str(work_gt),
            "--init-transform", str(init_tf),
            "--out-transform", str(icp_tf),
            "--out-json", str(icp_json),
            "--out-aligned-src", str(icp_aligned),
            "--sample-points", str(args.sample_points),
            "--downsample-voxel", str(args.icp_downsample_voxel),
            "--max-correspondence", str(args.icp_max_correspondence),
        ]
        icp_res = run_subprocess(icp_cmd, cwd=here)
        (icp_dir / "icp_stdout.log").write_text(icp_res.stdout)
        (icp_dir / "icp_stderr.log").write_text(icp_res.stderr)
        if icp_res.returncode != 0:
            status = {
                **base_status,
                "status": "icp_failed",
                "recon_src_used": str(recon_src),
                "gt_src_used": str(gt_src),
                "raw_recon": str(raw_recon),
                "raw_gt": str(raw_gt),
                "manual_cropped_recon_path": str(manual_cropped_recon) if manual_cropped_recon.exists() else "",
                "manual_recon_path": str(manual_recon) if manual_present else "",
                "manual_init_transform_path": str(manual_init_tf) if manual_init_tf.exists() else "",
                "work_recon": str(work_recon),
                "work_gt": str(work_gt),
                "bufferx_init_transform_path": str(init_tf),
                "bufferx_init_summary_path": str(init_json),
                "bufferx_aligned_source_path": str(init_aligned),
                "icp_transform_path": str(icp_tf),
                "icp_summary_path": str(icp_json),
                "icp_aligned_source_path": str(icp_aligned),
                "stderr": icp_res.stderr[-2000:],
            }
            write_scene_status(scene_out, status)
            summary_rows.append(status)
            print("[FAIL] ICP/Eval stage failed")

            if args.show_failed_viz:
                try:
                    show_overlay_interactive(
                        work_recon,
                        work_gt,
                        sample_points=args.sample_points,
                        title=f"{scene_name} :: ICP failed",
                    )
                except Exception as exc:
                    print(f"[WARN] Failed-scene visualization failed: {exc}")
            continue

        if args.save_viz and icp_aligned.exists():
            try:
                save_overlay_screenshot(
                    icp_aligned,
                    work_gt,
                    viz_dir / "04_icp_overlay.png",
                    sample_points=args.sample_points,
                    view_status_path=view_status_path,
                )
            except Exception as exc:
                print(f"[WARN] ICP visualization failed: {exc}")

        if args.show_final_viz:
            try:
                show_overlay_interactive(
                    icp_aligned,
                    work_gt,
                    sample_points=args.sample_points,
                    title=f"{scene_name} :: FINAL ICP vs GT",
                )
            except Exception as exc:
                print(f"[WARN] Final interactive visualization failed: {exc}")

        icp_metrics = json.loads(icp_json.read_text())
        init_metrics = json.loads(init_json.read_text())
        status = {
            **base_status,
            "status": "ok",
            "recon_src_used": str(recon_src),
            "gt_src_used": str(gt_src),
            "raw_recon": str(raw_recon),
            "raw_gt": str(raw_gt),
            "manual_cropped_recon_path": str(manual_cropped_recon) if manual_cropped_recon.exists() else "",
            "manual_recon_path": str(manual_recon) if manual_present else "",
            "manual_init_transform_path": str(manual_init_tf) if manual_init_tf.exists() else "",
            "work_recon": str(work_recon),
            "work_gt": str(work_gt),
            "bufferx_init_transform_path": str(init_tf),
            "bufferx_init_summary_path": str(init_json),
            "bufferx_aligned_source_path": str(init_aligned),
            "icp_transform_path": str(icp_tf),
            "icp_summary_path": str(icp_json),
            "icp_aligned_source_path": str(icp_aligned),
            "viz_raw": str(viz_dir / "01_raw_overlay.png") if (viz_dir / "01_raw_overlay.png").exists() else "",
            "viz_manual": str(viz_dir / "02_manual_overlay.png") if (viz_dir / "02_manual_overlay.png").exists() else "",
            "viz_bufferx": str(viz_dir / "03_bufferx_overlay.png") if (viz_dir / "03_bufferx_overlay.png").exists() else "",
            "viz_icp": str(viz_dir / "04_icp_overlay.png") if (viz_dir / "04_icp_overlay.png").exists() else "",
            "eval_mode": icp_metrics.get("mode", "unknown"),
            "icp_mode": icp_metrics.get("icp_mode", ""),
            "icp_fitness": str(icp_metrics.get("icp_fitness", "")),
            "icp_inlier_rmse": str(icp_metrics.get("icp_inlier_rmse", "")),
            "eval_rmse": str(icp_metrics.get("rmse", "")),
            "bufferx_inliers": str(init_metrics.get("num_inliers", "")),
        }
        write_scene_status(scene_out, status)
        summary_rows.append(status)
        print("[OK] Completed")

    summary_csv = run_dir / "batch_summary.csv"
    fieldnames = sorted({k for row in summary_rows for k in row.keys()})
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    had_failures = any(row.get("status") != "ok" for row in summary_rows)
    print(f"\nSummary written to: {summary_csv}")
    print(f"Run manifest: {run_dir / 'run_manifest.json'}")
    print(f"Run folder: {run_dir}")

    if had_failures:
        print("One or more scenes failed. See batch_summary.csv and scene_status.json files.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())