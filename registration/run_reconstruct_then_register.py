#!/usr/bin/env python3
"""Reconstruct scenes first (VGGT or Fast3R), then run cc_bufferx registration pipeline.

This wrapper keeps reconstruction separate from registration/evaluation.

Path contract:
- Input scenes:
    <scene-root>/<scene>/images/
- Generated reconstructions:
    <scene-root>/<scene>/recon_generated/<backend>/points.ply
- Wrapper outputs:
    <output-base>/<run-name>/_wrapper/
- Registration outputs:
    <output-base>/<run-name>/<scene>/{raw,manual,bufferx,icp,viz}/
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_REGISTER_SCRIPT = "/home/AP_PathMatters/path_matters/haroun/Pipeline/cc_bufferx_pipeline_package/run_cc_bufferx_pipeline.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reconstruct scenes, then run cc_bufferx registration pipeline")

    p.add_argument("--scene-root", required=True, help="Root folder with per-scene folders containing images/")
    p.add_argument("--gt-root", required=True, help="Root folder with per-scene GT folders")
    p.add_argument("--output-base", required=True, help="Base output folder for registration pipeline")
    p.add_argument("--run-name", default=None, help="Optional run folder name")

    p.add_argument(
        "--backend",
        choices=["none", "vggt", "fast3r"],
        default="none",
        help="Reconstruction backend",
    )

    p.add_argument(
        "--generated-relpath",
        default="recon_generated/{backend}/points.ply",
        help="Where reconstruction output is expected inside each scene folder",
    )
    p.add_argument(
        "--scene-image-subdir",
        default="images",
        help="Subfolder inside each scene containing images",
    )

    p.add_argument("--scene-names", nargs="*", default=None, help="Optional subset of scene names")
    p.add_argument("--skip-reconstruction-if-exists", action="store_true")
    p.add_argument("--continue-on-recon-failure", action="store_true")

    p.add_argument(
        "--register-script",
        default=DEFAULT_REGISTER_SCRIPT,
        help="Path to run_cc_bufferx_pipeline.py",
    )
    p.add_argument("--bufferx-root", required=True, help="BUFFER-X repo root")
    p.add_argument("--bufferx-env", default="bufferx", help="Conda env for BUFFER-X + Open3D helpers")

    p.add_argument("--experiment-id", default="threedmatch")
    p.add_argument("--manual-mode", default="prefer")
    p.add_argument("--manual-backend", default="cloudcompare")

    # Backend command templates
    p.add_argument(
        "--vggt-cmd-template",
        default=None,
        help=(
            "Command template used when --backend vggt. "
            "Available placeholders: {scene_dir}, {image_dir}, {out_ply}, {scene_name}, {backend}"
        ),
    )
    p.add_argument(
        "--fast3r-cmd-template",
        default=None,
        help=(
            "Command template used when --backend fast3r. "
            "Available placeholders: {scene_dir}, {image_dir}, {out_ply}, {scene_name}, {backend}"
        ),
    )

    # Everything after this flag is forwarded to run_cc_bufferx_pipeline.py
    p.add_argument(
        "--register-extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args forwarded verbatim to run_cc_bufferx_pipeline.py. Put this flag last.",
    )

    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def list_scene_names(scene_root: Path, gt_root: Path, requested: Optional[List[str]]) -> List[str]:
    scene_names = {p.name for p in scene_root.iterdir() if p.is_dir()}
    gt_names = {p.name for p in gt_root.iterdir() if p.is_dir()}
    names = sorted(scene_names & gt_names)
    if requested:
        requested_set = set(requested)
        names = [n for n in names if n in requested_set]
    return names


def run_cmd(cmd: List[str], cwd: Optional[Path] = None, dry_run: bool = False) -> subprocess.CompletedProcess:
    printable = " ".join(shlex.quote(x) for x in cmd)
    print(f"$ {printable}", flush=True)

    if dry_run:
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
    )


def make_recon_cmd(template: str, scene_dir: Path, image_dir: Path, out_ply: Path, scene_name: str, backend: str) -> List[str]:
    filled = template.format(
        scene_dir=str(scene_dir),
        image_dir=str(image_dir),
        out_ply=str(out_ply),
        scene_name=scene_name,
        backend=backend,
    )
    return shlex.split(filled)


def write_wrapper_run_manifest(
    wrapper_dir: Path,
    *,
    scene_root: Path,
    gt_root: Path,
    output_base: Path,
    run_name: str,
    backend: str,
    generated_relpath: str,
    scene_image_subdir: str,
    register_script: Path,
    bufferx_root: Path,
    bufferx_env: str,
    experiment_id: str,
    manual_mode: str,
    manual_backend: str,
    successful_scenes: List[str],
    selected_scenes: List[str],
) -> None:
    payload = {
        "run_name": run_name,
        "wrapper_dir": str(wrapper_dir),
        "scene_root": str(scene_root),
        "gt_root": str(gt_root),
        "output_base": str(output_base),
        "backend": backend,
        "scene_image_subdir": scene_image_subdir,
        "generated_relpath_template": generated_relpath,
        "generated_relpath_resolved_example": generated_relpath.format(backend=backend) if backend != "none" else "",
        "register_script": str(register_script),
        "bufferx_root": str(bufferx_root),
        "bufferx_env": bufferx_env,
        "experiment_id": experiment_id,
        "manual_mode": manual_mode,
        "manual_backend": manual_backend,
        "selected_scenes": selected_scenes,
        "successful_scenes": successful_scenes,
        "path_contract": {
            "images": "<scene-root>/<scene>/" + scene_image_subdir + "/",
            "generated_reconstruction": "<scene-root>/<scene>/" + generated_relpath,
            "wrapper_outputs": "<output-base>/<run-name>/_wrapper/",
            "registration_outputs": "<output-base>/<run-name>/<scene>/{raw,manual,bufferx,icp,viz}/",
        },
    }
    (wrapper_dir / "run_manifest.json").write_text(json.dumps(payload, indent=2))


def main() -> int:
    args = parse_args()

    scene_root = Path(args.scene_root).expanduser().resolve()
    gt_root = Path(args.gt_root).expanduser().resolve()
    output_base = Path(args.output_base).expanduser().resolve()
    register_script = Path(args.register_script).expanduser().resolve()
    bufferx_root = Path(args.bufferx_root).expanduser().resolve()

    run_name = args.run_name or f"reconstruct_then_register_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_root = output_base / run_name
    wrapper_dir = run_root / "_wrapper"
    wrapper_dir.mkdir(parents=True, exist_ok=True)

    scenes = list_scene_names(scene_root, gt_root, args.scene_names)
    print(f"Found {len(scenes)} scene(s): {', '.join(scenes) if scenes else '<none>'}")
    print(f"Scene root:   {scene_root}")
    print(f"GT root:      {gt_root}")
    print(f"Output base:  {output_base}")
    print(f"Run root:     {run_root}")
    print(f"Backend:      {args.backend}")
    print(f"Images subdir:{args.scene_image_subdir}")
    print(f"Gen relpath:  {args.generated_relpath}")

    if not scenes:
        print("No matching scenes found.")
        return 1

    template_map = {
        "vggt": args.vggt_cmd_template,
        "fast3r": args.fast3r_cmd_template,
    }

    if args.backend != "none" and not template_map.get(args.backend):
        raise SystemExit(f"--backend {args.backend} needs a matching command template")

    reconstruction_rows: List[Dict[str, str]] = []
    successful_scenes: List[str] = []

    for idx, scene_name in enumerate(scenes, start=1):
        print(f"\n=== [reconstruct {idx}/{len(scenes)}] {scene_name} ===")

        scene_dir = scene_root / scene_name
        gt_scene_dir = gt_root / scene_name
        image_dir = scene_dir / args.scene_image_subdir
        out_rel = args.generated_relpath.format(backend=args.backend)
        out_ply = scene_dir / out_rel

        row_base = {
            "scene": scene_name,
            "backend": args.backend,
            "scene_dir": str(scene_dir),
            "gt_scene_dir": str(gt_scene_dir),
            "image_dir": str(image_dir),
            "generated_relpath": out_rel,
            "out_ply": str(out_ply),
            "run_root": str(run_root),
            "wrapper_dir": str(wrapper_dir),
        }

        if not image_dir.exists():
            row = {
                **row_base,
                "status": "missing_images",
            }
            reconstruction_rows.append(row)
            print(f"[SKIP] Missing images dir: {image_dir}")
            if not args.continue_on_recon_failure:
                break
            continue

        if args.backend == "none":
            row = {
                **row_base,
                "status": "skipped_backend_none",
            }
            reconstruction_rows.append(row)
            successful_scenes.append(scene_name)
            continue

        if args.skip_reconstruction_if_exists and out_ply.exists():
            row = {
                **row_base,
                "status": "reused_existing",
            }
            reconstruction_rows.append(row)
            successful_scenes.append(scene_name)
            print(f"[OK] Reusing existing reconstruction: {out_ply}")
            continue

        out_ply.parent.mkdir(parents=True, exist_ok=True)

        cmd = make_recon_cmd(
            template=template_map[args.backend],
            scene_dir=scene_dir,
            image_dir=image_dir,
            out_ply=out_ply,
            scene_name=scene_name,
            backend=args.backend,
        )

        res = run_cmd(cmd, dry_run=args.dry_run)

        scene_log_dir = wrapper_dir / "reconstruction_logs" / scene_name
        scene_log_dir.mkdir(parents=True, exist_ok=True)
        (scene_log_dir / "stdout.log").write_text(res.stdout or "")
        (scene_log_dir / "stderr.log").write_text(res.stderr or "")

        if res.returncode != 0:
            row = {
                **row_base,
                "status": "reconstruction_failed",
                "stderr_tail": (res.stderr or "")[-2000:],
                "stdout_log": str(scene_log_dir / "stdout.log"),
                "stderr_log": str(scene_log_dir / "stderr.log"),
            }
            reconstruction_rows.append(row)
            print("[FAIL] Reconstruction failed")
            if not args.continue_on_recon_failure:
                break
            continue

        if not args.dry_run and not out_ply.exists():
            row = {
                **row_base,
                "status": "reconstruction_missing_output",
                "stdout_log": str(scene_log_dir / "stdout.log"),
                "stderr_log": str(scene_log_dir / "stderr.log"),
            }
            reconstruction_rows.append(row)
            print(f"[FAIL] Reconstruction did not create expected output: {out_ply}")
            if not args.continue_on_recon_failure:
                break
            continue

        row = {
            **row_base,
            "status": "ok",
            "stdout_log": str(scene_log_dir / "stdout.log"),
            "stderr_log": str(scene_log_dir / "stderr.log"),
        }
        reconstruction_rows.append(row)
        successful_scenes.append(scene_name)
        print(f"[OK] Reconstruction ready: {out_ply}")

    manifest_json = wrapper_dir / "reconstruction_manifest.json"
    manifest_csv = wrapper_dir / "reconstruction_manifest.csv"
    manifest_json.write_text(json.dumps(reconstruction_rows, indent=2))

    fieldnames = sorted({k for row in reconstruction_rows for k in row.keys()}) if reconstruction_rows else ["scene", "status"]
    with manifest_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(reconstruction_rows)

    write_wrapper_run_manifest(
        wrapper_dir=wrapper_dir,
        scene_root=scene_root,
        gt_root=gt_root,
        output_base=output_base,
        run_name=run_name,
        backend=args.backend,
        generated_relpath=args.generated_relpath,
        scene_image_subdir=args.scene_image_subdir,
        register_script=register_script,
        bufferx_root=bufferx_root,
        bufferx_env=args.bufferx_env,
        experiment_id=args.experiment_id,
        manual_mode=args.manual_mode,
        manual_backend=args.manual_backend,
        successful_scenes=successful_scenes,
        selected_scenes=scenes,
    )

    print(f"\nWrapper run manifest:    {wrapper_dir / 'run_manifest.json'}")
    print(f"Reconstruction manifest: {manifest_json}")
    print(f"Reconstruction CSV:      {manifest_csv}")

    if not successful_scenes:
        print("No successful scenes to register.")
        return 2

    # Prefer generated reconstruction first, then fall back to old candidates.
    recon_candidates = []
    if args.backend != "none":
        recon_candidates.append(args.generated_relpath.format(backend=args.backend))
    recon_candidates.extend(["textured.ply", "textured.obj", "sparse/points.ply"])

    register_cmd = [
        "conda", "run", "-n", args.bufferx_env,
        "python3",
        str(register_script),
        "--recon-root", str(scene_root),
        "--gt-root", str(gt_root),
        "--output-base", str(output_base),
        "--run-name", run_name,
        "--bufferx-root", str(bufferx_root),
        "--bufferx-env", args.bufferx_env,
        "--experiment-id", args.experiment_id,
        "--manual-mode", args.manual_mode,
        "--manual-backend", args.manual_backend,
        "--recon-candidates",
        *recon_candidates,
        "--scene-names",
        *successful_scenes,
    ]

    register_cmd.extend(args.register_extra_args)

    print(f"\n=== [register] {len(successful_scenes)} scene(s) ===")
    print(f"Registration outputs will be under: {run_root}")
    print(f"Registration will search recon candidates in this order: {recon_candidates}")

    reg_res = run_cmd(register_cmd, dry_run=args.dry_run)

    (wrapper_dir / "register_stdout.log").write_text(reg_res.stdout or "")
    (wrapper_dir / "register_stderr.log").write_text(reg_res.stderr or "")

    if reg_res.returncode != 0:
        print("[FAIL] Registration pipeline failed")
        print((reg_res.stderr or "")[-3000:])
        return reg_res.returncode

    print("[OK] Registration pipeline completed")
    print(f"Run folder: {run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())