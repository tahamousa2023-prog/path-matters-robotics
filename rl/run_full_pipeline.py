"""
experiments/taha/run_full_pipeline.py

Full pipeline wrapper — Taha.

Chains together:
    Step 1 → auto_crop.py               (automatic background removal)
    Step 2 → run_cc_bufferx_pipeline.py (BUFFER-X + ICP, original untouched)

Usage:
    conda activate bufferx_o3d

    # Single scene:
    python experiments/taha/run_full_pipeline.py \
        --scene-names air_conditioner_control_camera3 \
        --run-name taha_auto_01

    # All scenes:
    python experiments/taha/run_full_pipeline.py \
        --run-name taha_batch_01 \
        --cluster-eps 0.05 \
        --cluster-min-pts 30 \
        --max-fraction 0.15 \
        --min-fraction 0.05

    # Skip auto-crop (reuse existing cleaned file):
    python experiments/taha/run_full_pipeline.py \
        --scene-names air_conditioner_control_camera3 \
        --run-name taha_auto_01 \
        --skip-crop

    # Diagnose one scene without running pipeline:
    python experiments/taha/run_full_pipeline.py \
        --scene-names plastic_hammer_camera3 \
        --run-name debug \
        --diagnose-only
"""

from __future__ import annotations
import argparse
import subprocess
import sys
import json
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
# PATHS — edit these if your folder layout changes
# ══════════════════════════════════════════════════════════════════════════════

SCENE_ROOT = (
    "/home/AP_PathMatters/path_matters/datasets/"
    "Reallife_Dataset_Haroun_Aziz/scenes-others_SUBSAMPLED"
)

GT_ROOT = (
    "/home/AP_PathMatters/path_matters/datasets/"
    "Reallife_Dataset_Haroun_Aziz/scenes-others_SUBSAMPLED"
)

OUTPUT_BASE = "/home/AP_PathMatters/path_matters/runs/taha"

PIPELINE_SCRIPT = (
    "/home/AP_PathMatters/path_matters/haroun/Pipeline/"
    "cc_bufferx_pipeline_package/run_cc_bufferx_pipeline.py"
)

BUFFERX_ROOT = "/home/AP_PathMatters/BUFFER-X"

THIS_DIR = Path(__file__).parent

# Folder names to skip (not real scenes)
SKIP_NAMES = {"_batch_run", "_checks", "_logs_batch", "_logs_rebuild",
              "batch_summary.csv"}


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def run_cmd(cmd: list[str], step_name: str) -> bool:
    """Run a shell command, print output live, return True on success."""
    print(f"\n{'─'*65}")
    print(f"  STEP: {step_name}")
    print(f"  CMD:  {' '.join(cmd)}")
    print(f"{'─'*65}")

    result = subprocess.run(cmd, text=True)

    if result.returncode != 0:
        print(f"\n  ✗ FAILED: {step_name}  (exit code {result.returncode})")
        return False

    print(f"\n  ✓ OK: {step_name}")
    return True


def find_recon_file(scene_root: str, scene_name: str) -> Path | None:
    """
    Find the best reconstruction file for a scene.
    Priority:
        1. VGGT points.ply      — best quality (only exists for 1 scene)
        2. sparse/points.ply    — colmap sparse, works for all others
        3. textured.ply         — fallback mesh
        4. textured.obj         — last resort
    """
    candidates = [
        "recon_generated/vggt/points.ply",
        "sparse/points.ply",
        "textured.ply",
        "textured.obj",
    ]
    base = Path(scene_root) / scene_name
    for c in candidates:
        p = base / c
        if p.exists():
            print(f"    Using: {c}")
            return p
    return None


def read_icp_result(output_base: str, run_name: str,
                    scene_name: str) -> dict | None:
    """Read ICP summary JSON and return metrics."""
    icp_json = (Path(output_base) / run_name / scene_name /
                "icp" / "icp_summary.json")
    if not icp_json.exists():
        return None
    with open(icp_json) as f:
        data = json.load(f)

    # Handle all possible key names the pipeline may use
    fitness = (data.get("icp_fitness")
               or data.get("fitness")
               or data.get("icp_score")
               or "N/A")
    rmse = (data.get("icp_inlier_rmse")
            or data.get("inlier_rmse")
            or data.get("rmse")
            or data.get("eval_rmse")
            or "N/A")

    return {"fitness": fitness, "rmse": rmse, "raw": data}


def format_result_line(scene: str, result) -> str:
    """Format one summary line safely regardless of value types."""
    if not isinstance(result, dict):
        return f"  ✗  {scene:<50} {result}"

    fitness = result.get("fitness", "N/A")
    rmse    = result.get("rmse",    "N/A")

    f_str = f"{fitness:.4f}" if isinstance(fitness, float) else str(fitness)
    r_str = f"{rmse:.4f}"    if isinstance(rmse,    float) else str(rmse)

    return f"  ✓  {scene:<50} fitness={f_str}  rmse={r_str}"


# ══════════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSING
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Taha full pipeline: auto-crop → BUFFER-X → ICP"
    )

    # ── Scene selection ───────────────────────────────────────────────────
    parser.add_argument(
        "--scene-names", nargs="+", default=None,
        help="Scene names to process. Default: all scenes in SCENE_ROOT.",
    )
    parser.add_argument(
        "--run-name", type=str, default="taha_auto_01",
        help="Name for this pipeline run.",
    )

    # ── Auto-crop control ─────────────────────────────────────────────────
    parser.add_argument(
        "--skip-crop", action="store_true",
        help="Skip auto-crop and reuse existing points_cleaned.ply",
    )
    parser.add_argument(
        "--diagnose-only", action="store_true",
        help="Run auto-crop in diagnose mode only, do not run pipeline.",
    )
    parser.add_argument(
        "--std-ratio",        type=float, default=2.0,
        help="Outlier removal aggressiveness (lower = stricter)",
    )
    parser.add_argument(
        "--plane-distance",   type=float, default=0.01,
        help="RANSAC plane inlier distance in meters",
    )
    parser.add_argument(
        "--cluster-eps",      type=float, default=0.05,
        help="DBSCAN neighborhood radius in meters",
    )
    parser.add_argument(
        "--cluster-min-pts",  type=int,   default=30,
        help="DBSCAN minimum cluster size",
    )
    parser.add_argument(
        "--min-fraction",     type=float, default=0.05,
        help="Min cluster size as fraction of valid points",
    )
    parser.add_argument(
        "--max-fraction",     type=float, default=0.15,
        help="Max cluster size as fraction — set below background blob",
    )
    parser.add_argument(
        "--scene-root", type=str, default=SCENE_ROOT,
        help="Override the default SCENE_ROOT path",
    )
    parser.add_argument(
        "--gt-root", type=str, default=GT_ROOT,
        help="Override the default GT_ROOT path",
    )

    # ── Pipeline control ──────────────────────────────────────────────────
    parser.add_argument(
        "--show-viz", action="store_true",
        help="Open interactive visualization window after ICP",
    )

    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # ── Find scenes ───────────────────────────────────────────────────────
    scene_root = args.scene_root
    gt_root    = args.gt_root

    if args.scene_names:
        scenes = args.scene_names
    else:
        scenes = sorted([
            p.name for p in Path(scene_root).iterdir()
            if p.is_dir() and p.name not in SKIP_NAMES
        ])

    print("\n" + "═"*65)
    print("  TAHA — FULL AUTO PIPELINE")
    print("═"*65)
    print(f"  Run name    : {args.run_name}")
    print(f"  Scenes      : {len(scenes)}")
    print(f"  Output      : {OUTPUT_BASE}/{args.run_name}/")
    print(f"  Auto-crop   : {'SKIP (reuse existing)' if args.skip_crop else 'YES'}")
    print(f"  Diagnose    : {'YES (no pipeline)' if args.diagnose_only else 'NO'}")
    print(f"  eps         : {args.cluster_eps}")
    print(f"  min/max frac: {args.min_fraction} / {args.max_fraction}")
    print("═"*65)

    results     = {}
    n_success   = 0
    n_failed    = 0
    n_no_recon  = 0

    for scene_name in scenes:
        print(f"\n\n{'━'*65}")
        print(f"  SCENE: {scene_name}")
        print(f"{'━'*65}")

        # ── Step 1: Find reconstruction file ──────────────────────────────
        raw_recon = find_recon_file(SCENE_ROOT, scene_name)
        if raw_recon is None:
            print(f"  ✗ No reconstruction file found — skipping.")
            results[scene_name] = "no_recon"
            n_no_recon += 1
            continue

        print(f"  Found: {raw_recon}")

        # Cleaned output path (always same location regardless of input)
        cleaned_path = (
            Path(SCENE_ROOT) / scene_name /
            "recon_generated" / "vggt" / "points_cleaned.ply"
        )

        # ── Step 2: Auto-crop ──────────────────────────────────────────────
        if args.diagnose_only:
            # Diagnose only — print clusters, do not save, do not run pipeline
            run_cmd(
                cmd=[
                    sys.executable,
                    str(THIS_DIR / "auto_crop.py"),
                    "--input",            str(raw_recon),
                    "--output",           str(cleaned_path),
                    "--std-ratio",        str(args.std_ratio),
                    "--plane-distance",   str(args.plane_distance),
                    "--cluster-eps",      str(args.cluster_eps),
                    "--cluster-min-pts",  str(args.cluster_min_pts),
                    "--max-fraction",     str(args.max_fraction),
                    "--min-fraction",     str(args.min_fraction),
                    "--diagnose",
                ],
                step_name=f"Diagnose: {scene_name}",
            )
            results[scene_name] = "diagnose_only"
            continue

        elif not args.skip_crop:
            crop_ok = run_cmd(
                cmd=[
                    sys.executable,
                    str(THIS_DIR / "auto_crop.py"),
                    "--input",            str(raw_recon),
                    "--output",           str(cleaned_path),
                    "--std-ratio",        str(args.std_ratio),
                    "--plane-distance",   str(args.plane_distance),
                    "--cluster-eps",      str(args.cluster_eps),
                    "--cluster-min-pts",  str(args.cluster_min_pts),
                    "--max-fraction",     str(args.max_fraction),
                    "--min-fraction",     str(args.min_fraction),
                ],
                step_name=f"Auto-crop: {scene_name}",
            )

            if not crop_ok:
                # Auto-crop failed — fall back to raw file
                # BUFFER-X will attempt alignment on uncropped cloud
                print(f"  → Auto-crop failed — using raw reconstruction.")
                print(f"     BUFFER-X will attempt alignment without cropping.")
                cleaned_path = raw_recon

        else:
            # --skip-crop: reuse existing cleaned file
            if not cleaned_path.exists():
                print(f"  ✗ --skip-crop set but cleaned file does not exist:")
                print(f"    {cleaned_path}")
                results[scene_name] = "no_cleaned_file"
                n_failed += 1
                continue
            print(f"  Reusing: {cleaned_path}")

        # ── Step 3: BUFFER-X + ICP ─────────────────────────────────────────
        # Compute relative path of cleaned file for --recon-candidates
        try:
            cleaned_rel = str(
                cleaned_path.relative_to(Path(SCENE_ROOT) / scene_name)
            )
        except ValueError:
            # cleaned_path is outside scene dir (raw fallback) — use absolute
            cleaned_rel = str(cleaned_path)

        pipeline_cmd = [
            sys.executable,
            PIPELINE_SCRIPT,
            "--recon-root",        SCENE_ROOT,
            "--gt-root",           GT_ROOT,
            "--output-base",       OUTPUT_BASE,
            "--run-name",          args.run_name,
            "--bufferx-root",      BUFFERX_ROOT,
            "--bufferx-env",       "bufferx_o3d",
            "--scene-names",       scene_name,
            "--recon-candidates",
                cleaned_rel,
                "recon_generated/vggt/points.ply",
                "sparse/points.ply",
                "textured.ply",
            "--manual-mode",       "off",
            "--save-viz",
        ]

        if args.show_viz:
            pipeline_cmd.append("--show-final-viz")

        pipeline_ok = run_cmd(
            cmd=pipeline_cmd,
            step_name=f"BUFFER-X + ICP: {scene_name}",
        )

        if not pipeline_ok:
            results[scene_name] = "pipeline_failed"
            n_failed += 1
            continue

        # ── Step 4: Read ICP results ───────────────────────────────────────
        icp = read_icp_result(OUTPUT_BASE, args.run_name, scene_name)
        if icp:
            fitness = icp.get("fitness", "N/A")
            rmse    = icp.get("rmse",    "N/A")
            f_str   = f"{fitness:.4f}" if isinstance(fitness, float) else str(fitness)
            r_str   = f"{rmse:.4f}"    if isinstance(rmse,    float) else str(rmse)
            print(f"\n  ╔══ ICP RESULT: {scene_name}")
            print(f"  ║   Fitness   : {f_str}")
            print(f"  ║   RMSE      : {r_str}")
            print(f"  ╚══════════════")
            results[scene_name] = {"status": "ok", "fitness": fitness, "rmse": rmse}
            n_success += 1
        else:
            results[scene_name] = "no_icp_output"
            n_failed += 1

    # ── Final summary ──────────────────────────────────────────────────────
    print("\n\n" + "═"*65)
    print("  PIPELINE SUMMARY")
    print("═"*65)
    print(f"  Total: {len(scenes)}  |  "
          f"Success: {n_success}  |  "
          f"Failed: {n_failed}  |  "
          f"No recon: {n_no_recon}")
    print("─"*65)
    for scene, result in results.items():
        print(format_result_line(scene, result))
    print("═"*65 + "\n")


if __name__ == "__main__":
    main()