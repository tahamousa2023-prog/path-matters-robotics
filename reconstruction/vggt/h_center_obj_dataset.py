#!/usr/bin/env python3
"""
h_center_obj_dataset.py

Batch-center OBJ meshes so they:
- are centered in the horizontal plane (XY by default)
- remain above the ground plane by placing min(up_axis) = clearance

It preserves OBJ structure/material refs by rewriting ONLY 'v ' vertex lines.

Typical use for Isaac Sim:
- up axis is Z
- clearance ~ 0.001 to avoid z-fighting

Examples:
  # write to a new folder and copy sidecar files (mtl/textures/ply/etc.)
  python3 h_center_obj_dataset.py /data/Objaverse_named_OBJS \
    --out-root /data/Objaverse_named_OBJS_grounded \
    --copy-sidecars \
    --center-method centroid \
    --up z \
    --clearance 0.001

  # overwrite in place (creates .bak once)
  python3 h_center_obj_dataset.py /data/Objaverse_named_OBJS --inplace
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import trimesh


def axis_index(axis: str) -> int:
    axis = axis.lower()
    if axis == "x":
        return 0
    if axis == "y":
        return 1
    if axis == "z":
        return 2
    raise ValueError(f"Invalid axis: {axis} (use x/y/z)")


def load_baked_mesh(obj_path: Path) -> trimesh.Trimesh:
    # force scene so multi-part OBJs are handled consistently
    scene = trimesh.load(obj_path, force="scene", process=False)
    mesh = scene.to_mesh()  # bakes transforms + concatenates
    if mesh.is_empty:
        raise ValueError("Empty mesh after loading")
    return mesh


def compute_center(mesh: trimesh.Trimesh, method: str) -> np.ndarray:
    """
    method:
      - centroid: area-weighted triangle centroid
      - bounds: AABB center
    """
    if method == "centroid":
        return np.asarray(mesh.centroid, dtype=float)
    if method == "bounds":
        b = np.asarray(mesh.bounds, dtype=float)  # [[min],[max]]
        return b.mean(axis=0)
    raise ValueError(f"Unknown center method: {method}")


def compute_offset(mesh: trimesh.Trimesh, center_method: str, up: str, clearance: float) -> np.ndarray:
    """
    We subtract this offset from every vertex.

    - Center in the two horizontal axes
    - For the up axis, shift so min(up) becomes (0 + clearance)
    """
    up_i = axis_index(up)
    b = np.asarray(mesh.bounds, dtype=float)
    center = compute_center(mesh, center_method)

    # horizontal axes are the two axes that are NOT up
    horiz = [0, 1, 2]
    horiz.remove(up_i)

    offset = np.zeros(3, dtype=float)

    # center horizontally
    offset[horiz[0]] = center[horiz[0]]
    offset[horiz[1]] = center[horiz[1]]

    # ground on up axis: make min_up == clearance
    min_up = b[0, up_i]
    offset[up_i] = min_up - float(clearance)

    return offset


def rewrite_obj_vertices(in_obj: Path, out_obj: Path, offset: np.ndarray, decimals: int = 6) -> None:
    """
    Rewrite only geometric vertices ('v ' lines) and subtract offset.
    Preserves everything else (mtllib/usemtl/groups/faces/etc).
    """
    offset = np.asarray(offset, dtype=float)
    fmt = f"{{:.{decimals}f}}"

    out_obj.parent.mkdir(parents=True, exist_ok=True)

    with in_obj.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    out_lines = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("v "):
            # preserve indentation
            prefix = line[: len(line) - len(stripped)]
            parts = stripped.strip().split()
            # ["v", x, y, z, ...optional...]
            if len(parts) >= 4:
                try:
                    x, y, z = map(float, parts[1:4])
                except ValueError:
                    out_lines.append(line)
                    continue

                x -= offset[0]
                y -= offset[1]
                z -= offset[2]

                rest = parts[4:]  # keep w / vertex colors etc.
                new_line = (
                    prefix
                    + "v "
                    + fmt.format(x) + " "
                    + fmt.format(y) + " "
                    + fmt.format(z)
                )
                if rest:
                    new_line += " " + " ".join(rest)
                out_lines.append(new_line + "\n")
            else:
                out_lines.append(line)
        else:
            out_lines.append(line)

    with out_obj.open("w", encoding="utf-8") as f:
        f.writelines(out_lines)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Center OBJ meshes in horizontal plane and keep them above ground by placing min(up)=clearance."
    )
    p.add_argument("root", type=Path, help="Root directory to search recursively for .obj")
    p.add_argument("--inplace", action="store_true", help="Overwrite OBJ files in place (creates backups)")
    p.add_argument("--out-root", type=Path, default=None, help="Write results under this root (keeps rel paths)")
    p.add_argument("--copy-sidecars", action="store_true",
                   help="When using --out-root, copy other files in the same folder (mtl/textures/ply/etc.)")
    p.add_argument("--backup-ext", type=str, default=".bak", help="Backup extension for inplace mode")
    p.add_argument("--center-method", choices=["centroid", "bounds"], default="centroid",
                   help="How to center horizontally: centroid (surface centroid) or bounds (AABB center)")
    p.add_argument("--up", choices=["x", "y", "z"], default="z", help="Which axis is 'up'")
    p.add_argument("--clearance", type=float, default=0.001, help="Extra lift above the ground plane")
    p.add_argument("--decimals", type=int, default=6, help="Decimal places for rewritten vertex coords")
    args = p.parse_args()

    if not args.root.exists():
        print(f"Root not found: {args.root}", file=sys.stderr)
        return 2

    if not args.inplace and args.out_root is None:
        print("Pick one: --inplace OR --out-root <dir>", file=sys.stderr)
        return 2

    obj_files = sorted(list(args.root.rglob("*.obj")) + list(args.root.rglob("*.OBJ")))
    if not obj_files:
        print(f"No OBJ files found under {args.root}")
        return 0

    ok, fail = 0, 0
    for in_obj in obj_files:
        try:
            mesh = load_baked_mesh(in_obj)
            offset = compute_offset(mesh, args.center_method, args.up, args.clearance)

            if args.inplace:
                backup = in_obj.with_name(in_obj.name + args.backup_ext)
                if not backup.exists():
                    shutil.copy2(in_obj, backup)
                out_obj = in_obj
            else:
                rel = in_obj.relative_to(args.root)
                out_obj = args.out_root / rel
                if args.copy_sidecars:
                    out_dir = out_obj.parent
                    out_dir.mkdir(parents=True, exist_ok=True)
                    for f in in_obj.parent.iterdir():
                        if f.is_file() and f.suffix.lower() != ".obj":
                            dst = out_dir / f.name
                            if not dst.exists():
                                shutil.copy2(f, dst)

            rewrite_obj_vertices(in_obj, out_obj, offset, decimals=args.decimals)
            ok += 1
            print(f"[OK] {in_obj}")

        except Exception as e:
            fail += 1
            print(f"[FAIL] {in_obj}: {e}", file=sys.stderr)

    print(f"\nDone. OK={ok}, FAIL={fail}, total={len(obj_files)}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
