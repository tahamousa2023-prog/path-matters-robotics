#!/usr/bin/env python3
import math
import subprocess
from pathlib import Path

SCRIPT = Path("/home/AP_PathMatters/path_matters/Isaacsim/scripts/pcd_preprocess/14_12_convert_mesh.py")

ROOT = Path("/home/AP_PathMatters/path_matters/datasets/Synthetic_datasets_Haroun_Aziz/Objaverse_named_OBJ")
YODA = Path("//home/AP_PathMatters/path_matters/datasets/yoda/Baby_Yoda_v2.2.stl")  # <-- set this

N_REF = 50_000
N_MIN = 10_000
N_MAX = 200_000

# mode: "faces" (recommended with normalize=True) or "faces_area"
MODE = "faces"

def load_stats(obj_path: Path):
    """
    Returns (faces, area). Uses trimesh (reliable for area).
    """
    import trimesh
    mesh = trimesh.load(obj_path, force="mesh", process=False)
    # If OBJ loads as a Scene, merge:
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.dump()))
    faces = int(getattr(mesh, "faces", []).shape[0])
    area = float(getattr(mesh, "area", 0.0))
    return faces, area

def clip(x, lo, hi):
    return max(lo, min(hi, x))

def density_from_stats(faces, area, faces_ref, area_ref):
    if faces <= 0 or faces_ref <= 0:
        return N_REF  # fallback

    scale_faces = math.sqrt(faces / faces_ref)

    if MODE == "faces_area":
        # use area gently (sqrt) so it doesn't explode
        if area > 0 and area_ref > 0:
            scale_area = math.sqrt(area / area_ref)
        else:
            scale_area = 1.0
        n = N_REF * scale_faces * scale_area
    else:
        n = N_REF * scale_faces

    return int(round(clip(n, N_MIN, N_MAX)))

def main():
    faces_ref, area_ref = load_stats(YODA)
    print(f"[REF] yoda: faces={faces_ref}, area={area_ref:.6f}, N_REF={N_REF}")

    objs = sorted(ROOT.rglob("*.obj"))
    print(f"Found {len(objs)} obj files under {ROOT}")

    ok = 0
    for p in objs:
        faces, area = load_stats(p)
        dens = density_from_stats(faces, area, faces_ref, area_ref)

        print(f"\n{p}")
        print(f"  faces={faces} area={area:.6f} -> density={dens}")

        # THIS calls your exact existing script
        res = subprocess.run(["python", str(SCRIPT), str(p), str(dens)])
        if res.returncode == 0:
            ok += 1

    print(f"\nDone: {ok}/{len(objs)} converted")

if __name__ == "__main__":
    main()
