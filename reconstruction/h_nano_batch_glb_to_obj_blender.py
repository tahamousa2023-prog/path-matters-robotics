import bpy
import os
import re
import sys
from pathlib import Path

UID_RE = re.compile(r"^(.*)__[0-9a-f]{32}$", re.IGNORECASE)

def starting_name(stem: str) -> str:
    m = UID_RE.match(stem)
    return m.group(1) if m else stem

def unique_dir(path: Path) -> Path:
    if not path.exists():
        return path
    k = 2
    while True:
        cand = Path(f"{path}_{k}")
        if not cand.exists():
            return cand
        k += 1

def parse_args():
    # Blender passes args before/after "--"
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    # very small arg parser
    in_dir = None
    out_dir = None
    for i, a in enumerate(argv):
        if a == "--in_dir" and i + 1 < len(argv):
            in_dir = argv[i + 1]
        if a == "--out_dir" and i + 1 < len(argv):
            out_dir = argv[i + 1]
    if not in_dir or not out_dir:
        raise SystemExit("Usage: blender -b -P script.py -- --in_dir <DIR> --out_dir <DIR>")
    return Path(in_dir).expanduser().resolve(), Path(out_dir).expanduser().resolve()

def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)

def import_glb(glb_path: Path):
    # GLB/GLTF importer
    bpy.ops.import_scene.gltf(filepath=str(glb_path))

def select_all_meshes():
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.data.objects:
        obj.select_set(obj.type == "MESH")
    # set active object (needed for join sometimes)
    meshes = [o for o in bpy.context.selected_objects if o.type == "MESH"]
    if meshes:
        bpy.context.view_layer.objects.active = meshes[0]
    return meshes

def join_meshes_if_needed():
    meshes = [o for o in bpy.context.selected_objects if o.type == "MESH"]
    if len(meshes) > 1:
        bpy.ops.object.join()

def export_obj(obj_path: Path):
    # Blender 4+ uses bpy.ops.wm.obj_export (not export_scene.obj)
    # We keep a fallback just in case.
    obj_path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(bpy.ops.wm, "obj_export"):
        bpy.ops.wm.obj_export(
            filepath=str(obj_path),
            export_selected_objects=True,
            export_uv=True,
            export_normals=True,
            export_materials=True,
            path_mode='COPY',  # copy textures next to OBJ when possible
        )
    elif hasattr(bpy.ops.export_scene, "obj"):
        bpy.ops.export_scene.obj(
            filepath=str(obj_path),
            use_selection=True,
            use_uvs=True,
            use_normals=True,
            use_materials=True,
            path_mode='COPY',
        )
    else:
        raise RuntimeError("No OBJ export operator found in this Blender build.")

def main():
    in_dir, out_dir = parse_args()
    out_dir.mkdir(parents=True, exist_ok=True)

    glbs = sorted(in_dir.glob("*.glb"))
    if not glbs:
        raise SystemExit(f"No .glb files found in {in_dir}")

    print(f"Found {len(glbs)} GLBs in {in_dir}")
    print(f"Writing OBJ folders to {out_dir}")

    for idx, glb in enumerate(glbs, 1):
        stem = glb.stem
        name = starting_name(stem)

        target_folder = unique_dir(out_dir / name)
        target_folder.mkdir(parents=True, exist_ok=True)
        obj_path = target_folder / f"{name}.obj"

        reset_scene()
        import_glb(glb)

        meshes = select_all_meshes()
        if not meshes:
            print(f"[{idx}/{len(glbs)}] SKIP (no meshes): {glb.name}")
            continue

        # optional: join into one mesh (OBJ is easier to handle)
        join_meshes_if_needed()

        export_obj(obj_path)
        print(f"[{idx}/{len(glbs)}] OK: {glb.name} -> {obj_path}")

if __name__ == "__main__":
    main()
