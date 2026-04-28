"""
Single Camera Multi-Position Capture (Improved)
- Computes TRUE object center from WORLD AABB (not prim translation)
- Optionally grounds object so min Z sits on Z=0 (+ clearance)
- Auto-scales camera radius/height from object size for consistent framing
"""

import os
import time
import glob
import numpy as np
from datetime import datetime

import omni.kit.app
import omni.usd
from omni.isaac.core import World
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.bounds import compute_combined_aabb
from pxr import Usd, UsdGeom, Gf
from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file

from scipy.spatial.transform import Rotation as R

# ============================================================
# CONFIG
# ============================================================
TARGET_OBJECT_PATH = "/World/phonograph_record__gramophone__55adaf6069c048bfb4f7b5f47c98384d_2"
CAMERA_PATH = "/World/Camera"
OUTPUT_DIR = "/home/AP_PathMatters/path_matters/datasets/Synthetic_datasets_Haroun_Aziz/Objaverse_named_OBJ/phonograph_record__gramophone__55adaf6069c048bfb4f7b5f47c98384d_2/images"

# CAPTURE PARAMETERS
NUM_POSITIONS = 8

# If AUTO_SCALE=True, these become fallbacks / minimums
CIRCLE_RADIUS = 3.0
CAMERA_HEIGHT_OFFSET = 1.5

AUTO_SCALE = True
MIN_RADIUS = 3.0
RADIUS_MULT = 6.0          # multiply by object XY radius
HEIGHT_MULT = 1          # multiply by object Z size
EXTRA_LOOK_AT_LIFT = 0.05   # lift look-at point a bit (e.g. 0.05) if you want more "top-down"

# Grounding
GROUND_OBJECT = True
GROUND_Z = 0.0
GROUND_CLEARANCE = 0.001   # small lift to avoid z-fighting

# BBox purposes
INCLUDED_PURPOSES = [
    UsdGeom.Tokens.default_,
    UsdGeom.Tokens.render,
    UsdGeom.Tokens.proxy,
    UsdGeom.Tokens.guide,
]

# ============================================================
# HELPERS
# ============================================================
def get_world_aabb_and_center(stage, prim_path: str):
    """
    Returns:
      min_pt (np.array shape (3,))
      max_pt (np.array shape (3,))
      center (np.array shape (3,))
      size   (np.array shape (3,))
    """
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        INCLUDED_PURPOSES,
        useExtentsHint=True,
    )

    aabb = compute_combined_aabb(bbox_cache, [prim_path])  # [minx,miny,minz,maxx,maxy,maxz]
    min_pt = np.array(aabb[:3], dtype=float)
    max_pt = np.array(aabb[3:], dtype=float)
    center = (min_pt + max_pt) / 2.0
    size = (max_pt - min_pt)
    return min_pt, max_pt, center, size


def get_or_add_translate_op(xformable: UsdGeom.Xformable, op_suffix: str):
    """
    Reuse an existing xformOp:translate:<suffix> if present, else add it.
    """
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate and op.GetOpName().endswith(":" + op_suffix):
            return op
    return xformable.AddTranslateOp(opSuffix=op_suffix)


def maybe_ground_object(target_prim):
    """
    Adjust a dedicated translate op so that min Z becomes (GROUND_Z + GROUND_CLEARANCE).
    Safe to run multiple times (it reuses the op and updates its value).
    """
    xformable = UsdGeom.Xformable(target_prim)
    translate_op = get_or_add_translate_op(xformable, "grounding")

    # current translate value of grounding op
    cur = translate_op.GetAttr().Get()
    if cur is None:
        cur = Gf.Vec3d(0.0, 0.0, 0.0)
    cur_np = np.array([cur[0], cur[1], cur[2]], dtype=float)

    # compute current world AABB
    stage = omni.usd.get_context().get_stage()
    min_pt, max_pt, center, size = get_world_aabb_and_center(stage, TARGET_OBJECT_PATH)

    # how much we need to shift along world Z to put min_z at ground
    desired_min_z = GROUND_Z + GROUND_CLEARANCE
    dz = desired_min_z - float(min_pt[2])

    if abs(dz) > 1e-6:
        new_val = cur_np + np.array([0.0, 0.0, dz], dtype=float)
        translate_op.Set(Gf.Vec3d(float(new_val[0]), float(new_val[1]), float(new_val[2])))
        print(f"✓ Grounding applied: dz={dz:.6f} (translate:grounding now {new_val})")
    else:
        print("✓ Grounding not needed (already on/above plane)")


# ============================================================
# START
# ============================================================
print("=" * 70)
print("  SINGLE CAMERA MULTI-POSITION CAPTURE (IMPROVED)")
print("=" * 70)
print()

print("Analyzing target object...")
target_prim = get_prim_at_path(TARGET_OBJECT_PATH)

if not target_prim:
    print(f"❌ Object not found at {TARGET_OBJECT_PATH}")
    raise SystemExit(1)

app = omni.kit.app.get_app()
world = World.instance() or World()

if not world.is_playing():
    world.reset()
    world.play()

# Let sim settle
for _ in range(30):
    app.update()

# Optionally ground object
if GROUND_OBJECT:
    maybe_ground_object(target_prim)
    for _ in range(10):
        app.update()

# Compute accurate center/size AFTER grounding
stage = omni.usd.get_context().get_stage()
min_pt, max_pt, OBJECT_CENTER, OBJECT_SIZE = get_world_aabb_and_center(stage, TARGET_OBJECT_PATH)

print(f"✓ AABB min: {min_pt}")
print(f"✓ AABB max: {max_pt}")
print(f"✓ Object center (AABB): [{OBJECT_CENTER[0]:.3f}, {OBJECT_CENTER[1]:.3f}, {OBJECT_CENTER[2]:.3f}]")
print(f"✓ Object size: [{OBJECT_SIZE[0]:.3f}, {OBJECT_SIZE[1]:.3f}, {OBJECT_SIZE[2]:.3f}]")
print()

# Auto-scale camera distance/height for consistent framing
if AUTO_SCALE:
    # XY "radius" based on half diagonal in XY
    xy_radius = 0.5 * float(np.linalg.norm(OBJECT_SIZE[:2]))
    CIRCLE_RADIUS = max(MIN_RADIUS, RADIUS_MULT * xy_radius)

    # Height offset based on object height
    CAMERA_HEIGHT_OFFSET = max(0.2, HEIGHT_MULT * float(OBJECT_SIZE[2]))

print(f"✓ Camera radius: {CIRCLE_RADIUS:.3f} m")
print(f"✓ Camera height offset: {CAMERA_HEIGHT_OFFSET:.3f} m\n")

# Output folder
session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = os.path.join(OUTPUT_DIR, f"multicam_{session_timestamp}")
os.makedirs(output_path, exist_ok=True)
print(f"Output: {output_path}\n")

# ============================================================
# CREATE/GET CAMERA
# ============================================================
print("Setting up camera...")
camera_prim = stage.GetPrimAtPath(CAMERA_PATH)
if not camera_prim:
    camera_prim = stage.DefinePrim(CAMERA_PATH, "Camera")

camera = UsdGeom.Camera(camera_prim)
camera.GetFocalLengthAttr().Set(24.0)

print(f"✓ Camera ready at {CAMERA_PATH}\n")

# Switch viewport to this camera
try:
    vp_api = get_active_viewport()
    vp_api.set_active_camera(CAMERA_PATH)
    print("✓ Viewport switched to camera\n")
except Exception as e:
    print(f"⚠️ Viewport switch failed: {e}\n")

# ============================================================
# CALCULATE POSITIONS
# ============================================================
print("Calculating camera positions...")

# Look-at point (optionally lifted a bit)
LOOK_AT = OBJECT_CENTER.copy()
LOOK_AT[2] += float(EXTRA_LOOK_AT_LIFT)

camera_z = float(OBJECT_CENTER[2] + CAMERA_HEIGHT_OFFSET)
positions = []

for i in range(NUM_POSITIONS):
    angle = 2 * np.pi * i / NUM_POSITIONS

    cam_x = float(OBJECT_CENTER[0] + CIRCLE_RADIUS * np.cos(angle))
    cam_y = float(OBJECT_CENTER[1] + CIRCLE_RADIUS * np.sin(angle))
    cam_z = float(camera_z)

    positions.append({"position": np.array([cam_x, cam_y, cam_z], dtype=float), "angle": angle})
    print(f"  Position {i+1}: [{cam_x:.3f}, {cam_y:.3f}, {cam_z:.3f}] (angle {np.degrees(angle):.0f}°)")

print(f"\n✓ {len(positions)} positions calculated\n")

# ============================================================
# MOVE CAMERA AND CAPTURE
# ============================================================
print("=" * 70)
print("  CAPTURING IMAGES")
print("=" * 70)
print()

captured_count = 0

for i, pos_data in enumerate(positions):
    print(f"\nPosition {i+1}/{NUM_POSITIONS}")
    print("-" * 40)

    cam_pos = pos_data["position"]

    # Look-at rotation
    direction = LOOK_AT - cam_pos
    direction = direction / np.linalg.norm(direction)

    # Isaac world is Z-up; keep camera up aligned to +Z
    up = np.array([0.0, 0.0, 1.0], dtype=float)

    # Camera -Z points at target
    z_axis = -direction
    x_axis = np.cross(up, z_axis)

    if np.linalg.norm(x_axis) < 1e-3:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        x_axis = x_axis / np.linalg.norm(x_axis)

    y_axis = np.cross(z_axis, x_axis)

    rot_matrix = np.array(
        [
            [x_axis[0], y_axis[0], z_axis[0]],
            [x_axis[1], y_axis[1], z_axis[1]],
            [x_axis[2], y_axis[2], z_axis[2]],
        ],
        dtype=float,
    )

    rotation = R.from_matrix(rot_matrix)
    euler = rotation.as_euler("xyz", degrees=True)

    # Move camera
    xform = UsdGeom.Xformable(camera_prim)
    xform.ClearXformOpOrder()

    translate_op = xform.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])))

    rotate_op = xform.AddRotateXYZOp()
    rotate_op.Set(Gf.Vec3f(float(euler[0]), float(euler[1]), float(euler[2])))

    # Update viewport
    for _ in range(20):
        app.update()

    time.sleep(0.3)

    filepath = os.path.join(output_path, f"view_{i:02d}.png")

    try:
        vp_api = get_active_viewport()
        capture_viewport_to_file(vp_api, file_path=filepath)

        time.sleep(0.2)

        if os.path.exists(filepath):
            size_kb = os.path.getsize(filepath) / 1024
            print(f"  ✅ Captured: view_{i:02d}.png ({size_kb:.1f} KB)")
            captured_count += 1
        else:
            print("  ❌ Failed to save")
    except Exception as e:
        print(f"  ❌ Error: {e}")

# ============================================================
# SUMMARY
# ============================================================
print(f"\n{'=' * 70}")
print("  CAPTURE COMPLETE")
print(f"{'=' * 70}")
print(f"\nPositions: {NUM_POSITIONS}")
print(f"Captured: {captured_count}/{NUM_POSITIONS}")
print(f"Location: {output_path}")

final_images = sorted(glob.glob(os.path.join(output_path, "view_*.png")))
if final_images:
    print("\nCaptured views:")
    for img in final_images:
        print(f"  ✓ {os.path.basename(img)}")

print("\n✓ Ready for 3D reconstruction!")
print("=" * 70 + "\n")
