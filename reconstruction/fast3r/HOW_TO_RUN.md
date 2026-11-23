# Fast3R: how to run

Reconstruct a scene and align it to the ground-truth model with ICP:

```bash
python fast3r_reconstruction.py \
  --scene_dir /path/to/datasets/pix3d/img/misc/Bowl_white \
  --run_icp \
  --object_ply /path/to/datasets/pix3d/img/misc/Bowl_white/model.ply \
  --visualize_icp \
  --show_before
```

Convert a ground-truth mesh from OBJ to PLY:

```bash
python -c "import trimesh; m = trimesh.load('/path/to/model.obj'); m.export('/path/to/model.ply')"
```
