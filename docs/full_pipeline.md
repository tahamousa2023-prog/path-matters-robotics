How to use with new data
Step 1 — pre-scale reconstruction to match GT size:
bashpython3 -c "
import open3d as o3d, numpy as np
src = o3d.io.read_point_cloud('PATH/TO/RECON.ply')
tgt = o3d.io.read_point_cloud('PATH/TO/GT.ply')
scale = np.linalg.norm(tgt.get_axis_aligned_bounding_box().get_extent()) / np.linalg.norm(src.get_axis_aligned_bounding_box().get_extent())
print(f'Scale: {scale:.4f}')
src.scale(scale, center=src.get_center())
o3d.io.write_point_cloud('PATH/TO/RECON_scaled.ply', src)
"
Step 2 — run BUFFER-X + ICP:
bashpython /home/AP_PathMatters/path_matters/haroun/Pipeline/cc_bufferx_pipeline_package/run_cc_bufferx_pipeline.py \
  --recon-root  SCENE_FOLDER \
  --gt-root     SCENE_FOLDER \
  --output-base /home/AP_PathMatters/path_matters/runs/taha \
  --run-name    MY_RUN_NAME \
  --bufferx-root /home/AP_PathMatters/BUFFER-X \
  --bufferx-env  bufferx_o3d \
  --scene-names  SCENE_NAME \
  --recon-candidates sparse/points_scaled.ply sparse/points.ply \
  --gt-candidates    textured.ply textured.obj \
  --manual-mode off \
  --save-viz \
  --show-final-viz
Step 3 — read result:
bashcat /home/AP_PathMatters/path_matters/runs/taha/MY_RUN_NAME/SCENE_NAME/icp/icp_summary.json
Replace PATH/TO/RECON.ply, PATH/TO/GT.ply, SCENE_FOLDER, SCENE_NAME, MY_RUN_NAME with your actual paths.
