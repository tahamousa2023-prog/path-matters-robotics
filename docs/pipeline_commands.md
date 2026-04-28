# Full Pipeline Commands — Taha Mohammed

## Environment

    conda activate bufferx_o3d
    cd ~/UR5eVolumRecon

## Auto-crop (diagnose one scene)

    python experiments/taha/auto_crop.py \
      --input SCENE/sparse/points.ply \
      --output /tmp/test.ply \
      --cluster-eps 0.05 \
      --min-fraction 0.05 \
      --max-fraction 0.15 \
      --diagnose

## Auto-crop (save result)

    python experiments/taha/auto_crop.py \
      --input SCENE/sparse/points.ply \
      --output SCENE/recon_generated/vggt/points_cleaned.ply \
      --cluster-eps 0.05 \
      --min-fraction 0.05 \
      --max-fraction 0.15

## Run full batch (all 50 scenes)

    python experiments/taha/run_full_pipeline.py \
      --run-name taha_batch_03 \
      --cluster-eps 0.05 \
      --min-fraction 0.05 \
      --max-fraction 0.15

## Run one scene only

    python experiments/taha/run_full_pipeline.py \
      --scene-names SCENE_NAME \
      --run-name taha_test \
      --cluster-eps 0.05 \
      --min-fraction 0.05 \
      --max-fraction 0.15

## Run on new dataset

    python experiments/taha/run_full_pipeline.py \
      --scene-root PATH/TO/NEW/DATASET \
      --gt-root    PATH/TO/NEW/DATASET \
      --scene-names SCENE_NAME \
      --run-name new_dataset_test \
      --cluster-eps 0.05 \
      --min-fraction 0.05 \
      --max-fraction 0.15

## Pre-scale reconstruction to match GT

    python3 -c "
    import open3d as o3d, numpy as np
    src = o3d.io.read_point_cloud('RECON.ply')
    tgt = o3d.io.read_point_cloud('GT.ply')
    scale = np.linalg.norm(tgt.get_axis_aligned_bounding_box().get_extent()) / \
            np.linalg.norm(src.get_axis_aligned_bounding_box().get_extent())
    print(f'Scale: {scale:.4f}')
    src.scale(scale, center=src.get_center())
    o3d.io.write_point_cloud('RECON_scaled.ply', src)
    "

## Run BUFFER-X + ICP directly

    python path_matters/haroun/Pipeline/cc_bufferx_pipeline_package/run_cc_bufferx_pipeline.py \
      --recon-root  SCENE_FOLDER \
      --gt-root     SCENE_FOLDER \
      --output-base ~/path_matters/runs/taha \
      --run-name    MY_RUN_NAME \
      --bufferx-root ~/BUFFER-X \
      --bufferx-env  bufferx_o3d \
      --scene-names  SCENE_NAME \
      --recon-candidates sparse/points_scaled.ply sparse/points.ply \
      --gt-candidates    textured.ply textured.obj \
      --manual-mode off \
      --save-viz \
      --show-final-viz

## Check results

    # Read ICP result
    cat runs/taha/MY_RUN_NAME/SCENE_NAME/icp/icp_summary.json

    # View all fitness scores sorted
    grep -h "icp_fitness" \
      runs/taha/MY_RUN_NAME/*/icp/icp_summary.json \
      | sort -t: -k2 -rn

    # View batch summary
    cat runs/taha/MY_RUN_NAME/batch_summary.csv

    # Open result image
    eog runs/taha/MY_RUN_NAME/SCENE_NAME/viz/04_icp_overlay.png &

## Direct 2-file comparison

    python experiments/taha/compare_ply.py \
      --recon PATH/TO/RECON.ply \
      --gt    PATH/TO/GT.ply \
      --output-dir ~/path_matters/runs/taha/MY_RUN_NAME/
