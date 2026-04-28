# BUFFER-X — How to Run

## Setup

    conda activate bufferx
    cd ~/BUFFER-X

## Run 3DMatch benchmark

    python test.py --dataset 3DMatch --experiment_id threedmatch --verbose

Save results:

    python test.py --dataset 3DMatch --experiment_id threedmatch --verbose \
      2>&1 | tee my_results.txt

## Run ETH outdoor benchmark (zero-shot)

    python test.py --dataset ETH --experiment_id threedmatch --verbose \
      2>&1 | tee eth_log.txt

## Compare your own scans

    python compare.py Test/Baby_Yoda.ply Test/points.ply

Results:
- Fitness: closer to 1.0 is better
- RMSE: closer to 0.0 is better

## Visualize before/after alignment

    python results.py Test/Baby_Yoda.ply Test/points.ply

Press Q to toggle between windows.

## Visualize registration log

    python Visualize_registration.py \
      --src datasets/ThreeDMatch/test/3DMatch/fragments/7-scenes-redkitchen/cloud_bin_0.ply \
      --tgt datasets/ThreeDMatch/test/3DMatch/fragments/7-scenes-redkitchen/cloud_bin_1.ply \
      --log logs/log_3DMatch/7-scenes-redkitchen/03072246.log

## Convert RGB-D to point cloud

    python bufferx_tools.py rgbd \
      --color my_data/frame_001_color.png \
      --depth my_data/frame_001_depth.png \
      --output my_data/cloud_001.ply

## Analyse failed fragments

    python bufferx_tools.py failed \
      --log test_log.txt \
      --label "3DMatch Run"

## Plot metrics over time

    python bufferx_tools.py metrics \
      --logs test_log.txt \
      --labels "3DMatch-threedmatch"

## Our 3DMatch results (TU Berlin, March 2026)

| Metric | Value |
|--------|-------|
| Recall | 97.1% |
| RMSE Recall | 95.2% |
| RTE | 5.79 cm |
| RRE | 1.80 degrees |
| Failed / Total | 47 / 1623 |
| GPU | NVIDIA RTX A6000 |
| Runtime | 28 minutes |

## Why failures happen

Most of the 47 failures fall into two categories:
1. Symmetric scenes — RRE near 90, 125, or 180 degrees. The geometry
   is rotationally symmetric so no descriptor can distinguish orientations.
2. Low-overlap pairs — the two scans share very little surface.
   Insufficient matching geometry for a reliable transform.
