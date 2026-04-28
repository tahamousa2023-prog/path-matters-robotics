# BUFFER-X Quick Start


conda activate bufferx
cd ~/BUFFER-X
```

## Run Test

python test.py --dataset 3DMatch --experiment_id threedmatch --verbose
```

## Visualize Result

python Visualize_registration.py \
  --src ../datasets/ThreeDMatch/test/3DMatch/fragments/7-scenes-redkitchen/cloud_bin_0.ply \
  --tgt ../datasets/ThreeDMatch/test/3DMatch/fragments/7-scenes-redkitchen/cloud_bin_1.ply \
  --log logs/log_3DMatch/7-scenes-redkitchen/03072246.log
```
Press **Q** to move between before/after windows.

################################################################
################################################################

1. Convert your RGB-D images to point clouds:
bashpython bufferx_tools.py rgbd \
  --color my_data/frame_001_color.png \
  --depth my_data/frame_001_depth.png \
  --output my_data/cloud_001.ply
Adjust --fx --fy --cx --cy to match your camera (default is Kinect v1).

2. Plot Recall/RTE/RRE progress over the test run:
bashpython bufferx_tools.py metrics \
  --logs test_log.txt \
  --labels "3DMatch-threedmatch"
3. Compare two models or datasets:
bashpython bufferx_tools.py compare \
  --logs test_log.txt eth_log.txt \
  --labels "3DMatch" "ETH"
4. Analyze which fragments failed and why:
bashpython bufferx_tools.py failed \
  --log test_log.txt \
  --label "3DMatch Run"
5. Visualize before/after registration:

python bufferx_tools.py visualize \
  --src ../datasets/ThreeDMatch/test/3DMatch/fragments/7-scenes-redkitchen/cloud_bin_0.ply \
  --tgt ../datasets/ThreeDMatch/test/3DMatch/fragments/7-scenes-redkitchen/cloud_bin_1.ply
