# BUFFER-X Team Quick Start Guide



## 1. Start Every Session

conda activate bufferx
cd ~/BUFFER-X




## 2. Compare Your Own Samples

Place your files anywhere, then run:

python compare.py <groundtruth.ply> <reconstruction.ply>


**Example:**

python compare.py Test/Baby_Yoda.ply Test/points.ply

**Results_visualization**
python results.py Test/Baby_Yoda.ply Test/points.ply


Press **Q** to move between windows.

**What the numbers mean:**
- **Fitness** — closer to 1.0 is better
- **RMSE** — closer to 0.0 is better



## 3. Run the Full Benchmark Test


# Indoor scenes
python test.py --dataset 3DMatch --experiment_id threedmatch --verbose

# Outdoor scenes
python test.py --dataset ETH --experiment_id threedmatch --verbose

# Save results to file
python test.py --dataset 3DMatch --experiment_id threedmatch --verbose 2>&1 | tee my_results.txt



