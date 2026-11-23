# BUFFER-X quick start

## 1. Set up the session

```bash
conda activate bufferx
cd ~/BUFFER-X
```

## 2. Compare a reconstruction with its ground truth

```bash
python compare.py <ground_truth.ply> <reconstruction.ply>
# example
python compare.py Test/Baby_Yoda.ply Test/points.ply

# visualise the registration result (press Q to move between windows)
python results.py Test/Baby_Yoda.ply Test/points.ply
```

- **Fitness**: closer to 1.0 is better
- **RMSE**: closer to 0.0 is better

## 3. Run the benchmark tests

```bash
# indoor scenes
python test.py --dataset 3DMatch --experiment_id threedmatch --verbose

# outdoor scenes (zero-shot transfer)
python test.py --dataset ETH --experiment_id threedmatch --verbose

# save the log
python test.py --dataset 3DMatch --experiment_id threedmatch --verbose 2>&1 | tee results/test_log.txt
```
