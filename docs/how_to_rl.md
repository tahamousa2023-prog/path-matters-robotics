# Activate and navigate
conda activate env_isaaclab
cd ~/UR5eVolumRecon

# Train exp_09 from best checkpoint
python experiments/taha/run_experiment.py \
  --num_envs 16 \
  --max_iterations 5000 \
  --experiment_name exp_10 \
  --checkpoint logs/taha/exp_07_stricter_wasteful/2026-03-22_15-56-14/model_final.pt \
  --headless

# TensorBoard (open second terminal)
conda activate env_isaaclab
tensorboard --logdir ~/UR5eVolumRecon/logs/taha/ --port 6007

# Open browser
http://localhost:6007

# Filter to best runs only (paste in TensorBoard regex box)
exp_05|exp_06|exp_07|exp_09

# Evaluate trained policy + save poses
python scripts/rsl_rl/play.py \
  --task VolumeRecon-UR5e-v0 \
  --num_envs 1 \
  --num_episodes 3 \
  --checkpoint logs/taha/exp_07_stricter_wasteful/2026-03-22_15-56-14/model_final.pt \
  --save-poses \
  --poses-dir outputs/poses/exp_07

# Check saved poses
cat outputs/poses/exp_07/episode_01_poses.json

# Check training results
cat logs/taha/exp_09/$(ls -t logs/taha/exp_09/ | head -1)/training_log.json
