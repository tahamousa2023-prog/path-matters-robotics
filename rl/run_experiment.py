"""
experiments/taha/run_experiment.py

Personal training launcher — Taha.

This script:
    1. Creates the base VolumeReconEnvCfg from the original source
    2. Applies Taha's reward modifications via patch_rewards()
    3. Runs PPO training with RSL-RL
    4. Saves logs to logs/taha/<experiment_name>/<timestamp>/

Run:
    cd ~/UR5eVolumRecon
    conda activate env_isaaclab

    python experiments/taha/run_experiment.py \
        --num_envs 16 \
        --max_iterations 200 \
        --experiment_name exp_01_camera_facing

    # Headless (faster):
    python experiments/taha/run_experiment.py \
        --num_envs 32 \
        --max_iterations 2000 \
        --experiment_name exp_01_camera_facing \
        --headless

    # Resume from checkpoint:
    python experiments/taha/run_experiment.py \
        --num_envs 16 \
        --max_iterations 500 \
        --experiment_name exp_01_resumed \
        --checkpoint logs/taha/exp_01_camera_facing/2026-XX-XX/model_final.pt
"""

from __future__ import annotations
import argparse
import os
import sys

# ── Path setup ────────────────────────────────────────────────────────────────
# Add UR5eVolumRecon root so volume_recon package is found
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
# Add agents/ so rsl_rl_ppo_cfg is found
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../agents"))
# Add this folder so my_rewards is found
sys.path.insert(0, os.path.dirname(__file__))

from isaaclab.app import AppLauncher

# ── CLI arguments ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Taha's experiment runner")
parser.add_argument(
    "--num_envs",
    type=int,
    default=16,
    help="Number of parallel environments (more = faster but more GPU memory)",
)
parser.add_argument(
    "--max_iterations",
    type=int,
    default=200,
    help="Number of PPO training iterations",
)
parser.add_argument(
    "--experiment_name",
    type=str,
    default="exp_01",
    help="Name for this run — used as subfolder in logs/taha/",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Path to .pt checkpoint to resume training from",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── All other imports MUST come after AppLauncher ─────────────────────────────
import gymnasium as gym
import torch
from datetime import datetime

# RSL-RL import with fallback (same pattern as train.py)
try:
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunner, RslRlVecEnvWrapper
except ImportError:
    try:
        from isaaclab_rl.rsl_rl.runners import OnPolicyRunner as RslRlOnPolicyRunner
        from isaaclab_rl.rsl_rl.vecenv import RslRlVecEnvWrapper
    except ImportError:
        from rsl_rl.runners import OnPolicyRunner as RslRlOnPolicyRunner
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# Task registration (must import to register the gym env)
import volume_recon.tasks  # noqa: F401

# Original configs from teammate
from volume_recon.tasks.volume_scan.env_cfg import VolumeReconEnvCfg
from rsl_rl_ppo_cfg import VolumeReconPPORunnerCfg

# Taha's reward modifications
from my_rewards import patch_rewards


# ── Main ──────────────────────────────────────────────────────────────────────

def main():

    # ── Step 1: Create base environment config ────────────────────────────
    env_cfg = VolumeReconEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    # ── Step 2: Apply Taha's reward modifications ─────────────────────────
    # Apply Taha's reward modifications
    patch_rewards(env_cfg)

    # ── exp_02: reduce max captures to force quality over quantity ────────
    # With 50 captures the agent ends episodes cheaply.
    # 15 captures forces it to make each one count.
    env_cfg.terminations.max_captures.params["max_captures"] = 20

    

    # ── Step 3: Create environment ────────────────────────────────────────
    env = gym.make("VolumeRecon-UR5e-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # ── Step 4: Create PPO agent ──────────────────────────────────────────
    agent_cfg = VolumeReconPPORunnerCfg()
    agent_cfg.max_iterations = args_cli.max_iterations

    # ── Step 5: Set up log directory ──────────────────────────────────────
    # Goes to logs/taha/<experiment_name>/<timestamp>/
    # Completely separate from teammate's logs/rsl_rl/
    log_dir = os.path.join(
        os.path.dirname(__file__),          # experiments/taha/
        "../../logs/taha",                  # → logs/taha/
        args_cli.experiment_name,           # → logs/taha/exp_01_camera_facing/
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )
    os.makedirs(log_dir, exist_ok=True)

    # ── Step 6: Print summary ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("TAHA — VOLUME RECONSTRUCTION EXPERIMENT")
    print("=" * 65)
    print(f"  Experiment : {args_cli.experiment_name}")
    print(f"  Num envs   : {args_cli.num_envs}")
    print(f"  Iterations : {args_cli.max_iterations}")
    print(f"  Log dir    : {log_dir}")
    if args_cli.checkpoint:
        print(f"  Checkpoint : {args_cli.checkpoint}")
    print("=" * 65)

    # Print active reward weights so it's clear what changed
    print("\n  Active reward weights:")
    r = env_cfg.rewards
    for name, term in vars(r).items():
        if hasattr(term, "weight") and term.weight != 0.0:
            marker = "  NEW →" if name == "camera_facing_volume" else "       "
            print(f"  {marker} {name}: {term.weight}")
    print()

    # ── Step 7: Create runner ─────────────────────────────────────────────
    runner = RslRlOnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=log_dir,
        device="cuda:0",
    )

    # ── Step 8: Optionally load checkpoint ───────────────────────────────
    if args_cli.checkpoint:
        print(f"\nLoading checkpoint: {args_cli.checkpoint}")
        runner.load(args_cli.checkpoint)

    # ── Step 9: Train ────────────────────────────────────────────────────
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations,
        init_at_random_ep_len=True,
    )

    # ── Step 10: Save final model ─────────────────────────────────────────
    final_path = os.path.join(log_dir, "model_final.pt")
    runner.save(final_path)

    print("\n" + "=" * 65)
    print("TRAINING COMPLETE")
    print(f"  Model saved to : {final_path}")
    print(f"  TensorBoard    : tensorboard --logdir ~/UR5eVolumRecon/logs/taha/ --port 6007")
    print("=" * 65 + "\n")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()