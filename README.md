# Path Matters: RL-Driven Camera Trajectory Optimization for Robotic 3D Reconstruction

**TU Berlin Â· WiSe 2025/26 Â· Computational Engineering Science**

This repository documents the Path Matters project: a simulation-based study of whether a reinforcement-learning agent can learn better camera trajectories for robotic 3D reconstruction than fixed scan patterns, under a constrained viewpoint budget.

---

## Technical White Paper

The full technical white paper â€” covering the MDP formulation, PPO training, reward shaping, results across multiple seeds, and explicit limitations â€” is available here:

ðŸ“„ **[whitepaper.pdf](./whitepaper.pdf)** â€” *Path Matters: Reinforcement-Learning-Driven Camera Trajectory Optimization for Robotic 3D Reconstruction*

---

## Problem

Robotic 3D reconstruction pipelines typically fix the scan trajectory (circular orbit, raster sweep) independently of the object being scanned. For objects with concavities, occlusions, or asymmetric geometry, this wastes viewpoint budget on uninformative positions while leaving high-curvature or self-occluded regions under-sampled.

**Hypothesis:** A learned, geometry-aware viewpoint policy can outperform fixed trajectories under a constrained viewpoint budget.

---

## System

| Component | Specification |
|---|---|
| Robot arm | Universal Robots UR5e (eye-in-hand) |
| Simulator | NVIDIA Isaac Sim + Isaac Lab |
| Middleware | ROS 2 Humble, MoveIt2 |
| Reconstruction | VGGT, Fast3R, SAM3D |
| Registration | Open3D ICP + FPFH, BUFFER-X |
| RL algorithm | PPO (RSL-RL), 16 parallel envs |
| GPU | NVIDIA RTX A6000 |

---

## Approach

The viewpoint selection problem is formulated as a finite-horizon Markov Decision Process:

- **State**: End-effector pose, previously captured viewpoints, coarse occupancy estimate of surface coverage
- **Action**: Continuous delta in end-effector pose (translation + orientation), constrained to UR5e reachable workspace
- **Reward**: Shaped reward combining marginal surface coverage, penalty for revisiting covered regions, and per-step cost to discourage budget exhaustion without convergence

An initial pure-coverage reward produced degenerate behaviour (rapid large-amplitude motions nominally increasing coverage metrics but generating motion-blurred, unregisterable captures). Dense proximity shaping was required before sparse coverage rewards became a useful training signal.

---

## Results

### Reconstruction Backbone Comparison (28-object evaluation set)

| Method | Mean Fitness â†‘ | Mean RMSE â†“ | PCD Density | Inference |
|---|---|---|---|---|
| VGGT | **0.93** | **0.002 m** | ~8,200 pts | ~7 s |
| Fast3R | 0.89 | 0.010 m | ~5,600 pts | ~7 s |
| SAM3D | 0.91 | 0.008 m | ~45,000 pts | ~9 s |

VGGT selected as primary backbone for trajectory experiments.

### Camera Orientation Comparison (5 scan patterns, VGGT + ICP/FPFH fixed)

| Orientation | Mean ICP Fitness | Mean Inlier RMSE |
|---|---|---|
| Fixed-downward | 0.68 | 0.035 m |
| Object-pointing | **0.79** | **0.022 m** |
| Î” | +0.11 (+16.2%) | âˆ’0.013 m (âˆ’37.1%) |

### RL Coverage Policy vs Baselines

| Metric | Random Baseline | Sparse Reward Only | PPO + Proximity Shaping |
|---|---|---|---|
| Task success (â‰¥75% coverage) | 0% | 0.4% | **45.2%** |
| Mean episode return | â€” | 22.5 | **32.7** |
| Captures to reach target | Not reached | Rarely | ~35 |

Results averaged across multiple training seeds. Full variance reporting in [whitepaper.pdf](./whitepaper.pdf).

---

## Related Publication

> Altenbuchner et al. (2026). *Camera-Orientation Effects in Robotic Viewpoint Acquisition for Feed-Forward 3D Reconstruction in Manufacturing Inspection Workcells.* The International Journal of Advanced Manufacturing Technology (Springer). Under review.

[Group repository](https://github.com/Adam-yes/robotic-viewpoint-acquisition-for-3d-reconstruction)

---

## Author

**Taha Mohammed** Â· MSc Computational Engineering Science, TU Berlin  
[Website](https://tahamousa2023-prog.github.io) Â· [LinkedIn](https://linkedin.com/in/taha-mahmoud)
