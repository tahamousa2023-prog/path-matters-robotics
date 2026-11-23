# Path Matters: Learning Optimal Inference-Time Trajectories for Robotic 2D to 3D AI Reconstruction

**TU Berlin · Institut für Werkzeugmaschinen und Fabrikbetrieb (IWF), Industrial Automation Technology (IAT) · WiSe 2025/26**
Automation project, graded *sehr gut* (distinction)

Path Matters is a simulation-based study of how camera viewpoints and trajectories affect the quality of robotic 3D reconstruction, and whether a reinforcement-learning agent can learn better viewpoint behaviour than fixed scan patterns.

---

## System

| Component | Specification |
|---|---|
| Robot arm | Universal Robots UR5e with wrist-mounted camera (eye-in-hand) |
| Simulator | NVIDIA Isaac Sim + Isaac Lab |
| Middleware | ROS 2 Humble, MoveIt2 |
| Reconstruction | VGGT, Fast3R, SAM3D |
| Registration | Open3D ICP + FPFH, BUFFER-X |
| RL algorithm | PPO (RSL-RL), 16 parallel environments |
| GPU | NVIDIA RTX A6000 |

The pipeline has four modules that share one configuration: **trajectory** (viewpoint generation and execution in Isaac Sim via ROS 2), **data** (image export and point cloud conversion), **reconstruction** (VGGT, Fast3R and SAM3D behind a common interface) and **evaluation** (ICP registration with Fitness and RMSE against ground-truth meshes).

---

## Results

### Reconstruction backbone comparison (28-object evaluation set)

| Method | Mean Fitness ↑ | Mean RMSE ↓ | Inference |
|---|---|---|---|
| VGGT | **0.93** | **0.002 m** | ~7 s |
| Fast3R | 0.89 | 0.010 m | ~7 s |
| SAM3D | 0.91 | 0.008 m | ~9 s |

VGGT was selected as the backbone for the trajectory experiments.

### Camera orientation (5 scan patterns, VGGT + ICP/FPFH)

| Orientation | Mean ICP Fitness | Mean Inlier RMSE |
|---|---|---|
| Fixed-downward | 0.68 | 0.035 m |
| Object-pointing | **0.79** | **0.022 m** |

Camera orientation was the largest controllable factor: the improvement held for all five patterns (Lawnmower, Zigzag, Hemisphere, Spiral, Random) without changing the trajectory geometry or the number of viewpoints.

### Reinforcement learning: reward shaping

| Reward design | Task success (≥75% coverage) |
|---|---|
| Sparse coverage reward only | 0.4% |
| Coverage + dense proximity shaping | **45.2%** |

Sparse coverage rewards alone gave the agent no signal until it reached the scanning volume. Dense proximity shaping first teaches the agent to approach the workspace, then to scan it. The trained policy reached 75%+ coverage per episode, compared with 20.6% for random exploration. The policy has been evaluated in simulation only.

---

## Repository structure

```
preprocessing/     mesh conversion, centring, SAM masks
reconstruction/
  vggt/            VGGT reconstruction + ICP evaluation (archive/ holds earlier dated versions)
  fast3r/          Fast3R reconstruction
  sam3d/           SAM 3D Objects reconstruction
registration/      BUFFER-X + ICP pipeline, evaluation helpers
  bufferx/         BUFFER-X tools, benchmark logs and quick start
rl/                reward functions and experiment runners (Isaac Lab, PPO)
docs/              pipeline and RL command references
```

---

## Team

| Member | Focus |
|---|---|
| Artem Balatsiuk | Team member |
| Aziz Louati | Documentation and reporting |
| Haroun Lallouche | Point cloud evaluation |
| **Taha Mohammed** | Trajectory generation and evaluation, ROS 2 / MoveIt2 motion planning, RL for trajectory optimisation |
| Ziad Abouhalawa | Reconstruction and project coordination |

Supervisor: Adam Altenbuchner, IWF / IAT, TU Berlin

---

## Related publication

> Altenbuchner, A.M., Hartisch, R.M., Mohammed, T., Abouhalawa, Z., Balatsiuk, A., Louati, A., Lallouche, H., Krüger, J. (2026). *Camera-Orientation Effects in Robotic Viewpoint Acquisition for Feed-Forward 3D Reconstruction in Manufacturing Inspection Workcells.* The International Journal of Advanced Manufacturing Technology (Springer). Under review.

[Group repository](https://github.com/Adam-yes/robotic-viewpoint-acquisition-for-3d-reconstruction)

---

## Third-party code and licenses

This repository includes or adapts code from the following projects. Each is subject to its own license, included in the respective folder:

| Folder | Source | License |
|---|---|---|
| `reconstruction/vggt/` | [VGGT](https://github.com/facebookresearch/vggt) (Meta) | VGGT License (`LICENSE.txt`) |
| `reconstruction/fast3r/` | [Fast3R](https://github.com/facebookresearch/fast3r) (Meta) | FAIR Noncommercial Research License (`LICENSE`) |
| `reconstruction/sam3d/` | [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects) (Meta) | SAM License (`LICENSE`) |
| `registration/bufferx/` | [BUFFER-X](https://github.com/MIT-SPARK/BUFFER-X) (MIT SPARK Lab) | MIT License (`LICENSE`) |

The project-specific code was written by the Path Matters team for academic, non-commercial use.

---

## Contact

**Taha Mohammed** · [Website](https://tahamousa2023-prog.github.io) · [LinkedIn](https://www.linkedin.com/in/taha-mahmoud)
