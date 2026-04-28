"""
experiments/taha/my_rewards.py

exp_04: Fix proximity problem.

Root cause of exp_03 failure:
    CAMERA_PARAMS far_clip = 0.8m. If camera drifts > 0.8m from any
    voxel center, all captures return 0 new voxels regardless of orientation.
    The agent has no incentive to stay CLOSE to the volume, only to FACE it.

New rewards added:
    1. camera_proximity_reward (+5.0): reward for being within far_clip
       distance of the volume center. Dense gradient toward the volume.

    2. camera_in_volume_reward (+3.0): binary reward for camera being
       inside VOLUME_BOUNDS. Confirms the agent is in the right zone.

Training duration:
    200 iterations is not enough for this task.
    exp_04 runs 500 iterations to give the agent time to discover coverage.
"""

from __future__ import annotations
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import RewardTermCfg


# ══════════════════════════════════════════════════════════════════════════════
# PATCH FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def patch_rewards(env_cfg) -> None:
    r = env_cfg.rewards

    # ── NEW: proximity — reward for being close to volume ─────────────────
    # This is the missing signal. Agent must be within far_clip=0.8m
    # of the volume for any capture to register new voxels.
    r.camera_proximity = RewardTermCfg(
        func=camera_proximity_reward,
        weight=5.0,
    )

    # ── NEW: in-volume — binary reward for being inside bounds ────────────
    r.camera_in_volume = RewardTermCfg(
        func=camera_in_volume_reward,
        weight=3.0,
    )

    # ── SHAPING: correct X-axis, kept from exp_03 ─────────────────────────
    r.camera_facing_volume = RewardTermCfg(
        func=camera_facing_volume_reward,
        weight=3.0,
    )

    # ── MAIN COVERAGE SIGNAL: timing-safe delta ───────────────────────────
    r.coverage_increase = RewardTermCfg(
        func=coverage_delta_reward,
        weight=25.0,
    )

    # ── MULTI-VIEW ────────────────────────────────────────────────────────
    r.multi_view.weight = 3.0

    # ── PENALTIES ─────────────────────────────────────────────────────────
    r.wasteful_capture.weight   = -8.0
    r.action_smoothness.weight  = -0.001
    r.workspace_boundary.weight =  0.0   # disabled — conflicts with proximity
    r.self_collision.weight     = -1.0
    r.joint_limits.weight       = -0.5

    # ── DISABLED ──────────────────────────────────────────────────────────
    r.camera_orientation.weight = 0.0
    r.coverage_progress.weight  = 1.0
    r.task_completion.weight    = 100.0


# ══════════════════════════════════════════════════════════════════════════════
# NEW REWARD FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def camera_proximity_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Reward for camera being close to the volume center.

    Returns [0, 1]:
        1.0 = camera at volume center
        0.5 = camera 0.8m from center (at far_clip distance)
        0.0 = camera 1.6m+ from center

    This is a dense gradient the agent can follow to reach the volume.
    Critical: camera must be within far_clip=0.8m for any voxels to register.
    """
    from volume_recon.tasks.volume_scan.env_cfg import VOLUME_BOUNDS, CAMERA_PARAMS

    robot      = env.scene["robot"]
    camera_ids = robot.find_bodies("wrist_3_link")[0]
    cam_pos    = robot.data.body_pos_w[:, camera_ids[0], :]  # [N, 3]

    vol_center = torch.tensor([
        (VOLUME_BOUNDS["x_min"] + VOLUME_BOUNDS["x_max"]) / 2,
        (VOLUME_BOUNDS["y_min"] + VOLUME_BOUNDS["y_max"]) / 2,
        (VOLUME_BOUNDS["z_min"] + VOLUME_BOUNDS["z_max"]) / 2,
    ], device=env.device)

    distance = (cam_pos - vol_center.unsqueeze(0)).norm(dim=-1)  # [N]

    far_clip = CAMERA_PARAMS["far_clip"]  # 0.8m

    # Reward: 1.0 at center, decays linearly, 0.0 at 2 * far_clip
    reward = torch.clamp(1.0 - distance / (2.0 * far_clip), min=0.0)

    return reward  # [N] in [0, 1]


def camera_in_volume_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Binary reward: 1.0 if camera is inside VOLUME_BOUNDS, else 0.0.

    Confirms the camera is in the correct operating zone.
    From test_env.py: camera starts inside volume and immediately
    gets 35 new voxels — being inside is sufficient for coverage.
    """
    from volume_recon.tasks.volume_scan.env_cfg import VOLUME_BOUNDS

    robot      = env.scene["robot"]
    camera_ids = robot.find_bodies("wrist_3_link")[0]
    cam_pos    = robot.data.body_pos_w[:, camera_ids[0], :]  # [N, 3]

    in_x = ((cam_pos[:, 0] >= VOLUME_BOUNDS["x_min"]) &
             (cam_pos[:, 0] <= VOLUME_BOUNDS["x_max"]))
    in_y = ((cam_pos[:, 1] >= VOLUME_BOUNDS["y_min"]) &
             (cam_pos[:, 1] <= VOLUME_BOUNDS["y_max"]))
    in_z = ((cam_pos[:, 2] >= VOLUME_BOUNDS["z_min"]) &
             (cam_pos[:, 2] <= VOLUME_BOUNDS["z_max"]))

    return (in_x & in_y & in_z).float()  # [N]


def camera_facing_volume_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Dense shaping: [0,1] — camera +X axis pointing toward volume center.
    Uses LOCAL +X axis matching volumetric_coverage.py exactly.
    """
    from volume_recon.tasks.volume_scan.env_cfg import VOLUME_BOUNDS

    robot      = env.scene["robot"]
    camera_ids = robot.find_bodies("wrist_3_link")[0]
    cam_pos    = robot.data.body_pos_w[:, camera_ids[0], :]
    cam_quat   = robot.data.body_quat_w[:, camera_ids[0], :]

    w = cam_quat[:, 0]; x = cam_quat[:, 1]
    y = cam_quat[:, 2]; z = cam_quat[:, 3]

    cam_fwd = torch.stack([
        1 - 2 * (y * y + z * z),
        2 * (x * y + w * z),
        2 * (x * z - w * y),
    ], dim=-1)

    vol_center = torch.tensor([
        (VOLUME_BOUNDS["x_min"] + VOLUME_BOUNDS["x_max"]) / 2,
        (VOLUME_BOUNDS["y_min"] + VOLUME_BOUNDS["y_max"]) / 2,
        (VOLUME_BOUNDS["z_min"] + VOLUME_BOUNDS["z_max"]) / 2,
    ], device=env.device)

    to_vol = vol_center.unsqueeze(0) - cam_pos
    to_vol = to_vol / (to_vol.norm(dim=-1, keepdim=True) + 1e-6)

    dot = (cam_fwd * to_vol).sum(dim=-1)
    return torch.clamp(dot, min=0.0)


_prev_coverage: dict[int, torch.Tensor] = {}


def coverage_delta_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Timing-safe coverage increase — reads coverage_counts directly.
    Returns normalized delta [0,1] on steps where new voxels appear.
    """
    tracker = env.unwrapped._volume_tracker

    # Sum all spatial dims → [N]
    current_covered = (tracker.coverage_counts > 0).sum(dim=(1, 2, 3)).float()

    env_id = id(env)

    if env_id not in _prev_coverage:
        _prev_coverage[env_id] = current_covered.clone()
        return torch.zeros(env.num_envs, device=env.device)

    prev_covered = _prev_coverage[env_id]
    delta = torch.clamp(current_covered - prev_covered, min=0.0)
    _prev_coverage[env_id] = current_covered.clone()

    return delta / tracker.num_voxels