"""
Generates graphs and a full results report for your comparison.

Usage:
  python results.py <groundtruth.ply> <reconstruction.ply>

Example:
  python results.py Test/Baby_Yoda.ply Test/points.ply
"""

import sys
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

if len(sys.argv) != 3:
    print("Usage: python results.py <groundtruth.ply> <reconstruction.ply>")
    sys.exit(1)

gt_path  = Path(sys.argv[1])
rec_path = Path(sys.argv[2])

print(f"\nLoading files...")
gt  = o3d.io.read_point_cloud(str(gt_path))
rec = o3d.io.read_point_cloud(str(rec_path))

print(f"  Ground truth    : {len(gt.points)} points")
print(f"  Reconstruction  : {len(rec.points)} points")

# ── Align with ICP ──────────────────────────────────────────
print("\nRunning ICP alignment...")
voxel = 0.01
gt_d  = gt.voxel_down_sample(voxel)
rec_d = rec.voxel_down_sample(voxel)
gt_d.estimate_normals()
rec_d.estimate_normals()

result = o3d.pipelines.registration.registration_icp(
    rec_d, gt_d, 0.05, np.eye(4),
    o3d.pipelines.registration.TransformationEstimationPointToPlane()
)

rec_aligned = o3d.geometry.PointCloud(rec)
rec_aligned.transform(result.transformation)

fitness  = result.fitness
rmse     = result.inlier_rmse
T        = result.transformation
rotation = T[:3, :3]
translation = T[:3, 3]
rte      = np.linalg.norm(translation)
cos_a    = np.clip((np.trace(rotation) - 1) / 2, -1, 1)
rre      = np.degrees(np.arccos(cos_a))

print(f"\n── Results ─────────────────────────────")
print(f"  Fitness          : {fitness:.4f}  (1.0 = perfect)")
print(f"  RMSE             : {rmse:.6f}  (0.0 = perfect)")
print(f"  Translation error: {rte:.4f} m")
print(f"  Rotation error   : {rre:.4f} deg")
print(f"────────────────────────────────────────")

# ── Per-point distances ──────────────────────────────────────
print("\nComputing per-point distances...")
dists = np.array(rec_aligned.compute_point_cloud_distance(gt))
dists_gt = np.array(gt.compute_point_cloud_distance(rec_aligned))

# ── Build report figure ─────────────────────────────────────
print("Generating graphs...")

fig = plt.figure(figsize=(16, 12))
fig.suptitle(
    f"Registration Results\n{gt_path.name}  vs  {rec_path.name}",
    fontsize=15, fontweight="bold", y=0.98
)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── 1. Summary metrics bar chart ────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
metrics = ["Fitness", "1 - RMSE (normalized)", "1 - RTE (normalized)", "1 - RRE (normalized)"]
rte_norm  = max(0, 1 - rte)
rre_norm  = max(0, 1 - rre / 180)
rmse_norm = max(0, 1 - rmse * 100)
values = [fitness, rmse_norm, rte_norm, rre_norm]
colors = ["#4CAF50" if v >= 0.9 else "#FF9800" if v >= 0.7 else "#F44336" for v in values]
bars = ax1.bar(metrics, values, color=colors, edgecolor="black", alpha=0.85)
ax1.set_ylim(0, 1.1)
ax1.set_ylabel("Score (higher = better)")
ax1.set_title("Summary Metrics", fontweight="bold")
ax1.axhline(y=0.9, color="gray", linestyle="--", alpha=0.5, label="90% threshold")
ax1.legend(fontsize=9)
for bar, val in zip(bars, values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")
ax1.grid(True, alpha=0.3, axis="y")

# ── 2. Scorecard ─────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
ax2.axis("off")
score_color = "#4CAF50" if fitness >= 0.9 else "#FF9800" if fitness >= 0.7 else "#F44336"
score_text  = "GOOD" if fitness >= 0.9 else "OK" if fitness >= 0.7 else "POOR"
ax2.text(0.5, 0.75, score_text, ha="center", va="center",
         fontsize=32, fontweight="bold", color=score_color,
         transform=ax2.transAxes)
ax2.text(0.5, 0.45, f"Fitness:  {fitness:.4f}", ha="center", fontsize=11, transform=ax2.transAxes)
ax2.text(0.5, 0.32, f"RMSE:    {rmse:.6f}", ha="center", fontsize=11, transform=ax2.transAxes)
ax2.text(0.5, 0.19, f"RTE:     {rte:.4f} m", ha="center", fontsize=11, transform=ax2.transAxes)
ax2.text(0.5, 0.06, f"RRE:     {rre:.4f}°", ha="center", fontsize=11, transform=ax2.transAxes)
ax2.set_title("Overall Score", fontweight="bold")
ax2.patch.set_facecolor("#f5f5f5")
ax2.patch.set_alpha(0.5)

# ── 3. Distance histogram (reconstruction → GT) ──────────────
ax3 = fig.add_subplot(gs[1, :2])
ax3.hist(dists, bins=60, color="#2196F3", edgecolor="black", alpha=0.75)
ax3.axvline(np.mean(dists), color="red",    linestyle="--", linewidth=2, label=f"Mean: {np.mean(dists):.4f}")
ax3.axvline(np.median(dists), color="orange", linestyle="--", linewidth=2, label=f"Median: {np.median(dists):.4f}")
ax3.set_xlabel("Distance to nearest GT point (m)")
ax3.set_ylabel("Number of points")
ax3.set_title("Reconstruction → Ground Truth Distance Distribution", fontweight="bold")
ax3.legend()
ax3.grid(True, alpha=0.3)

# ── 4. Distance stats box ─────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 2])
ax4.axis("off")
stats = [
    ("Mean dist",   f"{np.mean(dists):.4f} m"),
    ("Median dist", f"{np.median(dists):.4f} m"),
    ("Std dev",     f"{np.std(dists):.4f} m"),
    ("Max dist",    f"{np.max(dists):.4f} m"),
    ("Min dist",    f"{np.min(dists):.4f} m"),
    ("Points",      f"{len(dists):,}"),
]
for i, (label, val) in enumerate(stats):
    y = 0.88 - i * 0.14
    ax4.text(0.05, y, label, fontsize=10, transform=ax4.transAxes, color="gray")
    ax4.text(0.95, y, val,   fontsize=10, transform=ax4.transAxes, ha="right", fontweight="bold")
ax4.set_title("Distance Stats", fontweight="bold")
ax4.patch.set_facecolor("#f5f5f5")
ax4.patch.set_alpha(0.5)

# ── 5. GT → Reconstruction distance histogram ────────────────
ax5 = fig.add_subplot(gs[2, :2])
ax5.hist(dists_gt, bins=60, color="#FF5722", edgecolor="black", alpha=0.75)
ax5.axvline(np.mean(dists_gt), color="blue",   linestyle="--", linewidth=2, label=f"Mean: {np.mean(dists_gt):.4f}")
ax5.axvline(np.median(dists_gt), color="green", linestyle="--", linewidth=2, label=f"Median: {np.median(dists_gt):.4f}")
ax5.set_xlabel("Distance to nearest reconstruction point (m)")
ax5.set_ylabel("Number of points")
ax5.set_title("Ground Truth → Reconstruction Distance Distribution", fontweight="bold")
ax5.legend()
ax5.grid(True, alpha=0.3)

# ── 6. Coverage pie chart ─────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 2])
threshold   = np.mean(dists) * 2
covered     = np.sum(dists_gt < threshold)
not_covered = len(dists_gt) - covered
coverage_pct = covered / len(dists_gt) * 100
ax6.pie(
    [covered, not_covered],
    labels=[f"Covered\n{coverage_pct:.1f}%", f"Missing\n{100-coverage_pct:.1f}%"],
    colors=["#4CAF50", "#F44336"],
    autopct="%1.1f%%",
    startangle=90,
    textprops={"fontsize": 10}
)
ax6.set_title("GT Coverage by Reconstruction", fontweight="bold")

# ── Save ─────────────────────────────────────────────────────
output_name = f"results_{gt_path.stem}_vs_{rec_path.stem}.png"
plt.savefig(output_name, dpi=150, bbox_inches="tight")
print(f"\nSaved report: {output_name}")
plt.show()