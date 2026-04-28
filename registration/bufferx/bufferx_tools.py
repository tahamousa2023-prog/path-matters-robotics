import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import open3d as o3d


# ─────────────────────────────────────────────
# 1. PARSE LOG FILE
# ─────────────────────────────────────────────

def parse_test_log(log_file):
    """Parse a BUFFER-X verbose test log file into structured data."""
    results = {
        "recall_per_100": [],
        "rte_per_100": [],
        "rre_per_100": [],
        "steps": [],
        "failed_fragments": [],
        "final_recall": None,
        "final_rte": None,
        "final_rre": None,
    }

    with open(log_file, "r") as f:
        for line in f:
            # Progress lines: [100/1623] Recall: 0.99 RTE: 0.04 RRE: 1.4
            m = re.search(r"\[(\d+)/\d+\] Recall: ([\d.]+) RTE: ([\d.]+) RRE: ([\d.]+)", line)
            if m:
                results["steps"].append(int(m.group(1)))
                results["recall_per_100"].append(float(m.group(2)))
                results["rte_per_100"].append(float(m.group(3)))
                results["rre_per_100"].append(float(m.group(4)))

            # Failed fragments
            m = re.search(r"(\d+)th fragment failed, RRE: ([\d.]+), RTE: ([\d.]+)", line)
            if m:
                results["failed_fragments"].append({
                    "id": int(m.group(1)),
                    "rre": float(m.group(2)),
                    "rte": float(m.group(3))
                })

            # Final results
            m = re.search(r"Recall: ([\d.]+)", line)
            if m and "Results" in line or "RMSE" in line:
                results["final_recall"] = float(m.group(1))

    if results["recall_per_100"]:
        results["final_recall"] = results["recall_per_100"][-1]
        results["final_rte"] = results["rte_per_100"][-1]
        results["final_rre"] = results["rre_per_100"][-1]

    return results


# ─────────────────────────────────────────────
# 2. PLOT: RECALL / RTE / RRE OVER TIME
# ─────────────────────────────────────────────

def plot_metrics_over_time(log_files, labels, output_path="metrics_over_time.png"):
    """Plot Recall, RTE, RRE progression during testing for one or multiple runs."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]

    for idx, (log_file, label) in enumerate(zip(log_files, labels)):
        data = parse_test_log(log_file)
        c = colors[idx % len(colors)]
        steps = data["steps"]
        axes[0].plot(steps, data["recall_per_100"], color=c, label=label, linewidth=2)
        axes[1].plot(steps, [x * 100 for x in data["rte_per_100"]], color=c, label=label, linewidth=2)
        axes[2].plot(steps, data["rre_per_100"], color=c, label=label, linewidth=2)

    axes[0].set_ylabel("Recall", fontsize=12)
    axes[0].set_ylim(0, 1.05)
    axes[0].axhline(y=0.9, color="gray", linestyle="--", alpha=0.5, label="90% threshold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Registration Performance Over Test Set", fontsize=14, fontweight="bold")

    axes[1].set_ylabel("RTE (cm)", fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].set_ylabel("RRE (degrees)", fontsize=12)
    axes[2].set_xlabel("Fragment Index", fontsize=12)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.show()


# ─────────────────────────────────────────────
# 3. PLOT: MODEL COMPARISON BAR CHART
# ─────────────────────────────────────────────

def plot_model_comparison(log_files, labels, output_path="model_comparison.png"):
    """Compare final Recall, RTE, RRE across multiple models/datasets as bar charts."""
    recalls, rtes, rres = [], [], []

    for log_file in log_files:
        data = parse_test_log(log_file)
        recalls.append(data["final_recall"] or 0)
        rtes.append((data["final_rte"] or 0) * 100)
        rres.append(data["final_rre"] or 0)

    x = np.arange(len(labels))
    width = 0.25
    colors = ["#2196F3", "#FF5722", "#4CAF50"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle("Model / Dataset Comparison", fontsize=16, fontweight="bold")

    axes[0].bar(x, recalls, width * 3, color=colors[0], alpha=0.8, edgecolor="black")
    axes[0].set_title("Recall (higher is better)", fontsize=12)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, ha="right")
    axes[0].set_ylim(0, 1.1)
    axes[0].set_ylabel("Recall")
    for i, v in enumerate(recalls):
        axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")

    axes[1].bar(x, rtes, width * 3, color=colors[1], alpha=0.8, edgecolor="black")
    axes[1].set_title("RTE in cm (lower is better)", fontsize=12)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15, ha="right")
    axes[1].set_ylabel("RTE (cm)")
    for i, v in enumerate(rtes):
        axes[1].text(i, v + 0.001, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")

    axes[2].bar(x, rres, width * 3, color=colors[2], alpha=0.8, edgecolor="black")
    axes[2].set_title("RRE in degrees (lower is better)", fontsize=12)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=15, ha="right")
    axes[2].set_ylabel("RRE (degrees)")
    for i, v in enumerate(rres):
        axes[2].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.show()


# ─────────────────────────────────────────────
# 4. PLOT: FAILED FRAGMENTS ANALYSIS
# ─────────────────────────────────────────────

def plot_failed_fragments(log_file, label, output_path="failed_fragments.png"):
    """Show which fragments failed and how badly (RTE/RRE scatter plot)."""
    data = parse_test_log(log_file)
    failed = data["failed_fragments"]

    if not failed:
        print("No failed fragments found.")
        return

    ids = [f["id"] for f in failed]
    rtes = [f["rte"] for f in failed]
    rres = [f["rre"] for f in failed]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Failed Fragments Analysis — {label}", fontsize=14, fontweight="bold")

    axes[0].scatter(ids, rres, color="#FF5722", alpha=0.7, s=50, edgecolors="black", linewidth=0.5)
    axes[0].axhline(y=5, color="red", linestyle="--", label="RRE threshold (5°)")
    axes[0].set_xlabel("Fragment Index")
    axes[0].set_ylabel("RRE (degrees)")
    axes[0].set_title("Rotation Error of Failed Fragments")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(ids, rtes, color="#2196F3", alpha=0.7, s=50, edgecolors="black", linewidth=0.5)
    axes[1].axhline(y=0.2, color="red", linestyle="--", label="RTE threshold (0.2m)")
    axes[1].set_xlabel("Fragment Index")
    axes[1].set_ylabel("RTE (meters)")
    axes[1].set_title("Translation Error of Failed Fragments")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.show()


# ─────────────────────────────────────────────
# 5. VISUALIZE: BEFORE / AFTER REGISTRATION
# ─────────────────────────────────────────────

def visualize_registration(src_path, tgt_path, transform=None):
    """Show before/after point cloud registration side by side."""
    def load(path):
        if path.endswith(".npy"):
            pts = np.load(path)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
        else:
            pcd = o3d.io.read_point_cloud(path)
        return pcd

    src = load(src_path)
    tgt = load(tgt_path)

    src.paint_uniform_color([1, 0, 0])        # Red = source
    tgt.paint_uniform_color([0, 0.6, 0.9])   # Blue = target

    print("BEFORE registration (Red=Source, Blue=Target) — Press Q to continue")
    o3d.visualization.draw_geometries([src, tgt], window_name="BEFORE", width=1280, height=720)

    if transform is not None:
        src_aligned = load(src_path)
        src_aligned.paint_uniform_color([0, 0.9, 0])  # Green = aligned
        src_aligned.transform(transform)
        print("AFTER registration (Green=Aligned Source, Blue=Target) — Press Q to close")
        o3d.visualization.draw_geometries([src_aligned, tgt], window_name="AFTER", width=1280, height=720)


# ─────────────────────────────────────────────
# 6. PREPARE CUSTOM RGB-D DATA
# ─────────────────────────────────────────────

def rgbd_to_pointcloud(color_path, depth_path, output_path,
                        fx=525.0, fy=525.0, cx=319.5, cy=239.5, depth_scale=1000.0):
    """
    Convert an RGB-D image pair to a .ply point cloud.

    Args:
        color_path: Path to color image (.png or .jpg)
        depth_path: Path to depth image (.png, 16-bit)
        output_path: Output .ply file path
        fx, fy: Focal lengths (default: Kinect v1)
        cx, cy: Principal point (default: Kinect v1)
        depth_scale: Depth scale factor (1000 for mm -> meters)
    """
    color = o3d.io.read_image(color_path)
    depth = o3d.io.read_image(depth_path)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth,
        depth_scale=depth_scale,
        depth_trunc=3.0,
        convert_rgb_to_intensity=False
    )

    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(
        width=640, height=480,
        fx=fx, fy=fy, cx=cx, cy=cy
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"Saved point cloud: {output_path} ({len(pcd.points)} points)")
    return pcd


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BUFFER-X Visualization & Comparison Tools")
    subparsers = parser.add_subparsers(dest="command")

    # metrics
    p1 = subparsers.add_parser("metrics", help="Plot Recall/RTE/RRE over time")
    p1.add_argument("--logs", nargs="+", required=True, help="Log file paths")
    p1.add_argument("--labels", nargs="+", required=True, help="Labels for each log")
    p1.add_argument("--output", default="metrics_over_time.png")

    # compare
    p2 = subparsers.add_parser("compare", help="Bar chart comparison of models/datasets")
    p2.add_argument("--logs", nargs="+", required=True, help="Log file paths")
    p2.add_argument("--labels", nargs="+", required=True, help="Labels for each log")
    p2.add_argument("--output", default="model_comparison.png")

    # failed
    p3 = subparsers.add_parser("failed", help="Analyze failed fragments")
    p3.add_argument("--log", required=True, help="Log file path")
    p3.add_argument("--label", default="Test Run")
    p3.add_argument("--output", default="failed_fragments.png")

    # visualize
    p4 = subparsers.add_parser("visualize", help="Visualize before/after registration")
    p4.add_argument("--src", required=True)
    p4.add_argument("--tgt", required=True)
    p4.add_argument("--transform", default=None, help="Path to .npy 4x4 transform matrix")

    # rgbd
    p5 = subparsers.add_parser("rgbd", help="Convert RGB-D image pair to point cloud")
    p5.add_argument("--color", required=True, help="Color image path")
    p5.add_argument("--depth", required=True, help="Depth image path")
    p5.add_argument("--output", required=True, help="Output .ply path")
    p5.add_argument("--fx", type=float, default=525.0)
    p5.add_argument("--fy", type=float, default=525.0)
    p5.add_argument("--cx", type=float, default=319.5)
    p5.add_argument("--cy", type=float, default=239.5)
    p5.add_argument("--depth_scale", type=float, default=1000.0)

    args = parser.parse_args()

    if args.command == "metrics":
        plot_metrics_over_time(args.logs, args.labels, args.output)

    elif args.command == "compare":
        plot_model_comparison(args.logs, args.labels, args.output)

    elif args.command == "failed":
        plot_failed_fragments(args.log, args.label, args.output)

    elif args.command == "visualize":
        T = np.load(args.transform) if args.transform else None
        visualize_registration(args.src, args.tgt, T)

    elif args.command == "rgbd":
        rgbd_to_pointcloud(
            args.color, args.depth, args.output,
            args.fx, args.fy, args.cx, args.cy, args.depth_scale
        )

    else:
        parser.print_help()