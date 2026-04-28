import open3d as o3d
import numpy as np
import argparse
import os


def visualize_registration(src_path, tgt_path, transform=None, log_file=None):
    """
    Visualize point cloud registration result.
    
    Args:
        src_path: Path to source point cloud (.pcd, .ply, .npy)
        tgt_path: Path to target point cloud (.pcd, .ply, .npy)
        transform: 4x4 transformation matrix (numpy array) or path to .log file
        log_file: Path to BUFFER-X output .log file containing transformations
    """

    # Load point clouds
    def load_pcd(path):
        if path.endswith(".npy"):
            pts = np.load(path)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
        else:
            pcd = o3d.io.read_point_cloud(path)
        return pcd

    src_pcd = load_pcd(src_path)
    tgt_pcd = load_pcd(tgt_path)

    # Color the point clouds
    src_pcd.paint_uniform_color([1, 0, 0])  # Red = source
    tgt_pcd.paint_uniform_color([0, 0.651, 0.929])  # Blue = target

    # Visualize BEFORE registration
    print("Showing BEFORE registration (Red=Source, Blue=Target)...")
    print("Press Q to continue to AFTER registration view.")
    o3d.visualization.draw_geometries(
        [src_pcd, tgt_pcd],
        window_name="BEFORE Registration",
        width=1280,
        height=720,
    )

    if transform is not None:
        if isinstance(transform, str) and transform.endswith(".npy"):
            T = np.load(transform)
        elif isinstance(transform, np.ndarray):
            T = transform
        else:
            print("Invalid transform format.")
            return

        # Apply transformation to source
        src_pcd_transformed = load_pcd(src_path)
        src_pcd_transformed.paint_uniform_color([0, 1, 0])  # Green = aligned source
        src_pcd_transformed.transform(T)

        # Visualize AFTER registration
        print("Showing AFTER registration (Green=Aligned Source, Blue=Target)...")
        print("Press Q to close.")
        o3d.visualization.draw_geometries(
            [src_pcd_transformed, tgt_pcd],
            window_name="AFTER Registration",
            width=1280,
            height=720,
        )


def parse_log_file(log_file):
    """Parse BUFFER-X output .log file and return list of (src_id, tgt_id, transform)."""
    results = []
    with open(log_file, "r") as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        parts = lines[i].strip().split()
        if len(parts) == 3:
            src_id, tgt_id = int(parts[0]), int(parts[1])
            T = np.eye(4)
            for row in range(4):
                T[row] = [float(x) for x in lines[i + 1 + row].strip().split()[:4]]
            results.append((src_id, tgt_id, T))
            i += 5
        else:
            i += 1
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize BUFFER-X Registration Results")
    parser.add_argument("--src", type=str, required=True, help="Path to source point cloud")
    parser.add_argument("--tgt", type=str, required=True, help="Path to target point cloud")
    parser.add_argument("--transform", type=str, default=None, help="Path to .npy transformation matrix")
    parser.add_argument("--log", type=str, default=None, help="Path to BUFFER-X output .log file")
    args = parser.parse_args()

    if args.log:
        # Parse log file and show first result
        results = parse_log_file(args.log)
        if results:
            src_id, tgt_id, T = results[0]
            print(f"Visualizing pair: src={src_id}, tgt={tgt_id}")
            print(f"Transformation matrix:\n{T}")
            visualize_registration(args.src, args.tgt, transform=T)
    else:
        T = np.load(args.transform) if args.transform else None
        visualize_registration(args.src, args.tgt, transform=T)