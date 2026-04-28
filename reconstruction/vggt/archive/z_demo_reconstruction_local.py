"""
VGGT 3D Reconstruction Module - Complete Export Suite
Exports: GLB, PLY, Colored Point Cloud (PLY/PCD/XYZ/NPY)
Optimized for ICP point cloud comparison
"""

import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Tuple, Dict
import trimesh
import argparse

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from visual_util import predictions_to_glb


class VGGTReconstructionModule:
    """
    Complete VGGT reconstruction with multiple export formats
    Optimized for visualization and ICP comparison
    """
    
    def __init__(self, device: str = 'cuda'):
        """Initialize VGGT reconstruction module"""
        self.device = device
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        
        print(f"Initializing VGGT on {device} with {self.dtype}")
        
        # Load model
        self.model = VGGT()
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        self.model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
        self.model.eval()
        self.model = self.model.to(device)
        
        print("✓ VGGT model loaded successfully")
    
    def load_images_from_folder(self, image_folder: str) -> Tuple[torch.Tensor, list]:
        """Load images from folder"""
        image_path_list = sorted(glob.glob(os.path.join(image_folder, "*")))
        
        if len(image_path_list) == 0:
            raise ValueError(f"No images found in {image_folder}")
        
        print(f"Found {len(image_path_list)} images")
        
        images = load_and_preprocess_images(image_path_list).to(self.device)
        image_names = [os.path.basename(p) for p in image_path_list]
        
        return images, image_names
    
    def run_vggt_inference(
        self, 
        images: torch.Tensor,
        vggt_resolution: int = 518
    ) -> Dict:
        """
        Run VGGT inference and return all predictions
        
        Returns:
            Dictionary with:
            - points_3d: 3D point coordinates [N, H, W, 3]
            - points_rgb: RGB colors [N, H, W, 3]
            - depth_conf: Confidence scores [N, H, W]
            - extrinsic, intrinsic: Camera parameters
            - images: Resized images for GLB export
        """
        print("Running VGGT reconstruction...")
        
        # Resize to VGGT's optimal resolution
        images_resized = F.interpolate(
            images, 
            size=(vggt_resolution, vggt_resolution), 
            mode="bilinear", 
            align_corners=False
        )
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.dtype):
                # Run full model
                images_batch = images_resized[None]
                predictions = self.model(images_batch)
        
        # Convert pose encoding to camera matrices
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            predictions["pose_enc"], 
            images_batch.shape[-2:]
        )
        
        # Remove batch dimension and convert to numpy
        results = {}
        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                results[key] = predictions[key].squeeze(0).cpu().numpy()
        
        results['extrinsic'] = extrinsic.squeeze(0).cpu().numpy()
        results['intrinsic'] = intrinsic.squeeze(0).cpu().numpy()
        
        # Compute 3D points from depth
        depth_map = results['depth']
        world_points_from_depth = unproject_depth_map_to_point_map(
            depth_map, results['extrinsic'], results['intrinsic']
        )
        results['world_points_from_depth'] = world_points_from_depth
        
        # Get RGB colors (matching resolution with points)
        points_rgb = images_resized.cpu().numpy()  # [N, 3, H, W]
        points_rgb = points_rgb.transpose(0, 2, 3, 1)  # [N, H, W, 3]
        results['points_rgb'] = points_rgb
        
        # Store images for GLB export (keep in CHW format)
        results['images'] = images_resized.cpu().numpy()
        
        print("✓ Reconstruction complete")
        
        return results
    
    def extract_point_cloud(
        self,
        predictions: Dict[str, np.ndarray],
        conf_threshold: float = 1.0,
        max_points: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract filtered and colored point cloud
        
        Args:
            predictions: Dictionary from run_vggt_inference
            conf_threshold: Confidence threshold (lower = more points)
            max_points: Maximum number of points (None = unlimited)
            
        Returns:
            points: Nx3 array of XYZ coordinates
            colors: Nx3 array of RGB colors (0-255)
        """
        print(f"Extracting point cloud (conf_threshold={conf_threshold})...")
        
        points_3d = predictions['world_points_from_depth']  # [N, H, W, 3]
        points_rgb = predictions['points_rgb']  # [N, H, W, 3]
        depth_conf = predictions['depth_conf']  # [N, H, W]
        
        # Flatten
        points_flat = points_3d.reshape(-1, 3)
        colors_flat = (points_rgb.reshape(-1, 3) * 255).astype(np.uint8)
        conf_flat = depth_conf.reshape(-1)
        
        print(f"  Total points before filtering: {len(points_flat):,}")
        print(f"  Confidence range: {conf_flat.min():.2f} - {conf_flat.max():.2f}")
        
        # Apply confidence filter
        conf_mask = conf_flat >= conf_threshold
        conf_mask &= np.all(np.isfinite(points_flat), axis=1)
        conf_mask &= np.linalg.norm(points_flat, axis=1) < 100  # Remove very far points
        
        points_filtered = points_flat[conf_mask]
        colors_filtered = colors_flat[conf_mask]
        conf_filtered = conf_flat[conf_mask]
        
        print(f"  Points after confidence filtering: {len(points_filtered):,}")
        
        if len(points_filtered) == 0:
            print("⚠️  WARNING: No points passed the filter!")
            print(f"   Try a lower confidence threshold (current: {conf_threshold})")
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)
        
        # Downsample if needed (keep highest confidence points)
        if max_points is not None and len(points_filtered) > max_points:
            indices = np.argsort(conf_filtered)[-max_points:]
            points_filtered = points_filtered[indices]
            colors_filtered = colors_filtered[indices]
            print(f"  Downsampled to: {len(points_filtered):,} points")
        
        return points_filtered, colors_filtered
    
    def export_glb(
        self,
        predictions: Dict[str, np.ndarray],
        output_path: str,
        conf_thres: float = 50.0,
        show_cam: bool = True,
        mask_black_bg: bool = False,
        mask_white_bg: bool = False
    ):
        """Export as GLB (web viewable format with cameras)"""
        print(f"\n📦 Exporting GLB to {output_path}...")
        
        # Convert images to [N, H, W, 3] format for visual_util
        images_np = predictions['images']
        if images_np.shape[1] == 3:
            images_for_glb = images_np.transpose(0, 2, 3, 1)
        else:
            images_for_glb = images_np
        
        pred_dict = {
            'world_points_from_depth': predictions['world_points_from_depth'],
            'depth_conf': predictions['depth_conf'],
            'images': images_for_glb,
            'extrinsic': predictions['extrinsic']
        }
        
        scene = predictions_to_glb(
            pred_dict,
            conf_thres=conf_thres,
            filter_by_frames="all",
            mask_black_bg=mask_black_bg,
            mask_white_bg=mask_white_bg,
            show_cam=show_cam,
            mask_sky=False,
            target_dir=None,
            prediction_mode="Depthmap and Camera Branch"
        )
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        scene.export(output_path)
        
        print(f"✓ GLB saved successfully")
    
    def export_ply(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        output_path: str
    ):
        """Export as PLY (standard point cloud format)"""
        print(f"\n📦 Exporting PLY to {output_path}...")
        
        if len(points) == 0:
            print("⚠️  Skipping PLY export (no points)")
            return
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        trimesh.PointCloud(points, colors=colors).export(output_path)
        
        print(f"✓ PLY saved with {len(points):,} colored points")
    
    def export_pcd(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        output_path: str
    ):
        """Export as PCD (Point Cloud Data - for Open3D/PCL)"""
        print(f"\n📦 Exporting PCD to {output_path}...")
        
        if len(points) == 0:
            print("⚠️  Skipping PCD export (no points)")
            return
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Normalize colors to 0-1 range
        colors_normalized = colors.astype(np.float32) / 255.0
        
        with open(output_path, 'w') as f:
            # PCD header
            f.write("# .PCD v0.7 - Point Cloud Data file format\n")
            f.write("VERSION 0.7\n")
            f.write("FIELDS x y z rgb\n")
            f.write("SIZE 4 4 4 4\n")
            f.write("TYPE F F F F\n")
            f.write("COUNT 1 1 1 1\n")
            f.write(f"WIDTH {len(points)}\n")
            f.write("HEIGHT 1\n")
            f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
            f.write(f"POINTS {len(points)}\n")
            f.write("DATA ascii\n")
            
            # Write points
            for point, color in zip(points, colors_normalized):
                # Pack RGB into single float (common PCD format)
                rgb_packed = (int(color[0] * 255) << 16) | (int(color[1] * 255) << 8) | int(color[2] * 255)
                f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {rgb_packed}\n")
        
        print(f"✓ PCD saved with {len(points):,} colored points")
    
    def export_xyz(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        output_path: str
    ):
        """Export as XYZ (simple ASCII format with colors)"""
        print(f"\n📦 Exporting XYZ to {output_path}...")
        
        if len(points) == 0:
            print("⚠️  Skipping XYZ export (no points)")
            return
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            for point, color in zip(points, colors):
                f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                       f"{color[0]} {color[1]} {color[2]}\n")
        
        print(f"✓ XYZ saved with {len(points):,} colored points")
    
    def export_npy(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        output_path: str
    ):
        """Export as NPY (NumPy format - best for Python ICP)"""
        print(f"\n📦 Exporting NPY to {output_path}...")
        
        if len(points) == 0:
            print("⚠️  Skipping NPY export (no points)")
            return
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save as dictionary with points and colors
        data = {
            'points': points,
            'colors': colors
        }
        np.save(output_path, data)
        
        print(f"✓ NPY saved with {len(points):,} colored points")
    
    def process_scene(
        self,
        scene_dir: str,
        export_glb: bool = True,
        export_ply: bool = True,
        export_pcd: bool = True,
        export_xyz: bool = True,
        export_npy: bool = True,
        conf_threshold: float = 1.0,
        conf_threshold_glb: float = 50.0,
        max_points: Optional[int] = None,
        show_cameras: bool = True
    ):
        """
        Complete pipeline: load -> reconstruct -> export all formats
        
        Args:
            scene_dir: Directory containing 'images/' subfolder
            export_glb: Export GLB (web viewable)
            export_ply: Export PLY (MeshLab/CloudCompare)
            export_pcd: Export PCD (Open3D/PCL)
            export_xyz: Export XYZ (simple ASCII)
            export_npy: Export NPY (Python/NumPy)
            conf_threshold: Confidence threshold for point clouds (0.1-10.0)
            conf_threshold_glb: Confidence threshold for GLB (0-100 percentage)
            max_points: Maximum points per export (None = unlimited)
            show_cameras: Show cameras in GLB
        """
        scene_path = Path(scene_dir)
        images_dir = scene_path / 'images'
        
        if not images_dir.exists():
            raise ValueError(f"Images directory not found: {images_dir}")
        
        print("="*70)
        print("VGGT 3D RECONSTRUCTION PIPELINE")
        print("="*70)
        
        # Load images
        images, image_names = self.load_images_from_folder(str(images_dir))
        
        # Run VGGT inference
        predictions = self.run_vggt_inference(images)
        
        # Extract point cloud
        points, colors = self.extract_point_cloud(
            predictions,
            conf_threshold=conf_threshold,
            max_points=max_points
        )
        
        print("\n" + "="*70)
        print("EXPORTING RESULTS")
        print("="*70)
        
        # Export GLB (uses its own confidence threshold)
        if export_glb:
            glb_path = scene_path / 'reconstruction.glb'
            self.export_glb(
                predictions,
                str(glb_path),
                conf_thres=conf_threshold_glb,
                show_cam=show_cameras
            )
        
        # Export point clouds in various formats
        if export_ply:
            ply_path = scene_path / 'point_cloud.ply'
            self.export_ply(points, colors, str(ply_path))
        
        if export_pcd:
            pcd_path = scene_path / 'point_cloud.pcd'
            self.export_pcd(points, colors, str(pcd_path))
        
        if export_xyz:
            xyz_path = scene_path / 'point_cloud.xyz'
            self.export_xyz(points, colors, str(xyz_path))
        
        if export_npy:
            npy_path = scene_path / 'point_cloud.npy'
            self.export_npy(points, colors, str(npy_path))
        
        return predictions, points, colors


def main():
    parser = argparse.ArgumentParser(
        description='VGGT 3D Reconstruction - Complete Export Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export all formats (recommended)
  python vggt_export.py --scene_dir ./my_scene
  
  # Export only GLB and PLY
  python vggt_export.py --scene_dir ./my_scene --formats glb ply
  
  # Adjust quality (lower = more points, higher = cleaner)
  python vggt_export.py --scene_dir ./my_scene --conf 0.5 --conf_glb 30.0
  
  # Limit points for faster processing
  python vggt_export.py --scene_dir ./my_scene --max_points 500000

Output Formats:
  - GLB: Web viewable 3D model with cameras
  - PLY: Standard point cloud (MeshLab, CloudCompare, Blender)
  - PCD: Point Cloud Data (Open3D, PCL)
  - XYZ: Simple ASCII (easy to parse)
  - NPY: NumPy format (best for Python ICP)
        """
    )
    
    parser.add_argument('--scene_dir', type=str, required=True,
                       help='Directory containing images/ subfolder')
    parser.add_argument('--formats', nargs='+', 
                       choices=['glb', 'ply', 'pcd', 'xyz', 'npy', 'all'],
                       default=['all'],
                       help='Export formats (default: all)')
    parser.add_argument('--conf', type=float, default=3.0,
                       help='Confidence threshold for point clouds (default: 1.0)')
    parser.add_argument('--conf_glb', type=float, default=50.0,
                       help='Confidence threshold for GLB percentage (default: 50.0)')
    parser.add_argument('--max_points', type=int, default=None,
                       help='Maximum number of points (default: unlimited)')
    parser.add_argument('--no_cameras', action='store_true',
                       help='Do not show cameras in GLB')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device: cuda or cpu')
    
    args = parser.parse_args()
    
    # Determine which formats to export
    export_all = 'all' in args.formats
    export_formats = {
        'glb': export_all or 'glb' in args.formats,
        'ply': export_all or 'ply' in args.formats,
        'pcd': export_all or 'pcd' in args.formats,
        'xyz': export_all or 'xyz' in args.formats,
        'npy': export_all or 'npy' in args.formats,
    }
    
    # Initialize module
    reconstructor = VGGTReconstructionModule(device=args.device)
    
    # Process scene
    predictions, points, colors = reconstructor.process_scene(
        scene_dir=args.scene_dir,
        export_glb=export_formats['glb'],
        export_ply=export_formats['ply'],
        export_pcd=export_formats['pcd'],
        export_xyz=export_formats['xyz'],
        export_npy=export_formats['npy'],
        conf_threshold=args.conf,
        conf_threshold_glb=args.conf_glb,
        max_points=args.max_points,
        show_cameras=not args.no_cameras
    )
    
    # Summary
    print("\n" + "="*70)
    print("✓ RECONSTRUCTION COMPLETE!")
    print("="*70)
    
    scene_path = Path(args.scene_dir)
    
    if export_formats['glb']:
        print(f"🌐 GLB (Web Viewer):      {scene_path / 'reconstruction.glb'}")
        print(f"   → View online: https://3dviewer.net/")
    
    if export_formats['ply']:
        print(f"📦 PLY (MeshLab):         {scene_path / 'point_cloud.ply'}")
    
    if export_formats['pcd']:
        print(f"📦 PCD (Open3D):          {scene_path / 'point_cloud.pcd'}")
    
    if export_formats['xyz']:
        print(f"📦 XYZ (ASCII):           {scene_path / 'point_cloud.xyz'}")
    
    if export_formats['npy']:
        print(f"📦 NPY (Python):          {scene_path / 'point_cloud.npy'}")
    
    print(f"\n💡 Total points exported: {len(points):,}")
    print("="*70)


if __name__ == '__main__':
    main()