"""
3D Format Converter Module
Converts between common 3D formats: OBJ, STL, PLY, PCD, USD, GLTF
Uses Open3D and trimesh for robust conversion

open a terminal
python /home/AP_PathMatters/path_matters/Isaacsim/scripts/pcd_preprocess/14_12_convert_mesh.py /path/to/model.obj 50000
"""


import os
import numpy as np
from pathlib import Path
from typing import Optional, Union, List
import sys

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("⚠️  Open3D not available. Install: pip install open3d")

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    print("⚠️  Trimesh not available. Install: pip install trimesh")


class FormatConverter:
    """
    Universal 3D format converter
    
    Supported formats:
    - Mesh: .obj, .stl, .ply, .off, .gltf, .glb
    - Point Cloud: .ply, .pcd, .xyz, .pts
    - USD: .usd, .usda, .usdc (requires pxr)
    """
    
    MESH_FORMATS = {'.obj', '.stl', '.ply', '.off', '.gltf', '.glb'}
    POINT_CLOUD_FORMATS = {'.ply', '.pcd', '.xyz', '.pts'}
    USD_FORMATS = {'.usd', '.usda', '.usdc'}
    
    def __init__(self):
        if not OPEN3D_AVAILABLE and not TRIMESH_AVAILABLE:
            raise RuntimeError("Need at least Open3D or Trimesh installed")
    
    @staticmethod
    def get_format(filepath: Union[str, Path]) -> str:
        """Extract file extension"""
        return Path(filepath).suffix.lower()
    
    def convert(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        sampling_density: Optional[int] = None,
        normalize: bool = False
    ) -> bool:
        """
        Convert between 3D formats
        
        Args:
            input_path: Source file
            output_path: Destination file
            sampling_density: For mesh->point cloud, number of points to sample
            normalize: Normalize to unit sphere
            
        Returns:
            True if successful
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        if not input_path.exists():
            print(f"❌ Input file not found: {input_path}")
            return False
        
        input_ext = self.get_format(input_path)
        output_ext = self.get_format(output_path)
        
        print(f"\n🔄 Converting {input_ext} → {output_ext}")
        print(f"   Input:  {input_path}")
        print(f"   Output: {output_path}")
        
        try:
            # Determine conversion pathway
            # FORCE point cloud conversion if output is .ply and sampling_density is specified
            if output_ext == '.ply' and sampling_density is not None and input_ext in self.MESH_FORMATS:
                return self._mesh_to_pointcloud(input_path, output_path, sampling_density, normalize)
            
            elif input_ext in self.MESH_FORMATS and output_ext in self.MESH_FORMATS:
                return self._mesh_to_mesh(input_path, output_path, normalize)
            
            elif input_ext in self.MESH_FORMATS and output_ext in self.POINT_CLOUD_FORMATS:
                return self._mesh_to_pointcloud(input_path, output_path, sampling_density, normalize)
            
            elif input_ext in self.POINT_CLOUD_FORMATS and output_ext in self.POINT_CLOUD_FORMATS:
                return self._pointcloud_to_pointcloud(input_path, output_path, normalize)
            
            elif input_ext in self.POINT_CLOUD_FORMATS and output_ext in self.MESH_FORMATS:
                return self._pointcloud_to_mesh(input_path, output_path, normalize)
            
            else:
                print(f"❌ Unsupported conversion: {input_ext} → {output_ext}")
                return False
                
        except Exception as e:
            print(f"❌ Conversion failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _mesh_to_mesh(self, input_path: Path, output_path: Path, normalize: bool) -> bool:
        """Convert mesh to mesh (format change only)"""
        if OPEN3D_AVAILABLE:
            mesh = o3d.io.read_triangle_mesh(str(input_path))
            
            if not mesh.has_vertices():
                print("❌ Failed to load mesh")
                return False
            
            if normalize:
                mesh = self._normalize_mesh(mesh)
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            success = o3d.io.write_triangle_mesh(str(output_path), mesh)
            
            if success:
                print(f"✅ Saved mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")
            return success
        
        elif TRIMESH_AVAILABLE:
            mesh = trimesh.load(str(input_path))
            
            if normalize:
                # Center and scale to unit sphere
                mesh.vertices -= mesh.vertices.mean(axis=0)
                scale = np.max(np.linalg.norm(mesh.vertices, axis=1))
                mesh.vertices /= scale
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            mesh.export(str(output_path))
            
            print(f"✅ Saved mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
            return True
        
        return False
    
    def _mesh_to_pointcloud(
        self,
        input_path: Path,
        output_path: Path,
        sampling_density: Optional[int],
        normalize: bool
    ) -> bool:
        """Convert mesh to point cloud by sampling"""
        if OPEN3D_AVAILABLE:
            mesh = o3d.io.read_triangle_mesh(str(input_path))
            
            if not mesh.has_vertices():
                print("❌ Failed to load mesh")
                return False
            
            # Sample points from mesh surface
            if sampling_density is None:
                # Auto-calculate based on mesh size
                sampling_density = max(10000, len(mesh.vertices) * 2)
            
            print(f"   Sampling {sampling_density} points from mesh...")
            pcd = mesh.sample_points_uniformly(number_of_points=sampling_density)
            
            if normalize:
                pcd = self._normalize_pointcloud(pcd)
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            success = o3d.io.write_point_cloud(str(output_path), pcd)
            
            if success:
                print(f"✅ Saved point cloud: {len(pcd.points)} points")
            return success
        
        elif TRIMESH_AVAILABLE:
            mesh = trimesh.load(str(input_path))
            
            if sampling_density is None:
                sampling_density = max(10000, len(mesh.vertices) * 2)
            
            print(f"   Sampling {sampling_density} points from mesh...")
            points, _ = trimesh.sample.sample_surface(mesh, sampling_density)
            
            if normalize:
                points -= points.mean(axis=0)
                scale = np.max(np.linalg.norm(points, axis=1))
                points /= scale
            
            # Save as PLY using trimesh
            point_cloud = trimesh.points.PointCloud(points)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            point_cloud.export(str(output_path))
            
            print(f"✅ Saved point cloud: {len(points)} points")
            return True
        
        return False
    
    def _pointcloud_to_pointcloud(self, input_path: Path, output_path: Path, normalize: bool) -> bool:
        """Convert point cloud to point cloud (format change only)"""
        if OPEN3D_AVAILABLE:
            pcd = o3d.io.read_point_cloud(str(input_path))
            
            if not pcd.has_points():
                print("❌ Failed to load point cloud")
                return False
            
            if normalize:
                pcd = self._normalize_pointcloud(pcd)
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            success = o3d.io.write_point_cloud(str(output_path), pcd)
            
            if success:
                print(f"✅ Saved point cloud: {len(pcd.points)} points")
            return success
        
        return False
    
    def _pointcloud_to_mesh(self, input_path: Path, output_path: Path, normalize: bool) -> bool:
        """Convert point cloud to mesh using surface reconstruction"""
        if not OPEN3D_AVAILABLE:
            print("❌ Open3D required for point cloud → mesh conversion")
            return False
        
        pcd = o3d.io.read_point_cloud(str(input_path))
        
        if not pcd.has_points():
            print("❌ Failed to load point cloud")
            return False
        
        print(f"   Reconstructing mesh from {len(pcd.points)} points...")
        
        # Estimate normals if not present
        if not pcd.has_normals():
            print("   Estimating normals...")
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
        
        # Poisson surface reconstruction
        print("   Running Poisson reconstruction...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9
        )
        
        # Remove low-density vertices
        vertices_to_remove = densities < np.quantile(densities, 0.01)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        if normalize:
            mesh = self._normalize_mesh(mesh)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        success = o3d.io.write_triangle_mesh(str(output_path), mesh)
        
        if success:
            print(f"✅ Saved mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")
        return success
    
    @staticmethod
    def _normalize_mesh(mesh):
        """Normalize mesh to unit sphere centered at origin"""
        vertices = np.asarray(mesh.vertices)
        vertices -= vertices.mean(axis=0)
        scale = np.max(np.linalg.norm(vertices, axis=1))
        vertices /= scale
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        return mesh
    
    @staticmethod
    def _normalize_pointcloud(pcd):
        """Normalize point cloud to unit sphere centered at origin"""
        points = np.asarray(pcd.points)
        points -= points.mean(axis=0)
        scale = np.max(np.linalg.norm(points, axis=1))
        points /= scale
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd
    
    def batch_convert(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        output_format: str,
        input_formats: Optional[List[str]] = None,
        **kwargs
    ) -> int:
        """
        Batch convert all files in directory
        
        Args:
            input_dir: Directory with source files
            output_dir: Directory for converted files
            output_format: Target format (e.g., '.ply')
            input_formats: List of formats to convert (None = all supported)
            **kwargs: Additional args for convert()
            
        Returns:
            Number of successfully converted files
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if input_formats is None:
            input_formats = self.MESH_FORMATS | self.POINT_CLOUD_FORMATS
        
        files = []
        for ext in input_formats:
            files.extend(input_dir.glob(f"*{ext}"))
        
        print(f"\n{'='*70}")
        print(f"  BATCH CONVERSION")
        print(f"{'='*70}")
        print(f"Input dir:  {input_dir}")
        print(f"Output dir: {output_dir}")
        print(f"Format:     → {output_format}")
        print(f"Files:      {len(files)}")
        
        success_count = 0
        
        for i, input_file in enumerate(files, 1):
            output_file = output_dir / f"{input_file.stem}{output_format}"
            
            print(f"\n[{i}/{len(files)}] {input_file.name}")
            
            if self.convert(input_file, output_file, **kwargs):
                success_count += 1
        
        print(f"\n{'='*70}")
        print(f"✅ Converted {success_count}/{len(files)} files")
        print(f"{'='*70}\n")
        
        return success_count


# ============================================================
# USAGE EXAMPLES
# ============================================================

# ============================================================
# SIMPLE CLI USAGE
# ============================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python converter.py <input_file> [output_format] [sampling_density]")
        print("Examples:")
        print("  python converter.py /path/to/model.obj .ply")
        print("  python converter.py /path/to/model.stl .pcd 50000")
        print("  python converter.py /path/to/model.obj .ply 50000")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    
    if not input_path.exists():
        print(f"❌ File not found: {input_path}")
        sys.exit(1)
    
    # Check if second arg is format or density
    if len(sys.argv) > 2:
        second_arg = sys.argv[2]
        # If it's a number, treat as density with default .ply format
        if second_arg.isdigit():
            output_format = ".ply"
            sampling_density = int(second_arg)
        else:
            # It's a format
            output_format = second_arg if second_arg.startswith('.') else '.' + second_arg
            sampling_density = int(sys.argv[3]) if len(sys.argv) > 3 else None
    else:
        output_format = ".ply"
        sampling_density = None
    
    # Output in same folder as input
    output_path = input_path.parent / f"{input_path.stem}{output_format}"
    
    print(f"\n{'='*70}")
    print(f"  3D FORMAT CONVERTER")
    print(f"{'='*70}")
    
    converter = FormatConverter()
    success = converter.convert(
        input_path=input_path,
        output_path=output_path,
        sampling_density=sampling_density,
        normalize=True
    )
    
    if success:
        print(f"\n✅ SUCCESS!")
        print(f"   Saved to: {output_path}")
    else:
        print(f"\n❌ FAILED")
        sys.exit(1)