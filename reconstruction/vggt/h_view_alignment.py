import open3d as o3d

scene = o3d.io.read_point_cloud("/home/AP_PathMatters/path_matters/datasets/Reallife_Dataset_Haroun_Aziz/scenes-others_SUBSAMPLED/eyelash_adhesive1_camera3/icp_results/scene_preprocessed.ply")
obj   = o3d.io.read_point_cloud("/home/AP_PathMatters/path_matters/datasets/Reallife_Dataset_Haroun_Aziz/scenes-others_SUBSAMPLED/eyelash_adhesive1_camera3/icp_results/object_aligned.ply")

scene.paint_uniform_color([1, 0, 0])  # red
obj.paint_uniform_color([0, 1, 0])    # green

o3d.visualization.draw_geometries([scene, obj], window_name="Alignment", width=1280, height=720)
