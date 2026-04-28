import trimesh

path = "/home/AP_PathMatters/path_matters/datasets/Taha_dataset/Dice Castle in halves - 7243491/files/castle_in_printable_halves.stl"

mesh = trimesh.load(path)

center = mesh.centroid
mesh.vertices -= center

mesh.export(path)
