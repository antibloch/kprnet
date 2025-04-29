import open3d as o3d
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Regrounding point cloud")
parser.add_argument("--input", type=str, help="Input point cloud file")
args = parser.parse_args()

pcd = o3d.io.read_point_cloud(args.input)

# Assume you already have pcd
plane_model, inliers = pcd.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=1000)
[a, b, c, d] = plane_model


print(f"Detected ground plane: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")
# Compute the normalization factor for the plane normal
norm_factor = np.sqrt(a**2 + b**2 + c**2)
# Convert the point cloud to a NumPy array of points
pcd_np = np.asarray(pcd.points)
# Compute the perpendicular distance from each point to the ground plane
distances = np.abs(a * pcd_np[:, 0] + b * pcd_np[:, 1] + c * pcd_np[:, 2] + d) / norm_factor
# Normalize distances to the range [0, 1] for color mapping]
# distances = np.log1p(np.log1p(np.log1p(np.log1p(distances))))
distances = np.log1p(distances)
distances =np.max(distances) - distances
min_dist = np.min(distances)
max_dist = np.max(distances)
normalized_intensity = (distances - min_dist) / (max_dist - min_dist)

color_np = np.zeros((pcd_np.shape[0], 3))
color_np[:, 0] = normalized_intensity

# view of ground plane via height encoding
#---------------------------------------------------------------------
pcd_new = o3d.geometry.PointCloud()
pcd_new.points = o3d.utility.Vector3dVector(pcd_np)
pcd_new.colors = o3d.utility.Vector3dVector(color_np)
o3d.io.write_point_cloud("colored.ply", pcd_new)
o3d.visualization.draw_geometries([pcd_new])
#---------------------------------------------------------------------

normal = np.array([a, b, c])
normal = normal / np.linalg.norm(normal)

target = np.array([0, 0, 1])
axis = np.cross(normal, target)
axis_norm = np.linalg.norm(axis)

if axis_norm < 1e-6:
    R = np.eye(3)
else:
    axis = axis / axis_norm
    angle = np.arccos(np.clip(np.dot(normal, target), -1.0, 1.0))
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

pcd.rotate(R, center=(0, 0, 0))

# Now ground is flat; align x and y
ground_points = np.asarray(pcd.points)[inliers]
xy_points = ground_points[:, :2]

xy_mean = xy_points.mean(axis=0)
xy_centered = xy_points - xy_mean

cov = np.cov(xy_centered, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eigh(cov)
sort_idx = np.argsort(eigenvalues)[::-1]
eigenvectors = eigenvectors[:, sort_idx]

x_axis = np.array([eigenvectors[0,0], eigenvectors[1,0], 0])
y_axis = np.array([eigenvectors[0,1], eigenvectors[1,1], 0])
z_axis = np.array([0, 0, 1])

R_align = np.stack([x_axis, y_axis, z_axis], axis=1)
pcd.rotate(R_align.T, center=(0,0,0))

# (Optional) Translate to put ground at z=0
ground_points = np.asarray(pcd.points)[inliers]
ground_z_mean = ground_points[:, 2].mean()
pcd.translate((0, 0, -ground_z_mean))



o3d.io.write_point_cloud("aligned.ply", pcd)