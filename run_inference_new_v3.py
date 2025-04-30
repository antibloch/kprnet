import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from scipy.spatial.ckdtree import cKDTree as kdtree

from datasets.semantic_kitti import (
    SemanticKitti,
    class_names,
    map_inv,
    splits,
)
from utils.evaluation import Eval
from models import deeplab
import cv2
import os
import open3d as o3d
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _transorm_train(depth, refl, labels, py, px, points_xyz):
    new_h = 289
    new_w = 4097

    py = new_h * py / 65.0
    px = new_w * px / 2049.0

    depth = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    refl = cv2.resize(refl, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    offset_x = np.random.randint(depth.shape[1] - 1025 + 1)
    offset_y = np.random.randint(depth.shape[0] - 289 + 1)

    depth = depth[offset_y : offset_y + 289, offset_x : offset_x + 1025]
    refl = refl[offset_y : offset_y + 289, offset_x : offset_x + 1025]

    py = (py - offset_y) / 289.0
    px = (px - offset_x) / 1025.0

    valid = (px >= 0) & (px <= 1) & (py >= 0) & (py <= 1)
    labels = labels[valid]
    px = px[valid]
    py = py[valid]
    points_xyz = points_xyz[valid, :]
    px = 2.0 * (px - 0.5)
    py = 2.0 * (py - 0.5)

    if np.random.uniform() > 0.5:
        depth = np.flip(depth, axis=1).copy()
        refl = np.flip(refl, axis=1).copy()
        px *= -1

    if px.shape[0] < 38_000:
        pad_len = 38_000 - px.shape[0]
        px = np.hstack([px, np.zeros((pad_len,))])
        py = np.hstack([py, np.zeros((pad_len,))])
        labels = np.hstack([labels, 255 * np.ones((pad_len,))])

    return depth, refl, labels, py, px, points_xyz



def _transorm_test(depth, refl, labels, py, px):
    depth = cv2.resize(depth, (4097, 289), interpolation=cv2.INTER_LINEAR)
    refl = cv2.resize(refl, (4097, 289), interpolation=cv2.INTER_LINEAR)
    py = 2 * (py / 65.0 - 0.5)
    px = 2 * (px / 2049.0 - 0.5)

    return depth, refl, labels, py, px




def do_range_projection_v3(points, reflectivity,
                        cols: int = 2048, channels: int = 64):
    """
    points:  (N,3)
    reflectivity: (N,)
    cols:  horizontal samples per revolution (512/1024/2048)
    channels: vertical scan lines (OS-1-128 ⇒ 128)
    """


    depth = np.linalg.norm(points, axis=1)
    yaw   = -np.arctan2(points[:,1], -points[:,0])

    yaw_norm = (yaw - yaw.min())/(yaw.max() - yaw.min())

    proj_x = (yaw_norm * cols- 1e-3).astype(np.int32)

    new_raw = np.nonzero((yaw_norm[1:] < 0.2) & (yaw_norm[:-1] > 0.8))[0] + 1

    proj_y = np.cumsum(new_raw)


    if proj_y.max() >= channels:
        proj_y = np.clip(proj_y, 0, channels - 1)
        H = channels
    else:
        H = int(proj_y.max()) + 1

    W = int(np.ceil(proj_x.max())) + 1



    order = np.argsort(depth)[::-1]                 # z-buffer
    depth, reflectivity, proj_x, proj_y = [a[order] for a in
                                           (depth, reflectivity, proj_x, proj_y)]

    proj_range        = np.zeros((H, W), dtype=np.float32)
    proj_reflectivity = np.zeros_like(proj_range)
    proj_range[proj_y, proj_x]        = 1.0 / np.maximum(depth, 1e-6)
    proj_reflectivity[proj_y, proj_x] = reflectivity

    # floating-point indices (needed later):
    px = proj_x.astype(np.float32)
    py = proj_y.astype(np.float32)

    return proj_range, proj_reflectivity, py, px, H, W




def do_range_projection_v2(points, reflectivity,
                        cols: int = 2048, channels: int = 64):
    """
    points:  (N,3)
    reflectivity: (N,)
    cols:  horizontal samples per revolution (512/1024/2048)
    channels: vertical scan lines (OS-1-128 ⇒ 128)
    """

    W = cols + 1            # add 1 so that last bin never wraps
    H = channels + 1        # +1 sentinel row, exactly like the HDL-64E code

    depth = np.linalg.norm(points, axis=1)
    yaw   = -np.arctan2(points[:,1], -points[:,0])
    proj_x_norm = (yaw / np.pi + 1.0) * 0.5         # [0,1]

    # --------------- scan-line–based row index ---------------
    wrap = (proj_x_norm[1:] < 0.2) & (proj_x_norm[:-1] > 0.8)
    proj_y = np.concatenate(([0], np.cumsum(wrap))).astype(np.int32)
    
    # ---------------------------------------------------------

    proj_x = (proj_x_norm * W - 1e-3).astype(np.int32)
    H = proj_y.max() + 1
    proj_y = np.minimum(proj_y, H - 1)              # safety clip

    order = np.argsort(depth)[::-1]                 # z-buffer
    depth, reflectivity, proj_x, proj_y = [a[order] for a in
                                           (depth, reflectivity, proj_x, proj_y)]

    proj_range        = np.zeros((H, W), dtype=np.float32)
    proj_reflectivity = np.zeros_like(proj_range)
    proj_range[proj_y, proj_x]        = 1.0 / np.maximum(depth, 1e-6)
    proj_reflectivity[proj_y, proj_x] = reflectivity

    # floating-point indices (needed later):
    px = proj_x.astype(np.float32)
    py = proj_y.astype(np.float32)

    return proj_range, proj_reflectivity, py, px, H, W




def do_range_projection_v1(
    points: np.ndarray, reflectivity: np.ndarray, W: int = 2048, H: int = 64
):
    
    W = W + 1  # add 1 so that last bin never wraps
    H = H + 1  # +1 sentinel row, exactly like the HDL-64E code
    # get depth of all points
    depth = np.linalg.norm(points, 2, axis=1)

    # get scan components
    scan_x = points[:, 0]
    scan_y = points[:, 1]
    scan_z = points[:, 2]

    # get angles of all points
    yaw = -np.arctan2(scan_y, -scan_x)
    proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]

    new_raw = np.nonzero((proj_x[1:] < 0.2) * (proj_x[:-1] > 0.8))[0] + 1
    proj_y = np.zeros_like(proj_x)
    proj_y[new_raw] = 1
    proj_y = np.cumsum(proj_y)
    # scale to image size using angular resolution
    proj_x = proj_x * W - 0.001

    proj_x = np.clip(proj_x, 0, W-1)
    proj_y = np.clip(proj_y, 0, H-1)

    # H = int(proj_y.max())+1
    # print(f"Height: {H}")

    px = proj_x.copy()
    py = proj_y.copy()

    proj_x = np.floor(proj_x).astype(np.int32)
    proj_y = np.floor(proj_y).astype(np.int32)

    # order in decreasing depth
    order = np.argsort(depth)[::-1]

    depth = depth[order]
    reflectivity = reflectivity[order]
    proj_y = proj_y[order]
    proj_x = proj_x[order]

    proj_range = np.zeros((H, W))
    depth[depth == 0] = 1e-6
    proj_range[proj_y, proj_x] = 1.0 / depth

    proj_reflectivity = np.zeros((H, W))
    proj_reflectivity[proj_y, proj_x] = reflectivity

    return (proj_range, proj_reflectivity, py, px)





def transform_points(pts, p, max_zoom):
    center = np.mean(pts, axis=0)

    # Calculate the rotation angle for this frame
    angle = p * 360.0
    
    # Create a rotation matrix around the y-axis
    rotation_matrix = np.array([
        [np.cos(np.radians(angle)), 0, np.sin(np.radians(angle))],
        [0, 1, 0],
        [-np.sin(np.radians(angle)), 0, np.cos(np.radians(angle))]
    ])
    
        # Apply a constant zoom factor to all frames
    current_zoom = max_zoom
    
    # Apply rotation to the points relative to the center
    centered_points = pts - center
    
    # Apply zoom by scaling the points (zoom in = points appear larger)
    zoomed_points = centered_points * current_zoom
    
    # Apply rotation and shift back
    pts = np.dot(zoomed_points, rotation_matrix.T) + center
    return pts


def reground(pts, dist_threshold=0.1, ransac_n = 3):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    # Assume you already have pcd
    plane_model, inliers = pcd.segment_plane(distance_threshold=dist_threshold, ransac_n=ransac_n, num_iterations=1000)
    [a, b, c, d] = plane_model


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
    ground_z_mean = ground_points[:, 2].mean()/2
    pcd.translate((0, 0, -ground_z_mean))

    final_points = np.asarray(pcd.points).astype(np.float32)

    return final_points


def main(args):
        auxil_transform = False
        view_img = False
        model = deeplab.resnext101_aspp_kp(19)
        model.to(device)
        model.load_state_dict(torch.load(args.checkpoint_path))
        print("Runnign validation")
        model.eval()
        point_paths = os.listdir(args.point_folder)
        point_paths = [path for path in point_paths if "points" in path]
        point_paths = sorted(point_paths, key=lambda x: int(x.split("_")[0]))

        depth_m_paths = os.listdir(args.point_folder)
        depth_m_paths = [path for path in depth_m_paths if "depth_m" in path]
        depth_m_paths = sorted(depth_m_paths, key=lambda x: int(x.split("_")[0]))

        refl_m_paths = os.listdir(args.point_folder)
        refl_m_paths = [path for path in refl_m_paths if "refl_m" in path]
        refl_m_paths = sorted(refl_m_paths, key=lambda x: int(x.split("_")[0]))

        proj_x_paths = os.listdir(args.point_folder)
        proj_x_paths = [path for path in proj_x_paths if "proj_x" in path]
        proj_x_paths = sorted(proj_x_paths, key=lambda x: int(x.split("_")[0]))

        proj_y_paths = os.listdir(args.point_folder)
        proj_y_paths = [path for path in proj_y_paths if "proj_y" in path]
        proj_y_paths = sorted(proj_y_paths, key=lambda x: int(x.split("_")[0]))
        

        if os.path.exists(args.output_path):
            os.system(f"rm -rf {args.output_path}")
        os.makedirs(args.output_path, exist_ok=True)

        with torch.no_grad():
            for i, (point_path, depth_m_path, refl_m_path, proj_x_path, proj_y_path) in enumerate(zip(point_paths, depth_m_paths, refl_m_paths, proj_x_paths, proj_y_paths)):
                point_name = point_path.split(".")[0]
                correct_point_path = os.path.join(args.point_folder, point_path)

                points = np.load(correct_point_path)
                points = points.astype(np.float32)

                points_xyz = points[:, :3]
                points_xyz = points_xyz[:, [1,0,2]]
                labels = np.zeros((points.shape[0],))

                points_refl = points[:, 3]

                depth_image = np.load(os.path.join(args.point_folder, depth_m_path))
                refl_image = np.load(os.path.join(args.point_folder, refl_m_path))
                px = np.load(os.path.join(args.point_folder, proj_x_path))
                py = np.load(os.path.join(args.point_folder, proj_y_path))



                depth_image, refl_image, labels, py, px = _transorm_test(
                    depth_image, refl_image, labels, py, px
                )

                if False:
                    # Visualize the range image

                    depth_image = np.clip(depth_image * 255, 0, 255).astype(np.uint8)
                    refl_image = np.clip(refl_image * 255, 0, 255).astype(np.uint8)


                    # depth_image = cv2.resize(depth_image, (224,224), interpolation=cv2.INTER_LINEAR)
                    # refl_image = cv2.resize(refl_image, (224,224), interpolation=cv2.INTER_LINEAR)

                    plt.imshow(depth_image, cmap='gray')
                    plt.title('Depth Image')
                    plt.axis('off')
                    plt.show()

                    plt.imshow(refl_image, cmap='gray')
                    plt.title('Reflectivity Image')
                    plt.axis('off')
                    plt.show()


                tree = kdtree(points_xyz)
                _, knns = tree.query(points_xyz, k=7)

                if points_xyz.shape[0] < px.shape[0]:
                    pad_len = px.shape[0] - points_xyz.shape[0]
                    points_xyz = np.vstack([points_xyz, np.zeros((pad_len, 3))])
                    knns = np.vstack([knns, np.zeros((pad_len, 7))])

                # normalize values to be between -10 and 10
                depth_image = 25 * (depth_image - 0.4)
                refl_image = 20 * (refl_image - 0.5)
                image = np.stack([depth_image, refl_image]).astype(np.float32)

                px = px[np.newaxis, :]
                py = py[np.newaxis, :]
                labels = labels[np.newaxis, :]

                images = torch.from_numpy(image).unsqueeze(0).to(device)   # (1, 2, H, W)
                px     = torch.from_numpy(px).unsqueeze(0).float().to(device)
                py     = torch.from_numpy(py).unsqueeze(0).float().to(device)
                labels = torch.from_numpy(labels).unsqueeze(0)
                pxyz   = torch.from_numpy(points_xyz).unsqueeze(0).float().to(device)
                knns   = torch.from_numpy(knns).unsqueeze(0).long().to(device)


                items = {
                    "image": images,
                    "labels": labels,
                    "py": py,
                    "px": px,
                    "points_xyz": pxyz,
                    "knns": knns,
                }

                images = items["image"].to(device)
                labels = items["labels"]
                py = items["py"].float().to(device)
                px = items["px"].float().to(device)
                pxyz = items["points_xyz"].float().to(device)
                knns = items["knns"].long().to(device)
                
                predictions = model(images, px, py, pxyz, knns)
                _, predictions_argmax = torch.max(predictions, 1)
                predictions_points = predictions_argmax.cpu().numpy()

                predictions_points = predictions_points.astype(np.uint32)

                out_file = os.path.join(args.output_path, f"{point_name}.npy")
                np.save(out_file, predictions_points)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run lidar bug inference")
    parser.add_argument("--checkpoint-path", required=True, type=Path)
    parser.add_argument("--output_path", required=True, type=Path)
    parser.add_argument("--point_folder", required=True, type=Path)

    args = parser.parse_args()
    main(args)
