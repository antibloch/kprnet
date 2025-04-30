from ouster.sdk import open_source
from ouster.sdk import client
import open3d as o3d
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


pcap_path = '1024x10-dual.pcap'
metadata_path = '1024x10-dual.json'
source = open_source(pcap_path, meta=[metadata_path])

xyz_lut = client.XYZLut(source.metadata)

def create_video_from_frames(frame_dir, output_path, fps=30):
    """
    Create a video from the saved frames using OpenCV.
    
    Args:
        frame_dir (str): Directory containing the frames
        output_path (str): Path to save the video
        fps (int): Frames per second for the output video
    """
    # Get all PNG files in the directory
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.png')])
    
    if not frame_files:
        print(f"No frames found in {frame_dir}")
        return
    
    # Read the first frame to get dimensions
    first_frame = cv2.imread(os.path.join(frame_dir, frame_files[0]))
    height, width, layers = first_frame.shape
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec for MP4 format
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Add each frame to the video
    for frame_file in frame_files:
        frame_path = os.path.join(frame_dir, frame_file)
        frame = cv2.imread(frame_path)
        
        # Check if frame was loaded properly
        if frame is None:
            print(f"Warning: Could not read frame {frame_path}")
            continue
            
        video.write(frame)
    
    # Release the video writer
    video.release()
    print(f"Video saved to {output_path}")




def view_frame(points, pred_colors, output_dir, i):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Calculate center for rotation
    center = np.mean(points, axis=0)
    
    # Find the bounding box to determine appropriate axis limits
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    extent = np.max(max_bound - min_bound)
    
    # Fixed axis limits for consistent view
    view_radius = extent / 1.5  # Adjust for good default view

    p = 0.1
    max_zoom = 6.0

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
    centered_points = points - center
    
    # Apply zoom by scaling the points (zoom in = points appear larger)
    zoomed_points = centered_points * current_zoom
    
    # Apply rotation and shift back
    rotated_points = np.dot(zoomed_points, rotation_matrix.T) + center

    # Create figure with two horizontal subplots
    fig = plt.figure(figsize=(20, 8), dpi=120)
    
    # First subplot - Predictions
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        rotated_points[:, 0], 
        rotated_points[:, 1], 
        rotated_points[:, 2],
        c=pred_colors, 
        s=0.5,
        alpha=0.7
    )
    ax.set_title('Scans', color='white', fontsize=14)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(center[0] - view_radius, center[0] + view_radius)
    ax.set_ylim(center[1] - view_radius, center[1] + view_radius)
    ax.set_zlim(center[2] - view_radius, center[2] + view_radius)
    ax.set_axis_off()
    ax.set_facecolor((0.1, 0.1, 0.1))
    elev = 20 + 10 * np.sin(np.radians(angle))
    ax.view_init(elev=elev, azim=angle)
    

    # Set dark background for entire figure
    fig.set_facecolor((0.1, 0.1, 0.1))

    ax.set_title(f'Scan: {i}', color='white', fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the image
    output_path = os.path.join(output_dir, f"frame_{i}.png")
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, facecolor=(0.1, 0.1, 0.1))
    plt.close()  # Close the figure to free memory




ctr= 0
thr = 20
if os.path.exists('results'):
    os.system('rm -rf results')

if os.path.exists('data'):
    os.system('rm -rf data')

os.makedirs('data', exist_ok=True)



def do_range_projection_v2(points, reflectivity, info):
    # --------- 0. use the organised (H×W) frame straight from the scan ---------
    # points must be shaped (H, W, 3) **before** flattening; keep (row, col)
    
    H = info.format.pixels_per_column
    W = info.format.columns_per_frame
    print(f"Image shape: {H}, {W}")
    points = points.reshape(H, W, 3)
    reflectivity = reflectivity.reshape(H, W)

    # --------- 1. real beam elevations from metadata, not wrap counting --------
    # row index is simply the laser channel id 0…H-1
    proj_y = np.repeat(np.arange(H)[:, None], W, axis=1).astype(np.int32)

    # --------- 2. azimuth (yaw) computed exactly like the firmware  ------------
    # 0 rad = +X; positive towards +Y
    yaw = np.arctan2(points[:, :, 1], points[:, :, 0])
    proj_x = (yaw / (2 * np.pi) * W).astype(np.int32) % W      # [0, W-1]

    # --------- 3. depth is raw metres, NOT 1/depth -----------------------------
    depth = np.linalg.norm(points, axis=2)                     # (H, W) m

    # --------- 4. put them into images -----------------------------------------
    proj_depth = depth
    proj_refl  = reflectivity
    return proj_depth, proj_refl, proj_y.astype(np.float32), proj_x.astype(np.float32), H, W




def do_range_projection_v1(points, reflectivity, info):
    """
    points:  (N,3)
    reflectivity: (N,)
    cols:  horizontal samples per revolution (512/1024/2048)
    channels: vertical scan lines (OS-1-128 ⇒ 128)
    """
    H = info.format.pixels_per_column
    W = info.format.columns_per_frame

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

    return (proj_range, proj_reflectivity, py, px, H, W)


def do_range_projection_v2(points, reflectivity, info):
    # --------- 0. use the organised (H×W) frame straight from the scan ---------
    # points must be shaped (H, W, 3) **before** flattening; keep (row, col)
    
    H = info.format.pixels_per_column
    W = info.format.columns_per_frame
    print(f"Image shape: {H}, {W}")
    points = points.reshape(H, W, 3)
    reflectivity = reflectivity.reshape(H, W)

    # --------- 1. real beam elevations from metadata, not wrap counting --------
    # row index is simply the laser channel id 0…H-1
    proj_y = np.repeat(np.arange(H)[:, None], W, axis=1).astype(np.int32)

    # --------- 2. azimuth (yaw) computed exactly like the firmware  ------------
    # 0 rad = +X; positive towards +Y
    yaw = np.arctan2(points[:, :, 1], points[:, :, 0])
    proj_x = (yaw / (2 * np.pi) * W).astype(np.int32) % W      # [0, W-1]

    # --------- 3. depth is raw metres, NOT 1/depth -----------------------------
    depth = np.linalg.norm(points, axis=2)                     # (H, W) m
    depth = depth/depth.max()  # Normalize depth to [0, 1]
    # --------- 4. put them into images -----------------------------------------
    proj_depth = depth
    proj_refl  = reflectivity
    return proj_depth, proj_refl, proj_y.astype(np.float32), proj_x.astype(np.float32), H, W




for scan in source:
    # Extract the RANGE field from the scan
    range_field = scan.field(client.ChanField.RANGE)

    # Compute the XYZ coordinates
    xyz = xyz_lut(range_field)  # Shape: (H, W, 3)

    # Optional: Flatten the XYZ array for further processing
    xyz_flat = xyz.reshape(-1, 3)

    reflectivity = scan.field(client.ChanField.REFLECTIVITY)
    signal = scan.field(client.ChanField.SIGNAL)

    range_field = scan.field(client.ChanField.RANGE)
    range_img = client.destagger(source.metadata, range_field)

    range_img_normalized = (range_img / np.max(range_img) * 255).astype(np.uint8)
    # plt.imshow(range_img_normalized, cmap='gray')
    # plt.title('Range Image')
    # plt.show()

    reflectivity = reflectivity.reshape(-1)
    signal = signal.reshape(-1)

    print(f"Scan {ctr}: Reflectivity shape: {reflectivity.shape}, Signal shape: {signal.shape}, XYZ shape: {xyz_flat.shape}")


    # map reflectivty to heatmap colors
    # reflectivity = (reflectivity - reflectivity.min()) / (reflectivity.max() - reflectivity.min())
    # reflectivity = (reflectivity * 255).astype(np.uint8)

    # ------------------------------------------------------------
    reflectivity = reflectivity.astype(np.uint8)
    normalized_reflectivity = reflectivity.astype(np.float32) - np.min(reflectivity)
    normalized_reflectivity = normalized_reflectivity.astype(np.float32) / (np.max(normalized_reflectivity)-np.min(normalized_reflectivity))
    print(f"Normalized reflectivity: {normalized_reflectivity.shape}")

    info = source.metadata

    rng_raw   = scan.field(client.ChanField.RANGE)        # still (H, W) staggered
    refl_raw  = scan.field(client.ChanField.REFLECTIVITY)

    # destagger so that col == azimuth sector
    rng  = client.destagger(info, rng_raw)                # shape (H, W)  ©Ouster
    refl = client.destagger(info, refl_raw)               # shape (H, W)

    refl = refl-refl.min()
    refl_m = refl.astype(np.float32) / (refl.max()-refl.min())

    depth_m = rng.astype(np.float32)
    depth_m = depth_m -depth_m.min()
    depth_m = depth_m.astype(np.float32) / (depth_m.max()-depth_m.min())

    H = info.format.pixels_per_column
    W = info.format.columns_per_frame

    # --------- 1. real beam elevations from metadata, not wrap counting --------
    # row index is simply the laser channel id 0…H-1
    proj_y = np.repeat(np.arange(H)[:, None], W, axis=1).astype(np.int32)

    # --------- 2. azimuth (yaw) computed exactly like the firmware  ------------
    # 0 rad = +X; positive towards +Y
    points_r = xyz_flat.reshape(H, W, 3)
    yaw = np.arctan2(points_r[:, :, 1], points_r[:, :, 0])
    proj_x = (yaw / (2 * np.pi) * W).astype(np.int32) % W      # [0, W-1]


    # ------------------------------------------------------------
    reflectivity = cv2.applyColorMap(reflectivity, cv2.COLORMAP_JET)
    reflectivity_colors = np.squeeze(reflectivity, axis=1)


    if ctr < thr:
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(xyz_flat)
        # pcd.colors = o3d.utility.Vector3dVector(reflectivity_colors / 255.0)

        # o3d.visualization.draw_geometries([pcd])

        view_frame(xyz_flat, reflectivity_colors / 255.0, 'results', ctr)
        data_points = np.concatenate((xyz_flat, np.expand_dims(normalized_reflectivity, axis=-1)), axis=1)

        print(f"Data points shape: {data_points.shape}")
        np.save(os.path.join('data', f"{ctr}.npy"), data_points)
        np.save(os.path.join('data', f"{ctr}_reflectivity.npy"), reflectivity_colors)
        np.save(os.path.join('data', f"{ctr}_depth_m.npy"), depth_m)
        np.save(os.path.join('data', f"{ctr}_refl_m.npy"), refl_m)
        np.save(os.path.join('data', f"{ctr}_proj_x.npy"), proj_x)
        np.save(os.path.join('data', f"{ctr}_proj_y.npy"), proj_y)


    ctr += 1
    if ctr > thr:
        break



create_video_from_frames('results', 'scans.mp4', fps=1)
