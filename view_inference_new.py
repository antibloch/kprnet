import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from datasets.semantic_kitti import (
    SemanticKitti,
    class_names,
    map_inv,
    splits,
)
from utils.evaluation import Eval
from models import deeplab
import os
import yaml
import cv2
import json
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
import shutil


parser = argparse.ArgumentParser("Run lidar bug inference")     
parser.add_argument("--points", required=True, type=Path)
parser.add_argument("--predictions", required=True, type=Path)
parser.add_argument("--results", required=True, type=Path)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

learning_map = {
    0: 255,  # "unlabeled"
    1: 255,  # "outlier" mapped to "unlabeled" --------------------------mapped
    10: 0,  # "car"
    11: 1,  # "bicycle"
    13: 4,  # "bus" mapped to "other-vehicle" --------------------------mapped
    15: 2,  # "motorcycle"
    16: 4,  # "on-rails" mapped to "other-vehicle" ---------------------mapped
    18: 3,  # "truck"
    20: 4,  # "other-vehicle"
    30: 5,  # "person"
    31: 6,  # "bicyclist"
    32: 7,  # "motorcyclist"
    40: 8,  # "road"
    44: 9,  # "parking"
    48: 10,  # "sidewalk"
    49: 11,  # "other-ground"
    50: 12,  # "building"
    51: 13,  # "fence"
    52: 255,  # "other-structure" mapped to "unlabeled" ------------------mapped
    60: 8,  # "lane-marking" to "road" ---------------------------------mapped
    70: 14,  # "vegetation"
    71: 15,  # "trunk"
    72: 16,  # "terrain"
    80: 17,  # "pole"
    81: 18,  # "traffic-sign"
    99: 255,  # "other-object" to "unlabeled" ----------------------------mapped
    252: 0,  # "moving-car" to "car" ------------------------------------mapped
    253: 6,  # "moving-bicyclist" to "bicyclist" ------------------------mapped
    254: 5,  # "moving-person" to "person" ------------------------------mapped
    255: 7,  # "moving-motorcyclist" to "motorcyclist" ------------------mapped
    256: 4,  # "moving-on-rails" mapped to "other-vehicle" --------------mapped
    257: 4,  # "moving-bus" mapped to "other-vehicle" -------------------mapped
    258: 3,  # "moving-truck" to "truck" --------------------------------mapped
    259: 4,  # "moving-other"-vehicle to "other-vehicle" ----------------mappe
}


class_names = [
    "car",
    "bicycle",
    "motorcycle",
    "truck",
    "other-vehicle",
    "person",
    "bicyclist",
    "motorcyclist",
    "road",
    "parking",
    "sidewalk",
    "other-ground",
    "building",
    "fence",
    "vegetation",
    "trunk",
    "terrain",
    "pole",
    "traffic-sign",
]

colors = {
    0: [0, 0, 0],       # unlabeled - black
    1: [0, 0, 1],       # car - blue
    2: [1, 0, 0],       # bicycle - red
    3: [1, 0, 1],       # motorcycle - magenta
    4: [0, 1, 1],       # truck - cyan
    5: [0.5, 0.5, 0],   # other-vehicle - olive
    6: [1, 0.5, 0],     # person - orange
    7: [1, 1, 0],       # bicyclist - yellow
    8: [1, 0, 0.5],     # motorcyclist - pink
    9: [0.5, 0.5, 0.5], # road - gray
    10: [0.5, 0, 0],    # parking - dark red
    11: [0, 0.5, 0],    # sidewalk - dark green
    12: [0, 0, 0.5],    # other-ground - dark blue
    13: [0, 0.5, 0.5],  # building - teal
    14: [0.5, 0, 0.5],  # fence - purple
    15: [0, 1, 0],      # vegetation - green
    16: [0.7, 0.7, 0.7],# trunk - light gray
    17: [0.7, 0, 0.7],  # terrain - light purple
    18: [0, 0.7, 0.7],  # pole - light cyan
    19: [0.7, 0.7, 0]   # traffic-sign - light yellow
}



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




def view_frame(points, pred_colors, legend_patches, output_dir,  i):
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

    # Add legend to the second subplot
    ax.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.15, 1),
              fancybox=True, shadow=True, ncol=1, fontsize='small',
              framealpha=0.8, facecolor='lightgray', edgecolor='black')
    
    

    # Set dark background for entire figure
    fig.set_facecolor((0.1, 0.1, 0.1))

    ax.set_title(f'Scan: {i}', color='white', fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the image
    output_path = os.path.join(output_dir, f"frame_{i}.png")
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, facecolor=(0.1, 0.1, 0.1))
    plt.close()  # Close the figure to free memory






def main():
    num_classes = 19

    point_fil_names = os.listdir(args.points)
    pred_file_names = os.listdir(args.predictions)

    point_fil_names = sorted(point_fil_names, key=lambda x: int(x.split('.')[0]))
    pred_file_names = sorted(pred_file_names, key=lambda x: int(x.split('.')[0]))

    # print(point_fil_names)
    # print(pred_file_names)
    # sdf

    if os.path.exists(args.results):
        shutil.rmtree(args.results)

    os.makedirs(args.results, exist_ok=True)


    for id, point_file, pred_file in zip(range(len(point_fil_names)), point_fil_names, pred_file_names):
        if id<len(point_fil_names):
            point_file_path = os.path.join(args.points, point_file)
            pred_file_path = os.path.join(args.predictions, pred_file)

            data = np.load(point_file_path)
            data = data.astype(np.float32)
            data = data[:, :3]  # Use only the XYZ coordinates


            predictions = np.load(pred_file_path)
            predictions = predictions

    
            points = data


            pred_colors = np.zeros((len(predictions), 3), dtype=np.float32)
            for label in range(num_classes):
                pred_colors[predictions == label] = colors[label]


            legend_patches = []
            for i, class_name in enumerate(class_names):
                rgb_color = colors[i]
                patch = mpatches.Patch(color=rgb_color, label=class_name)   
                legend_patches.append(patch)


            view_frame(points,  pred_colors, legend_patches, args.results, id)

    # Create a video from the saved frames
    create_video_from_frames(args.results, os.path.join(args.results, "output_video.mp4"), fps=2)






if __name__ == "__main__":
    main()
