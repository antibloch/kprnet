import numpy as np
import matplotlib.pyplot as plt
import numpy as np  



def do_range_projection(
    points: np.ndarray, reflectivity: np.ndarray, W: int = 2049, H: int = 65,
):
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

    # yaw = -np.arctan2(points[:, 1], -points[:, 0]) 
    ring = ((np.unwrap(yaw) - yaw[0]) / (2*np.pi)).astype(np.int32)
    # proj_y = (ring % H).astype(np.int32)   


    # scale to image size using angular resolution
    proj_x = proj_x * W - 0.001

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
    proj_range[proj_y, proj_x] = 1.0 / depth

    proj_reflectivity = np.zeros((H, W))
    proj_reflectivity[proj_y, proj_x] = reflectivity

    plt.figure(figsize=(12, 4))
    plt.subplot(2, 1, 1)
    plt.title("Depth (m)")
    plt.imshow(proj_range, cmap='gray',
            vmin=0, vmax=np.percentile(proj_range, 99))
    plt.axis('off')
    plt.subplot(2, 1, 2)
    plt.title("Reflectivity")
    plt.imshow(proj_reflectivity, cmap='gray',
            vmin=np.min(proj_reflectivity[proj_reflectivity > 0]),
            vmax=np.percentile(proj_reflectivity, 99))
    
    plt.axis('off')
    plt.tight_layout();   plt.show()



    return (proj_range, proj_reflectivity, py, px)



def indices_from_yaw(points, H):
    yaw = -np.arctan2(points[:, 1], -points[:, 0]) 
    ring = ((np.unwrap(yaw) - yaw[0]) / (2*np.pi)).astype(np.int32)
    return (ring % H).astype(np.int32)     


# def indices_from_yaw(points, H):
#     scan_x = points[:, 0]
#     scan_y = points[:, 1]
    
#     yaw = -np.arctan2(scan_y, -scan_x)
#     proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]

#     new_raw = np.nonzero((proj_x[1:] < 0.2) * (proj_x[:-1] > 0.8))[0] + 1
#     proj_y = np.zeros_like(proj_x)
#     proj_y[new_raw] = 1
#     proj_y = np.cumsum(proj_y)
#     proj_y = np.floor(proj_y).astype(np.int32)

#     return proj_y



def indices_from_elevation(points, beam_alt_deg):
    beam_alt = np.deg2rad(beam_alt_deg)           
    depth    = np.linalg.norm(points, axis=1) + 1e-6
    elev     = np.arcsin(points[:, 2] / depth)                
    return np.argmin(np.abs(elev[:, None] - beam_alt[None, :]), axis=1)


def do_range_projection_v2(
    points, 
    reflectivity, 
    W, 
    H,
    beam_alt_deg = None,
    fov_up = 2.0,
    fov_down = -24.9,
    ring_major = True,
    normalize = False,
    visualize = True,
    inverted_depth = True,
):
    print(f"Beam Altitude: {beam_alt_deg}")

    if ring_major:
        print("Ring Major")
        bin_indices = indices_from_yaw(points, H)  


    elif len(beam_alt_deg)>0:
        # asda
        beam_alts = np.asarray(beam_alt_deg)  # degrees
        bin_indices = indices_from_elevation(points, beam_alts) 

    elif beam_alt_deg is None and fov_up is not None and fov_down is not None:
        beam_alts = np.linspace(fov_down, fov_up, H)         
        bin_indices = indices_from_elevation(points, beam_alts) 
        bin_indices = np.flip(bin_indices, axis=0)

    else:
        num_p_bin = np.ceil(len(points)/H).astype(int)
        bin_indices = np.array([[i] * (num_p_bin) for i in range(H)]).flatten()
        bin_indices = bin_indices[:len(points)]
  

    cum_points =[]
    cum_depth_map = []
    cum_reflectance_map = []
    cum_reflectivity = []
    cum_depth = []
    proj_x = []
    proj_y = []

    ordered_bin_indices = np.sort(np.unique(bin_indices))

    # within each bin, acendingly sort the points by azimuth, and fill the rows of image form left to right
    for i in ordered_bin_indices:
        bin_mask = (bin_indices == i)
        # print(f"bin {i} has {np.sum(bin_mask)} points")
        bin_points = points[bin_mask]
        bin_reflectivity = reflectivity[bin_mask]
        bin_azimuth = -np.arctan2(bin_points[:, 1], - bin_points[:, 0])
        bin_azimuth = (bin_azimuth + np.pi) /(2 * np.pi)  # Normalize to [0, 1]

        order_sort = np.argsort(bin_azimuth)
        sorted_bin_azimuth = bin_azimuth[order_sort]
        sorted_bin_points = bin_points[order_sort]
        sorted_bin_reflectivity = bin_reflectivity[order_sort]
        sorted_bin_depth = np.linalg.norm(sorted_bin_points, 2, axis=1) + 1e-6

        bin_azimuth_indices = (sorted_bin_azimuth* (W-1)).astype(np.int32)
        depth_pixels = np.zeros(W).astype(np.float32)
        refl_pixels = np.zeros(W).astype(np.float32)

        if not inverted_depth:
            depth_pixels[bin_azimuth_indices] = sorted_bin_depth
        else:
            depth_pixels[bin_azimuth_indices] = 1.0/(0.5+sorted_bin_depth)
            
        refl_pixels[bin_azimuth_indices] = sorted_bin_reflectivity

        cum_depth_map.append(depth_pixels)
        cum_reflectance_map.append(refl_pixels)

        proj_x.append(bin_azimuth_indices)
        proj_y.append(np.ones_like(bin_azimuth_indices) * i)
        cum_points.append(sorted_bin_points)
        cum_reflectivity.append(sorted_bin_reflectivity)
        cum_depth.append(sorted_bin_depth)

    depth_img = np.vstack(cum_depth_map)  # shape (H, W)
    reflectance_img = np.vstack(cum_reflectance_map) # shape (H, W)
    proj_x = np.concatenate(proj_x)
    proj_y = np.concatenate(proj_y)
    cum_points = np.concatenate(cum_points)
    cum_reflectivity = np.concatenate(cum_reflectivity)
    cum_depth = np.concatenate(cum_depth)

    # in TOM-TOM's KPRNet, evaluation, inorder to get returning proj_x and proj_y
    # tom_order = np.argsort(cum_depth)[::-1]
    # proj_y = proj_y[tom_order]
    # proj_x = proj_x[tom_order]


    # normalize depth and reflectance values to [0, 1]
    if normalize:
        depth_img = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min())
        reflectance_img = (reflectance_img - reflectance_img.min()) / (reflectance_img.max() - reflectance_img.min())


    if visualize:
        plt.figure(figsize=(12, 4))
        plt.subplot(2, 1, 1)
        plt.title("Depth (m)")
        plt.imshow(depth_img, cmap='gray',
                vmin=0, vmax=np.percentile(depth_img, 99))
        plt.axis('off')
        plt.subplot(2, 1, 2)
        plt.title("Reflectivity")
        plt.imshow(reflectance_img, cmap='gray',
                vmin=np.min(reflectance_img[reflectance_img > 0]),
                vmax=np.percentile(reflectance_img, 99))
        
        plt.axis('off')
        plt.tight_layout();   plt.show()

    return (depth_img, reflectance_img, proj_x, proj_y, cum_points, cum_reflectivity)
  

