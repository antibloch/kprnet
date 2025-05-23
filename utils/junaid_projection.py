import numpy as np
import matplotlib.pyplot as plt
import numpy as np  





def do_range_projection_salsa(points, reflectance, fov_up, fov_down, H, W):
    # assumes fov up and down are in radians
    fov = abs(fov_down) + abs(fov_up)
    print("fov:", fov)
    print("fov_up:", fov_up)
    print("fov_down:", fov_down)
    depth = np.linalg.norm(points, 2, axis=1)+ 1e-6
    scan_x = points[:, 0]
    scan_y = points[:, 1]
    scan_z = points[:, 2]
    yaw = -np.arctan2(scan_y, -scan_x)
    pitch = np.arcsin(scan_z / depth)
    # proj_x = ((yaw+np.pi) / (2*np.pi))  # in [0.0, 1.0]
    proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]
    proj_x *= W  # in [0.0, W]
    proj_y *= H  # in [0.0, H]
    px = proj_x.copy()
    py = proj_y.copy()
    # round and clamp for use as index
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]
    proj_y = np.floor(proj_y)
    proj_y = np.minimum(H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]
    print("min / median / max reflectance:", reflectance.min(),
            np.median(reflectance), reflectance.max())
    print("min / median / max depth:", depth.min(),
            np.median(depth), depth.max())
    print("min / median / max proj_x:", proj_x.min(),
            np.median(proj_x), proj_x.max())
    print("min / median / max proj_y:", proj_y.min(),
            np.median(proj_y), proj_y.max())
    # order in decreasing depth
    indices = np.arange(depth.shape[0])
    order = np.argsort(depth)[::-1]
    depth = depth[order]
    indices = indices[order]
    reflectances = reflectance[order]
    proj_y = proj_y[order]
    proj_x = proj_x[order]
    depthmap   = np.full((H,W), 0, np.float32)
    reflmap = np.full((H,W), 0, np.float32)
    depthmap[proj_y, proj_x] = depth
    reflmap[proj_y, proj_x] = reflectances
    return depthmap, reflmap, px, py



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



    return (proj_range, proj_reflectivity, px, py)



def spherical_projection(points,
                         reflectance,
                         fov_up_deg: float,
                         fov_down_deg: float,
                         H: int,
                         W: int):
    # based on salsa's projection function
    # 1) convert FOV to radians
    fov_up   = np.deg2rad(fov_up_deg)
    fov_down = np.deg2rad(fov_down_deg)
    fov = abs(fov_down) + abs(fov_up)
    # 2) depths and angles
    x, y, z = points[:,0], points[:,1], points[:,2]
    depth = np.linalg.norm(points, axis=1) + 1e-6  # avoid zero
    yaw   = -np.arctan2(y, -x)                # [-π, +π]
    pitch = np.arcsin(z / depth)            # [-π/2, +π/2]
    # 3) normalized projection coords in [0..1]
    proj_x = 0.5 * (yaw/np.pi + 1.0)         # azimuth → [0..1]
    proj_x = (proj_x - np.min(proj_x))/ (np.max(proj_x) - np.min(proj_x))
    proj_y_sub = (pitch + abs(fov_down))/fov
    proj_y_sub = (proj_y_sub - np.min(proj_y_sub)) / (np.max(proj_y_sub) - np.min(proj_y_sub))
    proj_y = 1.0 - proj_y_sub  # elevation → [0..1]
    # 4) scale to image size
    proj_x = proj_x * W
    proj_y = proj_y * (H-1)
    pee_x = proj_x.copy()
    # 5) integer pixel indices
    px = np.floor(proj_x).astype(np.int32)
    py = np.floor(proj_y).astype(np.int32)
    pee_y = py.copy()
    pee_y = pee_y.astype(np.float32)
    # clamp into valid range
    np.clip(px, 0, W-1, out=px)
    np.clip(py, 0, H-1, out=py)
    # 6) prepare output maps
    depth_map = np.zeros((H, W), dtype=np.float32)
    refl_map  = np.zeros((H, W), dtype=np.float32)
    # 7) sort points far→near so nearer overwrite farther
    order = np.argsort(depth)[::-1]
    px_ord = px[order]
    py_ord = py[order]
    depth_ord = depth[order]
    refl_ord  = reflectance[order]
    # 8) fill
    depth_map[py_ord, px_ord] = depth_ord
    refl_map [py_ord, px_ord] = refl_ord
    return depth_map, refl_map, pee_x, pee_y


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
  



def pathetic_projection(
    points, 
    reflectivity, 
    W, 
    H,
    fov_up = 2.0,
    fov_down = -24.9,
    normalize = False,
    visualize = True,
    inverted_depth = True,
):

    r = np.linalg.norm(points, axis=1) + 1e-6
    elevation = np.arcsin(points[:, 2] / r)  # [-π/2, +π/2] 
    elevation_nrom = (elevation + abs(fov_down)) / (abs(fov_up) + abs(fov_down))  # [0, 1]
    order = np.argsort(elevation_nrom)[::-1]  # top to bottom
    points = points[order]
    reflectivity = reflectivity[order]        
    # num_p_bin = np.ceil(len(points)/H).astype(int)
    # bin_indices = np.array([[i] * (num_p_bin) for i in range(H)]).flatten()
    # bin_indices = bin_indices[:len(points)]

    beam_alts = np.linspace(fov_down, fov_up, H)         
    bin_indices = indices_from_elevation(points, beam_alts) 
    # bin_indices = np.flip(bin_indices, axis=0)
    

    cum_points =[]
    cum_depth_map = []
    cum_reflectance_map = []
    cum_reflectivity = []
    cum_depth = []
    proj_x = []
    proj_y = []

    ordered_bin_indices = np.sort(np.unique(bin_indices))
    ordered_bin_indices = np.flip(ordered_bin_indices, axis=0)  # top to bottom

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
        plt.title("V2 Depth (m)")
        plt.imshow(depth_img, cmap='gray',
                vmin=0, vmax=np.percentile(depth_img, 99))
        plt.axis('off')
        plt.subplot(2, 1, 2)
        plt.title("V2 VReflectivity")
        plt.imshow(reflectance_img, cmap='gray',
                vmin=np.min(reflectance_img[reflectance_img > 0]),
                vmax=np.percentile(reflectance_img, 99))
        
        plt.axis('off')
        plt.tight_layout();   plt.show()

    return (depth_img, reflectance_img, proj_x, proj_y, cum_points, cum_reflectivity)
  







def concrete_pathetic_projection(
    points, 
    reflectivity, 
    W, 
    H,
    fov_up = 2.0,
    fov_down = -24.9,
    normalize = False,
    visualize = True,
    inverted_depth = True,
):
    def indices_from_elevation(points, beam_alt_deg):
        beam_alt = np.deg2rad(beam_alt_deg)           
        depth    = np.linalg.norm(points, axis=1) + 1e-6
        elev     = np.arcsin(points[:, 2] / depth)                
        return np.argmin(np.abs(elev[:, None] - beam_alt[None, :]), axis=1)
    
        

    r = np.linalg.norm(points, axis=1) + 1e-6
    elevation = np.arcsin(points[:, 2] / r)  # [-π/2, +π/2] 
    elevation_nrom = (elevation + abs(fov_down)) / (abs(fov_up) + abs(fov_down))  # [0, 1]
    order = np.argsort(elevation_nrom)[::-1]  # top to bottom
    points = points[order]
    reflectivity = reflectivity[order]        
    # num_p_bin = np.ceil(len(points)/H).astype(int)
    # bin_indices = np.array([[i] * (num_p_bin) for i in range(H)]).flatten()
    # bin_indices = bin_indices[:len(points)]

    beam_alts = np.linspace(fov_down, fov_up, H)         
    bin_indices = indices_from_elevation(points, beam_alts) 
    # bin_indices = np.flip(bin_indices, axis=0)
    

    cum_points =[]
    cum_depth_map = []
    cum_reflectance_map = []
    cum_reflectivity = []
    cum_depth = []
    proj_x = []
    proj_y = []

    ordered_bin_indices = np.sort(np.unique(bin_indices))
    ordered_bin_indices = np.flip(ordered_bin_indices, axis=0)  # top to bottom
    max_ordered_bin_index = np.max(ordered_bin_indices)


    point_indices = []
    px_indx_ref = []
    py_indx_ref = []
    orcale_indices = np.arange(len(points))
    beeth_points = points.copy()


    # within each bin, acendingly sort the points by azimuth, and fill the rows of image form left to right
    for i in ordered_bin_indices:
        bin_mask = (bin_indices == i)
        # print(f"bin {i} has {np.sum(bin_mask)} points")
        bin_points = points[bin_mask]
        bin_reflectivity = reflectivity[bin_mask]
        curr_indices = orcale_indices[bin_mask]


        bin_azimuth = -np.arctan2(bin_points[:, 1], - bin_points[:, 0])
        bin_azimuth = (bin_azimuth + np.pi) /(2 * np.pi)  # Normalize to [0, 1]

        order_sort = np.argsort(bin_azimuth)
        sorted_bin_azimuth = bin_azimuth[order_sort]
        sorted_bin_points = bin_points[order_sort]
        sorted_bin_reflectivity = bin_reflectivity[order_sort]
        sorted_curr_indices = curr_indices[order_sort]
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
        proj_y.append(np.ones_like(bin_azimuth_indices) * (max_ordered_bin_index-i))
        cum_points.append(sorted_bin_points)
        cum_reflectivity.append(sorted_bin_reflectivity)
        cum_depth.append(sorted_bin_depth)
        point_indices.append(sorted_curr_indices)

    depth_img = np.vstack(cum_depth_map)  # shape (H, W)
    reflectance_img = np.vstack(cum_reflectance_map) # shape (H, W)
    proj_x = np.concatenate(proj_x)
    proj_y = np.concatenate(proj_y)
    cum_points = np.concatenate(cum_points)
    cum_reflectivity = np.concatenate(cum_reflectivity)
    cum_depth = np.concatenate(cum_depth)
    cum_point_indices = np.concatenate(point_indices)
    ref_proj_x = proj_x.copy()
    ref_proj_y = proj_y.copy()

    sort_order = np.argsort(cum_point_indices)
    cum_point_indices = cum_point_indices[sort_order]
    ref_proj_x = ref_proj_x[sort_order]
    ref_proj_y = ref_proj_y[sort_order]

    points_2_image_indices = {}

    for i in range(len(cum_point_indices)):
        points_2_image_indices[cum_point_indices[i]] = [ref_proj_y[i], ref_proj_x[i]]


    # normalize depth and reflectance values to [0, 1]
    if normalize:
        depth_img = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min())
        reflectance_img = (reflectance_img - reflectance_img.min()) / (reflectance_img.max() - reflectance_img.min())



    if visualize:
        plt.figure(figsize=(12, 4))
        plt.subplot(2, 1, 1)
        plt.title("V2 Depth (m)")
        plt.imshow(depth_img, cmap='gray',
                vmin=0, vmax=np.percentile(depth_img, 99))
        plt.axis('off')
        plt.subplot(2, 1, 2)
        plt.title("V2 VReflectivity")
        plt.imshow(reflectance_img, cmap='gray',
                vmin=np.min(reflectance_img[reflectance_img > 0]),
                vmax=np.percentile(reflectance_img, 99))
        
        plt.axis('off')
        plt.tight_layout();   plt.show()

    return (depth_img, reflectance_img, proj_x, proj_y, beeth_points, points_2_image_indices)
  



def concrete_pathetic_projection_extended(
    points, 
    reflectivity, 
    labels,
    num_classes,
    W, 
    H,
    fov_up = 2.0,
    fov_down = -24.9,
    normalize = False,
    visualize = True,
    inverted_depth = True,
):

    def indices_from_elevation(points, beam_alt_deg):
        beam_alt = np.deg2rad(beam_alt_deg)           
        depth    = np.linalg.norm(points, axis=1) + 1e-6
        elev     = np.arcsin(points[:, 2] / depth)                
        return np.argmin(np.abs(elev[:, None] - beam_alt[None, :]), axis=1)
    


    r = np.linalg.norm(points, axis=1) + 1e-6
    elevation = np.arcsin(points[:, 2] / r)  # [-π/2, +π/2] 
    elevation_nrom = (elevation + abs(fov_down)) / (abs(fov_up) + abs(fov_down))  # [0, 1]
    order = np.argsort(elevation_nrom)[::-1]  # top to bottom
    points = points[order]
    reflectivity = reflectivity[order]       
    labels = labels[order] 
    # num_p_bin = np.ceil(len(points)/H).astype(int)
    # bin_indices = np.array([[i] * (num_p_bin) for i in range(H)]).flatten()
    # bin_indices = bin_indices[:len(points)]

    beam_alts = np.linspace(fov_down, fov_up, H)         
    bin_indices = indices_from_elevation(points, beam_alts) 
    # bin_indices = np.flip(bin_indices, axis=0)
    

    cum_points =[]
    cum_depth_map = []
    cum_reflectance_map = []
    cum_reflectivity = []
    cum_depth = []
    proj_x = []
    proj_y = []

    ordered_bin_indices = np.sort(np.unique(bin_indices))
    ordered_bin_indices = np.flip(ordered_bin_indices, axis=0)  # top to bottom
    max_ordered_bin_index = np.max(ordered_bin_indices)


    point_indices = []
    orcale_indices = np.arange(len(points))

    beeth_points = points.copy()
    beeth_labels = labels.copy()


    # within each bin, acendingly sort the points by azimuth, and fill the rows of image form left to right
    for i in ordered_bin_indices:
        bin_mask = (bin_indices == i)
        # print(f"bin {i} has {np.sum(bin_mask)} points")
        bin_points = points[bin_mask]
        bin_reflectivity = reflectivity[bin_mask]
        curr_indices = orcale_indices[bin_mask]


        bin_azimuth = -np.arctan2(bin_points[:, 1], - bin_points[:, 0])
        bin_azimuth = (bin_azimuth + np.pi) /(2 * np.pi)  # Normalize to [0, 1]

        order_sort = np.argsort(bin_azimuth)
        sorted_bin_azimuth = bin_azimuth[order_sort]
        sorted_bin_points = bin_points[order_sort]
        sorted_bin_reflectivity = bin_reflectivity[order_sort]
        sorted_curr_indices = curr_indices[order_sort]
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
        proj_y.append(np.ones_like(bin_azimuth_indices) * (max_ordered_bin_index-i))
        cum_points.append(sorted_bin_points)
        cum_reflectivity.append(sorted_bin_reflectivity)
        cum_depth.append(sorted_bin_depth)
        point_indices.append(sorted_curr_indices)

    depth_img = np.vstack(cum_depth_map)  # shape (H, W)
    reflectance_img = np.vstack(cum_reflectance_map) # shape (H, W)
    proj_x = np.concatenate(proj_x)
    proj_y = np.concatenate(proj_y)
    cum_points = np.concatenate(cum_points)
    cum_reflectivity = np.concatenate(cum_reflectivity)
    cum_depth = np.concatenate(cum_depth)
    cum_point_indices = np.concatenate(point_indices)
    ref_proj_x = proj_x.copy()
    ref_proj_y = proj_y.copy()

    sort_order = np.argsort(cum_point_indices)
    cum_point_indices = cum_point_indices[sort_order]
    ref_proj_x = ref_proj_x[sort_order]
    ref_proj_y = ref_proj_y[sort_order]
    beeth_points = beeth_points[sort_order]
    beeth_labels = beeth_labels[sort_order]

    print(f"Size of Point Indices: {cum_point_indices.shape}")
    print(f"Size of Points: {beeth_points.shape}")
    print(f"Size of Labels: {beeth_labels.shape}")

    points_2_image_indices = {}

    for i in range(len(cum_point_indices)):
        points_2_image_indices[cum_point_indices[i]] = [ref_proj_y[i], ref_proj_x[i]]

    cl_maps = []
    unique_clsses = np.unique(beeth_labels)
    print(f"Unique classes in reference labels: {unique_clsses}")
    for cl in range(len(unique_clsses)):
        mask = (beeth_labels == unique_clsses[cl])
        mask = np.squeeze(mask, axis = 1)
        idx = cum_point_indices[mask]

        cl_map = np.zeros((H, W, 3), dtype=np.float32)
        
        coord_x = []
        coord_y = []
        for i in range(len(idx)):
            coord_x.append(points_2_image_indices[idx[i]][0])
            coord_y.append(points_2_image_indices[idx[i]][1])

        coord_x = np.array(coord_x)
        coord_y = np.array(coord_y)


        # Generate a single random color for the current class
        random_color = np.random.rand(3)
        cl_map[coord_x, coord_y, 0] = random_color[0]
        cl_map[coord_x, coord_y, 1] = random_color[1]
        cl_map[coord_x, coord_y, 2] = random_color[2]


        cl_maps.append(cl_map)


    # normalize depth and reflectance values to [0, 1]
    if normalize:
        depth_img = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min())
        reflectance_img = (reflectance_img - reflectance_img.min()) / (reflectance_img.max() - reflectance_img.min())



    if visualize:
        plt.figure(figsize=(12, 4))
        plt.subplot(2, 1, 1)
        plt.title("V2 Depth (m)")
        plt.imshow(depth_img, cmap='gray',
                vmin=0, vmax=np.percentile(depth_img, 99))
        plt.axis('off')
        plt.subplot(2, 1, 2)
        plt.title("V2 VReflectivity")
        plt.imshow(reflectance_img, cmap='gray',
                vmin=np.min(reflectance_img[reflectance_img > 0]),
                vmax=np.percentile(reflectance_img, 99))
        
        plt.axis('off')
        plt.tight_layout();   plt.show()


        total_cl_map = np.zeros((H, W, 3), dtype=np.float32)
        for i in range(len(cl_maps)):
            total_cl_map += cl_maps[i]


        plt.figure(figsize=(12, 12))
        plt.imshow(total_cl_map)
        plt.axis('off')
        plt.tight_layout()
        plt.show()


    return (depth_img, reflectance_img, proj_x, proj_y, beeth_points, points_2_image_indices)
  

