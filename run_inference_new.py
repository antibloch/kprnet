# infer_npy_semkitti_style.py
# ------------------------------------------------------------
import os, numpy as np, cv2, torch, open3d as o3d
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.ckdtree import cKDTree as kdtree
from torch.utils.data import Dataset, DataLoader
from utils.junaid_projection import do_range_projection_v2, do_range_projection
from models import deeplab
from utils.evaluation import Eval
import matplotlib.pyplot as plt
import shutil

def _transorm_test(depth, refl, labels, py, px):
    depth = cv2.resize(depth, (4097, 289), interpolation=cv2.INTER_LINEAR)
    refl = cv2.resize(refl, (4097, 289), interpolation=cv2.INTER_LINEAR)
    py = 2 * (py / 65.0 - 0.5)
    px = 2 * (px / 2049.0 - 0.5)

    return depth, refl, labels, py, px



# (If you already pasted those helpers in your file, just import them.)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ------------------------------------------------------------
# 1.  Dataset that mimics SemanticKitti.__getitem__ 100 %
# ------------------------------------------------------------
class NPYSemanticStyle(Dataset):
    """
    Imitates the original SemanticKitti Dataset class but reads
    arbitrary *.npy point clouds (N,4) = (x,y,z,intensity).
    Because you only do *inference*, it always behaves like the
    'test' split: `labels` is all zeros.
    """
    def __init__(self,
                 npy_dir,
                 split= "test",
                 W = 2048,
                 H = 64,
                fov_up = 2.0,
                fov_down = -24.9,
                beam_alt_deg  = None,
                ring_major = True,
                normalize = False,
                visualize = False,
                inverted_depth = True,
                 use_train_aug = False):
        """
        Args
        ----
        npy_dir :  folder that contains *.npy files
        split   :  'train' | 'val' | 'test' -- only affects transforms flag
        use_train_aug : if True, applies _transorm_train; otherwise _transorm_test
        """
        super().__init__()
        self.npy_paths = sorted(
            [Path(npy_dir) / f for f in os.listdir(npy_dir) if f.endswith(".npy")]
        )
        if not self.npy_paths:
            raise FileNotFoundError(f"No .npy files in {npy_dir}")
        self.split = split
        self.use_train_aug = use_train_aug and split == "train"
        self.W = W
        self.H = H
        self. fov_up = fov_up
        self.fov_down = fov_down
        self.beam_alt_deg = beam_alt_deg    
        self.ring_major = ring_major
        self.normalize = normalize  
        self.visualize = visualize
        self.inverted_depth = inverted_depth
    # ---------------------------- #
    def __len__(self):
        return len(self.npy_paths)
    # ---------------------------- #
    def __getitem__(self, index):
        npy_file = self.npy_paths[index]
        pts = np.load(npy_file).astype(np.float32).reshape(-1, 4)
        points_xyz = pts[:, :3]            # (N,3)
        points_refl = pts[:, 3]            # (N,)


        labels = np.zeros(points_xyz.shape[0], dtype=np.float32)   # dummy
        # projection (identical to reference)
        


        depth_img, refl_img, px, py, points_xyz , points_refl = do_range_projection_v2(
                                                                            points=points_xyz, 
                                                                            reflectivity = points_refl, 
                                                                            W = self.W, 
                                                                            H = self.H,
                                                                            beam_alt_deg = self.beam_alt_deg,
                                                                            fov_up = self.fov_up,
                                                                            fov_down = self.fov_down,
                                                                            ring_major = self.ring_major,
                                                                            normalize = self.normalize,
                                                                            visualize = self.visualize,
                                                                            inverted_depth = self.inverted_depth,
                                                                        )



        # depth_img, refl_img,  py, px = do_range_projection(
        #     points=points_xyz, 
        #     reflectivity = points_refl, 
        #     W = 2049, 
        #     H = 65,
        # )

        # choose the same transform block as reference (“train” vs “test”)

        depth_img, refl_img, labels, py, px = _transorm_test(
            depth_img, refl_img, labels, py, px
        )
        # k-NN exactly as reference
        tree = kdtree(points_xyz)
        _, knns = tree.query(points_xyz, k=7)
        if points_xyz.shape[0] < px.shape[0]:
            pad_len = px.shape[0] - points_xyz.shape[0]
            points_xyz = np.vstack([points_xyz, np.zeros((pad_len, 3))])
            knns = np.vstack([knns, np.zeros((pad_len, 7))])
        # normalise & pack (unchanged)
        depth_img = 25 * (depth_img - 0.4)
        refl_img  = 20 * (refl_img  - 0.5)
        image = np.stack([depth_img, refl_img]).astype(np.float32)
        px = px[np.newaxis, :]
        py = py[np.newaxis, :]
        labels = labels[np.newaxis, :]


        return {
            "image": image,                  # (2,289,4097) after _transorm_test
            "labels": labels,                # dummy, kept for API parity
            "px": px,
            "py": py,
            "points_xyz": points_xyz,
            "knns": knns,
            "fname": npy_file.name,
        }
# ------------------------------------------------------------
# 2.  Main inference loop (unchanged logic)
# ------------------------------------------------------------

def run_inference(args):

    point_output_path = str(args.output_path) +"_points"
    if os.path.exists(args.output_path):
        shutil.rmtree(args.output_path)

    os.makedirs(args.output_path, exist_ok=True)

    if os.path.exists(point_output_path ):
        shutil.rmtree(point_output_path )

    os.makedirs(point_output_path , exist_ok=True)


    do_quick_vis = False
    # intended for Os0 scan of ouster
    beam_alt_deg = [
            44.07,
            42.42,
            41.08,
            39.47,
            38.12,
            36.53,
            35.19,
            33.61,
            32.26,
            30.71,
            29.36,
            27.84,
            26.48,
            24.98,
            23.62,
            22.13,
            20.77,
            19.3,
            17.95,
            16.5,
            15.14,
            13.7,
            12.33,
            10.92,
            9.54,
            8.13,
            6.76,
            5.36,
            3.98,
            2.6,
            1.21,
            -0.17,
            -1.55,
            -2.93,
            -4.33,
            -5.7,
            -7.11,
            -8.47,
            -9.88,
            -11.25,
            -12.68,
            -14.04,
            -15.47,
            -16.84,
            -18.29,
            -19.64,
            -21.12,
            -22.46,
            -23.95,
            -25.3,
            -26.8,
            -28.15,
            -29.68,
            -31.02,
            -32.57,
            -33.91,
            -35.48,
            -36.83,
            -38.42,
            -39.77,
            -41.39,
            -42.74,
            -44.39,
            -45.73
        ]



    # model (same as before)
    from models import deeplab
    model = deeplab.resnext101_aspp_kp(19)
    model.to(device)
    model.load_state_dict(torch.load(args.checkpoint_path))
    print("Runnign validation")
    model.eval()
    eval_metric = Eval(19, 255)
    # dataset & loader *exactly like reference*
    ds = NPYSemanticStyle(
                            args.point_folder, 
                            split="test",
                            W = args.W,
                            H = args.H,
                            fov_up = args.fov_up,
                            fov_down = args.fov_down,
                            beam_alt_deg= beam_alt_deg,      # Oor None in case there isnot
                            ring_major= args.ring_major,
                            normalize= args.normalize,
                            visualize= args.visualize,
                            inverted_depth= args.inverted_depth
                          )
    dl = DataLoader(ds,
                    batch_size=1,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True,
                    drop_last=False)
    out_root = Path(args.output_path)
    out_root.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for batch in tqdm(dl, desc="predict"):
            images = batch["image"].to(device)               # (1,2,289,4097)
            px     = batch["px"].float().to(device)          # (1, N)
            py     = batch["py"].float().to(device)
            pxyz   = batch["points_xyz"].float().to(device)  # (N,3) → model expects (1,N,3) – add batch dim
            # pxyz   = pxyz.unsqueeze(0)
            knns   = batch["knns"].long().to(device)
            fname  = batch["fname"][0]

            print(f"py: {py.shape}")
            print(f"px: {px.shape}")
            print(f"pxyz: {pxyz.shape}")
            print(f"knns: {knns.shape}")
            print(f"images: {images.shape}")
            # print(f"labels: {labels.shape}")

            print(f"py dtype: {py.dtype}")
            print(f"px dtype: {px.dtype}")
            print(f"pxyz dtype: {pxyz.dtype}")
            print(f"knns dtype: {knns.dtype}")
            print(f"images dtype: {images.dtype}")
            # print(f"labels dtype: {labels.dtype}")




            predictions = model(images, px, py, pxyz, knns)
            _, predictions_argmax = torch.max(predictions, 1)
            predictions_points = predictions_argmax.cpu().numpy()
            if do_quick_vis:
                predictions_points = (predictions_points.astype(np.uint32)).flatten()
                print(f"predictions_points shape: {predictions_points.shape}")  

                unique_labels = np.unique(predictions_points)
                colors_predictions = np.zeros((len(predictions_points), 3), dtype=np.float32)
                for label in unique_labels:
                    colors_predictions[predictions_points == label] = np.random.rand(3)

                points_xyz = batch["points_xyz"].cpu().squeeze(0).numpy()
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points_xyz)
                pcd.colors = o3d.utility.Vector3dVector(colors_predictions)
                o3d.visualization.draw_geometries([pcd])

            else:
                points_xyz_ref = batch["points_xyz"].cpu().squeeze(0).numpy()
                predictions_points = (predictions_points.astype(np.uint32)).flatten()
                print(f"predictions_points shape: {predictions_points.shape}")
                out_file = os.path.join(args.output_path, f"{fname.split('.')[0]}.npy")
                out_file_points = os.path.join(point_output_path, f"{fname.split('.')[0]}.npy")
                np.save(out_file, predictions_points)
                np.save(out_file_points, points_xyz_ref)


        # comment next line if you don’t want the visor
        # o3d.visualization.draw_geometries([pcd])
# ------------------------------------------------------------
# 3.  CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Run lidar bug inference")
    parser.add_argument("--checkpoint_path", required=True, type=Path)
    parser.add_argument("--output_path", required=True, type=Path)
    parser.add_argument("--point_folder", required=True, type=Path)
    parser.add_argument("--W", default=2048, type=int)
    parser.add_argument("--H", default=64, type=int)
    parser.add_argument("--fov_up", default=2.0, type=float)
    parser.add_argument("--fov_down", default=-24.9, type=float)
    parser.add_argument("--ring_major", action='store_true')
    parser.add_argument("--normalize",action='store_true')
    parser.add_argument("--visualize", action='store_true')
    parser.add_argument("--inverted_depth", action='store_true')
    args = parser.parse_args()
    run_inference(args)