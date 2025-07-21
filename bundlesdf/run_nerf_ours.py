import os
import numpy as np
import json
from PIL import Image
from scipy.spatial.transform import Rotation as R
from nerf_runner import *

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{code_dir}/../")
from datareader import *
from bundlesdf.tool import *
import yaml, argparse


class LFDataset:
    def __init__(self, folder, is_ref=False, return_depth=False, return_segment=False):
        self.folder = folder
        self.is_ref = is_ref
        self.return_segment = return_segment
        self.camera_matrix = np.array(np.loadtxt(f"{self.folder}/camera_matrix.txt"))
        with open(f"{self.folder}/metadata.json", "r") as f:
            self.metadata = json.load(f)
        if self.is_ref:
            self.folder = f"{self.folder}/ref_views"
        self.frames = list(
            sorted([item for item in sorted(os.listdir(self.folder)) if "LF_" in item])
        )
        self.size = len(self.frames)
        self.return_depth = return_depth

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        frame_path = f"{self.folder}/{self.frames[idx]}"
        img_paths = [
            f"{frame_path}/imgs/{item}"
            for item in sorted(os.listdir(f"{frame_path}/imgs/"))
            if item.endswith(".png")
        ]
        imgs = [np.array(Image.open(img_path)) for img_path in img_paths]
        LF = np.stack(imgs, axis=0).reshape(
            self.metadata["n_views"][0], self.metadata["n_views"][1], *imgs[0].shape
        )
        LF = LF.astype(np.float32)
        LF /= LF.max()

        pose_paths = [
            f"{frame_path}/poses/{item}"
            for item in sorted(os.listdir(f"{frame_path}/poses/"))
            if item.endswith(".txt")
        ]
        poses = [np.loadtxt(pose_path) for pose_path in pose_paths]
        for i, pose in enumerate(poses):
            poses[i] = pose
        poses = np.stack(poses, axis=0).reshape(
            self.metadata["n_views"][0], self.metadata["n_views"][1], *poses[0].shape
        )
        result = (LF, poses)

        if os.path.exists(f"{frame_path}/depth/") and self.return_depth:
            depth_paths = [
                f"{frame_path}/depth/{item}"
                for item in sorted(os.listdir(f"{frame_path}/depth/"))
                if item.endswith(".npy")
            ]
            depths = [np.load(depth_path) for depth_path in depth_paths]
            depths = np.stack(depths, axis=0).reshape(
                self.metadata["n_views"][0],
                self.metadata["n_views"][1],
                *depths[0].shape,
            )
            if depths.dtype == np.uint16:
                depths = depths.astype(np.float32) / 1000.0
            result += (depths,)
        if os.path.exists(f"{frame_path}/masks/") and self.return_segment:
            mask_paths = [
                f"{frame_path}/masks/{item}"
                for item in sorted(os.listdir(f"{frame_path}/masks/"))
                if item.endswith(".png")
            ]
            masks = [np.array(Image.open(mask_path)) for mask_path in mask_paths]
            masks = np.stack(masks, axis=0).reshape(
                self.metadata["n_views"][0],
                self.metadata["n_views"][1],
                *masks[0].shape,
            )
            masks = masks.astype(np.float32)
            masks /= masks.max()
            result += (masks,)
        if self.is_ref:
            object_to_base = np.loadtxt(f"{self.folder}/obj_pose.txt")
        else:
            object_to_base = np.loadtxt(f"{frame_path}/obj_pose.txt")
        result += (object_to_base,)
        return result

    def load_all(self):
        glcam_in_cvcam = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        ).astype(float)
        rgbs = []
        depths = []
        cam_in_objs = []
        masks = []
        K = self.camera_matrix
        for i in range(self.size):
            (
                LF,
                LF_poses,
                LF_depths,
                LF_masks,
                object_to_base,
            ) = self[i]
            for s in range(LF.shape[0]):
                for t in range(LF.shape[1]):
                    rgbs.append((LF[s, t] * 255).astype(np.uint8))
                    cam_to_base = LF_poses[s, t]
                    cam_to_object = np.linalg.inv(object_to_base) @ cam_to_base
                    cam_to_object_gl = cam_to_object @ glcam_in_cvcam
                    cam_in_objs.append(cam_to_object_gl)
                    if self.return_depth:
                        depths.append(LF_depths[s, t])
                    if self.return_segment:
                        masks.append(LF_masks[s, t])
        rgbs = np.stack(rgbs, axis=0)
        depths = np.stack(depths, axis=0).astype(np.float32)
        masks = np.stack(masks, axis=0).astype(np.uint8) * 255
        cam_in_objs = np.stack(cam_in_objs, axis=0).astype(np.float32)
        result = rgbs, depths, masks, cam_in_objs, K
        return result


def run_neural_object_field(
    cfg,
    K,
    rgbs,
    depths,
    masks,
    cam_in_obs,
    save_dir,
):
    rgbs = np.asarray(rgbs)
    depths = np.asarray(depths)
    masks = np.asarray(masks)
    glcam_in_obs = np.asarray(cam_in_obs)

    cfg["save_dir"] = save_dir
    os.makedirs(save_dir, exist_ok=True)

    for i, rgb in enumerate(rgbs):
        imageio.imwrite(f"{save_dir}/rgb_{i:07d}.png", rgb)

    sc_factor, translation, pcd_real_scale, pcd_normalized = compute_scene_bounds(
        None,
        glcam_in_obs,
        K,
        use_mask=True,
        base_dir=save_dir,
        rgbs=rgbs,
        depths=depths,
        masks=masks,
        eps=cfg["dbscan_eps"],
        min_samples=cfg["dbscan_eps_min_samples"],
    )
    cfg["sc_factor"] = sc_factor
    cfg["translation"] = translation

    o3d.io.write_point_cloud(f"{save_dir}/pcd_normalized.ply", pcd_normalized)

    rgbs_, depths_, masks_, normal_maps, poses = preprocess_data(
        rgbs,
        depths,
        masks,
        normal_maps=None,
        poses=glcam_in_obs,
        sc_factor=cfg["sc_factor"],
        translation=cfg["translation"],
    )

    nerf = NerfRunner(
        cfg,
        rgbs_,
        depths_,
        masks_,
        normal_maps=None,
        poses=poses,
        K=K,
        occ_masks=None,
        build_octree_pcd=pcd_normalized,
    )
    nerf.train()

    mesh = nerf.extract_mesh(isolevel=0, voxel_size=cfg["mesh_resolution"])
    mesh = nerf.mesh_texture_from_train_images(mesh, rgbs_raw=rgbs, tex_res=1028)
    optimized_cvcam_in_obs, offset = get_optimized_poses_in_real_world(
        poses, nerf.models["pose_array"], cfg["sc_factor"], cfg["translation"]
    )
    mesh = mesh_to_real_world(
        mesh,
        pose_offset=offset,
        translation=nerf.cfg["translation"],
        sc_factor=nerf.cfg["sc_factor"],
    )
    return mesh


if __name__ == "__main__":
    with open("bundlesdf/config_ycbv.yml", "r") as ff:
        cfg = yaml.safe_load(ff)
    dataset_dir = "/home/ngoncharov/LFPose/data/parrot_rs"
    dataset = LFDataset(
        folder=dataset_dir,
        return_depth=True,
        return_segment=True,
        is_ref=True,
    )
    rgbs, depths, masks, cam_in_objs, K = dataset.load_all()
    mesh = run_neural_object_field(
        cfg,
        K,
        rgbs,
        depths,
        masks,
        cam_in_objs,
        save_dir=dataset_dir + "/mesh",
    )
    out_file = f"{dataset_dir}/model.obj"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    mesh.export(out_file)
