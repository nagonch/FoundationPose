import os
import numpy as np
import json
from PIL import Image
from scipy.spatial.transform import Rotation as R


class LFDataset:
    def __init__(self, folder, return_depth=False, return_segment=False):
        self.folder = folder
        self.return_segment = return_segment
        self.camera_matrix = np.array(np.loadtxt(f"{self.folder}/camera_matrix.txt"))
        self.frames = list(
            sorted([item for item in sorted(os.listdir(self.folder)) if "LF_" in item])
        )
        self.size = len(self.frames)
        with open(f"{self.folder}/metadata.json", "r") as f:
            self.metadata = json.load(f)
        self.H_pix_to_rays, self.H_rays_to_pix = self.get_H()
        self.return_depth = return_depth

    def get_H(self):
        s_spacing = self.metadata["x_spacing"]
        t_spacing = self.metadata["y_spacing"]
        s_size = self.metadata["n_views"][0]
        t_size = self.metadata["n_views"][1]

        cam_to_image = self.camera_matrix
        image_to_cam = np.linalg.inv(cam_to_image)
        pixels_to_rays_rel = np.eye(5)
        pixels_to_rays_rel[2:, 2:] = image_to_cam
        pixels_to_rays_rel[2, 2] = image_to_cam[1, 1]
        pixels_to_rays_rel[3, 3] = image_to_cam[0, 0]
        pixels_to_rays_rel[2, -1] = image_to_cam[1, -1]
        pixels_to_rays_rel[3, -1] = image_to_cam[0, -1]
        pixels_to_rays_rel[:2, :2] = np.array([s_spacing, t_spacing]) * np.eye(2)
        pixels_to_rays_rel[0, -1] = -(s_size // 2) * s_spacing
        pixels_to_rays_rel[1, -1] = -(t_size // 2) * t_spacing

        pixels_to_rays_abs = np.copy(pixels_to_rays_rel)
        pixels_to_rays_abs[2, 0] = s_spacing
        pixels_to_rays_abs[3, 1] = t_spacing
        pixels_to_rays_abs[2, -1] -= pixels_to_rays_rel[0, -1]
        pixels_to_rays_abs[3, -1] -= pixels_to_rays_rel[1, -1]

        H_pix_to_rays = np.array(pixels_to_rays_abs)
        H_rays_to_pix = np.linalg.inv(H_pix_to_rays)
        return H_pix_to_rays, H_rays_to_pix

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
            trans = pose[:3, 3]
            rot = R.from_matrix(pose[:3, :3]).as_rotvec()
            poses[i] = np.concatenate((trans, rot))
        poses = np.stack(poses, axis=0).reshape(
            self.metadata["n_views"][0], self.metadata["n_views"][1], *poses[0].shape
        )
        result = (LF, self.H_pix_to_rays, self.H_rays_to_pix, poses)

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
            LF *= masks[..., None]
            depths *= masks
        result += (depths,)
        return result


if __name__ == "__main__":
    dataset = LFDataset(
        folder="data_ours/dynamic_dataset_orb_farther",
        return_depth=True,
        return_segment=True,
    )
    LF, H_pix_to_rays, H_rays_to_pix, poses, depths = dataset[0]
    print(LF.shape, H_pix_to_rays.shape, H_rays_to_pix.shape, poses.shape, depths.shape)
