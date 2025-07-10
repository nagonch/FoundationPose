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


if __name__ == "__main__":
    dataset = LFDataset(
        folder="bundlesdf/data_ours/dynamic_dataset_orb_farther",
        return_depth=True,
        return_segment=True,
    )
    rgbs, depths, masks, cam_in_objs, K = dataset.load_all()
