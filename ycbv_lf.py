import torch
import os
import json
import numpy as np
from PIL import Image

sequence_names = [
    "bleach_hard_00_03_chaitanya",
    "bleach0",
    "cracker_box_reorient",
    "cracker_box_yalehand0",
    "mustard_easy_00_02",
    "mustard0",
    "sugar_box_yalehand0",
    "sugar_box1",
    "tomato_soup_can_yalehand",
]
model_names = [
    "021_bleach_cleanser",
    "021_bleach_cleanser",
    "024_cracker_box",
    "024_cracker_box",
    "037_mustard_bottle",
    "037_mustard_bottle",
    "035_sugar_box",
    "035_sugar_box",
    "048_tomato_soup_can",
]

sequence_to_model = {seq: model for seq, model in zip(sequence_names, model_names)}


class YCBV_LF:
    def __init__(self, dataset_path, sequence_name):
        assert os.path.exists(
            dataset_path
        ), f"Dataset path {dataset_path} does not exist."
        assert os.path.exists(
            os.path.join(dataset_path, sequence_name)
        ), f"Sequence {sequence_name} does not exist in dataset path {dataset_path}"
        self.dataset_path = dataset_path
        self.sequence_name = sequence_name
        self.sequence_path = os.path.join(dataset_path, sequence_name)
        self.model_name = sequence_to_model[sequence_name]
        assert os.path.exists(
            os.path.join(dataset_path, "models", self.model_name)
        ), f"Model {self.model_name} does not exist in dataset path {dataset_path}"

        self.camera_poses_paths = [
            os.path.join(self.sequence_path, "camera_poses", item)
            for item in list(
                sorted(os.listdir(os.path.join(self.sequence_path, "camera_poses")))
            )
        ]
        with open(os.path.join(self.sequence_path, "metadata.json"), "r") as f:
            self.metadata = json.load(f)
            self.n_views = self.metadata["n_views"]
            self.baseline = self.metadata["x_spacing"]
        self.n_cameras = self.n_views[0] * self.n_views[1]
        self.camera_poses = (
            torch.stack(
                [torch.tensor(np.loadtxt(path)) for path in self.camera_poses_paths]
            )
            .cuda()
            .float()
            .reshape(*self.n_views, 4, 4)
        )
        self.camera_poses = (
            self.camera_poses.reshape(-1, 4, 4)[self.n_cameras // 2].cpu().numpy()
        )
        self.camera_matrix = (
            torch.tensor(
                np.loadtxt(os.path.join(self.sequence_path, "camera_matrix.txt"))
            )
            .cuda()
            .float()
        )

        self.depth_dir = os.path.join(self.sequence_path, "depth")
        self.depth_paths = [
            os.path.join(self.depth_dir, item)
            for item in list(sorted(os.listdir(self.depth_dir)))
        ]
        self.object_poses_dir = os.path.join(self.sequence_path, "object_poses")
        self.object_poses_paths = [
            os.path.join(self.object_poses_dir, item)
            for item in list(sorted(os.listdir(self.object_poses_dir)))
        ]
        self.lf_paths = [
            os.path.join(self.sequence_path, item)
            for item in list(sorted(os.listdir(os.path.join(self.sequence_path))))
            if "LF_" in item
        ]

    def __len__(self):
        return len(self.lf_paths)

    def __getitem__(self, idx):
        lf_path = self.lf_paths[idx]
        depth_path = self.depth_paths[idx]
        object_pose_path = self.object_poses_paths[idx]
        frame_id = int(lf_path[-4:])
        rgb_image = np.array(
            Image.open(f"{lf_path}/{self.n_cameras//2:04d}.png")
        ).astype(np.uint8)
        object_mask = np.array(
            Image.open(f"{lf_path}/masks/{self.n_cameras//2:04d}.png")
        ).astype(np.uint8)
        depth_image = np.array(Image.open(depth_path), dtype=np.uint16)
        # depth_image = depth_image.astype(np.float32) / 1000.0
        # depth_image[depth_image == 0] = np.inf
        object_pose = np.loadtxt(object_pose_path)
        return {
            "rgb_image": rgb_image,
            "object_mask": object_mask,
            "depth_image": depth_image,
            "object_pose": object_pose,
            "frame_id": frame_id,
        }


class LFReader:
    def __init__(self, dataset):
        self.dataset = dataset
        self.K = torch.clone(self.dataset.camera_matrix)
        self.id_strs = [str(i).zfill(4) for i in range(len(dataset))]
        self.colors = []
        self.depths = []
        self.masks = []
        for i in range(len(dataset)):
            frame = self.dataset[i]
            self.colors.append(frame["rgb_image"])
            self.depths.append(frame["depth_image"])
            self.masks.append(frame["object_mask"])

    def get_color(self, id):
        # idx = self.id_strs.index(id)
        return self.colors[id]

    def get_depth(self, id):
        # idx = self.id_strs.index(id)
        return self.depths[id]

    def get_mask(self, id):
        # idx = self.id_strs.index(id)
        return self.masks[id]


if __name__ == "__main__":
    DATASET_PATH = "/home/ngoncharov/cvpr2026/ycbv-eoat-lf/dataset"
    SEQUENCE_NAME = "bleach_hard_00_03_chaitanya"
    dataset = YCBV_LF(DATASET_PATH, SEQUENCE_NAME)
    from utilities import *

    visualizer = Visualizer()
    for i, frame in enumerate(dataset):
        camera_matrix = dataset.camera_matrix
        baseline = dataset.baseline
        depth = frame["depth_image"] / 1000.0
        points = (
            backproject_depth_to_pointcloud(
                None, torch.tensor(depth).cuda(), torch.tensor(camera_matrix).cuda()
            )
            .cpu()
            .numpy()
        )
        colors = frame["rgb_image"].reshape(-1, 3)
        if "masks" in frame and frame.get("masks", None) is not None:
            mask = frame["object_mask"].reshape(-1)
            points = points.reshape(-1, 3)[mask.reshape(-1) > 0]
            colors = colors.reshape(-1, 3)[mask.reshape(-1) > 0]

        object_to_base_pose = frame["object_pose"]
        visualizer.add_point_cloud(
            f"points_{i}", points, colors=colors, point_size=1e-3
        )
        visualizer.add_frame(name=f"obj_{i}", frame_T=object_to_base_pose)
    visualizer.run()
