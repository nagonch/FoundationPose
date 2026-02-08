import torch
import os
import json
import numpy as np
from PIL import Image
import trimesh
import copy

sequence_names = [
    "box_motion_prod",
    "car_prod",
    "car_shiny_prod",
    "jug_motion_prod",
    "jug_tilt_prod",
    "jug_translation_z_prod",
    "shiny_box_tilt_prod",
    "teabox_tilt_prod",
    "teabox_translation_prod",
]
model_names = [
    "box_ref_prod",
    "car_ref_prod",
    "car_shiny_ref_prod",
    "jug_ref_prod",
    "jug_ref_prod",
    "jug_ref_prod",
    "shiny_box_ref",
    "teabox_ref_prod",
    "teabox_ref_prod",
]

sequence_to_model = {seq: model for seq, model in zip(sequence_names, model_names)}


class LIFT_DATASET:
    def __init__(self, dataset_path, sequence_name, reference_mesh_path=None):
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

        if reference_mesh_path is not None:
            self.mesh = trimesh.load(
                f"{reference_mesh_path}/{self.model_name.strip('_ref_prod')}/model.obj"
            )
            self.gt_mesh = copy.deepcopy(self.mesh)

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
        # self.camera_matrix = self.camera_matrix

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
        depth_image = np.array(Image.open(depth_path), dtype=np.float32) / 1000.0
        object_to_world = np.loadtxt(object_pose_path)
        object_to_cam = np.linalg.inv(self.camera_poses) @ object_to_world
        return {
            "rgb_image": rgb_image,
            "object_mask": object_mask,
            "depth_image": depth_image,
            "object_pose": object_to_cam.astype(np.float32),
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
    DATASET_PATH = "/home/ngoncharov/cvpr2026/datasets/LiFT_dataset"
    SEQUENCE_NAME = "box_motion_prod"
    dataset = LIFT_DATASET(DATASET_PATH, SEQUENCE_NAME)
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


if __name__ == "__main__":
    pass
