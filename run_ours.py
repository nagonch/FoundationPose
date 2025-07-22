from Utils import *
import json, uuid, joblib, os, sys, argparse
from datareader import *
from estimater import *
import numpy as np
from PIL import Image

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{code_dir}/mycpp/build")
import yaml

CODE_DIR = os.path.dirname(os.path.realpath(__file__))
DEBUG = False
DEBUG_DIR = f"{CODE_DIR}/debug"


def get_model(device: int = 0):
    glctx = dr.RasterizeCudaContext()
    mesh_tmp = trimesh.primitives.Box(
        extents=np.ones((3)), transform=np.eye(4)
    ).to_mesh()
    est = FoundationPose(
        model_pts=mesh_tmp.vertices.copy(),
        model_normals=mesh_tmp.vertex_normals.copy(),
        symmetry_tfs=None,
        mesh=mesh_tmp,
        scorer=None,
        refiner=None,
        glctx=glctx,
        debug_dir=DEBUG_DIR,
        debug=DEBUG,
    )
    torch.cuda.set_device(device)
    est.to_device(f"cuda:{device}")
    est.glctx = dr.RasterizeCudaContext(device)
    return est


class LFDataset:
    def __init__(self, folder):
        self.folder = folder
        self.mesh = trimesh.load(f"{folder}/model.obj")
        self.frames = list(
            sorted([item for item in sorted(os.listdir(self.folder)) if "LF_" in item])
        )
        self.size = len(self.frames)
        self.camera_matrix = np.array(np.loadtxt(f"{self.folder}/camera_matrix.txt"))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        frame_path = f"{self.folder}/{self.frames[idx]}"
        img_paths = [
            f"{frame_path}/imgs/{item}"
            for item in sorted(os.listdir(f"{frame_path}/imgs/"))
            if item.endswith(".png")
        ]
        mask_paths = [
            f"{frame_path}/masks/{item}"
            for item in sorted(os.listdir(f"{frame_path}/masks/"))
            if item.endswith(".png")
        ]
        depth_paths = [
            f"{frame_path}/depth/{item}"
            for item in sorted(os.listdir(f"{frame_path}/depth/"))
            if item.endswith(".npy")
        ]
        cam_pose_paths = [
            f"{frame_path}/poses/{item}"
            for item in sorted(os.listdir(f"{frame_path}/poses/"))
            if item.endswith(".txt")
        ]
        center_image_path = img_paths[len(img_paths) // 2]
        image_center = np.array(Image.open(center_image_path))
        depth_center = np.load(depth_paths[len(depth_paths) // 2]) / 1000.0
        mask_center = np.array(Image.open(mask_paths[len(mask_paths) // 2]))
        cam_to_world = np.loadtxt(cam_pose_paths[len(cam_pose_paths) // 2])

        object_to_world = np.loadtxt(f"{frame_path}/obj_pose.txt")
        object_to_cam = np.linalg.inv(cam_to_world) @ object_to_world

        return image_center, depth_center, mask_center, object_to_cam


def set_object(model, mesh):
    model.reset_object(
        model_pts=mesh.vertices.copy(),
        model_normals=mesh.vertex_normals.copy(),
        mesh=mesh,
    )
    return model


def infer_poses(model, dataset):
    gt_poses = []
    poses = []
    for i in range(len(dataset)):
        img, depth_image, mask_image, gt_pose = dataset[i]
        depth_image = np.ma.masked_equal(depth_image, 0)
        if i == 0:
            pose = model.register(
                K=dataset.camera_matrix,
                rgb=img,
                depth=depth_image,
                ob_mask=mask_image,
                ob_id=0,
                iteration=5,
            )
        else:
            pose = model.track_one(
                rgb=img,
                depth=depth_image,
                K=dataset.camera_matrix,
                iteration=5,
            )
        print(pose[2, 3])
        print(depth_image[mask_image].mean())
        gt_poses.append(gt_pose)
        poses.append(pose)
    return gt_poses, poses


if __name__ == "__main__":
    dataset_path = "/home/ngoncharov/LFPose/data/parrot_rs2"
    dataset = LFDataset(dataset_path)
    model = get_model()
    model = set_object(model, dataset.mesh)
    gt_poses, poses = infer_poses(model, dataset)
    for i in range(len(gt_poses)):
        print(f"Frame {i}:")
        print("Ground Truth Pose:")
        print(gt_poses[i])
        print("Estimated Pose:")
        print(poses[i])
