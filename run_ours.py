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
DEBUG = True
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

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        frame_path = f"{self.folder}/{self.frames[idx]}"
        img_paths = [
            f"{frame_path}/imgs/{item}"
            for item in sorted(os.listdir(f"{frame_path}/imgs/"))
            if item.endswith(".png")
        ]
        depth_paths = [
            f"{frame_path}/depth/{item}"
            for item in sorted(os.listdir(f"{frame_path}/depth/"))
            if item.endswith(".npy")
        ]
        center_image_path = img_paths[len(img_paths) // 2]
        image_center = Image.open(center_image_path)
        depth_center = np.load(depth_paths[len(depth_paths) // 2])
        gt_pose = np.loadtxt(f"{frame_path}/obj_pose.txt")

        return image_center, depth_center, gt_pose


def infer_pose(model, img, depth_img, gt_pose, mesh):
    pass


if __name__ == "__main__":
    dataset_path = "/home/ngoncharov/LFPose/data/parrot"
    dataset = LFDataset(dataset_path)
    print(dataset.mesh)
    print(len(dataset))
    img, depth_image, gt_pose = dataset[0]
    print(img.size)
    print(depth_image.shape)
    print(gt_pose)
    # model = get_model()
