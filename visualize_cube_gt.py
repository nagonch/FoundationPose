"""
Visualize cube_0.0 / bleach0 frame 0:
  - Full scene point cloud colored by RGB
  - Cube mesh transformed into ground-truth pose
Both are in camera frame so they should align perfectly.
"""

import sys, os
import numpy as np
import torch
import trimesh
import copy

sys.path.insert(0, os.path.dirname(__file__))

from ycbv_lf import YCBV_LF_Prod
from utilities import Visualizer, backproject_depth_to_pointcloud, clean_point_cloud

DATASET_ROOT = "/home/ngoncharov/cvpr2026/ycbv-eoat-lf/prod_dataset_new"
MESH_ROOT = f"{DATASET_ROOT}/object_meshes_reconstructed"

SEQ_PATH = f"{DATASET_ROOT}/cube_0.0/bleach0"
MESH_PATH = f"{MESH_ROOT}/cube_r0.0_dgt/model.obj"

dataset = YCBV_LF_Prod(SEQ_PATH, MESH_PATH, depth_mode="gt")
frame = dataset[0]

depth_image = np.ma.masked_equal(frame["depth_image"], 0).filled(0)
rgb_image = frame["rgb_image"]  # (H, W, 3) uint8
object_pose = frame["object_pose"]  # 4x4 float32, camera_T_object

K = dataset.camera_matrix  # cuda tensor

# --- full scene point cloud ---
depth_t = torch.tensor(depth_image, device=K.device)
points = backproject_depth_to_pointcloud(None, depth_t, K.double()).cpu().numpy()
valid = (depth_image.reshape(-1) > 0) & clean_point_cloud(points)
points = points[valid]
colors = rgb_image.reshape(-1, 3)[valid]
# --- cube mesh at GT pose ---
mesh = copy.deepcopy(dataset.mesh)
mesh.apply_transform(object_pose)

viz = Visualizer()
viz.add_point_cloud("scene", points, colors=colors, point_size=1e-3)
viz.scene.add_mesh_trimesh("cube_gt", mesh)
viz.add_frame("pose_gt", object_pose, frames_scale=0.05)

print("Viser running — open http://localhost:8080 in your browser.")
print("Ctrl-C to stop.")
viz.run()
