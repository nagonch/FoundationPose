from Utils import *
import json, uuid, joblib, os, sys, argparse
from datareader import *
from estimater import *
import numpy as np
from PIL import Image
import viser
import time
from scipy.spatial.transform import Rotation as R
from Utils import add_err, adds_err

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{code_dir}/mycpp/build")
import yaml

CODE_DIR = os.path.dirname(os.path.realpath(__file__))
DEBUG = False
DEBUG_DIR = f"{CODE_DIR}/debug"


def quat_scalar_first(q):
    return np.array([q[3], q[0], q[1], q[2]])


def convert_depth_to_pc(
    color_img, depth_map, camera_matrix, depth_thresh=0, depth_trunc=200
):
    color_img = (color_img * 255.0).astype(np.uint8)
    depth_img = depth_map.astype(np.float32)
    color_o3d, depth_o3d = o3d.geometry.Image(color_img), o3d.geometry.Image(depth_img)
    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width=color_img.shape[1],
        height=color_img.shape[0],
        fx=camera_matrix[0, 0],
        fy=camera_matrix[1, 1],
        cx=camera_matrix[0, -1],
        cy=camera_matrix[1, -1],
    )
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d,
        depth_o3d,
        depth_scale=1.0,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False,
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        intrinsics,
        project_valid_depth_only=False,
    )
    points_3D = np.array(pcd.points)
    points_3D = np.nan_to_num(
        points_3D,
    )
    point_colors = np.array(pcd.colors)
    points_mask = points_3D[:, 2] > np.nanmin(points_3D[:, 2]) + depth_thresh
    points_3D = points_3D[points_mask]
    point_colors = point_colors[points_mask]
    pcd.points = o3d.utility.Vector3dVector(points_3D)
    pcd.colors = o3d.utility.Vector3dVector(point_colors)
    return pcd


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

        return image_center, depth_center, mask_center, object_to_cam, cam_to_world


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
        img, depth_image, mask_image, gt_pose, _ = dataset[i]
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
        gt_poses.append(gt_pose)
        poses.append(pose)
    return gt_poses, poses


def vis_results(dataset, estimated_poses, frames_scale=0.05, apply_mask=True):
    server = viser.ViserServer()

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        gui_info = client.gui.add_text("Client ID", initial_value=str(client.client_id))
        gui_info.disabled = True

    camera_matrix = dataset.camera_matrix
    for i in range(len(dataset)):
        image_center, depth_center, mask, object_to_cam, cam_to_world = dataset[i]
        # 2D vis
        vis = draw_xyz_axis(
            image_center,
            ob_in_cam=estimated_poses[i],
            scale=0.1,
            K=camera_matrix,
            thickness=3,
            transparency=0,
            is_input_rgb=True,
        )
        os.makedirs(f"{DEBUG_DIR}/track_vis", exist_ok=True)
        imageio.imwrite(f"{DEBUG_DIR}/track_vis/{str(i).zfill(4)}.png", vis)

        # 3D vis
        if apply_mask:
            mask = mask.astype(bool)
            image_center = image_center * mask[:, :, None].astype(image_center.dtype)
            depth_center = depth_center * mask.astype(depth_center.dtype)
        image_center = image_center.astype(np.float32) / 255.0
        object_to_cam_est = estimated_poses[i]
        object_to_world_est = cam_to_world @ object_to_cam_est
        object_to_world_gt = cam_to_world @ object_to_cam
        pc = convert_depth_to_pc(
            image_center, depth_center, camera_matrix, depth_thresh=0.1
        )
        pc.transform(cam_to_world)
        server.scene.add_point_cloud(
            f"my_point_cloud_{i}",
            np.array(pc.points),
            np.array(pc.colors),
            point_size=1e-4,
        )
        server.scene.add_camera_frustum(
            name=f"{i}_cam",
            aspect=image_center.shape[1] / image_center.shape[0],
            scale=1e-1,
            fov=np.arctan2(
                image_center.shape[1] / 2,
                dataset.camera_matrix[0, 0],
            )
            * 2,
            line_width=0.5,
            image=image_center,
            position=cam_to_world[:3, 3],
            wxyz=quat_scalar_first(R.from_matrix(cam_to_world[:3, :3]).as_quat()),
        )
        server.scene.add_frame(
            name=f"{i}_obj_est",
            position=object_to_world_est[:3, 3],
            wxyz=quat_scalar_first(
                R.from_matrix(object_to_world_est[:3, :3]).as_quat()
            ),
            axes_length=frames_scale * 2,
            origin_radius=frames_scale / 5,
            axes_radius=frames_scale / 10,
        )
        server.scene.add_frame(
            name=f"{i}_obj_gt",
            position=object_to_world_gt[:3, 3],
            wxyz=quat_scalar_first(R.from_matrix(object_to_world_gt[:3, :3]).as_quat()),
            axes_length=frames_scale * 2,
            origin_radius=frames_scale / 5,
            axes_radius=frames_scale / 10,
        )
    try:
        while True:
            time.sleep(2.0)
    except KeyboardInterrupt:
        pass


def get_metrics(dataset, estimated_poses):
    gt_pc = dataset.mesh.vertices.copy()
    adds_vals = []
    add_vals = []
    for i in range(len(dataset)):
        _, _, _, object_to_cam, _ = dataset[i]
        estimated_pose = estimated_poses[i]
        add_val = add_err(estimated_pose, object_to_cam, gt_pc)
        adds_val = adds_err(estimated_pose, object_to_cam, gt_pc)
        add_vals.append(add_val)
        adds_vals.append(adds_val)
    return adds_vals, add_vals


if __name__ == "__main__":
    dataset_path = "/home/ngoncharov/LFPose/data/parrot_rs2"
    dataset = LFDataset(dataset_path)
    model = get_model()
    model = set_object(model, dataset.mesh)
    gt_poses, poses = infer_poses(model, dataset)
    adds_vals, add_vals = get_metrics(dataset, poses)
    vis_results(dataset, poses, apply_mask=True)
