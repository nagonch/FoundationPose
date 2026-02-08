from Utils import *
import json, uuid, joblib, os, sys, argparse
from datareader import *
from estimater import *
import numpy as np
from PIL import Image

# import viser
import time
from scipy.spatial.transform import Rotation as R
from Utils import add_err, adds_err
from sklearn.metrics import auc
import yaml
from ycbv_lf import sequence_to_model


code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{code_dir}/mycpp/build")
import yaml

CODE_DIR = os.path.dirname(os.path.realpath(__file__))
DEBUG = False
DEBUG_DIR = f"{CODE_DIR}/debug"


class YCBV_EOAT:
    def __init__(
        self,
        dataset_path,
        sequence_name,
        reference_mesh_path=None,
        keyframes_only=False,
        key_frames_dataset_path=None,
    ):
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

        self.gt_mesh = trimesh.load(
            f"{dataset_path}/models/{self.model_name}/textured.obj"
        )
        if reference_mesh_path is not None:
            self.mesh = trimesh.load(
                f"{reference_mesh_path}/{self.model_name[4:]}/model.obj"
            )
        else:
            self.mesh = copy.deepcopy(self.gt_mesh)

        self.camera_poses = np.eye(4)
        self.camera_matrix = (
            torch.tensor(np.loadtxt(os.path.join(self.sequence_path, "cam_K.txt")))
            .cuda()
            .float()
        )

        self.depth_dir = os.path.join(self.sequence_path, "depth")
        self.depth_paths = [
            os.path.join(self.depth_dir, item)
            for item in list(sorted(os.listdir(self.depth_dir)))
        ]
        self.object_poses_dir = os.path.join(self.sequence_path, "annotated_poses")
        self.object_poses_paths = [
            os.path.join(self.object_poses_dir, item)
            for item in list(sorted(os.listdir(self.object_poses_dir)))
        ]

        self.rgb_dir = os.path.join(self.sequence_path, "rgb")
        self.rgb_paths = [
            os.path.join(self.rgb_dir, item)
            for item in list(sorted(os.listdir(self.rgb_dir)))
        ]

        self.mask_dir = os.path.join(self.sequence_path, "gt_mask")
        self.mask_paths = [
            os.path.join(self.mask_dir, item)
            for item in list(sorted(os.listdir(self.mask_dir)))
        ]

        if keyframes_only:
            self.keyframes_sequence_path = os.path.join(
                key_frames_dataset_path, sequence_name
            )
            keyframe_indices = [
                int(filename[3:])
                for filename in sorted(os.listdir(self.keyframes_sequence_path))
                if "LF_" in filename
            ]
            self.rgb_paths = [self.rgb_paths[idx] for idx in keyframe_indices]
            self.depth_paths = [self.depth_paths[idx] for idx in keyframe_indices]
            self.mask_paths = [self.mask_paths[idx] for idx in keyframe_indices]
            self.object_poses_paths = [
                self.object_poses_paths[idx] for idx in keyframe_indices
            ]

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        rgb_path = self.rgb_paths[idx]
        depth_path = self.depth_paths[idx]
        mask_path = self.mask_paths[idx]

        object_pose_path = self.object_poses_paths[idx]
        frame_id = int(rgb_path.split("/")[-1].split(".")[0])

        rgb_image = np.array(Image.open(rgb_path).convert("RGB")).astype(np.uint8)
        object_mask = (np.array(Image.open(mask_path)) * 255.0).astype(np.uint8)
        depth_image = np.array(Image.open(depth_path), dtype=np.float32) / 1000.0
        object_pose = np.loadtxt(object_pose_path)
        return {
            "rgb_image": rgb_image,
            "object_mask": object_mask,
            "depth_image": depth_image,
            "object_pose": object_pose.astype(np.float32),
            "frame_id": frame_id,
        }


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
        frame = dataset[i]
        img = frame["rgb_image"]
        depth_image = frame["depth_image"]
        mask_image = frame["object_mask"]
        gt_pose = frame["object_pose"]
        depth_image = np.ma.masked_equal(depth_image, 0)
        if i == 0:
            pose = model.register(
                K=dataset.camera_matrix.cpu().numpy().astype(np.float64),
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
                K=dataset.camera_matrix.cpu().numpy().astype(np.float64),
                iteration=5,
            )
        gt_poses.append(gt_pose)
        poses.append(pose)

    pose_est_0 = poses[0]
    pose_gt_0 = gt_poses[0]
    est_to_gt = (
        np.linalg.inv(pose_est_0) @ pose_gt_0
    )  # only evaluate tracking, without pose estiamtion
    poses = [p @ est_to_gt for p in poses]
    return gt_poses, poses


def vis_results(
    dataset, mesh_points, estimated_poses, frames_scale=0.05, apply_mask=True
):
    server = viser.ViserServer()

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        gui_info = client.gui.add_text("Client ID", initial_value=str(client.client_id))
        gui_info.disabled = True

    camera_matrix = dataset.camera_matrix.cpu().numpy()
    for i in range(len(dataset)):
        frame = dataset[i]
        image_center = frame["rgb_image"]
        depth_center = frame["depth_image"]
        mask = frame["object_mask"]
        object_to_cam = frame["object_pose"]
        cam_to_world = np.eye(4)
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
        mesh_pc = toOpen3dCloud(mesh_points)
        mesh_pc.transform(object_to_world_est)
        server.scene.add_point_cloud(
            f"my_point_cloud_{i}",
            np.array(pc.points),
            np.array(pc.colors),
            point_size=1e-4,
        )
        server.scene.add_point_cloud(
            f"mesh_pc_{i}",
            np.array(mesh_pc.points),
            colors=[1.0, 0.0, 0.0],
            point_size=1e-4,
        )
        server.scene.add_camera_frustum(
            name=f"{i}_cam",
            aspect=image_center.shape[1] / image_center.shape[0],
            scale=1e-1,
            fov=np.arctan2(
                image_center.shape[1] / 2,
                dataset.camera_matrix[0, 0].cpu().numpy(),
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


def rotation_angle_deg(R_err):
    trace = R_err.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos_theta = ((trace - 1) / 2).clamp(-1.0, 1.0)
    return torch.acos(cos_theta) * (180.0 / torch.pi)


def compute_pose_errors(gt_poses, est_poses):
    assert gt_poses.shape == est_poses.shape
    assert gt_poses.shape[-2:] == (4, 4)

    N = gt_poses.shape[0]

    # Absolute errors
    R_gt = gt_poses[:, :3, :3]
    t_gt = gt_poses[:, :3, 3]
    R_est = est_poses[:, :3, :3]
    t_est = est_poses[:, :3, 3]

    R_err_abs = R_est @ R_gt.transpose(-1, -2)  # (N, 3, 3)
    rot_err_abs = rotation_angle_deg(R_err_abs)  # (N,)
    trans_err_abs = torch.norm(t_est - t_gt, dim=1)  # (N,)
    ate_rmse = torch.sqrt((trans_err_abs**2).mean())

    # Relative errors
    rel_rot_errs = []
    rel_trans_errs = []
    for i in range(N - 1):
        T_gt_rel = torch.linalg.inv(gt_poses[i]) @ gt_poses[i + 1]
        T_est_rel = torch.linalg.inv(est_poses[i]) @ est_poses[i + 1]

        R_gt_rel = T_gt_rel[:3, :3]
        R_est_rel = T_est_rel[:3, :3]
        t_gt_rel = T_gt_rel[:3, 3]
        t_est_rel = T_est_rel[:3, 3]

        R_err_rel = R_est_rel @ R_gt_rel.transpose(-1, -2)
        rel_rot_errs.append(rotation_angle_deg(R_err_rel.unsqueeze(0)))
        rel_trans_errs.append(torch.norm(t_est_rel - t_gt_rel).unsqueeze(0))

    rel_rot_errs = torch.cat(rel_rot_errs)
    rel_trans_errs = torch.cat(rel_trans_errs)

    return {
        "mean_abs_rot_deg": rot_err_abs.mean().item(),
        "mean_abs_trans": trans_err_abs.mean().item(),
        "mean_rel_rot_deg": rel_rot_errs.mean().item(),
        "mean_rel_trans": rel_trans_errs.mean().item(),
        "ate_rmse": ate_rmse.item(),
    }


def project_frame_to_image(object_to_cam, camera_matrix, image):
    frame_3d = np.array(
        [
            [0, 0, 0, 1],
            [0.1, 0, 0, 1],
            [0, 0.1, 0, 1],
            [0, 0, 0.1, 1],
        ]
    ).T

    frame_in_cam = object_to_cam @ frame_3d

    points_3d = frame_in_cam[:3, :].T

    points_2d = (camera_matrix @ points_3d.T).T
    points_2d = points_2d[:, :2] / points_2d[:, 2:]

    img_vis = image.copy()
    origin = tuple(points_2d[0].astype(int))
    x_axis = tuple(points_2d[1].astype(int))
    y_axis = tuple(points_2d[2].astype(int))
    z_axis = tuple(points_2d[3].astype(int))

    cv2.line(img_vis, origin, x_axis, (0, 0, 255), 2)
    cv2.line(img_vis, origin, y_axis, (0, 255, 0), 2)
    cv2.line(img_vis, origin, z_axis, (255, 0, 0), 2)

    return img_vis


def visualize_tracking(dataset, object_poses, camera_matrix, save_folder):
    rgb_paths = dataset.rgb_paths
    os.makedirs(save_folder, exist_ok=True)
    camera_matrix = camera_matrix.cpu().numpy().astype(np.float64)
    for i, (frame, pose) in enumerate(zip(rgb_paths, object_poses)):
        mid_frame = np.array(Image.open(frame).convert("RGB"))
        img_vis = project_frame_to_image(pose, camera_matrix, mid_frame)
        Image.fromarray(img_vis).save(f"{save_folder}/{str(i).zfill(4)}.png")


def get_metrics(dataset, estimated_poses, threshold_max=0.1):
    thresholds_space = np.linspace(0, threshold_max, 100)
    gt_pc = dataset.gt_mesh.vertices.copy()
    adds_vals = []
    add_vals = []
    for i in range(len(dataset)):
        object_to_cam = dataset[i]["object_pose"]
        estimated_pose = estimated_poses[i]
        add_val = add_err(estimated_pose, object_to_cam, gt_pc)
        adds_val = adds_err(estimated_pose, object_to_cam, gt_pc)
        add_vals.append(add_val)
        adds_vals.append(adds_val)
    adds_vals = np.array(adds_vals)
    add_vals = np.array(add_vals)
    adds_accuracies = [(adds_vals < t).mean() for t in thresholds_space]
    add_accuracies = [(add_vals < t).mean() for t in thresholds_space]
    adds_auc = auc(np.linspace(0, 1, 100), adds_accuracies)
    add_auc = auc(np.linspace(0, 1, 100), add_accuracies)
    return adds_vals, add_vals, adds_auc, add_auc


if __name__ == "__main__":
    dataset_path = "/home/ngoncharov/cvpr2026/datasets/ycb_in_eoat"
    keyframe_dataset_path = "/home/ngoncharov/cvpr2026/ycbv-eoat-lf/dataset"
    use_keyframes = False

    save_dir = "results_ycb_in_eoat"
    if use_keyframes:
        save_dir += "_keyframes"
    folder_names = os.listdir(dataset_path)

    for folder_name in folder_names:
        if folder_name in ["models", "ref_views", "download_ycbv.sh"]:
            continue
        print("RUNNING ON SEQUENCE ", folder_name)
        sequence_name = folder_name
        out_folder = f"{save_dir}/{sequence_name}"
        try:
            os.makedirs(out_folder)
        except FileExistsError:
            print(f"Skipping {sequence_name}, already exists.")
            continue
        dataset = YCBV_EOAT(
            dataset_path,
            sequence_name,
            reference_mesh_path="bundlesdf/output",
            keyframes_only=use_keyframes,
            key_frames_dataset_path=keyframe_dataset_path,
        )
        camera_matrix = torch.tensor(dataset.camera_matrix).float()
        gt_poses = [dataset[i]["object_pose"] for i in range(len(dataset))]
        gt_poses = torch.stack([torch.tensor(p).float() for p in gt_poses])

        model = get_model()
        model = set_object(model, dataset.mesh)
        gt_poses, poses = infer_poses(model, dataset)
        adds_vals, add_vals, adds_auc, add_auc = get_metrics(dataset, poses)

        gt_poses = torch.stack([torch.tensor(p).float() for p in gt_poses])
        poses = torch.stack([torch.tensor(p).float() for p in poses])

        pose_errors = compute_pose_errors(gt_poses, poses)
        pose_errors.update({"adds_auc": float(adds_auc), "add_auc": float(add_auc)})

        with open(f"{save_dir}/{sequence_name}/metrics.yaml", "w") as file:
            yaml.dump(pose_errors, file, sort_keys=False)

        gt_poses = gt_poses.cpu().numpy()
        poses = poses.cpu().numpy()
        visualize_tracking(
            dataset,
            gt_poses,
            camera_matrix,
            f"{save_dir}/{sequence_name}/gt",
        )
        visualize_tracking(
            dataset,
            poses,
            camera_matrix,
            f"{save_dir}/{sequence_name}/est",
        )
