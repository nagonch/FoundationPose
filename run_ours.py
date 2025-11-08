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
from sklearn.metrics import auc
import yaml
from plenpy.lightfields import LightField


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


def robust_sigma_mad(x):
    x = np.asarray(x)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad


def get_disparity_range(disparity, sigma=1.5):
    disparity = disparity.reshape(-1)
    disp_centroid = np.median(disparity)
    disp_std = robust_sigma_mad(disparity)
    disp_min, disp_max = (
        disp_centroid - sigma * disp_std,
        disp_centroid + sigma * disp_std,
    )
    return disp_min, disp_max


def weighted_fusion(disp, conf):
    confidence = np.nanmean(conf**2, axis=-1)
    disparity = np.nanmean(disp * conf**2, axis=-1) / confidence
    confidence = np.sqrt(confidence)
    return disparity, confidence


def denoise_disparity(disparity, disp_min, disp_max, denoise_param=35):
    disparity = np.nan_to_num(disparity, nan=0.0)
    disparity = (disparity - disp_min) / (disp_max - disp_min) * 255
    disparity = disparity.astype(np.uint8)
    disparity = cv2.fastNlMeansDenoising(disparity, h=denoise_param)
    disparity = disparity / 255 * (disp_max - disp_min) + disp_min
    return disparity


def get_LF_disparity(LF, denoise_param=35, sigma=1.5):
    disparity, confidence = LightField(LF).get_disparity(
        vmin=-100, vmax=100, fusion_method="no_fusion"
    )
    disparity = np.abs(disparity)
    disparity = np.nan_to_num(disparity, nan=0.0)
    disp_min, disp_max = get_disparity_range(disparity, sigma=sigma)
    disparity = np.clip(disparity, disp_min, disp_max)
    disparity, confidence = weighted_fusion(disparity, confidence)
    disparity = denoise_disparity(
        disparity,
        disp_min,
        disp_max,
        denoise_param=denoise_param,
    )
    return disparity, confidence


def fuse_disparities(
    disparity,
    dam_disparity,
    sanity_mask,
    max_disparity=100,
):
    sanity_mask = sanity_mask & (disparity < max_disparity)
    reliable_disparities = disparity[sanity_mask].reshape(-1).float()
    corresponding_dam_disparities = dam_disparity[sanity_mask].reshape(-1).float()
    X = torch.stack(
        [corresponding_dam_disparities, torch.ones_like(corresponding_dam_disparities)],
        dim=1,
    )
    sol = torch.linalg.lstsq(X, reliable_disparities).solution
    alpha, beta = sol[0], sol[1]
    result_disparities = alpha * dam_disparity + beta
    return result_disparities


def get_frame_disparity(LF, dam_disparity, min_fit_confidence=0.9):
    LF_disparity, confidence = get_LF_disparity(
        LF,
    )
    LF_disparity, confidence = (
        torch.tensor(LF_disparity).cuda(),
        torch.tensor(confidence).cuda(),
    )
    dam_disparity = torch.tensor(dam_disparity, device=LF_disparity.device)
    sanity_mask = confidence > min_fit_confidence
    result_disparity = fuse_disparities(
        LF_disparity, dam_disparity, sanity_mask=sanity_mask
    )
    return result_disparity


class LFDataset:
    def __init__(self, folder, mesh_folder, use_dam=False):
        self.folder = folder
        self.use_dam = use_dam
        self.mesh = trimesh.load(f"{mesh_folder}/model.obj")
        self.frames = list(
            sorted([item for item in sorted(os.listdir(self.folder)) if "LF_" in item])
        )[1:]
        self.size = len(self.frames)
        self.camera_matrix = np.array(np.loadtxt(f"{self.folder}/camera_matrix.txt"))
        with open(f"{self.folder}/metadata.json", "r") as f:
            self.metadata = json.load(f)

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

        object_to_world = np.loadtxt(f"{frame_path}/object_pose.txt")
        object_to_cam = np.linalg.inv(cam_to_world) @ object_to_world
        if self.use_dam:
            LF = np.stack([np.array(Image.open(p)) for p in img_paths]).reshape(
                self.metadata["n_views"][0],
                self.metadata["n_views"][1],
                *image_center.shape,
            )
            dam_disparity = np.load(f"{frame_path}/DAM_depth.npy")
            disparity = get_frame_disparity(LF, dam_disparity).cpu().numpy()
            depth_dam = (
                self.camera_matrix[0, 0]
                * self.metadata["x_spacing"]
                / (disparity + 1e-8)
            )
        else:
            depth_dam = None

        return (
            image_center,
            depth_center,
            mask_center,
            object_to_cam,
            cam_to_world,
            depth_dam,
        )


def set_object(model, mesh):
    model.reset_object(
        model_pts=mesh.vertices.copy(),
        model_normals=mesh.vertex_normals.copy(),
        mesh=mesh,
    )
    return model


def infer_poses(model, dataset, use_dam=False):
    gt_poses = []
    poses = []
    for i in range(len(dataset)):
        img, depth_image, mask_image, gt_pose, _, dam_depth = dataset[i]
        if use_dam:
            depth_image = dam_depth
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

    pose_est_0 = poses[0]
    pose_gt_0 = gt_poses[0]
    est_to_gt = (
        np.linalg.inv(pose_est_0) @ pose_gt_0
    )  # only evaluate tracking, without pose estiamtion
    poses = [p @ est_to_gt for p in poses]
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


def visualize_tracking(dataset_path, object_poses, camera_matrix, save_folder):
    os.makedirs(save_folder, exist_ok=True)
    frames = list(sorted(os.listdir(dataset_path)))
    frames = [f for f in frames if "LF_" in f]
    camera_matrix = camera_matrix.cpu().numpy()
    for i, (frames, pose) in enumerate(zip(frames, object_poses)):
        pose = pose.cpu().numpy()
        all_frames = sorted(os.listdir(f"{dataset_path}/{frames}/imgs"))
        mid_frame_path = (
            f"{dataset_path}/{frames}/imgs/{all_frames[len(all_frames) // 2]}"
        )
        mid_frame = np.array(Image.open(mid_frame_path))
        img_vis = project_frame_to_image(pose, camera_matrix, mid_frame)
        Image.fromarray(img_vis).save(f"{save_folder}/{str(i).zfill(4)}.png")


def get_metrics(dataset, estimated_poses, threshold_max=0.1):
    thresholds_space = np.linspace(0, threshold_max, 100)
    gt_pc = dataset.mesh.vertices.copy()
    adds_vals = []
    add_vals = []
    for i in range(len(dataset)):
        _, _, _, object_to_cam, _, _ = dataset[i]
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
    dataset_path = "/home/ngoncharov/LFTracking/data/jug_motion"
    mesh_path = "/home/ngoncharov/LFTracking/data/jug_ref"
    use_dam = False

    dataset = LFDataset(dataset_path, mesh_path, use_dam)
    camera_matrix = torch.tensor(dataset.camera_matrix).float()
    model = get_model()
    model = set_object(model, dataset.mesh)
    gt_poses, poses = infer_poses(model, dataset, use_dam)

    adds_vals, add_vals, adds_auc, add_auc = get_metrics(dataset, poses)
    # vis_results(dataset, poses, frames_scale=0.05, apply_mask=True)
    gt_poses = torch.stack([torch.tensor(p).float() for p in gt_poses])
    poses = torch.stack([torch.tensor(p).float() for p in poses])

    pose_errors = compute_pose_errors(gt_poses, poses)

    pose_errors.update({"adds_auc": float(adds_auc), "add_auc": float(add_auc)})
    with open("metrics_jug.yaml", "w") as file:
        yaml.dump(pose_errors, file, sort_keys=False)
    print(pose_errors)
    visualize_tracking(dataset_path, poses, camera_matrix, "results_vis_jug")
    visualize_tracking(dataset_path, poses, camera_matrix, "results_vis_gt")
