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
import OpenEXR
import quaternion


def pose_blender_to_opencv(pose):
    transform_rot = R.from_euler("xyz", [180, 0, 0], degrees=True).as_matrix()
    transform_4x4 = np.eye(4)
    transform_4x4[:3, :3] = transform_rot
    return pose @ torch.tensor(transform_4x4, device=pose.device, dtype=pose.dtype)


def open_exr(f_name: str):
    with OpenEXR.File(f_name) as infile:
        header = infile.header()
        channels = infile.channels()
        keys = list(channels.keys())
        if keys == ["RGB"]:
            img = channels["RGB"].pixels
        elif keys == ["X", "Y", "Z"]:
            X = channels["X"].pixels
            Y = channels["Y"].pixels
            Z = channels["Z"].pixels

            height, width = Z.shape

            img = np.empty((height, width, 3))
            img[..., 0] = X
            img[..., 1] = Y
            img[..., 2] = Z

    return img, header


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


class LFSynthData:
    def __init__(self, folder, start_idx=5):
        self.folder = folder
        self.start_idx = start_idx
        self.lf_dir = os.path.join(folder, "LF_images")
        self.cam_dir = os.path.join(folder, "Cam_params")
        self.depth_dir = os.path.join(folder, "depth")
        self.target_pose_dir = os.path.join(folder, "Poses")
        self.dam_depth_dir = os.path.join(folder, "dam_depths")

        frame_dirs = os.listdir(self.lf_dir)
        frame_nums = []
        for frame_dir in frame_dirs:
            if "frame_" in frame_dir:
                frame_num = int(frame_dir[6:])
                frame_nums.append(frame_num)

        self.frame_nums = sorted(frame_nums)
        self.frames = []
        for f in self.frame_nums:
            self.frames.append("frame_" + str(f))

        self.size = len(self.frames)

        cam_data_path = os.path.join(self.cam_dir, self.frames[0]) + "/params.json"
        with open(cam_data_path, "r") as file:
            cam_data = json.load(file)

        img = np.array(Image.open(os.path.join(self.lf_dir, frame_dir, "00-00.jpg")))
        self.height, self.width, channels = img.shape
        focal_length_px = (
            cam_data["focal_length"] * self.width / cam_data["sensor_width"]
        )

        self.camera_matrix = (
            torch.tensor(
                [
                    [focal_length_px, 0, self.width / 2],
                    [0, focal_length_px, self.height / 2],
                    [0, 0, 1],
                ]
            )
            .double()
            .cpu()
            .numpy()
        )

        self.n_view_x = cam_data["n_cams_x"]
        self.n_view_y = cam_data["n_cams_y"]
        self.clip_start = cam_data["clip_start"]
        self.clip_end = cam_data["clip_end"]
        self.baseline_x = cam_data["unit_baseline_x"]
        self.baseline_y = cam_data["unit_baseline_y"]

    def __len__(self):
        return 5

    def __getitem__(self, idx):
        idx += self.start_idx
        img_dir = os.path.join(self.lf_dir, self.frames[idx])
        lf_arr = np.empty((self.n_view_y, self.n_view_x, self.height, self.width, 3))
        depth_arr = np.empty((self.n_view_y, self.n_view_x, self.height, self.width))
        masks_arr = np.empty((self.n_view_y, self.n_view_x, self.height, self.width))

        for u in range(self.n_view_y):
            for v in range(self.n_view_x):
                # Load light field images
                view_name = f"{u:02d}-{v:02d}"
                img_name = os.path.join(img_dir, view_name)
                img = np.array(Image.open(img_name + ".jpg"))
                lf_arr[u, v] = img

                # Load light field depths
                depth_name = os.path.join(
                    self.depth_dir, self.frames[idx], view_name + ".exr"
                )
                depth_map, _ = open_exr(depth_name)
                depth_arr[u, v] = depth_map[:, :, 0]
        masks = []
        mask_dir = img_dir.replace("LF_images", "sam_masks")
        for fname in list(sorted(os.listdir(mask_dir))):
            mask = Image.open(os.path.join(mask_dir, fname))
            mask = np.array(mask)
            masks.append(mask)
        masks = np.stack(masks, axis=0)
        masks_arr = masks.reshape(self.n_view_y, self.n_view_x, self.height, self.width)

        LF = torch.tensor(lf_arr) / lf_arr.max()
        depth_arr = np.where(depth_arr > self.clip_end, np.nan, depth_arr)
        depths = torch.tensor(depth_arr)

        cam_data_path = os.path.join(self.cam_dir, self.frames[idx]) + "/params.json"
        with open(cam_data_path, "r") as file:
            cam_data = json.load(file)

        cam_loc = cam_data["loc"]
        cam_q = quaternion.quaternion(*cam_data["rot"])
        cam_rot_mat = quaternion.as_rotation_matrix(cam_q)

        n_centre_x = (self.n_view_x - 1) / 2
        n_centre_y = (self.n_view_y - 1) / 2

        cam_poses = np.zeros((self.n_view_y, self.n_view_x, 4, 4))
        cam_poses[..., 0:3, 0:3] = cam_rot_mat
        cam_poses[..., 3, 3] = 1

        for u in range(self.n_view_y):
            shift_y = u - n_centre_y
            cam_shift_y = shift_y * self.baseline_y
            for v in range(self.n_view_x):
                shift_x = v - n_centre_x
                cam_shift_x = shift_x * self.baseline_x
                cam_shift = np.array([cam_shift_x, cam_shift_y, 0])

                cam_shift_world = quaternion.rotate_vectors(cam_q, cam_shift)
                cam_poses[u, v, 0:3, 3] = cam_loc + cam_shift_world

        s_mid, t_mid = LF.shape[0] // 2, LF.shape[1] // 2
        cam_poses = torch.tensor(cam_poses)
        cam_poses = pose_blender_to_opencv(cam_poses)

        with open(self.target_pose_dir + "/frame_" + str(idx) + ".json", "r") as file:
            obj_data = json.load(file)
        obj_loc = obj_data["loc"]
        obj_q = quaternion.quaternion(*obj_data["rot"])
        obj_rot_mat = quaternion.as_rotation_matrix(obj_q)
        obj_pose = np.zeros((4, 4))
        obj_pose[0:3, 0:3] = obj_rot_mat
        obj_pose[0:3, 3] = obj_loc
        obj_pose[3, 3] = 1
        obj_pose = torch.tensor(obj_pose).cuda()
        obj_pose = pose_blender_to_opencv(obj_pose)

        image_center = LF[s_mid, t_mid].cpu().numpy()
        image_center = (image_center * 255).astype(np.uint8)
        depth_center = depths[s_mid, t_mid].cpu().numpy()
        mask_center = masks_arr[s_mid, t_mid]
        cam_to_world = cam_poses[s_mid, t_mid].double().cpu().numpy()
        object_to_cam = np.linalg.inv(cam_to_world) @ obj_pose.double().cpu().numpy()
        return (
            image_center,
            depth_center,
            mask_center,
            object_to_cam,
            cam_to_world,
            None,
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


def visualize_tracking(images_rgb, object_poses, camera_matrix, save_folder):
    os.makedirs(save_folder, exist_ok=True)
    for i, (image, pose) in enumerate(zip(images_rgb, object_poses)):
        pose = pose.cpu().numpy()
        img_vis = project_frame_to_image(pose, camera_matrix, image)
        Image.fromarray(img_vis).save(f"{save_folder}/{str(i).zfill(4)}.png")


def get_metrics(dataset, mesh, estimated_poses, threshold_max=0.1):
    thresholds_space = np.linspace(0, threshold_max, 100)
    gt_pc = mesh.vertices.copy()
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
    dataset_path = "/home/ngoncharov/LFTracking/data/toy_car"
    mesh_path = "bundlesdf/data_jim/car_diffuse"
    use_dam = False
    idx_start = 5
    mesh = trimesh.load(f"{mesh_path}/model.obj")

    dataset = LFSynthData(dataset_path, start_idx=idx_start)
    camera_matrix = torch.tensor(dataset.camera_matrix).float()
    model = get_model()
    model = set_object(model, mesh)
    gt_poses, poses = infer_poses(model, dataset, use_dam)

    adds_vals, add_vals, adds_auc, add_auc = get_metrics(dataset, mesh, poses)
    # vis_results(dataset, poses, frames_scale=0.05, apply_mask=True)
    gt_poses = torch.stack([torch.tensor(p).float() for p in gt_poses])
    poses = torch.stack([torch.tensor(p).float() for p in poses])

    pose_errors = compute_pose_errors(gt_poses, poses)

    pose_errors.update({"adds_auc": float(adds_auc), "add_auc": float(add_auc)})
    with open("metrics_toy_car.yaml", "w") as file:
        yaml.dump(pose_errors, file, sort_keys=False)
    print(pose_errors)
    visualize_tracking(
        [dataset[i][0] for i in range(len(dataset))],
        poses,
        camera_matrix.cpu().numpy(),
        "results_toy_car",
    )
    visualize_tracking(
        [dataset[i][0] for i in range(len(dataset))],
        gt_poses,
        camera_matrix.cpu().numpy(),
        "results_toy_car_gt",
    )
