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
import OpenEXR, Imath, numpy as np


def evenly_spaced_elements(array, k=16):
    n = len(array)
    if n <= k:
        return array
    indices = np.linspace(0, n - 1, k, dtype=int)
    return array[indices]


def exr_depth_to_meters(path):
    file = OpenEXR.InputFile(path)
    dw = file.header()["dataWindow"]
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    raw = file.channel("Z", pt)
    depth = np.frombuffer(raw, dtype=np.float32).reshape(height, width)
    result = np.copy(depth)
    result[result <= 0] = np.nan
    return result


def convert_pose(pose_dict):
    transform_rot = R.from_euler("xyz", [180, 0, 0], degrees=True).as_matrix()
    transform_4x4 = np.eye(4)
    transform_4x4[:3, :3] = transform_rot
    location = np.array(pose_dict["loc"])
    rotation = pose_dict["rot"][1:] + [pose_dict["rot"][0]]
    rotation = R.from_quat(rotation).as_matrix()
    scale = pose_dict.get("scale", 1)
    scale = np.linalg.norm(np.array([scale, scale, scale]))
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = location
    return pose @ transform_4x4, scale


def compute_intrinsic_matrix(camera_data, image_width, image_height):
    focal_length = camera_data["focal_length"]
    sensor_width = camera_data["sensor_width"]
    sensor_height = sensor_width * (image_height / image_width)

    fx = focal_length * (image_width / sensor_width)
    fy = focal_length * (image_height / sensor_height)
    cx = image_width / 2.0
    cy = image_height / 2.0

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)

    return K


def load_data(dataset_dir, downscale=10):
    glcam_in_cvcam = np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    ).astype(float)

    imgs_folder = f"{dataset_dir}/images"
    images = []
    for fname in sorted(os.listdir(imgs_folder)):
        img = np.array(Image.open(f"{imgs_folder}/{fname}"))
        images.append(img)
    images = np.stack(images, axis=0)

    cam_params_folder = f"{dataset_dir}/cam_params"
    Ks = []
    cam_poses = []
    for fname in sorted(os.listdir(cam_params_folder)):
        with open(f"{cam_params_folder}/{fname}", "r") as f:
            cam_param = json.load(f)
        Ks.append(compute_intrinsic_matrix(cam_param, img.shape[1], img.shape[0]))
        cam_poses.append(convert_pose(cam_param)[0])

    obj_poses_folder = f"{dataset_dir}/Poses"
    obj_poses = []
    for fname in sorted(os.listdir(obj_poses_folder)):
        with open(f"{obj_poses_folder}/{fname}", "r") as f:
            obj_pose = json.load(f)
            pose, scale = convert_pose(obj_pose)
        obj_poses.append(pose)

    images = np.stack(images, axis=0)
    cam_poses = np.stack(cam_poses, axis=0)
    Ks = np.stack(Ks, axis=0)
    obj_poses = np.stack(obj_poses, axis=0)
    obj_to_cam = np.linalg.inv(cam_poses) @ obj_poses
    cam_in_objs = np.linalg.inv(obj_to_cam)
    cam_in_objs = cam_in_objs @ glcam_in_cvcam

    depths_folder = f"{dataset_dir}/depth"
    depths = []
    for fname in sorted(os.listdir(depths_folder)):
        depth = exr_depth_to_meters(f"{depths_folder}/{fname}")
        depths.append(depth)

    depths = np.stack(depths, axis=0)

    depth_masks_folder = f"{dataset_dir}/depth_masks"
    depth_masks = []
    for fname in sorted(os.listdir(depth_masks_folder)):
        depth_mask = exr_depth_to_meters(f"{depth_masks_folder}/{fname}")
        depth_masks.append(depth_mask)

    depth_masks = np.stack(depth_masks, axis=0)
    depth_masks = (depth_masks < 1e8).astype(np.uint8) * 255
    cam_in_objs[:, :3, 3] /= downscale
    depths /= downscale
    return (
        evenly_spaced_elements(images),
        evenly_spaced_elements(depths),
        evenly_spaced_elements(depth_masks),
        evenly_spaced_elements(cam_in_objs).astype(np.float64),
        Ks[0],
    )


def run_neural_object_field(
    cfg,
    K,
    rgbs,
    depths,
    masks,
    cam_in_obs,
    save_dir,
):
    rgbs = np.asarray(rgbs)
    depths = np.asarray(depths)
    masks = np.asarray(masks)
    glcam_in_obs = np.asarray(cam_in_obs)

    cfg["save_dir"] = save_dir
    os.makedirs(save_dir, exist_ok=True)

    for i, rgb in enumerate(rgbs):
        imageio.imwrite(f"{save_dir}/rgb_{i:07d}.png", rgb)

    sc_factor, translation, pcd_real_scale, pcd_normalized = compute_scene_bounds(
        None,
        glcam_in_obs,
        K,
        use_mask=True,
        base_dir=save_dir,
        rgbs=rgbs,
        depths=depths,
        masks=masks,
        eps=cfg["dbscan_eps"],
        min_samples=cfg["dbscan_eps_min_samples"],
    )
    cfg["sc_factor"] = sc_factor
    cfg["translation"] = translation

    o3d.io.write_point_cloud(f"{save_dir}/pcd_normalized.ply", pcd_normalized)

    rgbs_, depths_, masks_, normal_maps, poses = preprocess_data(
        rgbs,
        depths,
        masks,
        normal_maps=None,
        poses=glcam_in_obs,
        sc_factor=cfg["sc_factor"],
        translation=cfg["translation"],
    )

    nerf = NerfRunner(
        cfg,
        rgbs_,
        depths_,
        masks_,
        normal_maps=None,
        poses=poses,
        K=K,
        occ_masks=None,
        build_octree_pcd=pcd_normalized,
    )
    nerf.train()

    mesh = nerf.extract_mesh(isolevel=0, voxel_size=cfg["mesh_resolution"])
    mesh = nerf.mesh_texture_from_train_images(mesh, rgbs_raw=rgbs, tex_res=1028)
    optimized_cvcam_in_obs, offset = get_optimized_poses_in_real_world(
        poses, nerf.models["pose_array"], cfg["sc_factor"], cfg["translation"]
    )
    mesh = mesh_to_real_world(
        mesh,
        pose_offset=offset,
        translation=nerf.cfg["translation"],
        sc_factor=nerf.cfg["sc_factor"],
    )
    return mesh


if __name__ == "__main__":
    with open("bundlesdf/config_ycbv.yml", "r") as ff:
        cfg = yaml.safe_load(ff)
    dataset_dir = "bundlesdf/data_jim/car_diffuse"
    downscale = 10
    depth_orruption = 0.2
    rgbs, depths, masks, cam_in_objs, K = load_data(dataset_dir, downscale)
    mesh = run_neural_object_field(
        cfg,
        K,
        rgbs,
        depths,
        masks,
        cam_in_objs,
        save_dir=dataset_dir + "/mesh",
    )
    out_file = f"{dataset_dir}/model.obj"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    mesh.export(out_file)
