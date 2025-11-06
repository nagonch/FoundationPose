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


def convert_pose(pose_dict):
    location = np.array(pose_dict['loc'])
    rotation = pose_dict['rot'][1:] + [pose_dict['rot'][0]]
    rotation = R.from_quat(rotation).as_matrix()
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = location
    return pose


def load_data(dataset_dir):
    imgs_folder = f'{dataset_dir}/images'
    images = []
    for fname in sorted(os.listdir(imgs_folder)):
        img = np.array(Image.open(f'{imgs_folder}/{fname}'))
        images.append(img)
    images = np.stack(images, axis=0)
    
    cam_params_folder = f'{dataset_dir}/cam_params'
    cam_params = []
    cam_poses = []
    for fname in sorted(os.listdir(cam_params_folder)):
        with open(f'{cam_params_folder}/{fname}', 'r') as f:
            cam_param = json.load(f)
        cam_params.append(cam_param)
        cam_poses.append(convert_pose(cam_param))
    print(cam_poses)
    raise
    obj_poses_folder = f'{dataset_dir}/Poses'
    obj_poses = []
    for fname in sorted(os.listdir(obj_poses_folder)):
        with open(f'{obj_poses_folder}/{fname}', 'r') as f:
            obj_pose = json.load(f)
        obj_poses.append(convert_pose(obj_pose))
    print(obj_poses)


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
    dataset_dir = 'bundlesdf/data_jim/teapot_clutter'
    rgbs, depths, masks, cam_in_objs, K = load_data(dataset_dir)
    print(rgbs)
    print(depths)
    print(masks)
    print(cam_in_objs)  
    # mesh = run_neural_object_field(
    #     cfg,
    #     K,
    #     rgbs,
    #     depths,
    #     masks,
    #     cam_in_objs,
    #     save_dir=dataset_dir + "/mesh",
    # )
    # out_file = f"{dataset_dir}/model.obj"
    # os.makedirs(os.path.dirname(out_file), exist_ok=True)
    # mesh.export(out_file)
