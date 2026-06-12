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


def load_all(object_path):
    glcam_in_cvcam = np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    ).astype(float)
    rgbs = []
    depths = []
    cam_in_objs = []
    masks = []
    K = np.loadtxt(f"{object_path}/K.txt").astype(np.float32)
    rgb_filenames = list(sorted(os.listdir(f"{object_path}/rgb")))

    for filename in range(len(rgb_filenames)):
        rgbs.append(
            np.array(Image.open(f"{object_path}/rgb/{filename:07d}.png")).astype(
                np.uint8
            )
        )
        depth = Image.open(f"{object_path}/depth/{filename:07d}.png")
        depth = np.array(depth).astype(np.float32) / 1000.0
        depths.append(depth)
        mask = Image.open(f"{object_path}/mask/{filename:07d}.png")
        mask = np.array(mask).astype(np.uint8)
        masks.append(mask)
        cam_in_obj = np.loadtxt(f"{object_path}/cam_in_ob/{filename:07d}.txt").astype(
            np.float32
        )
        cam_in_obj_gl = cam_in_obj @ glcam_in_cvcam
        cam_in_objs.append(cam_in_obj_gl)
    rgbs = np.stack(rgbs, axis=0)
    depths = np.stack(depths, axis=0).astype(np.float32)
    masks = np.stack(masks, axis=0).astype(np.uint8) * 255
    cam_in_objs = np.stack(cam_in_objs, axis=0).astype(np.float32)
    result = rgbs, depths, masks, cam_in_objs, K
    return result


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
    import os

    with open("config_ycbv.yml", "r") as ff:
        cfg = yaml.safe_load(ff)
    ref_views_path = "/home/ngoncharov/cvpr2026/ycbv-eoat-lf/dataset/ref_views"
    object_name = "tomato_soup"
    for object_name in [
        "bleach_cleanser",
        "cracker_box",
        "mustard_bottle",
        "sugar_box",
        "tomato_soup",
    ]:
        object_path = f"{ref_views_path}/{object_name}"

        rgbs, depths, masks, cam_in_objs, K = load_all(object_path)
        mesh = run_neural_object_field(
            cfg,
            K,
            rgbs,
            depths,
            masks,
            cam_in_objs,
            save_dir=f"output/{object_name}/mesh",
        )
        out_file = f"output/{object_name}/model.obj"
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        mesh.export(out_file)
