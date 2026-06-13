from Utils import *
import os, sys, argparse
from estimater import *
import numpy as np

from ycbv_lf import YCBV_LF_Prod, PROD_SEQUENCE_TO_OBJECT

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{code_dir}/mycpp/build")

DATASET_ROOT = "/home/ngoncharov/cvpr2026/ycbv-eoat-lf/prod_dataset_new"
MESH_ROOT = f"{DATASET_ROOT}/object_meshes_reconstructed"
OUTPUT_ROOT = "/home/ngoncharov/cvpr2026/ReLiFT-6DoF/baselines/results_fp"

REFLECTIVITIES = ["0.0", "0.5", "0.7", "1.0"]
DEPTH_MODES = ["gt", "synth"]

DEBUG = False
DEBUG_DIR = f"{code_dir}/debug"


def depth_tag(depth_mode: str) -> str:
    return "dgt" if depth_mode == "gt" else "dsim"


def mesh_path_for(object_name: str, reflectivity: str, depth_mode: str) -> str:
    return (
        f"{MESH_ROOT}/{object_name}_r{reflectivity}_{depth_tag(depth_mode)}/model.obj"
    )


def get_model(device: int):
    glctx = dr.RasterizeCudaContext()
    mesh_tmp = trimesh.primitives.Box(extents=np.ones(3), transform=np.eye(4)).to_mesh()
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
    poses = []
    gt_poses = []
    for i in range(len(dataset)):
        frame = dataset[i]
        depth_image = np.ma.masked_equal(frame["depth_image"], 0)
        K = dataset.camera_matrix.cpu().numpy().astype(np.float64)
        if i == 0:
            pose = model.register(
                K=K,
                rgb=frame["rgb_image"],
                depth=depth_image,
                ob_mask=frame["object_mask"],
                ob_id=0,
                iteration=5,
            )
        else:
            pose = model.track_one(
                rgb=frame["rgb_image"],
                depth=depth_image,
                K=K,
                iteration=5,
            )
        poses.append(pose)
        gt_poses.append(frame["object_pose"])

    # align first estimate to GT so we evaluate tracking only
    try:
        est_to_gt = np.linalg.inv(poses[0]) @ gt_poses[0]
    except np.linalg.LinAlgError:
        print(f"[warn] singular pose[0], skipping sequence")
        return None
    poses = [p @ est_to_gt for p in poses]
    return poses


def run_sequences(prefix: str, reflectivity: str, depth_mode: str, device: int):
    """Run all sequences for one (prefix, reflectivity, depth_mode) combination."""
    split_dir = f"{DATASET_ROOT}/{prefix}_{reflectivity}"
    sequences = sorted(
        s
        for s in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, s)) and s != "models"
    )

    out_dir = f"{OUTPUT_ROOT}/{depth_mode}/{prefix}_{reflectivity}"
    os.makedirs(out_dir, exist_ok=True)

    model = get_model(device)

    for seq_name in sequences:
        out_path = f"{out_dir}/{seq_name}.npy"
        if os.path.exists(out_path):
            print(f"[skip] {prefix}_{reflectivity} / {depth_mode} / {seq_name}")
            continue

        object_name = "cube" if prefix == "cube" else PROD_SEQUENCE_TO_OBJECT[seq_name]
        mesh_path = mesh_path_for(object_name, reflectivity, depth_mode)

        if not os.path.exists(mesh_path):
            print(f"[warn] mesh not found: {mesh_path}")
            continue

        seq_path = os.path.join(split_dir, seq_name)
        print(
            f"[run]  {prefix}_{reflectivity} / {depth_mode} / {seq_name}  mesh={object_name}"
        )

        dataset = YCBV_LF_Prod(seq_path, mesh_path, depth_mode=depth_mode)
        set_object(model, dataset.mesh)

        poses = infer_poses(model, dataset)
        if poses is None:
            continue
        np.save(out_path, np.array(poses, dtype=np.float32))
        print(f"[done] saved {out_path}  ({len(poses)} frames)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["cube", "objects", "all"],
        default="all",
        help="Which sequence group to run",
    )
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index")
    args = parser.parse_args()

    prefixes = {"cube": ["cube"], "objects": ["objects"], "all": ["cube", "objects"]}[
        args.mode
    ]

    for prefix in prefixes:
        for reflectivity in REFLECTIVITIES:
            for depth_mode in DEPTH_MODES:
                run_sequences(prefix, reflectivity, depth_mode, args.gpu)


if __name__ == "__main__":
    main()
