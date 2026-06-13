# Running FoundationPose on YCBV-LF

Two-stage pipeline. Stage 1 builds per-object meshes. Stage 2 runs pose estimation on every sequence.

---

## Stage 1 — Build reconstructed meshes (NeRF)

**Script:** `bundlesdf/run_reference_views_spectrack.py`
**Run from:** `bundlesdf/` (it opens `config_ycbv.yml` from cwd)

```bash
cd /home/ngoncharov/cvpr2026/FoundationPose/bundlesdf
python run_reference_views_spectrack.py
```

### Where the data comes from

Reference views are in `/home/ngoncharov/SpecTrack_dataset/reference_views/`.
Each folder is `{object}_{reflectivity}/` (e.g. `bleach_cleanser_0.5/`) and contains:

```
rgb/           ← PNG frames
depth_gt/      ← GT depth (uint16, mm → /1000 = meters)
depth_sim/     ← Synthetic/simulated depth (same format)
mask/          ← Binary masks (0/255)
camera_poses/  ← ob_in_cam .txt files, one per frame
camera_matrix.txt
```

The script loops over all 6 objects × 4 reflectivities × 2 depth types = **48 jobs**.

### What it produces

Each job writes to `bundlesdf/output/{object}_r{refl}_d{depth}/model.obj`.
The reconstructed meshes then get **symlinked or copied** to:

```
/home/ngoncharov/cvpr2026/ycbv-eoat-lf/prod_dataset_new/object_meshes_reconstructed/
  {object}_r{refl}_dgt/model.obj
  {object}_r{refl}_dsim/model.obj
```

> **Quirk:** The depth tag naming differs between scripts.
> `run_reference_views_spectrack.py` uses `d{gt,sim}` internally, but
> `run_ycbvlf_prod.py` calls `depth_tag()` which maps `"gt"→"dgt"` and `"synth"→"dsim"`.
> The folder names on disk under `object_meshes_reconstructed/` use `_dgt`/`_dsim`.
> Make sure reconstructed mesh dirs match that pattern or Stage 2 will silently skip sequences.

### Quirks and hacks

- **Coordinate flip:** poses are stored as `ob_in_cam`, so the loader inverts them to get `cam_in_obj`, then right-multiplies by `glcam_in_cvcam = diag(1,-1,-1,1)` to convert OpenCV → OpenGL convention for NeRF.
- **Depth units:** raw PNG values are millimetres; divided by 1000 on load. Don't double-convert.
- **Masks rescaled:** raw mask pixels are 0/1 (not 0/255). The loader does `* 255` after stacking to turn them into proper uint8 masks. If masks look inverted downstream that's why.
- **Idempotent:** if `output/{tag}/model.obj` already exists the job is skipped. Safe to re-run after crashes.
- **Cleanup on failure:** failed jobs delete their partial `output/{tag}/` dir automatically.

---

## Stage 2 — Run FoundationPose tracking

**Script:** `run_ycbvlf_prod.py`
**Run from:** repo root

```bash
cd /home/ngoncharov/cvpr2026/FoundationPose

# All objects + cube, all reflectivities, all depth modes, on GPU 0
python run_ycbvlf_prod.py --mode all --gpu 0

# Only the cube sequences
python run_ycbvlf_prod.py --mode cube --gpu 0

# Only the named YCB objects (no cube)
python run_ycbvlf_prod.py --mode objects --gpu 0
```

### Where the data comes from

Dataset root: `/home/ngoncharov/cvpr2026/ycbv-eoat-lf/prod_dataset_new/`

Top-level layout:
```
cube_0.0/          ← cube sequences at reflectivity 0.0
cube_0.5/ ...
objects_0.0/       ← named-object sequences at reflectivity 0.0
objects_0.5/ ...
object_meshes_reconstructed/   ← meshes from Stage 1
```

Each split dir (`{prefix}_{reflectivity}/`) contains per-sequence folders plus a `models/` dir (not used in prod path):
```
bleach0/
bleach_hard_00_03_chaitanya/
...
```

Each sequence folder:
```
camera_matrix.txt
metadata.json          ← has n_views: [5,5] (5×5 light field grid)
LF_0000/               ← one light field per timestep
  0000.png ... 0024.png   ← 25 sub-aperture views
  masks/
    0000.png ... 0024.png
LF_0138/ ...
depth/                 ← GT depth, one file per LF timestep (uint16 mm)
depth_synth/           ← Simulated depth
object_poses/          ← GT object poses (.txt, 4×4)
camera_poses/          ← camera poses (not used by YCBV_LF_Prod)
```

### How frames are picked from the light field

The 5×5 grid = 25 views. The code always uses the **center view** (`n_cameras // 2 = 12`, 0-indexed → file `0012.png`). Everything else in the LF folder is unused by FoundationPose.

### Sequence → object mapping

`PROD_SEQUENCE_TO_OBJECT` in `ycbv_lf.py` maps sequence name to object name for mesh lookup:

| Sequence | Object |
|---|---|
| bleach0, bleach_hard_00_03_chaitanya | bleach_cleanser |
| cracker_box_reorient, cracker_box_yalehand0 | cracker_box |
| mustard0, mustard_easy_00_02 | mustard_bottle |
| sugar_box1, sugar_box_yalehand0 | sugar_box |
| tomato_soup_can_yalehand0 | tomato_soup_can |

> **Hack:** `cube` sequences bypass this map entirely — if `prefix == "cube"` the object name is hardcoded to `"cube"`.

### What it produces

Results land in `/home/ngoncharov/cvpr2026/ReLiFT-6DoF/baselines/results_fp/`:
```
gt/
  objects_0.0/
    bleach0.npy          ← (N, 4, 4) float32 pose array
    ...
synth/
  objects_0.5/
    mustard0.npy
    ...
```

### Evaluation alignment hack

The first estimated pose is almost never perfectly initialized. Rather than evaluating absolute pose error, `infer_poses()` computes `est_to_gt = inv(poses[0]) @ gt_poses[0]` and applies it to **all** estimated poses. This anchors the trajectory to GT at frame 0 so evaluation only measures **tracking drift**, not initialization error.

### Quirks and hacks

- **Depth mode naming mismatch:** the dataset has folders `depth/` (GT) and `depth_synth/` (sim). The `DEPTH_MODES` list in the script uses `["gt", "synth"]`, and `YCBV_LF_Prod` maps them to the correct folder names internally. Don't confuse with Stage 1 which uses `["gt", "sim"]` (not `"synth"`).
- **Depth masking:** zero-depth pixels are masked via `np.ma.masked_equal(frame["depth_image"], 0)` before passing to FoundationPose. This matters for reflective objects where depth is missing.
- **Model is reused across sequences:** `get_model()` is called once per `(prefix, reflectivity, depth_mode)` combination, then `set_object()` swaps the mesh per sequence. The CUDA context is not re-created between sequences.
- **Idempotent:** if `{seq_name}.npy` already exists, the sequence is skipped.
- **Singular pose guard:** if the first registered pose is singular (degenerate init), the whole sequence is dropped with a warning rather than crashing.
- **`models/` dirs are skipped:** the directory listing filters out `"models"` so it doesn't try to treat the models folder as a sequence.

---

## End-to-end checklist

1. Confirm reference views exist: `ls /home/ngoncharov/SpecTrack_dataset/reference_views/`
2. Run Stage 1 from `bundlesdf/`: `python run_reference_views_spectrack.py`
3. Confirm mesh output: `ls /home/ngoncharov/cvpr2026/ycbv-eoat-lf/prod_dataset_new/object_meshes_reconstructed/ | head`
   - Folders must be named `{object}_r{refl}_dgt` or `{object}_r{refl}_dsim`
4. Run Stage 2 from repo root: `python run_ycbvlf_prod.py --mode all --gpu 0`
5. Results in `/home/ngoncharov/cvpr2026/ReLiFT-6DoF/baselines/results_fp/{gt,synth}/`
