# ExMesh in MeshSuite

ExMesh uses an indexed mesh with independent UV indices and a learned RGB texture.
Its nvdiffrast renderer, reference losses and discrete topology operations are packaged
under `msbench.vendor.exmesh`; normal operation does not import the adjacent ExMesh repository.
See that package's `UPSTREAM.md`, source notices and license for provenance.
Measured DTU scan24 and NeRF-Synthetic lego results, test evidence and limitations
are recorded in [exmesh_validation.md](exmesh_validation.md).

## Installation

Use the `msbench` environment and its matching CUDA development toolkit. Install the
optional `exmesh` extra with build isolation disabled. Open3D is pinned to 0.19.0:
0.20.0 produced empty legacy `ScalableTSDFVolume` geometry in a standalone regression probe.

```bash
conda activate msbench
export CUDA_HOME="$CONDA_PREFIX"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export TORCH_CUDA_ARCH_LIST=8.6  # RTX 3060; adjust for another GPU
# PyTorch pip wheels keep these headers under their NVIDIA packages.
export CPATH="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cusparse/include:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cublas/include:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cusolver/include${CPATH:+:$CPATH}"
MSBENCH_SKIP_CUDA=1 MAX_JOBS=2 python -m pip install -e '.[exmesh]' --no-build-isolation
```

The header paths above match the tested Python 3.12 environment; use its actual
Python site-packages directory for another Python version. `MSBENCH_SKIP_CUDA`
skips rebuilding MeshSuite's other backends; nvdiffrast still builds its own extension.

## Inputs and protocols

Training requires a coarse mesh, foreground masks (RGBA alpha or the reference
`train_mask/<image_name>_gtmask.png` layout), and, when `lambda_depth` is nonzero,
`mono_priors/da3/<image_name>.npz` containing a two-dimensional finite `depth` array.
Image names, not COLMAP numeric IDs, determine depth association.

- `configs/exmesh/dtu/scanXX.yaml` is now the default DTU paper-parameter preset:
  independent 2DGS 7,000-step initialization, TSDF resolution 256, 800×600 images,
  all views, white alpha compositing, ExMesh 8,000 steps and the paper's loss weights. All 15 scenes are
  configured. Unpublished settings use the official 2DGS DTU recommendations;
  algorithm details retain the author's code, as selected by the user. See
  [the protocol audit](exmesh_paper_protocol.md) for exact choices and results.
- `configs/exmesh/reproduction/dtu/scan24.yaml` follows the checked-out repository's
  scan24 settings: all 49 views, black compositing, full image resolution, 10,000 steps,
  `lambda_depth=0.1`, `lambda_smooth=3000`, `lambda_deviation=10000`.
- `configs/exmesh/benchmark/dtu/scan24.yaml` uses 42 training and 7 held-out views,
  white foreground compositing and half image resolution. Its PGSR initializer must
  be trained only on those 42 training views. The all-view initializer is not reused.

All ExMesh configurations currently retain the reference centered-camera projection and
its interpolated vertex-distance depth. They are explicitly recorded in checkpoints.
The DTU `cameras.npz` principal point differs from the centered projection. A corrected
camera model would be a separate experiment requiring an explicit choice.

Prepare the held-out benchmark initializer independently:

```bash
python tools/prepare_exmesh_pgsr.py --pgsr-repo ../ExMesh/PGSR \
  --dataset data/DTU/scan24 --output outputs/exmesh-preprocess/scan24-benchmark --holdout 8
```

This tool stages COLMAP inputs inside the output directory. PGSR's generated sparse
PLY and source backups therefore do not modify the shared dataset. It requires the
independently installed PGSR rasterizer/KNN and PyTorch3D transforms. The core ExMesh
runtime only needs the resulting mesh, not PGSR or PyTorch3D.

The paper and checked-out repository differ in initialization and several schedules
and weights. The default DTU configuration uses the paper's initialization, total
steps and weights. Historical repository comparisons retain their 5,000-step PGSR
initializers. Unpublished settings and retained code differences are recorded in
[the protocol audit](exmesh_paper_protocol.md), so parameter alignment alone does
not establish exact reproduction of every published experiment.

## CLI

The standard DTU entry point automatically initializes the mesh, trains ExMesh,
renders images, evaluates image/DTU geometry metrics and exports a video:

```bash
conda activate msbench
./single_train.sh exmesh dtu/scan24 0
./single_train.sh exmesh dtu/all 0
```

The final argument selects the GPU. Existing completed training is skipped;
`--retrain` removes the scene's ExMesh output and rebuilds it. The independently
cached initializer is reused only when its settings, source revision and input
fingerprint match. Paper outputs live in `outputs/exmesh/dtu/scanXX`, distinct from
the historical repository/held-out experiments. Initializer logs and provenance
live in `outputs/exmesh-preprocess/paper-white/dtu/scanXX`.

RGBA alpha is retained as the silhouette mask. RGB targets use
`alpha * RGB + (1 - alpha) * white` throughout training and image evaluation;
rendering uses the checkpoint's white background. Official 2DGS's COLMAP reader
does not composite alpha into RGB, so the initializer stages white-composited RGB
while preserving alpha for TSDF masking and passes `--white_background`.
The earlier black-background paper-parameter run is archived under
`outputs/exmesh/history/paper-black/dtu/scan24`; its initialization remains under
`outputs/exmesh-preprocess/paper/dtu/scan24`. Its CD is not the corrected preset's result.

The 2DGS initializer additionally requires the official `diff_surfel_rasterization`
extension; it is installed in the tested msbench environment. For a fresh setup,
check out the revision recorded in `configs/base/exmesh_paper_dtu.yaml`, initialize
its git submodules, and install `submodules/diff-surfel-rasterization` using the
same CUDA/header environment as above. Direct ExMesh rendering only needs its
native nvdiffrast/xatlas dependencies.

DTU `full` controls evaluation scope; an explicit white background remains white
in the adapter as well. It no longer unconditionally overrides the renderer to black.

Explicit CLI commands for the historical repository-parameter comparison:

```bash
msbench train --config configs/exmesh/reproduction/dtu/scan24.yaml
msbench render images --config configs/exmesh/reproduction/dtu/scan24.yaml
msbench eval images --config configs/exmesh/reproduction/dtu/scan24.yaml
msbench eval mesh --config configs/exmesh/reproduction/dtu/scan24.yaml
msbench inspect --config configs/exmesh/reproduction/dtu/scan24.yaml
msbench profile --config configs/exmesh/reproduction/dtu/scan24.yaml
msbench export textured-mesh --config configs/exmesh/reproduction/dtu/scan24.yaml
```

Use the benchmark config for held-out metrics. Image metrics on the reproduction
config's `all` split are fitting metrics, not held-out novel-view metrics.
DTU geometry uses masks, the scene scale matrix, ObsMask, Plane and the official
point cloud; inspect the output `protocol` rather than assuming a Chamfer fallback
is the same protocol.

`--override trainer.resume=true` resumes a completed iteration from a numbered
checkpoint. A resume checkpoint contains topology/UV/texture, optimizer state,
geometry references, per-face/vertex statistics, RNG state and camera-pool order.
Saving is deferred until topology and optimizer updates for the iteration finish.
The original final-iteration optimizer-step convention is retained.
`ckpt/final.pt` is an inference snapshot without optimizer/RNG state; numbered
checkpoints retain the complete resumable state. Reported checkpoint size therefore
refers to the learned model, not its Adam momentum buffers.

The reference `max_shapes` parameter is a split-selection target, not a strict hard
face limit: paired edge splits can overshoot it. Actual face counts are recorded in
`topology.jsonl` and statistics. A hard-cap variant would require a separate algorithm choice.

`ExMeshAdapter.export_textured_mesh(path)` writes OBJ/MTL/PNG while preserving
independent geometry and UV indices. `to_primitive()` exposes shared-vertex geometry
for MeshSuite statistics and PLY export; it does not encode the UV texture as SH.

## Validation

```bash
python -m pytest -q msbench/tests/test_exmesh.py
PYTHONPATH=. python tools/validate_exmesh_parity.py \
  --reference ../ExMesh --output outputs/exmesh-validation/parity.json
```

The reference path is used only by the opt-in comparison tool. The CUDA tests exercise
real rasterization, nonzero geometry/texture gradients, UV topology and optimizer binding.
The parity report also measures repeated-reference floating-point reduction noise.
Run these checks with a visible CUDA device; a skipped CUDA test is not validation.

## NeRF-Synthetic lego

The user selected DA3 predicted depth, independent PGSR initialization at 5,000 steps,
and the paper's 8,000-step ExMesh schedule with depth/smoothness/offset weights
0.01/1000/1000. Unspecified learning-rate/topology options retain the reference code's
defaults. `configs/exmesh/benchmark/nerf_synthetic/lego.yaml` records this choice.
It uses all 100 training images, the standard 200-image test split, white backgrounds
and 800×800 images. PGSR remains the selected initializer rather than the paper's 2DGS;
this distinction must accompany any comparison with published numbers.

```bash
python tools/prepare_exmesh_synthetic.py --dataset data/nerf_synthetic/lego \
  --output outputs/exmesh-preprocess/lego
python tools/prepare_exmesh_pgsr.py --pgsr-repo ../ExMesh/PGSR \
  --dataset outputs/exmesh-preprocess/lego/colmap-input \
  --output outputs/exmesh-preprocess/lego/pgsr --holdout 0 --white-background
python tools/prepare_exmesh_synthetic.py --dataset data/nerf_synthetic/lego \
  --output outputs/exmesh-preprocess/lego --depth --batch-size 4 --disable-autocast-cache \
  --model /path/to/DA3NESTED-GIANT-LARGE
msbench train --config configs/exmesh/benchmark/nerf_synthetic/lego.yaml
msbench render images --config configs/exmesh/benchmark/nerf_synthetic/lego.yaml
msbench eval images --config configs/exmesh/benchmark/nerf_synthetic/lego.yaml
```

The synthetic preparation tool converts only training cameras to COLMAP/OpenCV and
composites their RGB on white. This bypasses the reference PGSR Blender reader's
outdated `CameraInfo` construction without changing PGSR optimization. The staged
camera transforms are tested against MeshSuite's synthetic loader. The source
`points3d.ply` is an initialization cloud, **not geometry ground truth**. No synthetic
Chamfer score can be reported from it.

DA3 is a separate preprocessing dependency, not an ExMesh runtime dependency. The
checked official source is commit `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`;
the downloaded nested giant/large model revision is
`8615eefb62f2db4f8d6ebaa59160086981672829`. Dependencies are staged under
`outputs/exmesh-preprocess/da3-deps` to preserve the tested msbench PyTorch/Open3D
installation. Add that directory to `PYTHONPATH` when using this local installation.
The depth tool conditions DA3 on known training poses/intrinsics at processing
resolution 504, matching the reference API settings, and writes finite per-image
NPZ priors under the output directory. It does not read test images or overwrite
shared datasets. All-view inference is the default; a change to view batching after
a memory failure requires recording the chosen batch size, because it changes
multi-view context. The user selected batches of four after all-100-view inference
failed with CUDA OOM on the 12GB RTX 3060; the original failure log is retained.
Four-view inference also needed the autocast weight cache disabled and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. These affect memory allocation,
preserving the API's BF16/FP16 computation, model weights, resolution and view groups.

The reference repository's README describes DA3 predictions, and the
[paper's sections 3.5/7.2](https://arxiv.org/html/2606.07288#S3.SS5) use a Pearson
depth consistency loss. Ground-truth depth is not required. Disabling this term
(`lambda_depth=0`) is a separate ablation, not the selected full-loss baseline.
