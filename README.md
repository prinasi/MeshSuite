<!-- @format -->

# MeshSuite: A Unified Benchmark for Triangle- and Mesh-Based Neural Rendering

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2609.01306-b31b1b.svg)](https://arxiv.org/abs/2609.01306)
[![Project Page](https://img.shields.io/badge/Project-Page-2563eb.svg)](https://x-up-lab.github.io/research/meshsuite)

**MeshSuite** is a unified, Python-first benchmarking toolkit and evaluation framework for comparing triangle-splatting-style radiance field reconstruction methods, bridging academic novel view synthesis research and real-time industrial game engine deployment (Unity). Visit the [Project Page](https://x-up-lab.github.io/research/meshsuite) for interactive comparisons and full leaderboards.

## Overview

MeshSuite provides a single, consistent evaluation framework for comparing different triangle-based radiance field reconstruction methods. Instead of wrangling per-repo scripts and ad-hoc metrics, you get:

- **Standardized metrics** -- PSNR, SSIM, LPIPS, and geometry quality, computed the same way for every method.
- **Adapter pattern** -- plug any triangle-splatting method into the same API.
- **Reproducible experiments** -- YAML configs lock down every hyperparameter.
- **CLI-first workflow** -- inspect, profile, evaluate, and compare from the terminal.
- **Industrial Engine Parity** -- export method-preserving `.triasset` packages and evaluate real-time framerates and rendering fidelity natively in Unity.

## Supported Methods

| Method                       | Adapter                    | Status                                   |
| ---------------------------- | -------------------------- | ---------------------------------------- |
| 2D Triangle Splatting (2DTS) | `D2TSAdapter`              | Config/train/render/eval/export support  |
| Triangle Splatting           | `TriangleSplattingAdapter` | Config/train/render/eval/profile/export support |
| MeshSplatting                | `MeshSplattingAdapter`     | Config/train/render/eval/profile/export support |
| DiffSoup                     | `DiffSoupAdapter`          | Config/train/render/eval/export support; In-shader Micro-MLP support |
| ExMesh                       | `ExMeshAdapter`            | Native training/render/eval/profile/export; DTU and NeRF-Synthetic configs; beyond the paper |

### ExMesh

ExMesh goes beyond the MeshSuite paper (arXiv v5), which does not evaluate it; its
results are not part of the paper's tables.

ExMesh uses the same dataset configs as every other method (e.g. DTU from
`configs/base/dtu.yaml`: foreground evaluation on white, 1/2 resolution, all views
used for training), so its results are directly comparable:

```bash
bash single_train.sh exmesh dtu/all 0
bash single_train.sh exmesh nerf_synthetic/all 0
```

ExMesh is benchmarked only on the object-centric datasets (DTU and NeRF-Synthetic), as
in its paper. It refines one bounded, textured mesh from a coarse initialization, using
object masks and a fixed face budget, so it does not apply to unbounded Mip-NeRF 360 and
Tanks & Temples scenes; MeshSuite ships no ExMesh configs for them.

On NeRF-Synthetic its method settings follow the ExMesh paper: 8k iterations and a
coarse initial mesh from 2DGS (7k iterations, TSDF at 256³) built on the training
split, as the authors confirmed; the pinned 2DGS source is cloned to
`outputs/exmesh-preprocess/2dgs_source` on first use, and its
`diff_surfel_rasterization` extension must be installed. NeRF-Synthetic is trained
without depth supervision, as in the paper.

On DTU the paper's setup reaches only mean CD 0.98 on the shared protocol (paper:
0.58), so MeshSuite uses the best-performing settings it found instead
(`configs/base/exmesh-dtu.yaml`): the PGSR initializer bundled with the official
ExMesh repo, extracted with its depth filter at TSDF voxel 0.003 and decimated to 30k
faces; the repo's 10k iterations; and a 16x offset (`lambda_deviation`) weight.
Per-scene `lambda_smooth`/`lambda_depth` and the LR schedule come from the official
`scripts/run_dtu.py`. The pinned ExMesh source is cloned to
`outputs/exmesh-preprocess/exmesh_source` on first use, and PGSR's
`diff_plane_rasterization` extension must be installed. ExMesh on DTU also reads
monocular depth priors from `data/dtu/<scan>/mono_priors/da3/`.

The initialization is built automatically before training, on the same views as
ExMesh training, and cached under `outputs/exmesh-preprocess/`. ExMesh needs the
`exmesh` extra (`pip install -e ".[exmesh]"`).

## Installation

### Option 1: Conda Environment (Recommended)

```bash
git clone https://github.com/prinasi/MeshSuite.git
cd MeshSuite
conda env create -f environment.yml
conda activate msbench

# Use this environment's CUDA development toolkit, matching PyTorch cu118.
export CUDA_HOME="$CONDA_PREFIX"
export CUDACXX="$CUDA_HOME/bin/nvcc"

# Build for the visible GPU (e.g. RTX 3060: 8.6 / 86).
export TORCH_CUDA_ARCH_LIST="$(python -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability())')"
export MSBENCH_CUDA_ARCHS="${TORCH_CUDA_ARCH_LIST/./}"
python -m pip install -e ".[dev,cuda]" --no-build-isolation
```

### Option 2: Pip Install / Editable Development Install

If you already have a Conda or Python environment with PyTorch (>= 2.0 with CUDA support), install MeshSuite in editable mode:

```bash
git clone https://github.com/prinasi/MeshSuite.git
cd MeshSuite
```

#### 1. Full Installation (Compiling All Bundled CUDA Extensions)
To build all bundled CUDA rendering backends (Triangle Splatting, MeshSplatting, 2DTS, Simple-KNN, and DiffSoup):

```bash
# For PyTorch cu118, install the CUDA 11.8 compiler and development headers.
# The runtime-only `cudatoolkit` package does not provide nvcc.
conda install -c nvidia/label/cuda-11.8.0 cuda-nvcc=11.8 cuda-cudart-dev=11.8 cuda-cccl=11.8 cuda-driver-dev=11.8 cuda-nvtx=11.8
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export CUDACXX="$CUDA_HOME/bin/nvcc"

export TORCH_CUDA_ARCH_LIST="$(python -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability())')"
export MSBENCH_CUDA_ARCHS="${TORCH_CUDA_ARCH_LIST/./}"

# --no-build-isolation requires build dependencies to be installed first.
python -m pip install "setuptools>=68" wheel "nanobind>=2.11.0" "cmake>=3.20" ninja
python -m pip install -r requirements.txt

python -m pip install -e ".[dev,cuda]" --no-build-isolation
```

> **Tip for Multi-CUDA Systems**:
> Match the development toolkit to `python -c 'import torch; print(torch.version.cuda)'`; “11.8 or newer” alone is insufficient for PyTorch cu118. Set both `CUDA_HOME` and `CUDACXX` so PyTorch and CMake (used by DiffSoup) use the same compiler. For a system toolkit, replace `CUDA_HOME` with its actual installation directory. Without a visible GPU, set the architecture variables manually (e.g. `8.6` / `86` for RTX 3060, `8.9` / `89` for RTX 4090).
>
> To retain these settings across Conda activations, run `conda env config vars set CUDA_HOME="$CUDA_HOME" CUDACXX="$CUDACXX" TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" MSBENCH_CUDA_ARCHS="$MSBENCH_CUDA_ARCHS"`, then reactivate the environment.

#### 2. Fast Python-Only Install (Skip CUDA Compilation)
If you only need to work on Python code, CLI tools, evaluation scripts, or tests without compiling C++/CUDA extensions:

```bash
MSBENCH_SKIP_CUDA=1 pip install -e ".[dev]" --no-build-isolation
```

#### 3. Selective Backend Compilation
If you only require specific CUDA backends (e.g., skipping DiffSoup CMake compilation or only building Triangle Splatting):

```bash
# Compile only triangle-splatting, mesh-splatting, 2dts, and simple-knn (skip DiffSoup)
MSBENCH_CUDA_BACKENDS=triangle-splatting,mesh-splatting,2dts,simple-knn pip install -e ".[dev]" --no-build-isolation

# Or compile only Triangle Splatting and its KNN dependency
MSBENCH_CUDA_BACKENDS=triangle-splatting,simple-knn pip install -e ".[dev]" --no-build-isolation
```

Supported backend identifiers: `triangle-splatting`, `mesh-splatting`, `2dts`, `simple-knn`, `diffsoup`.

#### 4. Additional Build Environment Variables
| Environment Variable | Description | Default |
| ------------------- | ----------- | ------- |
| `MSBENCH_SKIP_CUDA` | Skip compiling all CUDA extensions (`1`, `true`, `yes`) | Unset |
| `MSBENCH_CUDA_BACKENDS` | Comma-separated list of backends to build | All |
| `MSBENCH_CUDA_ARCHS` | Semicolon-separated CUDA architectures for CMake (DiffSoup) | `89` |
| `MSBENCH_CCACHE` | Enable `ccache` compilation caching for NVCC (`1`, `true`, `yes`) | Unset |
| `MAX_BUILD_JOBS` | Maximum parallel compilation jobs | Auto |

## Dataset Setup

MeshSuite evaluates methods on standard novel view synthesis benchmarks: **Mip-NeRF 360**, **NeRF-Synthetic (Blender)**, **Tanks & Temples**, and **DTU**.

Readers should download the datasets from their official sources and place or symlink them into the `data/` directory according to the default format:

```bash
mkdir -p data
ln -s /path/to/MipNeRF360 data/mipnerf360
ln -s /path/to/nerf_synthetic data/nerf_synthetic
ln -s /path/to/tandt data/tandt
ln -s /path/to/DTU data/dtu
```

Each scene directory is expected to look like this:

```text
data/
├── mipnerf360/<scene>/       images/ (images_2, images_4, ...), sparse/0/
├── nerf_synthetic/<scene>/   train/, test/, transforms_{train,test}.json
├── tandt/<scene>/            images/, sparse/0/
└── dtu/
    ├── ObsMask/, Points/stl/ # official DTU evaluation data (Chamfer distance)
    └── scan<id>/             images/, mask/, sparse/0/, cameras.npz
                              # ExMesh also needs mono_priors/da3/
```

The 15 DTU evaluation scans are 24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110,
114, 118, and 122. Config paths are lowercase (`data/dtu`), so the symlink name
must match on case-sensitive file systems.

DTU is evaluated as a mesh benchmark. Every method trains on all views of each scan,
as in the methods' papers, and only DTU mesh metrics (Chamfer distance) are reported;
`single_train.sh` skips image rendering and NVS metrics on DTU.

## Quick Start

```bash
# Train, render, evaluate, inspect, and profile from one experiment config
msbench train --config configs/triangle-splatting/mipnerf360/bicycle.yaml
msbench render images --config configs/triangle-splatting/mipnerf360/bicycle.yaml
msbench eval images --config configs/triangle-splatting/mipnerf360/bicycle.yaml
msbench inspect --config configs/triangle-splatting/mipnerf360/bicycle.yaml
msbench profile --config configs/triangle-splatting/mipnerf360/bicycle.yaml

# Config values can still be overridden from the CLI
msbench render video --config configs/triangle-splatting/mipnerf360/bicycle.yaml --output-dir outputs/demo-video

# Train complete datasets with the scene pipeline helper
bash single_train.sh triangle-splatting mipnerf360/all 0
bash single_train.sh triangle-splatting tandt/all 0
bash single_train.sh triangle-splatting dtu/all 0
bash single_train.sh mesh-splatting mipnerf360/all 0
bash single_train.sh mesh-splatting tandt/all 0
bash single_train.sh mesh-splatting dtu/all 0
bash single_train.sh 2dts mipnerf360/all 0
bash single_train.sh 2dts tandt/all 0
bash single_train.sh 2dts dtu/all 0

# Export feature-preserving Unity-native assets from completed runs
bash single_export_unity.sh triangle-splatting mipnerf360/all 0
bash single_export_unity.sh mesh-splatting mipnerf360/garden 0

# Export missing Unity assets, capture Unity frames, profile FPS/memory, and format report tables
bash single_unity_eval.sh 2dts mipnerf360/bicycle 0 --unity "$UNITY" --unity-project unity
```

## Profiling

`msbench profile` measures the rendering performance of a trained checkpoint
and writes a versioned JSON report. It reuses the same config resolution as
train/render/eval, so a completed run can be profiled with the same config:

```bash
msbench profile --config configs/triangle-splatting/mipnerf360/bicycle.yaml
```

The defaults in `configs/base/_base_.yaml` write the report to
`outputs/{method}/{dataset}/{scene}/profile.json` and measure the `test` split
with 100 timed repeats after 5 untimed warmup renders.

### Measured metrics

| Metric | Top-level key | Notes |
| ------ | ------------- | ----- |
| Forward latency | `forward_latency_ms` | Mean of the timed eval renders |
| FPS | `fps` | `1000 / forward_latency_ms` |
| Peak CUDA memory | `peak_cuda_memory_gb` | Peak device allocation during one eval render |
| Primitive count | `primitive_count` | Triangles/faces/primitives in the checkpoint |
| Checkpoint size | `checkpoint_size_mb` | On-disk state-dict size |
| Backward latency | `backward_latency_ms` | Mean train-mode backward time; opt-in |

The `forward` object (and `backward`, when enabled) carries the full timing
summary (`mean_ms`, `std_ms`, `p50_ms`, `p95_ms`, `min_ms`, `max_ms`, `fps`),
and `memory` reports both allocated and reserved peaks. Timings use CUDA events
when the adapter runs on a CUDA device and wall-clock time otherwise. The
report shape is versioned under `schema_version`.

### Options and config keys

Explicit command-line flags override the config; otherwise the `profile:`
block from `--config` is used:

```yaml
profile:
  split: test
  repeats: 100
  warmup: 5
  backward: false
  output: outputs/{method}/{dataset}/{scene}/profile.json
```

| Flag | Description |
| ---- | ----------- |
| `--method/-m`, `--checkpoint/-c`, `--dataset/-d` | Override adapter method, checkpoint path, or dataset root |
| `--scene` | Scene name; resolves `{scene}` path templates |
| `--split/-s` | Dataset split (train/test/all) |
| `--repeats/-r` | Number of timed repetitions |
| `--warmup/-w` | Number of untimed warmup renders |
| `--backward/--no-backward` | Also measure backward-pass latency |
| `--output/-o` | JSON report path |

Fast smoke run on the same config:

```bash
msbench profile \
  --config configs/triangle-splatting/mipnerf360/bicycle.yaml \
  --repeats 3 \
  --warmup 1 \
  --backward \
  --output /tmp/bicycle-profile.json
```

### Example report

Triangle Splatting `bicycle` (4.89 M triangles, RTX 4090):

```json
{
  "schema_version": 1,
  "method": "triangle-splatting",
  "dataset": "data/mipnerf360/bicycle",
  "split": "test",
  "device": "cuda:0",
  "camera_name": "_DSC8679",
  "repeats": 100,
  "warmup": 5,
  "status": "complete",
  "forward_latency_ms": 14.847,
  "fps": 67.35,
  "peak_cuda_memory_gb": 3.399,
  "primitive_count": 4888414,
  "checkpoint_size_mb": 1100.22,
  "forward": {
    "mean_ms": 14.847,
    "std_ms": 0.151,
    "p50_ms": 14.850,
    "p95_ms": 15.037,
    "min_ms": 14.557,
    "max_ms": 15.632,
    "fps": 67.35
  },
  "memory": {
    "peak_memory_allocated_gb": 3.399,
    "peak_memory_reserved_gb": 3.697
  }
}
```

Running with `--backward` adds a `backward` summary with the same shape plus
`backward_latency_ms` (about 27.7 ms for this scene).

### CUDA extension rebuilds

Native rasterization requires the bundled CUDA extensions. Rebuild them
whenever the active PyTorch/CUDA environment changes; an ABI mismatch surfaces
as an extension import error when the adapter loads:

```bash
conda activate msbench
MSBENCH_BUILD_CUDA=1 \
  MSBENCH_CUDA_BACKENDS=triangle-splatting,simple-knn \
  python setup.py build_ext --inplace --force
```

Set `MSBENCH_CUDA_BACKENDS` to the backends you need and `MSBENCH_CUDA_ARCHS`
to your GPU architecture (default `89`); set `MSBENCH_SKIP_CUDA=1` to skip
extension builds entirely.

## Unity-native Evaluation

MeshSuite can export method-preserving Unity `.triasset` packages, render Unity
test-view images for PSNR/SSIM/LPIPS, and profile GPU FPS/memory through a
Linux standalone Development Player. The workflow is designed for completed
config-driven runs under `outputs/{method}/{dataset}/{scene}`.

### Unity project setup

Install Unity with Linux standalone build support. 
Set the usual path:

```bash
export UNITY=/path/to/Unity/Editor/Unity
```

`single_unity_eval.sh` applies the Vulkan compatibility patch automatically on
Linux. It also builds or reuses a Linux standalone Development Player at:

```text
outputs/unity_player/linux/MeshSuiteProfilePlayer.x86_64
```

### Export Unity assets only

```bash
./single_export_unity.sh 2dts mipnerf360/all cpu
./single_export_unity.sh triangle-splatting mipnerf360/garden cpu
```

Each scene writes:

```text
outputs/{method}/{dataset}/{scene}/unity_native/{method}.triasset/
```

Existing valid packages are skipped; use `--force` to re-export.

### Method-aware Unity benchmark

```bash
./single_unity_eval.sh 2dts mipnerf360/all 0 \
  --unity "$UNITY" \
  --unity-project unity
```

For all methods:

```bash
for method in 2dts triangle-splatting mesh-splatting diffsoup; do
  ./single_unity_eval.sh "$method" mipnerf360/all 0 \
    --unity "$UNITY" \
    --unity-project unity
done
```

The default condition is `method-aware`; outputs are written under each scene's
`unity_method_aware/` directory and summarized in:

```text
outputs/unity_reports/{method}/unity_method_aware/unity_metrics_report.csv
outputs/unity_reports/{method}/unity_method_aware/unity_metrics_report.json
outputs/unity_reports/{method}/unity_method_aware/unity_metrics_report.md
```

The terminal summary reports:

```text
PSNR | SSIM | LPIPS | GPU FPS | GPU P50 ms | GPU P95 ms | Render MiB | Asset MiB
```

`GPU FPS` is computed as `1000 / GPU P50 ms`. CPU/engine FPS is deliberately
not used for paper tables.

### General-purpose Unity renderer

Use `--general-purpose` and a separate output name:

```bash
for method in 2dts triangle-splatting mesh-splatting diffsoup; do
  ./single_unity_eval.sh "$method" mipnerf360/all 0 \
    --unity "$UNITY" \
    --unity-project unity \
    --general-purpose \
    --output-name unity_general_purpose
done
```

Methods whose exported manifest does not declare a comparable general-purpose
appearance are rejected instead of producing misleading numbers.

### MeshSplatting topology ablation

To isolate the value of MeshSplatting's shared-vertex topology, run
`single_unity_eval.sh` for `mesh-splatting` with a soup deployment layout
(command below):

- `materialized-soup`: exports a separate `.triasset` with per-triangle
  duplicated positions/SH/opacity attributes and sequential soup indices. This
  is intended for asset-memory, load-feasibility, and CG compatibility evidence.

Reports are written under:

```text
outputs/unity_reports/mesh-splatting/unity_mesh_topology/materialized_soup/
```

For true exported soup assets, use a separate asset subdirectory:

```bash
./single_unity_eval.sh mesh-splatting mipnerf360/bicycle 0 \
  --unity "$UNITY" \
  --unity-project unity \
  --asset-subdir unity_native_materialized_soup \
  --export-topology soup \
  --output-name unity_method_aware_materialized_soup
```

### Runtime profile details

By default, GPU speed is measured with the standalone Player:

```text
--profile-runtime player
--profile-runs 3
--profile-views 3
--profile-warmup 60
--profile-frames 180
--gpu-timing-min-fraction 0.5
```

This means each scene profiles three independent Player runs, three held-out
views per run, and 180 timed frames per view after warmup. Unity's
`FrameTimingManager` may not return GPU timing on every frame, so
`--gpu-timing-min-fraction` controls the minimum valid GPU sample coverage. A
profile with no valid GPU timing fails rather than falling back to CPU time.

For fast smoke tests:

```bash
./single_unity_eval.sh diffsoup mipnerf360/bicycle 0 \
  --unity "$UNITY" \
  --unity-project unity \
  --profile-runs 1 \
  --profile-views 1 \
  --profile-warmup 20 \
  --profile-frames 60 \
  --fps-warmup 3 \
  --fps-frames 12
```

If quality images and metrics already exist and only GPU profiles need to be
rebuilt:

```bash
./single_unity_eval.sh diffsoup mipnerf360/all 0 \
  --unity "$UNITY" \
  --unity-project unity \
  --skip-capture \
  --skip-metrics \
  --force-profile
```

### Resolution and dataset rules

The wrapper reads each scene config. For Mip-NeRF 360, `image_dir: images` plus
`resolution: 4` resolves to `images_4`; `resolution: 2` resolves to `images_2`.
This keeps Unity quality/profile resolution aligned with native evaluation.
The Unity camera replay currently requires COLMAP `sparse/0/cameras.bin` and
`sparse/0/images.bin`.

## Config Inheritance

Experiment configs use `_base_` inheritance. Common defaults live in
`configs/base/_base_.yaml`, dataset-level defaults live in files such as
`configs/base/mipnerf360.yaml`, method-level shared settings live in files such
as `configs/base/triangle-splatting.yaml`, and method/dataset/scene configs
only keep the fields that are unique to that run.

Scene configs are grouped by method and dataset:

```text
configs/triangle-splatting/
├── dtu/scan24.yaml
├── mipnerf360/bicycle.yaml
└── tandt/truck.yaml
configs/mesh-splatting/
├── dtu/scan24.yaml
├── mipnerf360/bicycle.yaml
└── tandt/truck.yaml
configs/2dts/
├── dtu/scan24.yaml
├── mipnerf360/bicycle.yaml
├── native/dtu.yaml
└── tandt/truck.yaml
configs/exmesh/
├── dtu/scan24.yaml
└── nerf_synthetic/lego.yaml
```

For example, `configs/triangle-splatting/mipnerf360/bicycle.yaml` inherits the
MipNeRF360 dataset defaults plus Triangle Splatting defaults, then only sets the
scene and scene-specific options:

```yaml
_base_:
  - ../../base/mipnerf360.yaml
  - ../../base/triangle-splatting.yaml

experiment:
  name: triangle_splatting_bicycle

dataset:
  scene: bicycle
  image_dir: images_4
  resolution: 1

trainer:
  outdoor: true
```

`dataset.root` and `dataset.scene` are resolved together, so a dataset base can
declare `root: data/mipnerf360` while a scene config declares `scene: bicycle`;
the effective dataset path becomes `data/mipnerf360/bicycle`. String templates
such as `outputs/{method}/{dataset}/{scene}` are resolved after all inherited
configs are merged.

To run a whole dataset, use `single_train.sh` with a dataset target ending in
`/all`. It expands the configured scene list, trains each scene, renders/evals,
exports videos, and prints a summary table:

```bash
bash single_train.sh triangle-splatting mipnerf360/all 0
bash single_train.sh triangle-splatting tandt/all 0
bash single_train.sh triangle-splatting dtu/all 0
bash single_train.sh mesh-splatting mipnerf360/all 0
bash single_train.sh mesh-splatting tandt/all 0
bash single_train.sh mesh-splatting dtu/all 0
bash single_train.sh 2dts mipnerf360/all 0
bash single_train.sh 2dts tandt/all 0
bash single_train.sh 2dts dtu/all 0
```

Use `all` as the target to run all built-in datasets in one pass.

> **DTU Evaluation Modes**:
> DTU training and evaluation defaults to masked foreground evaluation composited on a white background (`--dtu-foreground` or alias `--dtu-masked`). A config that sets `dataset.background` explicitly keeps that background in foreground mode. To train and evaluate on unmasked full images with natural backgrounds (Mode A), pass `--dtu-full` or alias `--dtu-unmasked`:
> ```bash
> # Unmasked full-image evaluation (Mode A)
> bash single_train.sh triangle-splatting dtu/all 0 --dtu-unmasked
>
> # Masked white-background evaluation (Mode B, default)
> bash single_train.sh triangle-splatting dtu/all 0 --dtu-masked
> ```

Triangle Splatting configs inherit scene-specific triangle count caps from
`configs/base/triangle-splatting-caps.yaml` through
`configs/base/triangle-splatting.yaml`. During config finalization, the cap for
`dataset.scene` is written to `trainer.max_shapes` unless the scene config
explicitly overrides it.

MeshSplatting configs live under `configs/mesh-splatting/` for MipNeRF360,
Tanks&Temples, and DTU. They use the bundled MeshSplatting renderer and a
method-specific trainer that preserves shared-vertex topology updates. DTU
configs follow the upstream DTU settings, including restricted Delaunay at
iteration 11000; that stage requires the MeshSplatting `effrdel` dependency to
be installed in the active environment.

## Unified Output Layout

Every method now writes to one consistent per-scene directory so training
checkpoints, rendered images, exported geometry, videos, and logs always live
in the same place regardless of backend.

### Target layout

```text
outputs/{method}/{dataset}/{scene}/
├── ckpt/                 # all training checkpoints (one folder, every method)
│   ├── 30000.ckpt                       # 2DTS
│   ├── point_cloud/{N}.ply              # 2DTS exported point clouds
│   ├── point_cloud/iteration_{N}/*.pt   # triangle-/mesh-splatting
│   └── final_params.pt                  # DiffSoup
├── renders/              # rendered images + matching ground truth, by split
│   ├── train/
│   │   ├── renders/      # predicted RGB
│   │   ├── gt/           # dataset ground truth
│   │   ├── manifest.json
│   │   └── metrics.json
│   └── test/
│       ├── renders/
│       ├── gt/
│       ├── manifest.json
│       └── metrics.json
├── mesh/                 # exported geometry used for mesh/DTU metrics
│   ├── fuse_post.ply                    # TSDF mesh (mip360/tandt/dtu)
│   └── {N}_pcd.ply                      # 2DTS DTU point-cloud export
├── video/                # trajectory video
│   ├── render_traj.mp4
│   └── frames/
├── logs/                 # every log: pipeline *.log, training stdout,
│                         # tensorboard events, cfg dumps, loss curves
├── config.yaml           # resolved config snapshot
├── metrics.json          # aggregated image metrics (test split)
├── mesh_metrics.json     # DTU/mesh geometry metrics
├── profile.json          # renderer profiling report (msbench profile)
├── stats.json            # model statistics (msbench inspect)
└── train_stats.json      # training time / peak GPU memory
```

### Adapter checkpoint paths

The `adapter.checkpoint` template in each method base config points at `ckpt/`:

| Method             | `adapter.checkpoint`                                            |
| ------------------ | -------------------------------------------------------------- |
| 2DTS               | `outputs/{method}/{dataset}/{scene}/ckpt`                      |
| Triangle Splatting | `outputs/{method}/{dataset}/{scene}/ckpt/point_cloud/iteration_{max_steps}` |
| MeshSplatting      | `outputs/{method}/{dataset}/{scene}/ckpt/point_cloud/iteration_{max_steps}` |
| DiffSoup           | `outputs/{method}/{dataset}/{scene}/ckpt/final_params.pt`     |
| ExMesh             | `outputs/{method}/{dataset}/{scene}/ckpt/final.pt`             |


## Project Structure


```
MeshSuite/
├── msbench/
│   ├── core/           # Config, datasets, cameras, metrics, mesh eval, profiling, viewer
│   ├── primitives/     # Triangle primitive types (independent, mesh, convex)
│   ├── renderers/      # Adapter wrappers for each method
│   ├── trainers/       # Training loop, per-method trainers, losses, hooks
│   ├── cli/            # Typer-based CLI commands
│   ├── vendor/         # Bundled upstream method code and CUDA extensions
│   └── tests/          # pytest test suite
├── configs/            # YAML experiment configs (base/ + <method>/<dataset>/<scene>.yaml)
├── tools/              # Metrics formatting, Unity export/eval, preprocessing scripts
├── unity/              # Unity project used for engine-side evaluation
├── single_*.sh         # Scene/dataset pipeline wrappers (train, export, Unity eval)
└── pyproject.toml
```


## Citation

If you find this benchmark, codebase, or results helpful in your research, please cite our paper:

```bibtex
@misc{zhang2026meshsuiteunifiedbenchmarktriangle,
      title={MeshSuite: A Unified Benchmark for Triangle- and Mesh-Based Neural Rendering}, 
      author={Kaixuan Zhang and Minxian Li and Mingwu Ren and Xiatian Zhu},
      year={2026},
      eprint={2609.01306},
      archivePrefix={arXiv},
      primaryClass={cs.GR},
      url={https://arxiv.org/abs/2609.01306}, 
}
```

## License

This project is licensed under the [MIT License](LICENSE). Third-party submodules and vendor components under `msbench/vendor/` and `submodules/` are subject to their respective original licenses.
