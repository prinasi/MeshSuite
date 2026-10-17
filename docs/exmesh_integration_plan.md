# ExMesh 原生集成实施方案

状态：已完成 ExMesh 原生接入、CUDA 数值对照，以及 scan24 两种协议的 10,000 步训练、图像/正式几何评测、性能统计和网格/纹理导出。NeRF-Synthetic lego 也已完成 DA3 训练深度生成、独立 PGSR 5,000 步初始化、ExMesh 8,000 步训练及 200 个测试视图的图像指标/性能验收。实测结果、未执行的原仓库完整轨迹对照及 lego 分裂未触发的限制见 [exmesh_validation.md](exmesh_validation.md)。

调查日期：2026-10-16。ExMesh 参考提交：`0bf121895ebd74642c1bab824a992c48f95296e8`。

## 1. 已确认的范围

- 在 MeshSuite 内原生实现 `exmesh`，运行时不依赖相邻的 `../ExMesh` 目录。
- 首版读取已有粗网格；PGSR 训练和粗网格提取保留为独立前处理。
- 先保留原版数值行为建立对照；相机、深度等算法修正使用独立配置，逐项验证。
- 分别提供“原版复现”和“MeshSuite 基准”配置，独立保存结果。
- 首版包含 Python 训练、渲染、图像/几何评测、性能统计、网格及 UV 纹理导出。
- Unity 的纹理资源和渲染支持后续单独实施。
- 真实验收使用用户提供的 DTU scan24，进行原版对照及完整训练。

数据已准备：`data/DTU -> /scratch/datasets/DTU`（即用户的 `~/datasets/DTU`）。
已将 `~/datasets/DTU_depth_da3.zip` 合并到该目录，新增并逐一校验 1,710 个文件，无覆盖冲突。
scan24 的 49 张图像与 49 份 `mono_priors/da3/*.npz` 文件名完全对应，深度数组均有限且包含正值。
原图大小为 1554×1162，深度数组为 378×504，训练读取时需沿用确认的缩放方式。
msbench 的 DTU 前景模式加载检查通过（诊断 resolution=8），训练/测试划分为 42/7。
`Points/stl/stl024_total.ply` 真值存在；用户更新数据后，15 个场景的真值、ObsMask 和 Plane 均已补齐。
`ObsMask` 通过用户建立的软链接指向 `SampleSet/MVS Data/ObsMask/`。
scan24 的 ObsMask/BB/Res/P 字段读取正常，真值 5,169,152 个点均有限，抽样点能通过观测区域及平面过滤。
初始 `scan24/mesh.ply` 确认通过独立 PGSR 前处理生成。
数据准备与读取报告位于 `outputs/exmesh-validation/data_setup.json` 和 `scan24_data_check.json`。
正式几何评测输入检查报告位于 `outputs/exmesh-validation/dtu_evaluation_data_check.json`。

PGSR 复现参数已确认：5,000 步、`-r2 --ncc_scale 0.5`；网格提取使用
`--num_cluster 1 --voxel_size 0.01 --max_depth 5.0`，按 README 不启用深度过滤。
独立工作副本和输入视图位于 `outputs/exmesh-preprocess/`；输入图像只链接，
COLMAP 中间文件写入工作区，避免改写共享原始数据。首个初始化对应原版全部视图复现；
后续 MeshSuite 留出测试视图的基准须单独生成只用训练视图的初始化，避免测试信息泄漏。

PGSR 环境选择：用户确认使用官方 PyTorch3D 0.7.8 的 Python-only 构建，
固定提交 `75ebeeaea0908c5527e7b1e305fbc7681382db47`，保留原版四元数转换函数。
该安装不提供 PyTorch3D 自身的 C++/CUDA 算子；PGSR 渲染与 KNN 使用其独立 CUDA 扩展。
完整 PyTorch3D 构建曾因缺少 `cusparse.h` 失败，未将更改 CUDA/PyTorch 版本作为解决方式。

额外相机核对：COLMAP 与 DTU 相机位姿一致（抽查误差约 `6e-8`），但相机主点不同。
例如诊断 resolution=8 下，图像 0000 的 DTU 主点约 `(102.77,77.25)`，
COLMAP 公共加载器及原版 PGSR 的居中主点为 `(97,72.5)`。
首次原版复现保留 PGSR 行为；后续统一基准与修正配置需明确处理主点约定，不能静默混用。
比较报告位于 `outputs/exmesh-validation/scan24_camera_comparison.json`。

PGSR 实测进展：5,000 步训练已完成并保存模型，但第一次提取得到空 TSDF。
对已训练模型的 0000/0024/0048 视图检查显示前景深度有限，约为 1.52–3.02，未被 `max_depth=5` 截除。
进一步使用无 PGSR 依赖的 2 米平面测试复现：Open3D 0.20.0 可以反投影出 3,072 个点，
但 TSDF 体素和网格均为零。与上游 [Open3D #7592](https://github.com/isl-org/Open3D/issues/7592)
报告的问题一致。保留已训练结果，只需处理 Open3D 兼容性后重跑提取。
复现脚本和结果为 `outputs/exmesh-validation/check_open3d_tsdf.py`、`tsdf_smoke_0.20.0.json`；
PGSR 深度诊断为 `outputs/exmesh-preprocess/scan24/depth_diagnostic.json`。

用户确认后，已将 msbench 的 Open3D 固定为 0.19.0，并同步写入 `requirements.txt` 与 `environment.yml`。
同一平面测试恢复正常，生成 7,836 个体素点、1,900 个顶点与 3,626 个面。
只重跑提取后，scan24 粗网格生成成功：30,165 个顶点、58,206 个三角面，带顶点颜色；
顶点有限、面索引有效、所有面面积大于零。`data/DTU/scan24/mesh.ply` 已链接到
`outputs/exmesh-preprocess/scan24/model/mesh/tsdf_fusion_post.ply`。
完整过程及参数记录于 `outputs/exmesh-preprocess/scan24/status.json`，
训练时环境与提取时环境分别保存在 `environment.json` 和 `extraction_environment.json`。
版本调整后的全量回归仍为 439 passed / 5 failed / 5 skipped，与集成前基线相比无新增失败；
验证报告为 `outputs/exmesh-validation/open3d019-validation.json`。

## 2. 源码调查结论

ExMesh 是带共享顶点和 UV 纹理的显式网格优化，不是现有 MeshSplatting 渲染器的一组参数。
需要新增独立模型、nvdiffrast 渲染器及训练方法，复用 MeshSuite 的公共接口。

| ExMesh 源文件 | 职责 | 集成方式 |
| --- | --- | --- |
| `scene/mesh_model.py` | 顶点、面、UV、纹理、分裂/合并/边界裁剪、优化器重绑定 | 移入独立 vendor 包，保留算法和出处 |
| `renderer/__init__.py` | nvdiffrast 光栅化、UV 采样、抗锯齿、深度、法线、面覆盖统计 | 原生后端 + `ExMeshAdapter` |
| `train.py` | 损失、梯度统计、拓扑与 UV 更新时间表 | `ExMeshTrainingMethod` + 公共 `TrainingLoop` |
| `utils/graphics_utils.py` | xatlas 展 UV、纹理烘焙及顶点颜色采样 | 隔离到 ExMesh vendor 命名空间 |
| `utils/loss_utils.py` | 图像、轮廓、Pearson 深度、Laplacian、位移差正则 | 保留训练损失公式；评测走公共 metrics |
| `utils/mono_utils.py` | 按图像名读取 DA3 `depth` 数组 | 与公共 dataset sample 对齐的深度读取器 |
| `scene/dataset_readers.py` / `scene/cameras.py` | COLMAP/Blender 与相机矩阵 | 用公共 CameraBatch 封装，显式保留兼容转换 |
| `scripts/eval_dtu/` | 遮罩裁剪、尺度还原、DTU 几何评测 | 对照公共 `core/mesh_eval.py` 的协议差异 |

核心迁移所需第三方库包括 `nvdiffrast`、`xatlas`、Open3D、SciPy、OpenCV 等。
源码中 `pytorch3d` 的引用位于 PGSR 的四元数转换；不属于首版 ExMesh 核心依赖。
PGSR 的 `diff-plane-rasterization`、`simple-knn` 同样不应成为读取粗网格模式的强制依赖。

## 3. 计划的文件与接口

新增文件的具体拆分可随实现调整，外部接口固定为独立的 `exmesh` 方法。

```text
msbench/vendor/exmesh/
    __init__.py
    mesh_model.py
    renderer.py
    geometry.py
    losses.py
    priors.py
    LICENSE.md / 来源与修改说明
msbench/renderers/exmesh_adapter.py
msbench/trainers/exmesh_method.py
configs/base/exmesh.yaml
configs/exmesh/reproduction/dtu/scan24.yaml
configs/exmesh/benchmark/dtu/scan24.yaml
msbench/tests/test_exmesh_adapter.py
msbench/tests/test_exmesh_training.py
msbench/tests/test_exmesh_topology.py
msbench/tests/test_exmesh_cuda.py
tools/validate_exmesh_parity.py
```

公共接入点：

- `renderers/backends.py`：增加 backend、依赖诊断及名称解析。
- `core/registry.py` 和 `trainers/registry.py`：增加惰性发现。
- `pyproject.toml`：adapter / training method entry points 与 ExMesh 可选依赖。
- `core/builder.py`、`cli/train.py`：优先沿用通用构建路径；检查配置传参是否完整。
- CLI 的 primitive count、检查点解析、导出路径：验证新检查点结构得到正确识别。
- `single_train.sh` 的方法白名单及分支：在 CLI 闭环之后接入，避免依赖尚未实现的 Unity 阶段。

模型数据必须同时保存：`vertices[V,3]`、`faces[F,3]`、`uvs[K,2]`、
`uv_indices[F,3]`、`vmapping[K]`、`texture[3,Ht,Wt]`、`texture_mask[Ht,Wt]`。
几何顶点索引与 UV 索引分开，避免破坏纹理接缝。

Adapter 实现 `load_checkpoint`、`load_scene`、`configure`、`render`、
`model_stats`、`profile_metadata`、`to_primitive` 和方法自己的纹理导出接口。
`to_primitive` 可复用 `IndexedMeshTriangle` 提供几何，完整 UV 外观仍以 ExMesh 模型为准。
不把纹理伪装成 SH 或 splat opacity。

渲染输出统一为 `RenderOutput`：RGB/法线使用 HWC/BHWC，alpha/深度使用 HW/BHW；
原版面覆盖量放在 `extras` 中。兼容深度需标注其为插值的相机空间顶点距离，
不能默认当成公共工具所需的相机 Z 深度使用。

## 4. 必须单独验证的语义差异

### 相机、坐标和深度

- 原版 Camera 存储的 W2C 与 renderer 的乘法方式需逐式核对。
- renderer 内法线与深度路径分别使用 `view` 和 `view.t()`；这属于待验证差异，不能静默统一。
- 原版将投影 Z 从 `[0,1]` 转到 `[-1,1]`；需要验证裁剪、主点和图像 Y 方向。
- 使用带平移、旋转、非居中主点和非对称纹理的场景测试，单位矩阵不足以证明正确。
- DTU 的 COLMAP 坐标、`cameras.npz` 归一化坐标、粗网格坐标和评测尺度必须核对；避免重复应用 `scale_mat`。
- 原版一致性与解析几何正确性分别报告。原版存在疑点时，匹配原版不等于证明物理约定正确。

### 训练与拓扑更新

原版默认：10,000 迭代；每 500 步进行结构事件；第 1 步初始合并；
第 1,000–7,000 步分裂/合并；第 500、2,000、4,000、6,000 步重建 UV。
实际事件必须按 `train.py` 的条件与顺序实现，而不是只参考注释。

- 早期先对轮廓损失反传，累计顶点梯度幅度，再清除顶点/纹理梯度，随后对总损失反传。
  该梯度统计需要显式接入，不能简单把所有损失相加后仅反传一次。
- 公共循环的顺序是 backward → 日志/保存 → 结构更新 → optimizer step。
  原版最终步不做 optimizer step。需要在方法 hooks 中明确保存与恢复的步语义。
- 每次替换 Parameter 后，优化器必须绑定新对象，动量按原版映射或重置规则处理。
- 拓扑改变后同步 `image_size`、退化度、`mask_grad_ema`、`vertices_ref`、UV 索引及映射。
- 保留原版的采样与随机数消耗顺序，比较时固定相机顺序和 RNG 状态。
- 空网格、全部退化、零概率采样、空深度有效区域必须形成专门用例；发现需要改变行为时先说明并确认。

### 检查点

原版 `save/load` 可以保存模型外观，但不足以恢复训练全过程。
`capture/restore` 还引用当前网格模型没有初始化的 Gaussian 字段，且调用 `training_setup` 的参数不匹配。

原生实现应区分推理快照和可恢复训练状态：后者包含步数、模型与拓扑、优化器、学习率、
参考顶点、统计数组、Python/NumPy/Torch/CUDA RNG、相机采样池与配置版本。
增加完整训练状态属于保存能力修复，不应改变从头训练的更新公式。
保存时间点应明确定义，并通过连续训练与断点训练对照验证，避免恢复后重复执行结构事件。

### 配置与评测

- 原版复现配置：按原版训练视图、背景、scan24 场景参数建立对照。
  `scripts/run_dtu.py` 的 scan24 深度权重是 `0.1`，不同于参数类默认的 `0.003`，必须显式记录来源。
- MeshSuite 基准配置：固定 train/test 划分、DTU 背景和公共图像评测协议，结果单独输出。
- 不直接继承 `configs/base/dtu.yaml` 的整段 trainer：其中有其他方法的损失和拓扑参数。
  仅复用适用的数据与评测设置，ExMesh 参数显式配置。
- 记录 mask 来源（RGBA、`mask/` 或原版的 `train_mask/*_gtmask.png`）、图像排序及深度文件对应关系。
- DTU 原版遮罩裁剪和公共实现的取样/裁剪行为需要对照；协议不同的结果不能混为同一组。
- 正式 DTU 验收要求真值、ObsMask、Plane 与尺度文件完整，结果须报告实际协议；
  通用 Chamfer 回退结果不计作正式 DTU 验收通过。

## 5. 实施阶段与完成条件

| 阶段 | 工作 | 完成条件 |
| --- | --- | --- |
| P0 环境与数据 | 固定源码版本；补齐 nvdiffrast/xatlas；核对 scan24、mesh、mask、DA3、真值坐标 | msbench 中真实 CUDA 前向/反向及 UV 烘焙成功；数据清单完整 |
| P1 模型与推理 | 迁移模型/纹理/光栅化，新增 adapter、registry、CPU 检查点检查 | 同一模型和相机下对照原版 RGB/alpha/深度/法线/面覆盖统计 |
| P2 固定拓扑训练 | 接入损失、轮廓梯度统计、LR、EMA；诊断用例暂不触发结构事件 | 各损失、梯度及一步参数更新与原版对齐，短训练有限且可优化 |
| P3 完整拓扑训练 | 分裂、合并、裁剪、UV 重建、动量映射、完整 checkpoint/resume | 每类事件均实际触发；网格/UV/优化器状态有效；断点对照通过 |
| P4 CLI 与导出 | 两套配置、train/render/eval/profile/inspect、PLY 与纹理资产、批处理入口 | 从配置训练到评测形成闭环，导出重载外观一致且无相邻仓库依赖 |
| P5 scan24 验收 | 原版与移植版相同输入完整训练；分别输出两种协议报告；全项目回归 | 提供逐层对照、最终图像/DTU 指标、耗时与显存、测试报告及已知限制 |

所有阶段只在 `msbench` 环境中执行，不切换到 `ts` 或另外新建 ExMesh 环境。
12GB 显存下可先降低诊断用例纹理分辨率；正式复现配置仍使用确认的原版参数，
若实际 OOM 需要改变分辨率、CPU/GPU 缓存或面数预算，先给出实测证据与选项。

## 6. msbench 验证矩阵

| 层级 | 用例 | 通过依据 |
| --- | --- | --- |
| CPU 接口 | registry、惰性依赖、配置、坏索引、checkpoint、统计、几何导出 | 不依赖 CUDA 也能做相应检查；缺依赖报错明确；既有方法照常加载 |
| CUDA 解析场景 | 非对称 UV 四边形/立方体、遮挡面、非单位相机、背景与近远裁剪 | 覆盖、遮挡、纹理方向、深度、法线满足已声明约定；顶点及纹理梯度有限且非零 |
| 原版数值对照 | 同一 tensors、相机、纹理、种子；固定 UV，不重复独立运行 xatlas | 比较输出、分项损失、梯度、一步优化；容差按重复运行噪声校准并写入报告 |
| 拓扑与 UV | 指定分裂/合并候选；边界裁剪；原版时间表与缩短诊断时间表 | 索引有效、属性长度一致、面数预算可核查；变更后仍可渲染/反传/优化 |
| 状态恢复 | 普通步及结构事件前后的 checkpoint；连续 N 步 vs K 步保存恢复再 N-K 步 | RNG、相机序列、事件执行次数和状态一致；数值误差在已校准容差内 |
| CLI 闭环 | 小场景 train → reload → render → eval → profile → export/reload | 输出文件可解析，指标来自匹配视图；profile 使用真实 CUDA 同步计时 |
| scan24 | 原版与移植版短对照，再完整 10,000 步；两种协议分别报告 | 无 NaN/坏拓扑；最终质量差异与数值波动有证据，真实 DTU 数据和协议完整 |
| 全项目回归 | 全量现有 pytest + 新增 ExMesh 用例 | 无新增失败；既有失败逐项列明；关键 ExMesh CUDA 测试不能跳过冒充通过 |

损失下降只能证明部分优化流程可用，不能单独证明算法移植正确。
不要求随机拓扑运行跨不同环境逐位一致；首先在同一环境固定输入、候选与 RNG 建立可比较证据。
具体数值容差和最终质量允许差异是待校准的验收参数；若原版噪声大或结果有明显偏差，
提交重复运行数据及建议阈值，请用户确认，不自行放宽标准。

已实现的主要验证命令（早期规划中的分文件测试现已集中到 `test_exmesh.py`）：

```bash
conda activate msbench
python -m pytest -q msbench/tests/test_exmesh.py
PYTHONPATH=. python tools/validate_exmesh_parity.py --reference ../ExMesh --output outputs/exmesh-validation/parity.json
msbench train --config configs/exmesh/reproduction/dtu/scan24.yaml
msbench render images --config configs/exmesh/reproduction/dtu/scan24.yaml
msbench eval mesh --config configs/exmesh/reproduction/dtu/scan24.yaml
msbench train --config configs/exmesh/benchmark/dtu/scan24.yaml
msbench render images --config configs/exmesh/benchmark/dtu/scan24.yaml
msbench eval images --config configs/exmesh/benchmark/dtu/scan24.yaml
msbench eval mesh --config configs/exmesh/benchmark/dtu/scan24.yaml
msbench inspect --config configs/exmesh/benchmark/dtu/scan24.yaml
msbench profile --config configs/exmesh/benchmark/dtu/scan24.yaml
python -m pytest -q --junitxml=outputs/exmesh-validation/regression.xml
```

## 7. 已实际执行的基线测试

环境：`/scratch/miniconda3/envs/msbench/bin/python`，Python 3.12.15，
PyTorch 2.7.0+cu118。GPU 在沙箱外可访问，为 RTX 3060 12GB，驱动 550.163.01。
基线采集时缺少 nvdiffrast、xatlas；当时 pytorch3d 也未安装，但不在首版核心迁移范围内。
后续为 PGSR 安装了前述 PyTorch3D Python-only 构建及 PGSR CUDA 扩展；ExMesh 核心依赖仍待补齐。

沙箱内运行得到 435 passed / 6 failed / 8 skipped，其中一个失败是 socket 权限限制。
在沙箱外使用同一环境重跑：

```bash
/scratch/miniconda3/envs/msbench/bin/python -m pytest -q --tb=short --junitxml=/tmp/exmesh-msbench-baseline-host.xml
```

实际结果：**439 passed / 5 failed / 5 skipped**，449 项，共 9.28 秒。
五个失败均涉及缺失的既有 `configs/opaque_2dts_mipnerf360/2dts/mipnerf360/` 配置：

1. `test_2dts_native_config_auto_resumes_from_latest_checkpoint`
2. `test_2dts_native_config_skips_resume_when_checkpoint_reaches_max_steps`
3. `test_structured_train_cli_max_steps_overrides_config`
4. `test_2dts_mipnerf360_opaque_experiment_native_overrides_match_dtu_style`
5. `test_2dts_mipnerf360_opaque_experiment_configs_resolve_to_isolated_outputs`

跳过项包括 3 项 Triangle Splatting 后端条件测试和 2 项已移除 Unity 工具的测试。
上述结果是集成前基线，**不代表 ExMesh 已集成或已验证正确**。

当前工作树已有 README、环境、rendering、D2TS 等修改，实施时保留这些改动。
初次规划仅新增本方案与基线报告，不修改既有算法代码；后续按用户要求固定了 Open3D 依赖版本并完成独立 PGSR 初始化。

## 8. 最终交付

2026-10-17 实施进展：核心 vendor、adapter、训练 hooks、两套 DTU 配置、完整断点恢复、
纹理导出 CLI 均已实现；实际测试集中于 `msbench/tests/test_exmesh.py`。
固定 UV 和非单位相机下，与原版 RGB/alpha/深度/法线/损失对照的误差为零，
梯度差异与原版重复运行噪声相同。跨分裂事件的断点恢复测试已通过。
初始 PGSR 网格使用完整 DTU 几何协议得到 accuracy=0.4845、completeness=0.6165、overall=0.5505。
这些是初始化指标；现已完成的最终结果见独立验收记录。
原版对照范围为固定输入下前向、五项损失、梯度及一次 Adam 更新；两套完整训练都运行于移植版。
早期 P5 规划中的原仓库独立完整训练轨迹对照尚未执行，不能据此声称随机拓扑轨迹等价。

- 可独立运行的原生 ExMesh 方法及两套配置。
- 完整可恢复检查点、几何导出，以及保留 UV 接缝的网格/材质/纹理资产。
- 来源、许可证及修改说明，保留迁入源码已有声明。
- `outputs/exmesh-validation/` 下的环境与数据清单、原版对照、CUDA 测试、
  拓扑/恢复测试、scan24 完整训练、图像/DTU 指标、时间/显存与回归报告。
- 清楚区分“原版复现通过”“统一基准可运行”“算法修正验证通过”，不合并为笼统的成功结论。
