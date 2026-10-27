# AGENTS.md - MeshSuite 协作与开发规范

本文档为参与本项目（MeshSuite）开发与维护的 AI Agent 及开发者制定严格的工作流程与质量控制规范。

---

## 1. 核心工作流程原则（Core Workflow Mandate）

1. **每次代码修改后必须进行 Git 提交**：
   - 任何功能开发、缺陷修复、配置重构或测试更新完成后，必须及时执行 Git 提交，保持项目版本历史完整可追溯。
   - 保持原子性提交（Atomic Commits）：每个提交应对应一个逻辑清晰、单一明确的变更。

2. **提交前必须进行独立检查（Pre-Commit Independent Validation）**：
   - **绝对禁止盲目提交**。在执行 `git commit` 之前，必须主动执行独立验证步骤，确保没有语法错误、没有逻辑缺陷、未破坏现有测试套件。
   - 若独立检查失败，必须先定位并修复问题，直至验证完全通过后方可执行提交。

---

## 2. 提交前独立检查清单（Pre-Commit Checklist）

在执行 `git add` 和 `git commit` 之前，必须依次完成以下 2 项独立检查：

### 2.1 自动化测试套件验证
运行相关测试，确保所有测试用例 100% 通过（pytest 会导入所有被测模块，语法错误会直接导致失败）：
```bash
# 针对 ExMesh 及相关组件的单元与集成测试
/scratch/miniconda3/envs/msbench/bin/pytest -v msbench/tests/test_exmesh.py

# 涉及其他模块时执行对应测试
/scratch/miniconda3/envs/msbench/bin/pytest -v msbench/tests/
```

### 2.2 变更范围与脏文件核查
检查 `git status` 和 `git diff`：
- 确认只有预期的代码和配置文件被修改。
- 严禁将临时调试文件（如 `*.log`、`log.txt`、临时 dump 的 `.ply`/`.pt`/`.npy`）带入版本库。
- 确认无未处理的 debug print 或临时注释掉的核心逻辑。

---

## 3. Git 提交规范（Commit Standards）

采用清晰规范的 Conventional Commits 格式：

```
<type>(<scope>): <subject>

[optional body]
```

- **类型（Type）**：
  - `feat`: 新功能或新配置
  - `fix`: 缺陷修复（如背景通道对齐、超参数修正）
  - `refactor`: 代码重构（不改变外部行为）
  - `test`: 测试用例新增或调整
  - `docs`: 文档变更
  - `config`: 配置文件新增或优化
- **示例**：
  - `fix(exmesh): align dtu paper config with black background and pgsr init`
  - `config(exmesh): inherit the shared DTU dataset protocol`
  - `test(exmesh): update published paper hyperparameter assertions`

---

## 4. 数据集与方法配置协议规范

MeshSuite 中所有方法在同一数据集上使用**同一套数据集配置**，以保证横向对比公平。Agent 在创建或修改配置时必须严格遵循：

1. **数据集协议只来自共享数据集基础配置**：
   - 每个场景配置按 `configs/<method>/<dataset>/<scene>.yaml` 组织，并依次继承 `configs/base/<dataset>.yaml`（数据集协议）与 `configs/base/<method>.yaml`（方法默认值），必要时再加 `configs/base/<method>-<dataset>.yaml`（方法在该数据集上的专属流程）。
   - 方法基础配置（`configs/base/<method>.yaml`、`<method>-<dataset>.yaml`）**禁止**设置数据集协议字段：分辨率、`image_dir`、背景、`dtu_eval_mode`、`eval_split`/`eval_every`、`render.split`/`eval.split`。
   - 场景配置中的数据集字段（如 Mip-NeRF 360 每个场景的 `image_dir`/`resolution`）在所有方法间必须一致。
   - `configs/base/<dataset>.yaml` 只放数据集协议，不放某个方法的训练超参数（例如 Triangle Splatting / MeshSplatting 的 DTU 训练参数位于 `configs/base/splatting-dtu.yaml`）。
   - 方法基础配置（`configs/base/<method>.yaml`）不继承 `_base_.yaml`，否则会以全局默认值覆盖数据集协议。

2. **DTU 统一协议**（`configs/base/dtu.yaml`）：
   - `dtu_eval_mode: foreground`，前景模式下训练与渲染使用白色背景。
   - `resolution: 2`（2 倍降采样）。
   - 留出评估：`eval_split: true`，`eval_every: 8`，`render.split: test`，`eval.split: test`。
   - 训练器的 `eval_split`、`resolution`、`dtu_eval_mode` 由 `msbench.cli.train.inherit_dataset_defaults` 从 `dataset` 继承；方法基础配置不要显式设置这些字段。
   - 任何初始化/预处理（如粗网格）必须使用相同的留出划分，不得接触测试视点。

3. **方法专属超参数**：迭代次数、损失权重、初始化方式等按方法论文设置；论文明确给出的设置优先于官方仓库默认值，论文未说明的细节沿用官方仓库。
   - 例：ExMesh（`configs/base/exmesh.yaml`、`configs/base/exmesh-dtu.yaml`、`configs/exmesh/dtu/scan*.yaml`）：8k 迭代（论文 Sec. 4.1）、2DGS 7k iters + 256³ TSDF 初始化（论文 Sec. 4.1，`preprocess.holdout: 8`），逐场景 `lambda_smooth`/`lambda_depth` 与学习率调度沿用官方 `scripts/run_dtu.py` 与默认参数。
   - 测试 `msbench/tests/test_exmesh.py` 断言 ExMesh 的 DTU 数据集配置与其他方法完全一致；新增方法时应添加同类断言。
