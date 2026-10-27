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

在执行 `git add` 和 `git commit` 之前，必须依次完成以下 3 项独立检查：

### 2.1 语法与静态检查
确保所有新增或修改的 Python 文件无语法错误：
```bash
/scratch/miniconda3/envs/msbench/bin/python -m py_compile <modified_python_files>
```

### 2.2 自动化测试套件验证
运行相关测试，确保所有测试用例 100% 通过：
```bash
# 针对 ExMesh 及相关组件的单元与集成测试
/scratch/miniconda3/envs/msbench/bin/pytest -v msbench/tests/test_exmesh.py

# 涉及其他模块时执行对应测试
/scratch/miniconda3/envs/msbench/bin/pytest -v msbench/tests/
```

### 2.3 变更范围与脏文件核查
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
  - `config(exmesh): add dual benchmark and reproduction dtu configs`
  - `test(exmesh): update published paper hyperparameter assertions`

---

## 4. DTU 数据集与算法配置协议规范

在 MeshSuite 中，DTU 数据集的处理严格划分为两个不同版本，Agent 在创建或修改配置时必须严格遵循：

1. **论文官方复现版（Reproduction Version）**：
   - **定位**：沿用方法官方实现的训练设置（ExMesh 官方 DTU 脚本使用黑色背景），但在留出视点上评测前景指标。
   - **核心属性**：
     - 背景模式：`dtu_eval_mode: foreground`，`background: black`，`white_background: false`（显式的 `dataset.background` 优先于前景模式默认的白色背景）。
     - 分辨率：`resolution: 1`（全分辨率 1554×1162）。
     - 初始化：官方推荐粗网格（PGSR 5000 iters 提取的 mesh），PGSR 使用相同留出划分（`preprocess.holdout: 8`），不接触测试视点。
     - 视点划分：留出评估模式（`dataset.eval_split: true`，`eval_every: 8`，`trainer.eval_split: true`，`render.split: test`，`eval.split: test`）。注意 `trainer.eval_split` 不会从 `dataset` 继承，必须显式设置。
   - **配置路径**：`configs/base/exmesh_paper_dtu.yaml`、`configs/exmesh/reproduction/dtu/scan*.yaml`（`single_train.sh exmesh dtu/<scene>` 默认指向此版本）。

2. **标准基准评测版（Benchmark Version）**：
   - **定位**：MeshSplatBench 统一横向对比评测协议（与其他方法如 2DTS、Triangle-Splatting 等在同一基准线上对比）。
   - **核心属性**：
     - 背景模式：`dtu_eval_mode: foreground`，`background: white`，`white_background: true`。
     - 分辨率：`resolution: 2`（2 倍降采样）。
     - 视点划分：留出评估模式（`eval_split: true`，`eval_every: 8`）。
     - 初始化：基准预处理网格（如 2DGS 降采样提取的 mesh）。
   - **配置路径**：`configs/base/exmesh_benchmark_dtu.yaml`、`configs/exmesh/benchmark/dtu/scan*.yaml`。
