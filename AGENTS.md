# AI 协作规则

本文件适用于仓库根目录及所有子目录。Codex、Claude、GitHub Copilot 等 AI 在读取、修改、运行或提交本项目时，必须遵守以下规则。若子目录以后增加更具体的 `AGENTS.md`，则子目录规则优先，但不得违反本文件的数据和结果安全边界。

## 1. 项目目标

本项目依据论文 *Deep spiking neural networks based on model fusion technology for remote sensing image classification*，独立复现 BM-IF、S-ResNet、Repeat Encoding、ANN 到 SNN 权重复制和 MFT 训练流程。

- 论文没有提供完整官方源码，不得把本实现描述为论文作者的原始代码。
- 当前主要运行环境是 Apple Silicon Mac，训练优先支持 PyTorch MPS。
- 已完成结果与论文结果必须明确区分，不得夸大复现程度或伪造缺失实验。

开始工作前至少阅读：

1. `README.md`
2. `CONTRIBUTING.md`
3. 当前任务涉及目录中的 `README.md`
4. 涉及数据或实验时，再阅读 `docs/datasets.md` 和 `docs/results.md`

## 2. 语言与可读性

- 面向用户的说明、代码注释、文档和 PR 描述默认使用中文。
- Python 标识符、命令行参数和通用技术名词可保留英文。
- 代码应适合学习和复查：命名清楚，控制流程直接，不使用不必要的技巧或过度抽象。
- 只在复杂逻辑、论文公式对应关系或设备兼容处理处添加有价值的中文注释。
- 新增或调整命令行参数时，必须同步更新帮助文本和相关文档。

## 3. 目录职责

| 路径 | 允许完成的任务 |
| --- | --- |
| `snn_mft/` | BM-IF、数据加载、ANN/SNN 模型和权重复制等可复用核心实现。 |
| `train.py` | ANN 与 SNN 的统一训练、验证、保存和断点恢复入口。 |
| `scripts/` | 可重复执行的实验队列与批处理脚本。 |
| `tests/` | 不依赖真实遥感数据集的快速测试。 |
| `docs/` | 数据来源、实验设置、结果汇总和复现说明。 |
| `results/completed/` | 已完成实验的轻量参数、逐轮指标和任务说明。 |
| `data/` | 本地数据目录，只允许 Git 跟踪 `README.md`。 |
| `outputs/` | 本地训练输出目录，只允许 Git 跟踪 `README.md`。 |
| `.github/` | PR 模板、GitHub Actions 和仓库协作配置。 |

- 不要把模型实现放入 `scripts/`，也不要把实验结果混入 `snn_mft/`。
- 新增功能目录时，必须同时添加 `README.md`，说明该目录完成的任务、主要文件和边界。
- 不做与当前任务无关的重构、重命名或格式化。

## 4. 数据与大文件边界

以下内容禁止提交到 Git：

- 遥感数据集、图片、压缩包和解压后的数据目录
- 模型权重和 checkpoint，包括 `*.pt`、`*.pth`、`*.ckpt`
- 完整的本地 `outputs/`、缓存、虚拟环境和系统临时文件
- 密钥、令牌、账号信息和机器专用绝对路径

数据只保存在 `data/`，训练产物只保存在 `outputs/`。修改 `.gitignore` 时不得破坏以上边界。提交前必须使用 `git status` 检查是否误纳入大文件。

## 5. 训练与设备规则

- 使用项目本地 `.venv`，推荐 Python 3.10；不要修改系统 Python 环境。
- 设备参数必须继续支持 `mps`、`cpu` 和 `cuda`，不得为了单一环境移除其他后端。
- MPS 训练期间，参与同一计算的模型和张量必须位于同一设备。
- 转换为 NumPy 或使用 Matplotlib 前，张量必须先执行 `detach().cpu()`。
- 恢复训练必须优先使用 `latest` checkpoint，并保持优化器、调度器、轮次和历史最佳指标连续。
- 不得擅自降低完整复现实验的 120 轮设置来冒充最终结果；快速验证必须明确标记为 smoke test 或 dry-run。

## 6. 实验与结果归档

- 实验命名应包含数据集、划分比例、模型、时间步、图像尺寸、轮数和随机种子等关键信息。
- 已完成实验必须达到预定轮数后，才可进入 `results/completed/`。
- 每个完成实验目录至少包含：
  - `README.md`：完成任务、配置、最佳轮次、最佳与最终验证指标
  - `args.json`：实际启动或恢复参数
  - `history.csv`：逐轮训练和验证指标
- 新增或修正结果时，同时更新 `results/completed/README.md` 和 `docs/results.md`。
- 指标必须从 `history.csv` 或实际运行日志计算，不得仅凭目录名、记忆或论文表格填写。
- 不覆盖已有实验记录。参数不同的实验应使用新的运行目录，并说明差异。
- 未完成、失败或仅预览的实验不得标记为“已完成”。

## 7. 代码修改要求

- 优先沿用仓库现有 PyTorch 实现、参数命名和文件边界。
- 修改 BM-IF、替代梯度、状态复位、时间步循环或权重复制时，必须核对论文公式及现有测试。
- 数据划分必须保持随机种子可控，避免训练集和验证集泄漏。
- checkpoint 加载必须明确处理设备映射和参数兼容性，不静默忽略关键权重缺失。
- shell 脚本必须支持从仓库根目录运行，并保留 `--help`、`--dry-run` 和断点恢复行为。
- 不引入无必要的新依赖；新增依赖必须更新 `requirements.txt` 和安装说明。

## 8. 必须执行的验证

修改提交前，根据变更范围运行以下检查：

```bash
.venv/bin/python tests/smoke_test.py
bash -n scripts/run_recommended_experiments.sh
scripts/run_recommended_experiments.sh --dry-run
git diff --check
git status --short
```

- 修改 BM-IF 演示时，再运行 `.venv/bin/python demo_bmif.py`。
- 修改训练、数据或模型逻辑时，至少增加或调整一个能覆盖该行为的测试。
- 修改文档时，检查 Markdown 本地链接是否有效，并核对实验数字与 CSV 一致。
- 无法执行某项验证时，必须在提交或回复中说明原因，不得写成已通过。

## 9. Git 与 GitHub 协作

- 从最新 `main` 创建范围单一的功能分支，不直接在 `main` 上开发。
- 提交信息应简短描述实际变更，不混入数据、权重或无关文件。
- 推送前检查 staged diff；不得覆盖或回退其他人已有的未相关修改。
- 默认通过 Pull Request 合并，并在 PR 中写明变更、验证和数据边界。
- 自动检查通过且无冲突后，优先使用 `Squash and merge` 保持主分支历史简洁。
- 合并后可以删除已完成的功能分支。

## 10. 完成标准

AI 只有在以下条件全部满足时，才能声称任务完成：

1. 实现或文档已经落到正确目录。
2. 相关说明、索引和参数记录已经同步。
3. 必要测试已经运行并如实报告结果。
4. `git status` 中没有误提交的数据、权重或本地输出。
5. 已明确说明未完成内容、剩余实验或环境限制。
