# 贡献说明

本仓库以论文复现的可读性、可重复性和结果可追溯性为优先目标。

## 开发流程

1. 从最新 `main` 创建功能分支。
2. 保持修改范围单一，不混入数据集、模型权重或本地训练输出。
3. 新增或修改模型逻辑时，同步补充 `tests/` 中的快速验证。
4. 修改实验参数或结果时，同步更新 `docs/results.md` 和对应的 `results/completed/<run>/README.md`。
5. 提交前运行：

```bash
python tests/smoke_test.py
bash -n scripts/run_recommended_experiments.sh
scripts/run_recommended_experiments.sh --dry-run
git diff --check
```

## 文件边界

- `snn_mft/`：论文方法实现。
- `scripts/`：可重复执行的实验编排。
- `tests/`：不依赖真实数据集的快速检查。
- `docs/`：面向读者的数据和结果说明。
- `results/completed/`：已完成实验的轻量参数与逐轮指标。
- `data/`、`outputs/`：仅保留本地内容，不提交大文件。

## 提交信息

提交信息应简短描述结果，例如：

```text
add RSSCN7 T=6 completed metrics
fix MPS checkpoint resume
document AID dataset verification
```
