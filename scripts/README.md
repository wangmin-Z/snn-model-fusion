# 实验脚本

本目录保存可重复执行的实验编排脚本，不放模型实现。

## `run_recommended_experiments.sh`

该脚本完成以下任务：

1. 检查三个遥感数据集和 ANN 初始权重是否存在。
2. 优先从 `latest` checkpoint 恢复未完成实验。
3. 按顺序运行 RSSCN7 的缺失时间窗口。
4. 训练 AID ANN，并继续运行 AID 的 `T=2/4/6/8` SNN。
5. 将所有输出写入已被 Git 忽略的 `outputs/`。

先查看队列：

```bash
./scripts/run_recommended_experiments.sh --dry-run
```

正式启动：

```bash
./scripts/run_recommended_experiments.sh
```

该脚本用于单机 MPS 长时间训练，不由 GitHub Actions 执行正式实验。

## `archive_completed_experiment.py`

该脚本只归档最后一轮达到 120 轮的实验。它会复制轻量的 `args.json` 和 `history.csv`，生成实验说明，并同步 `results/completed/README.md` 与 `docs/results.md`；数据集、checkpoint 和完整 `outputs/` 不会被归档。

示例：

```bash
python scripts/archive_completed_experiment.py \
  --source-run /path/to/outputs/rsscn7_80_snn_mft_resnet18_T2_img224_e120_seed42 \
  --dataset RSSCN7 \
  --model SNN-MFT-ResNet-18 \
  --time-steps 2 \
  --output-path outputs/paper_recommended_20260711/rsscn7_80_snn_mft_resnet18_T2_img224_e120_seed42
```

## `publish_completed_results.sh`

该监控脚本用于训练期间持续发布完成实验。每 5 分钟检查推荐队列；某组实验的 `history.csv` 到达第 120 轮后，脚本才会归档结果、更新索引，并只提交 `results/completed/<run>/`、`results/completed/README.md` 与 `docs/results.md`。

```bash
./scripts/publish_completed_results.sh \
  --source-root /path/to/training/snn-model-fusion
```

使用 `--once` 可只检查一次。该脚本必须从专用发布分支运行，不能在正在训练的工作目录中运行。
