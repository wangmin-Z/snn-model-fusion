# 本地训练输出

本目录由 `train.py` 和实验队列自动创建，用于保存：

- `args.json`：训练参数。
- `history.csv`：逐轮指标。
- `best_*.pt`：最佳验证精度 checkpoint。
- `latest_*.pt`：最近完整轮次 checkpoint，用于断点恢复。

完整输出和模型权重不提交到 GitHub。实验完成后，只把必要的 `args.json` 和 `history.csv` 整理到 `results/completed/<run>/`，并在该目录补充任务说明。

除本说明文件外，`outputs/` 下所有内容都由 `.gitignore` 排除。
