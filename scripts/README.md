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
