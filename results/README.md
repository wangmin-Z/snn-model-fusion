# 可复现实验结果

`completed/` 保存已经完成 120 轮训练的轻量结果文件：

- `history.csv`：每轮学习率、训练损失、训练准确率、验证损失、验证准确率和历史最佳准确率。
- `args.json`：最后一次启动或断点恢复时使用的命令行参数。

模型 checkpoint 和数据集不直接写入 Git 历史。AID 完整数据通过仓库的 `aid-dataset-v1` Release 发布，其他数据来源及校验信息见 `DATASETS.md`。

汇总指标见仓库根目录的 `RESULTS.md`。仍在运行或未完成 120 轮的实验不会放入 `completed/`。
