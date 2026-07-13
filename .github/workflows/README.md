# 自动检查任务

本目录包含 GitHub Actions 工作流。

`smoke-test.yml` 在推送和 Pull Request 时完成以下任务：

1. 创建 Python 3.10 环境。
2. 安装 `requirements.txt` 中的依赖。
3. 执行 BM-IF 和 S-ResNet 前向/反向冒烟测试。
4. 检查长训练脚本语法。
5. 使用 `--dry-run` 验证实验队列，不启动正式训练。

正式 120 轮训练不会在 GitHub Actions 中运行。
