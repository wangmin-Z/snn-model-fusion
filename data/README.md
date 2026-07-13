# 本地数据目录

本目录用于保存训练数据，不提交图片或压缩包到 GitHub。

推荐结构：

```text
data/
├── raw/          # 原始压缩包
└── processed/    # 解压并整理成 ImageFolder 的数据
```

已使用数据集、下载来源、图片数量和哈希值见 [`docs/datasets.md`](../docs/datasets.md)。

除本说明文件外，`data/` 下所有内容都由 `.gitignore` 排除。
