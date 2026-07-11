# 数据集准备与校验

训练脚本使用 `torchvision.datasets.ImageFolder`。数据集均保存在本地 `data/` 目录，该目录已被 `.gitignore` 排除，不上传到 GitHub。

## 当前状态

| 数据集 | 类别数 | 图片数 | 本地训练路径 | 状态 |
| --- | ---: | ---: | --- | --- |
| UCMerced LandUse | 21 | 2,100 | `data/processed/UCMerced_LandUse/UCMerced_LandUse/Images` | 已校验 |
| RSSCN7 | 7 | 2,800 | `data/processed/RSSCN7/RSSCN7-master` | 已校验 |
| AID | 30 | 10,000 | `data/processed/AID` | 已校验，0 个损坏文件 |

## 目录格式

每个数据根目录应直接包含类别子目录：

```text
data_root/
├── class_a/
│   ├── image_001.jpg
│   └── ...
├── class_b/
└── ...
```

`train.py` 会按类别分层随机划分训练集和验证集。也可以显式提供：

```text
data_root/
├── train/class_name/*.jpg
└── val/class_name/*.jpg
```

## UCMerced LandUse

- 原始压缩包：`data/raw/UCMerced_LandUse.zip`
- 每类图片数：100
- 来源：TorchGeo/Hugging Face 镜像
- 原始数据集：UC Merced Land Use Dataset

## RSSCN7

- 原始压缩包：`data/raw/RSSCN7-master.zip`
- 每类图片数：400
- 来源：公开 GitHub 镜像 `palewithout/RSSCN7`

## AID

- 原始压缩包：`data/raw/AID_scene_classification.zip`
- 官方页面：`https://captain-whu.github.io/AID/`
- OneDrive：`https://1drv.ms/u/s!AthY3vMZmuxChNR0Co7QHpJ56M-SvQ`
- BaiduPan：`https://pan.baidu.com/s/1mifOBv6`
- 本次使用的公开镜像：`https://www.kaggle.com/datasets/jiayuanchengala/aid-scene-classification-datasets`
- MD5：`d0e4f31114c7b5c377d629a0016c5470`
- SHA-256：`efd8dfb3ef38b7306f5d94c59d26e55d1cea77096c6a8d79cd8bb14db333aa88`

AID 压缩包约 2.5 GB，解压后约 2.6 GB。由于体积较大，数据集本体不写入 Git 历史。重新准备环境时，应从上述入口下载并核对至少一种哈希值。

训练命令和断点恢复方式见仓库根目录的 [README](../README.md)。
