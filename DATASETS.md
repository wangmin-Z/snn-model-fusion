# 数据集下载状态

## 已下载

### UCMerced LandUse

- 原始压缩包：`data/raw/UCMerced_LandUse.zip`
- 训练路径：`data/processed/UCMerced_LandUse/UCMerced_LandUse/Images`
- 类别数：21
- 图片数：2100
- 每类图片数：100
- 来源：TorchGeo/Hugging Face 镜像，原始数据集为 UC Merced Land Use Dataset。

训练示例：

```bash
../.venv/bin/python train.py \
  --model-type snn \
  --depth 18 \
  --time-steps 4 \
  --data-root data/processed/UCMerced_LandUse/UCMerced_LandUse/Images \
  --epochs 30 \
  --batch-size 16
```

### RSSCN7

- 原始压缩包：`data/raw/RSSCN7-master.zip`
- 训练路径：`data/processed/RSSCN7/RSSCN7-master`
- 类别数：7
- 图片数：2800
- 每类图片数：400
- 来源：公开 GitHub 镜像 `palewithout/RSSCN7`。

训练示例：

```bash
../.venv/bin/python train.py \
  --model-type snn \
  --depth 18 \
  --time-steps 4 \
  --data-root data/processed/RSSCN7/RSSCN7-master \
  --epochs 30 \
  --batch-size 16
```

### AID

AID 已下载并解压完成。官方页面提供 OneDrive 和 BaiduPan 下载入口：

- 官方页面：`https://captain-whu.github.io/AID/`
- OneDrive：`https://1drv.ms/u/s!AthY3vMZmuxChNR0Co7QHpJ56M-SvQ`
- BaiduPan：`https://pan.baidu.com/s/1mifOBv6`
- 本次使用的公开镜像：`https://www.kaggle.com/datasets/jiayuanchengala/aid-scene-classification-datasets`
- 原始压缩包：`data/raw/AID_scene_classification.zip`
- 训练路径：`data/processed/AID`
- 压缩包 MD5：`d0e4f31114c7b5c377d629a0016c5470`
- 压缩包 SHA-256：`efd8dfb3ef38b7306f5d94c59d26e55d1cea77096c6a8d79cd8bb14db333aa88`
- 校验结果：30 个类别、10,000 张图片、0 个损坏文件

数据集体积较大，不上传到 GitHub 仓库。需要重新准备环境时，请从上面的官方入口或公开镜像下载，并使用 MD5 或 SHA-256 校验压缩包。

官方下载尝试记录：

- OneDrive `?download=1` 返回 `403 Forbidden`。
- OneDrive 分享 API 返回 `404 Not Found`。
- OneDrive 跳转诊断在 `onedrive.live.com` 上超时。

数据已保持为 PyTorch `ImageFolder` 结构：

```text
data/processed/AID/
  Airport/
  BareLand/
  ...
```
