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

## 未自动下载

### AID

AID 官方页面可以访问，但官方下载入口是 OneDrive 和 BaiduPan：

- 官方页面：`https://captain-whu.github.io/AID/`
- OneDrive：`https://1drv.ms/u/s!AthY3vMZmuxChNR0Co7QHpJ56M-SvQ`
- BaiduPan：`https://pan.baidu.com/s/1mifOBv6`

脚本下载尝试结果：

- OneDrive `?download=1` 返回 `403 Forbidden`。
- OneDrive 分享 API 返回 `404 Not Found`。
- OneDrive 跳转诊断在 `onedrive.live.com` 上超时。

因此 AID 需要用浏览器或 BaiduPan 客户端手动下载，然后解压到：

```text
data/processed/AID/
```

解压后应保持 `ImageFolder` 结构：

```text
data/processed/AID/
  Airport/
  BareLand/
  ...
```
