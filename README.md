# Deep Spiking Neural Networks Based on Model Fusion Technology

这是对论文 `Deep spiking neural networks based on model fusion technology for remote sensing image classification` 的 PyTorch 复现骨架。

论文没有在 PDF 中提供完整源码，所以这里复现的是方法本身：BM-IF 脉冲神经元、S-ResNet18/S-ResNet34、Repeat Encoding、ANN -> SNN 权重拷贝、以及用替代梯度继续训练的 MFT 流程。

## 论文方法对应关系

MFT 的训练流程是：

1. 先训练普通 ANN ResNet。
2. 把 ReLU 激活替换成 BM-IF 脉冲神经元，直接复制能匹配的 ANN 权重。
3. 对 SNN 做短时间窗训练，使用 STBP/替代梯度继续优化。
4. 输出层不发放脉冲，只累积膜电位/分类 logits。

BM-IF 神经元的核心逻辑：

```text
membrane = membrane + current
spike = floor(membrane / V_thr)              if membrane >= V_thr
spike = -floor(-membrane / V_thr)            if membrane <= -V_thr
spike = 0                                    otherwise
membrane = membrane - spike * V_thr
```

论文中 ReLU 斜率 `k = 1`，因此转换时使用 `V_thr = 1`。替代梯度窗口使用 `alpha = 1`。

## 文件结构

```text
snn-model-fusion/
  snn_mft/
    bmif.py        # BM-IF 神经元和替代梯度
    models.py      # ANN ResNet 与 S-ResNet18/S-ResNet34
    data.py        # ImageFolder 数据集读取与 80/20、50/50 划分
  train.py         # ANN 训练和 SNN/MFT 训练入口
  demo_bmif.py     # BM-IF 单神经元演示
  tests/
    smoke_test.py  # 随机数据前向/反向测试
```

## 快速验证

在项目根目录运行：

```bash
cd /Users/zhuwangfeng/Documents/Study/deep-learning-study/snn-model-fusion
../.venv/bin/python tests/smoke_test.py
../.venv/bin/python demo_bmif.py
```

## 数据集格式

训练脚本使用 `torchvision.datasets.ImageFolder`。可以直接准备成：

```text
UCM/
  airplane/
  beach/
  ...
```

脚本会按类别分层随机划分训练集和验证集。也可以手动准备：

```text
UCM/
  train/class_name/*.jpg
  val/class_name/*.jpg
```

## 训练 ANN

```bash
cd /Users/zhuwangfeng/Documents/Study/deep-learning-study/snn-model-fusion
../.venv/bin/python train.py \
  --model-type ann \
  --depth 18 \
  --data-root /path/to/UCM \
  --train-ratio 0.8 \
  --epochs 100 \
  --batch-size 32 \
  --lr 0.1
```

## 训练 MFT/SNN

把上一步保存的 ANN checkpoint 传给 SNN：

```bash
cd /Users/zhuwangfeng/Documents/Study/deep-learning-study/snn-model-fusion
../.venv/bin/python train.py \
  --model-type snn \
  --depth 18 \
  --time-steps 4 \
  --data-root /path/to/UCM \
  --ann-checkpoint outputs/snn_mft_runs/ann_resnet18_YYYYMMDD_HHMMSS/best_ann_resnet18.pt \
  --train-ratio 0.8 \
  --epochs 30 \
  --batch-size 16 \
  --lr 0.1
```

论文中的遥感迁移训练参数大致是：

- optimizer: `SGD + momentum`
- momentum: `0.9`
- lr: `0.1`
- lr scheduler: 每 10 个 epoch 乘以 `0.3`
- weight decay: `0`
- max epoch: `120`
- time window: `T = 2, 4, 6, 8`

## 复现限制

这份代码可以复现论文的算法结构和训练入口，但不能保证直接复现论文表格中的 SOTA 数字。论文实验依赖 ImageNet-1K 预训练、UCM/RSSCN7/AID 原始数据、数据增强细节、随机种子平均 5 次实验，以及 8 张 RTX3090 的训练环境。
