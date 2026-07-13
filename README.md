# SNN Model Fusion 论文复现

本仓库复现论文 *Deep spiking neural networks based on model fusion technology for remote sensing image classification* 的核心方法，包括 BM-IF 脉冲神经元、S-ResNet18/S-ResNet34、Repeat Encoding、ANN 到 SNN 权重复制，以及使用替代梯度继续训练的 MFT 流程。

论文没有提供完整源码，因此本项目是依据论文公式、伪代码和实验设置完成的独立 PyTorch 实现。当前目标是先完成 Apple MPS 上的 ResNet-18、80/20 划分、单随机种子复现；它不能直接等同于论文使用 8 张 RTX 3090 和 5 次重复实验得到的 SOTA 数字。

## 快速导航

- [数据集准备与校验](docs/datasets.md)
- [已完成实验与指标](docs/results.md)
- [逐轮训练记录](results/README.md)
- [推荐实验队列](scripts/run_recommended_experiments.sh)
- [AI 协作规则](AGENTS.md)
- [贡献与提交规范](CONTRIBUTING.md)

## 方法流程

1. 使用 ImageNet 预训练权重初始化并训练普通 ANN ResNet。
2. 将 ReLU 替换为 BM-IF 脉冲神经元，复制名称和形状匹配的 ANN 权重。
3. 使用 Repeat Encoding 将同一输入运行 `T` 个时间步。
4. 通过 STBP 和替代梯度继续优化 SNN。
5. 输出层不发放脉冲，只累积膜电位或分类 logits。

BM-IF 神经元的核心更新为：

```text
membrane = membrane + current
spike = floor(membrane / V_thr)              if membrane >= V_thr
spike = -floor(-membrane / V_thr)            if membrane <= -V_thr
spike = 0                                    otherwise
membrane = membrane - spike * V_thr
```

论文设置中 `V_thr = 1`，替代梯度窗口宽度 `alpha = 1`。

## 项目结构

```text
snn-model-fusion/
├── .github/                         # GitHub Actions 和 PR 模板
├── data/                            # 本地数据放置约定，不跟踪数据本体
├── docs/                            # 数据集说明和实验汇总
├── outputs/                         # 本地训练输出约定，不跟踪权重
├── results/completed/               # 已完成实验的参数、指标和任务说明
├── scripts/                         # 可恢复的长时间实验队列
├── snn_mft/                         # BM-IF、数据加载和 S-ResNet 实现
├── tests/                           # 快速前向与反向测试
├── train.py                         # ANN 与 SNN 统一训练入口
├── demo_bmif.py                     # BM-IF 单神经元演示
├── AGENTS.md                       # Codex、Claude 等 AI 的统一协作规则
├── CLAUDE.md                       # Claude 的规则入口
├── GEMINI.md                       # Gemini 的规则入口
├── CONTRIBUTING.md                  # 开发、测试与提交规范
└── requirements.txt                 # Python 依赖
```

每个功能目录都有独立的 `README.md`，说明该目录完成的任务、文件边界和使用方法。数据集、模型权重、checkpoint 和完整 `outputs/` 不写入 Git 历史。

## 环境安装

推荐使用 Python 3.10：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

在 Apple Silicon Mac 上，训练命令使用 `--device mps`。其他环境可以改为 `cpu`、`cuda` 或 `cuda:0`。

## 快速验证

```bash
python tests/smoke_test.py
python demo_bmif.py
```

冒烟测试使用随机数据，不需要下载遥感数据集。

## 训练示例

先训练 ANN：

```bash
python train.py \
  --model-type ann \
  --depth 18 \
  --pretrained-imagenet \
  --data-root data/processed/UCMerced_LandUse/UCMerced_LandUse/Images \
  --output-dir outputs/examples \
  --run-name ucm_ann_resnet18 \
  --train-ratio 0.8 \
  --epochs 120 \
  --batch-size 8 \
  --device mps
```

再复制 ANN 权重并训练 SNN：

```bash
python train.py \
  --model-type snn \
  --depth 18 \
  --time-steps 4 \
  --data-root data/processed/UCMerced_LandUse/UCMerced_LandUse/Images \
  --ann-checkpoint outputs/examples/ucm_ann_resnet18/best_ann_resnet18.pt \
  --output-dir outputs/examples \
  --run-name ucm_snn_resnet18_t4 \
  --train-ratio 0.8 \
  --epochs 120 \
  --batch-size 8 \
  --device mps
```

从中断位置恢复：

```bash
python train.py \
  --model-type snn \
  --depth 18 \
  --time-steps 4 \
  --data-root /path/to/imagefolder \
  --resume-checkpoint /path/to/latest_snn_resnet18.pt \
  --epochs 120 \
  --device mps
```

依次完成当前推荐实验矩阵：

```bash
./scripts/run_recommended_experiments.sh --dry-run
./scripts/run_recommended_experiments.sh
```

先用 `--dry-run` 查看队列而不启动训练。正式运行时，脚本会优先读取每个实验的 `latest` checkpoint，中断后可直接再次运行。

## 论文实验设置

- 优化器：SGD + momentum
- 学习率：`0.1`
- momentum：`0.9`
- weight decay：`0`
- 学习率调度：每 10 轮乘以 `0.3`
- 最大轮数：120
- 时间窗口：`T = 2, 4, 6, 8`
- 数据划分：80/20 和 50/50

本仓库已完成和正在运行的具体实验以 [实验结果文档](docs/results.md) 为准。
