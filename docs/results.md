# 复现实验记录

本文件记录当前机器上已经完整跑完的论文相关复现实验。大文件数据集、训练输出和 checkpoint 已通过 `.gitignore` 排除，不提交到代码仓库。

## 环境

- 设备: Apple MPS
- 当前可用本地 PyTorch: `torch==2.7.1`, `torchvision==0.22.1`
- 数据集: UCMerced LandUse, RSSCN7, AID
- 划分: 80% 训练 / 20% 验证
- 主干: ResNet-18
- 图像尺寸: 224
- 完整训练轮数: 120
- 优化器设置: SGD, `lr=0.1`, `momentum=0.9`, `weight_decay=0`
- 学习率衰减: 每 10 轮乘以 `0.3`
- 随机种子: 42

## 已完成结果

| 实验 | 输出目录 | 最佳轮数 | 最佳验证准确率 | 第 120 轮验证准确率 |
| --- | --- | ---: | ---: | ---: |
| UCM ANN ResNet-18 | `outputs/paper_full_120_20260704/ucm80_ann_resnet18_img224_e120_seed42` | 89 | 66.67% | 64.76% |
| UCM SNN-MFT ResNet-18, `T=2` | `outputs/paper_full_120_20260705/ucm80_snn_mft_resnet18_T2_img224_e120_seed42` | 87 | 46.90% | 43.57% |
| UCM SNN-MFT ResNet-18, `T=4` | `outputs/paper_full_120_20260704/ucm80_snn_mft_resnet18_T4_img224_e120_seed42` | 44 | 56.43% | 54.05% |
| UCM SNN-MFT ResNet-18, `T=6` | `outputs/paper_full_120_20260705/ucm80_snn_mft_resnet18_T6_img224_e120_seed42` | 101 | 52.86% | 45.95% |
| UCM SNN-MFT ResNet-18, `T=8` | `outputs/paper_full_120_20260706/ucm80_snn_mft_resnet18_T8_img224_e120_seed42` | 118 | 49.05% | 46.19% |
| RSSCN7 ANN ResNet-18 | `outputs/paper_full_120_20260704/rsscn7_80_ann_resnet18_img224_e120_seed42` | 88 | 75.36% | 71.96% |
| RSSCN7 SNN-MFT ResNet-18, `T=4` | `outputs/paper_full_120_20260704/rsscn7_80_snn_mft_resnet18_T4_img224_e120_seed42` | 95 | 72.50% | 70.36% |
| RSSCN7 SNN-MFT ResNet-18, `T=2` | `outputs/paper_recommended_20260711/rsscn7_80_snn_mft_resnet18_T2_img224_e120_seed42` | 102 | 73.57% | 69.82% |
| RSSCN7 SNN-MFT ResNet-18, `T=6` | `outputs/paper_recommended_20260711/rsscn7_80_snn_mft_resnet18_T6_img224_e120_seed42` | 68 | 75.00% | 71.43% |
| RSSCN7 SNN-MFT ResNet-18, `T=8` | `outputs/paper_recommended_20260711/rsscn7_80_snn_mft_resnet18_T8_img224_e120_seed42` | 45 | 76.61% | 76.43% |

推荐实验队列由 `scripts/run_recommended_experiments.sh` 管理。仍在训练的实验不计入上面的完整 120 轮结果。

## Checkpoint 文件

- UCM ANN 最佳模型: `outputs/paper_full_120_20260704/ucm80_ann_resnet18_img224_e120_seed42/best_ann_resnet18.pt`
- UCM SNN `T=2` 最佳模型: `outputs/paper_full_120_20260705/ucm80_snn_mft_resnet18_T2_img224_e120_seed42/best_snn_resnet18.pt`
- UCM SNN `T=4` 最佳模型: `outputs/paper_full_120_20260704/ucm80_snn_mft_resnet18_T4_img224_e120_seed42/best_snn_resnet18.pt`
- UCM SNN `T=6` 最佳模型: `outputs/paper_full_120_20260705/ucm80_snn_mft_resnet18_T6_img224_e120_seed42/best_snn_resnet18.pt`
- UCM SNN `T=8` 最佳模型: `outputs/paper_full_120_20260706/ucm80_snn_mft_resnet18_T8_img224_e120_seed42/best_snn_resnet18.pt`
- RSSCN7 ANN 最佳模型: `outputs/paper_full_120_20260704/rsscn7_80_ann_resnet18_img224_e120_seed42/best_ann_resnet18.pt`
- RSSCN7 SNN 最佳模型: `outputs/paper_full_120_20260704/rsscn7_80_snn_mft_resnet18_T4_img224_e120_seed42/best_snn_resnet18.pt`

## 说明

这不是论文完整实验矩阵。论文还包含多个训练划分、不同深度网络和多随机种子。当前仓库已经实现可运行的复现代码，完成了上表所列实验，并已准备 AID 数据集和后续自动训练队列。

在 Codex 沙箱内，系统 Metal 设备不可见，`torch.backends.mps.is_available()` 会返回 `False`。正式 MPS 训练需要在沙箱外执行；沙箱外验证结果为 MPS 可用。
