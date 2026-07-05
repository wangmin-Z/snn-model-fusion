# 复现实验记录

本文件记录当前机器上已经完整跑完的论文相关复现实验。大文件数据集、训练输出和 checkpoint 已通过 `.gitignore` 排除，不提交到代码仓库。

## 环境

- 设备: Apple MPS
- 数据集: UCMerced LandUse, RSSCN7
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
| UCM SNN-MFT ResNet-18, `T=4` | `outputs/paper_full_120_20260704/ucm80_snn_mft_resnet18_T4_img224_e120_seed42` | 44 | 56.43% | 54.05% |
| RSSCN7 ANN ResNet-18 | `outputs/paper_full_120_20260704/rsscn7_80_ann_resnet18_img224_e120_seed42` | 88 | 75.36% | 71.96% |
| RSSCN7 SNN-MFT ResNet-18, `T=4` | `outputs/paper_full_120_20260704/rsscn7_80_snn_mft_resnet18_T4_img224_e120_seed42` | 95 | 72.50% | 70.36% |

## Checkpoint

- ANN 最佳模型: `outputs/paper_full_120_20260704/ucm80_ann_resnet18_img224_e120_seed42/best_ann_resnet18.pt`
- SNN 最佳模型: `outputs/paper_full_120_20260704/ucm80_snn_mft_resnet18_T4_img224_e120_seed42/best_snn_resnet18.pt`
- RSSCN7 ANN 最佳模型: `outputs/paper_full_120_20260704/rsscn7_80_ann_resnet18_img224_e120_seed42/best_ann_resnet18.pt`
- RSSCN7 SNN 最佳模型: `outputs/paper_full_120_20260704/rsscn7_80_snn_mft_resnet18_T4_img224_e120_seed42/best_snn_resnet18.pt`

## 说明

这不是论文完整实验矩阵。论文还包含 AID、多个训练划分、多个时间窗 `T=2/4/6/8`、不同深度网络和多随机种子。当前仓库已经实现可运行的复现代码，并完成 UCM 与 RSSCN7 80/20 子集的 120 轮 ANN 与 SNN-MFT 训练。
