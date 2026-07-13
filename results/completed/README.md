# 已完成实验索引

本目录归档已训练满 120 轮的轻量实验结果。每个子目录的 `README.md` 说明完成任务、训练配置和关键指标。

| 数据集 | 模型 | 时间步 | 最佳轮次 | 最佳验证准确率 | 实验目录 |
| --- | --- | ---: | ---: | ---: | --- |
| RSSCN7 | ANN ResNet-18 | - | 88 | 75.36% | [`rsscn7_80_ann_resnet18_img224_e120_seed42/`](rsscn7_80_ann_resnet18_img224_e120_seed42/) |
| RSSCN7 | SNN-MFT ResNet-18 | 4 | 95 | 72.50% | [`rsscn7_80_snn_mft_resnet18_T4_img224_e120_seed42/`](rsscn7_80_snn_mft_resnet18_T4_img224_e120_seed42/) |
| UCMerced | ANN ResNet-18 | - | 89 | 66.67% | [`ucm80_ann_resnet18_img224_e120_seed42/`](ucm80_ann_resnet18_img224_e120_seed42/) |
| UCMerced | SNN-MFT ResNet-18 | 2 | 87 | 46.90% | [`ucm80_snn_mft_resnet18_T2_img224_e120_seed42/`](ucm80_snn_mft_resnet18_T2_img224_e120_seed42/) |
| UCMerced | SNN-MFT ResNet-18 | 4 | 44 | 56.43% | [`ucm80_snn_mft_resnet18_T4_img224_e120_seed42/`](ucm80_snn_mft_resnet18_T4_img224_e120_seed42/) |
| UCMerced | SNN-MFT ResNet-18 | 6 | 101 | 52.86% | [`ucm80_snn_mft_resnet18_T6_img224_e120_seed42/`](ucm80_snn_mft_resnet18_T6_img224_e120_seed42/) |
| UCMerced | SNN-MFT ResNet-18 | 8 | 118 | 49.05% | [`ucm80_snn_mft_resnet18_T8_img224_e120_seed42/`](ucm80_snn_mft_resnet18_T8_img224_e120_seed42/) |
| RSSCN7 | SNN-MFT ResNet-18 | 2 | 102 | 73.57% | [`rsscn7_80_snn_mft_resnet18_T2_img224_e120_seed42/`](rsscn7_80_snn_mft_resnet18_T2_img224_e120_seed42/) |

这里只保留可审查的配置和逐轮指标，不保留数据集或 checkpoint。
