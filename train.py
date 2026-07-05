"""在 ImageFolder 数据集上训练 ANN 或 MFT SNN 模型。

典型 MFT 流程：

1. 训练普通 ANN ResNet。
2. 构建带 BM-IF 神经元的 SNN ResNet。
3. 把匹配的 ANN 权重复制到 SNN 中。
4. 使用替代梯度 STBP 继续训练 SNN。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from tqdm.auto import tqdm

from snn_mft.data import make_dataloaders
from snn_mft.models import build_ann_resnet, build_snn_resnet, copy_matching_state

HISTORY_FIELDNAMES = ["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc", "best_val_acc"]


def set_seed(seed: int) -> None:
    """固定 Python、NumPy 和 PyTorch 随机种子。

    参数
    ----------
    seed:
        本次实验的随机种子，影响数据划分、数据增强和模型初始化。
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    """根据用户请求选择训练设备。

    参数
    ----------
    requested:
        可选 auto、cpu、mps、cuda 或 cuda:0 等字符串。
    """

    if requested != "auto":
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("用户指定了 CUDA，但当前 PyTorch 没有可用 CUDA 设备。")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("用户指定了 MPS，但当前运行环境无法访问 Metal/MPS 设备。")
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    """从不同格式的 checkpoint 中取出模型权重字典。

    参数
    ----------
    checkpoint:
        torch.load 读取出的对象，可能是完整 checkpoint，也可能已经是 state_dict。
    """

    if isinstance(checkpoint, dict):
        for key in ("model_state", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError("checkpoint 中没有可用的 state dict。")


def accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    """计算一个 batch 的 top-1 准确率。"""

    prediction = logits.argmax(dim=1)
    return (prediction == target).float().mean().item()


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    show_progress: bool = True,
) -> tuple[float, float]:
    """运行一个训练或验证 epoch。

    参数
    ----------
    model:
        ANN 或 SNN 模型。
    loader:
        当前阶段的数据加载器。
    criterion:
        损失函数，默认训练脚本中使用交叉熵。
    device:
        训练设备。
    optimizer:
        传入优化器时表示训练阶段；为 None 时表示验证阶段。
    show_progress:
        是否显示 batch 级进度条。
    """

    is_train = optimizer is not None
    model.train(is_train)

    # 用样本数加权累计，避免最后一个 batch 较小时影响均值。
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    progress = tqdm(loader, desc="训练" if is_train else "验证", leave=False, disable=not show_progress)

    for images, labels in progress:
        images = images.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, labels)
            if is_train:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size
        if show_progress:
            progress.set_postfix(loss=total_loss / total_samples, acc=total_correct / total_samples)

    return total_loss / total_samples, total_correct / total_samples


def build_model(args: argparse.Namespace, num_classes: int) -> nn.Module:
    """根据命令行参数构建 ANN 或 SNN，并在需要时复制 ANN 权重。"""

    if args.model_type == "ann":
        return build_ann_resnet(
            depth=args.depth,
            num_classes=num_classes,
            pretrained_imagenet=args.pretrained_imagenet,
        )

    model = build_snn_resnet(
        depth=args.depth,
        num_classes=num_classes,
        time_steps=args.time_steps,
        threshold=args.threshold,
        alpha=args.alpha,
    )
    if args.ann_checkpoint:
        checkpoint = torch.load(args.ann_checkpoint, map_location="cpu")
        state_dict = extract_state_dict(checkpoint)
        copied, skipped = copy_matching_state(
            state_dict,
            model,
            include_classifier=args.copy_classifier,
        )
        print(f"已从 ANN checkpoint 复制 {len(copied)} 个张量。")
        print(f"跳过 {len(skipped)} 个张量。是否复制分类器: {args.copy_classifier}")
    else:
        print("没有提供 ANN checkpoint，SNN 将从随机权重开始训练。")
    return model


def save_checkpoint(
    path: Path,
    model: nn.Module,
    args: argparse.Namespace,
    class_names: list[str],
    epoch: int,
    val_acc: float,
) -> None:
    """保存训练 checkpoint。

    参数
    ----------
    path:
        checkpoint 文件路径。
    model:
        当前模型。
    args:
        本次运行的命令行参数。
    class_names:
        ImageFolder 类别名称，后续推理时用于还原标签。
    epoch:
        当前 epoch。
    val_acc:
        当前验证准确率。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "args": vars(args),
            "class_names": class_names,
            "epoch": epoch,
            "val_acc": val_acc,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    """解析训练脚本命令行参数。"""

    parser = argparse.ArgumentParser(description="MFT SNN 复现训练脚本")
    parser.add_argument("--data-root", required=True, help="ImageFolder 数据集路径")
    parser.add_argument("--output-dir", default="outputs/snn_mft_runs", help="checkpoint 保存目录")
    parser.add_argument("--run-name", default="", help="本次运行的目录名；为空时自动使用时间戳")
    parser.add_argument("--model-type", choices=["ann", "snn"], default="snn")
    parser.add_argument("--depth", type=int, choices=[18, 34], default=18)
    parser.add_argument("--time-steps", type=int, default=4, help="SNN 的 Repeat Encoding 时间窗口 T")
    parser.add_argument("--threshold", type=float, default=1.0, help="BM-IF 阈值 V_thr")
    parser.add_argument("--alpha", type=float, default=1.0, help="替代梯度窗口宽度")
    parser.add_argument("--ann-checkpoint", default="", help="用于 ANN -> SNN 权重复制的 ANN checkpoint")
    parser.add_argument("--copy-classifier", action="store_true", help="形状匹配时复制 fc.* 分类器参数")
    parser.add_argument("--pretrained-imagenet", action="store_true", help="ANN 使用 torchvision ImageNet 预训练权重")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--train-ratio", type=float, default=0.8, help="当 data-root 没有 train/val 划分时使用")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--scheduler-step", type=int, default=10)
    parser.add_argument("--scheduler-gamma", type=float, default=0.3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="可选 auto、cpu、mps、cuda、cuda:0 等")
    parser.add_argument("--disable-progress", action="store_true", help="关闭 batch 级进度条，只保留每轮指标")
    return parser.parse_args()


def main() -> None:
    """训练入口：准备数据、模型、优化器，并记录每轮指标。"""

    args = parse_args()
    set_seed(args.seed)
    device = select_device(args.device)
    print(f"使用设备: {device}")

    train_loader, val_loader, class_names = make_dataloaders(
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    print(f"类别数量 ({len(class_names)}): {class_names}")

    model = build_model(args, num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)

    run_name = args.run_name or datetime.now().strftime(f"{args.model_type}_resnet{args.depth}_%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "args.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")
    history_path = output_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=HISTORY_FIELDNAMES)
        writer.writeheader()
    print(f"输出目录: {output_dir}")

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f"\n第 {epoch}/{args.epochs} 轮")
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            show_progress=not args.disable_progress,
        )
        val_loss, val_acc = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            show_progress=not args.disable_progress,
        )
        scheduler.step()

        print(
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        if val_acc >= best_acc:
            best_acc = val_acc
            save_checkpoint(
                output_dir / f"best_{args.model_type}_resnet{args.depth}.pt",
                model,
                args,
                class_names,
                epoch,
                val_acc,
            )
            print(f"已保存最佳 checkpoint: val_acc={best_acc:.4f}")
        save_checkpoint(
            output_dir / f"latest_{args.model_type}_resnet{args.depth}.pt",
            model,
            args,
            class_names,
            epoch,
            val_acc,
        )
        with history_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=HISTORY_FIELDNAMES)
            writer.writerow(
                {
                    "epoch": epoch,
                    "lr": current_lr,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "best_val_acc": best_acc,
                }
            )


if __name__ == "__main__":
    main()
