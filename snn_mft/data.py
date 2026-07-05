"""遥感场景分类数据集辅助函数。

脚本使用 torchvision 的 ImageFolder 数据格式。你可以显式提供 train/val 文件夹：

    data_root/train/class_name/*.jpg
    data_root/val/class_name/*.jpg

也可以只提供一个包含类别子文件夹的目录：

    data_root/class_name/*.jpg

第二种情况下，代码会按类别做分层随机划分，对应论文中的 80/20 和 50/50 划分协议。
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def build_transforms(image_size: int = 224) -> tuple[transforms.Compose, transforms.Compose]:
    """构建训练和验证图像预处理流程。

    参数
    ----------
    image_size:
        输入模型前裁剪到的正方形边长。论文复现实验中使用 224。
    """

    # 训练阶段加入常见遥感图像增强，对应论文里的翻转、旋转和样本扩展思想。
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.ColorJitter(brightness=0.25, contrast=0.25),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    # 验证阶段只做确定性 resize/center crop，保证评估结果可重复。
    eval_transform = transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return train_transform, eval_transform


def stratified_indices(targets: list[int], train_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    """按类别分层划分训练集和验证集索引。

    参数
    ----------
    targets:
        ImageFolder 生成的类别编号列表。
    train_ratio:
        每个类别进入训练集的比例，例如 0.8 对应论文的 80/20 划分。
    seed:
        控制每个类别内部打乱顺序的随机种子。
    """

    by_class: dict[int, list[int]] = defaultdict(list)
    for index, target in enumerate(targets):
        by_class[int(target)].append(index)

    rng = random.Random(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    for _, indices in sorted(by_class.items()):
        rng.shuffle(indices)
        # 每个类别都独立计算切分点，避免类别不均衡时某些类只落在训练或验证中。
        split = max(1, int(round(len(indices) * train_ratio)))
        if split >= len(indices) and len(indices) > 1:
            split = len(indices) - 1
        train_indices.extend(indices[:split])
        val_indices.extend(indices[split:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def make_dataloaders(
    data_root: str | Path,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 0,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, list[str]]:
    """创建训练/验证 DataLoader。

    参数
    ----------
    data_root:
        ImageFolder 数据根目录，可以是 class 子目录结构，也可以包含 train/val 子目录。
    image_size:
        输入图像尺寸。
    batch_size:
        每个 batch 的样本数。
    num_workers:
        DataLoader 子进程数量。macOS/MPS 上为了稳定默认使用 0。
    train_ratio:
        当数据没有显式 train/val 目录时使用的训练集比例。
    seed:
        分层划分随机种子。
    """

    data_root = Path(data_root)
    train_transform, eval_transform = build_transforms(image_size)

    train_dir = data_root / "train"
    val_dir = data_root / "val"
    test_dir = data_root / "test"

    if train_dir.exists() and (val_dir.exists() or test_dir.exists()):
        # 用户已经准备好显式划分时，优先尊重磁盘上的 train/val 或 train/test 结构。
        eval_dir = val_dir if val_dir.exists() else test_dir
        train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
        val_dataset = datasets.ImageFolder(eval_dir, transform=eval_transform)
        class_names = train_dataset.classes
    else:
        # 只有 class 子目录时，按类别分层随机切分，复现论文的 80/20、50/50 协议。
        train_base = datasets.ImageFolder(data_root, transform=train_transform)
        eval_base = datasets.ImageFolder(data_root, transform=eval_transform)
        train_indices, val_indices = stratified_indices(train_base.targets, train_ratio=train_ratio, seed=seed)
        train_dataset = Subset(train_base, train_indices)
        val_dataset = Subset(eval_base, val_indices)
        class_names = train_base.classes

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return train_loader, val_loader, class_names
