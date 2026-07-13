#!/usr/bin/env python3
"""将已完成训练的轻量结果归档到仓库，并同步结果索引。"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
COMPLETED_ROOT = REPO_ROOT / "results" / "completed"
COMPLETED_INDEX = COMPLETED_ROOT / "README.md"
RESULTS_DOC = REPO_ROOT / "docs" / "results.md"


@dataclass(frozen=True)
class Metrics:
    best_epoch: int
    best_val_acc: float
    final_val_acc: float


def read_metrics(history_path: Path, expected_epochs: int) -> Metrics:
    with history_path.open(newline="", encoding="utf-8") as history_file:
        rows = list(csv.DictReader(history_file))

    if not rows:
        raise ValueError(f"训练记录为空: {history_path}")

    final_epoch = int(rows[-1]["epoch"])
    if final_epoch != expected_epochs:
        raise ValueError(
            f"训练尚未完成: 最后一轮为 {final_epoch}，期望为 {expected_epochs}。"
        )

    best_row = max(rows, key=lambda row: float(row["val_acc"]))
    return Metrics(
        best_epoch=int(best_row["epoch"]),
        best_val_acc=float(best_row["val_acc"]),
        final_val_acc=float(rows[-1]["val_acc"]),
    )


def insert_table_row(path: Path, header: str, row: str, duplicate: str) -> None:
    text = path.read_text(encoding="utf-8")
    if duplicate in text:
        raise ValueError(f"结果索引已包含: {duplicate}")

    lines = text.splitlines(keepends=True)
    try:
        header_index = next(index for index, line in enumerate(lines) if line.rstrip() == header)
    except StopIteration as exc:
        raise ValueError(f"未找到结果表头: {path}") from exc

    insert_index = header_index + 2
    while insert_index < len(lines) and lines[insert_index].lstrip().startswith("|"):
        insert_index += 1
    lines.insert(insert_index, row + "\n")
    path.write_text("".join(lines), encoding="utf-8")


def build_readme(args: argparse.Namespace, metrics: Metrics) -> str:
    time_steps = f"时间步 `T={args.time_steps}` 的 " if args.time_steps else ""
    model_label = f"{args.model}（T={args.time_steps}）" if args.time_steps else args.model
    return f"""# {args.dataset} {model_label}

## 完成任务

在 {args.dataset} 数据集上完成 {time_steps}{args.model} 的完整训练，用于论文复现结果对比。

- 状态：已完成 {args.expected_epochs}/{args.expected_epochs} 轮
- 配置：{args.train_ratio} 训练验证划分，图像尺寸 {args.image_size}，随机种子 {args.seed}，MPS 训练
- 最佳结果：第 {metrics.best_epoch} 轮，验证准确率 {metrics.best_val_acc:.2%}
- 最终结果：第 {args.expected_epochs} 轮，验证准确率 {metrics.final_val_acc:.2%}

`args.json` 保存实际训练参数，`history.csv` 保存逐轮训练与验证指标。本目录不包含数据集、模型权重或 checkpoint。
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="归档已完成实验，并同步 Markdown 结果索引。")
    parser.add_argument("--source-run", type=Path, required=True, help="本地 outputs 中的实验目录")
    parser.add_argument("--dataset", required=True, help="数据集显示名称，例如 RSSCN7")
    parser.add_argument("--model", required=True, help="模型显示名称，例如 SNN-MFT ResNet-18")
    parser.add_argument("--time-steps", type=int, help="SNN 时间步；ANN 实验不填写")
    parser.add_argument("--output-path", required=True, help="写入 docs/results.md 的本地输出目录")
    parser.add_argument("--expected-epochs", type=int, default=120)
    parser.add_argument("--train-ratio", default="80/20")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="只校验并打印指标，不写入文件")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_run = args.source_run.resolve()
    run_name = source_run.name
    args_path = source_run / "args.json"
    history_path = source_run / "history.csv"

    for required_path in (args_path, history_path):
        if not required_path.is_file():
            raise FileNotFoundError(f"缺少归档文件: {required_path}")

    metrics = read_metrics(history_path, args.expected_epochs)
    print(
        f"{run_name}: best epoch={metrics.best_epoch}, "
        f"best val={metrics.best_val_acc:.2%}, final val={metrics.final_val_acc:.2%}"
    )

    destination = COMPLETED_ROOT / run_name
    if destination.exists():
        raise FileExistsError(f"归档目录已存在，拒绝覆盖: {destination}")

    if args.dry_run:
        print("dry-run: 训练已完成，归档前置校验通过。")
        return

    destination.mkdir()
    shutil.copy2(args_path, destination / "args.json")
    shutil.copy2(history_path, destination / "history.csv")
    (destination / "README.md").write_text(build_readme(args, metrics), encoding="utf-8")

    time_steps = str(args.time_steps) if args.time_steps else "-"
    completed_row = (
        f"| {args.dataset} | {args.model} | {time_steps} | {metrics.best_epoch} | "
        f"{metrics.best_val_acc:.2%} | [`{run_name}/`]({run_name}/) |"
    )
    insert_table_row(COMPLETED_INDEX, "| 数据集 | 模型 | 时间步 | 最佳轮次 | 最佳验证准确率 | 实验目录 |", completed_row, run_name)

    experiment_label = f"{args.dataset} {args.model}"
    if args.time_steps:
        experiment_label += f", `T={args.time_steps}`"
    docs_row = (
        f"| {experiment_label} | `{args.output_path}` | {metrics.best_epoch} | "
        f"{metrics.best_val_acc:.2%} | {metrics.final_val_acc:.2%} |"
    )
    insert_table_row(
        RESULTS_DOC,
        "| 实验 | 输出目录 | 最佳轮数 | 最佳验证准确率 | 第 120 轮验证准确率 |",
        docs_row,
        f"`{args.output_path}`",
    )

    # 检查保存的参数仍然是可解析 JSON，避免把损坏的恢复记录提交到 Git。
    json.loads((destination / "args.json").read_text(encoding="utf-8"))
    print(f"归档完成: {destination.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
