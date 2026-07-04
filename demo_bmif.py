"""小型 BM-IF 神经元演示。

运行：
    ../.venv/bin/python demo_bmif.py
"""

from __future__ import annotations

from pathlib import Path

import torch

from snn_mft.bmif import BmIfActivation

PROJECT_ROOT = Path(__file__).resolve().parent


def points_to_svg_polyline(values: list[float], color: str, width: int, height: int) -> str:
    left, right, top, bottom = 48, 24, 18, 34
    min_y, max_y = -3.0, 3.0
    plot_w = width - left - right
    plot_h = height - top - bottom

    def sx(index: int) -> float:
        if len(values) == 1:
            return left + plot_w / 2
        return left + index * plot_w / (len(values) - 1)

    def sy(value: float) -> float:
        ratio = (value - min_y) / (max_y - min_y)
        return top + (1.0 - ratio) * plot_h

    points = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(values))
    return f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>'


def save_svg(currents: list[float], spikes: list[float], membranes: list[float], path: Path) -> None:
    width, height = 760, 360
    left, right, top, bottom = 48, 24, 18, 34
    plot_w = width - left - right
    plot_h = height - top - bottom

    def sy(value: float) -> float:
        min_y, max_y = -3.0, 3.0
        ratio = (value - min_y) / (max_y - min_y)
        return top + (1.0 - ratio) * plot_h

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#333" stroke-width="1"/>
  <line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#333" stroke-width="1"/>
  <line x1="{left}" y1="{sy(1.0):.1f}" x2="{left + plot_w}" y2="{sy(1.0):.1f}" stroke="#999" stroke-dasharray="5 5"/>
  <line x1="{left}" y1="{sy(-1.0):.1f}" x2="{left + plot_w}" y2="{sy(-1.0):.1f}" stroke="#999" stroke-dasharray="5 5"/>
  <text x="8" y="{sy(1.0) + 4:.1f}" font-size="12" fill="#555">+Vthr</text>
  <text x="8" y="{sy(-1.0) + 4:.1f}" font-size="12" fill="#555">-Vthr</text>
  {points_to_svg_polyline(currents, "#2f6fdd", width, height)}
  {points_to_svg_polyline(spikes, "#d1495b", width, height)}
  {points_to_svg_polyline(membranes, "#2a9d8f", width, height)}
  <text x="56" y="330" font-size="13" fill="#2f6fdd">输入电流</text>
  <text x="190" y="330" font-size="13" fill="#d1495b">脉冲强度</text>
  <text x="340" y="330" font-size="13" fill="#2a9d8f">复位后膜电位</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def main() -> None:
    currents = torch.tensor([0.4, 0.8, -2.4, 1.2, 2.7, -0.6, -1.1])
    neuron = BmIfActivation(threshold=1.0, alpha=1.0, transmit_negative=True)

    membranes: list[float] = []
    spikes: list[float] = []
    for current in currents:
        spike = neuron(current.view(1))
        spikes.append(float(spike.item()))
        membranes.append(float(neuron.mem.item()))

    print("t | 输入电流 | 脉冲强度 | 复位后膜电位")
    print("--|---------------|----------------|---------------------")
    for i, (current, spike, membrane) in enumerate(zip(currents.tolist(), spikes, membranes), start=1):
        print(f"{i:1d} | {current:13.2f} | {spike:14.2f} | {membrane:19.2f}")

    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    fig_path = output_dir / "bmif_demo.svg"

    steps = list(range(1, len(currents) + 1))
    _ = steps
    save_svg(currents.tolist(), spikes, membranes, fig_path)
    print(f"\n已保存图像: {fig_path.resolve()}")


if __name__ == "__main__":
    main()
