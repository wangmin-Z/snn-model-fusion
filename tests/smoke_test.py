"""MFT SNN 复现代码的冒烟测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from snn_mft.bmif import BmIfActivation
from snn_mft.models import build_snn_resnet


def test_bmif_sequence() -> None:
    neuron = BmIfActivation(threshold=1.0, alpha=1.0, transmit_negative=True)
    currents = torch.tensor([0.4, 0.8, -2.4, 1.2])
    spikes = []
    for current in currents:
        spikes.append(float(neuron(current.view(1)).item()))

    assert spikes == [0.0, 1.0, -2.0, 1.0], spikes


def test_snn_resnet_forward_backward() -> None:
    torch.set_num_threads(1)
    model = build_snn_resnet(depth=18, num_classes=3, time_steps=2)
    model.train()

    images = torch.randn(2, 3, 64, 64)
    labels = torch.tensor([0, 1])
    logits = model(images)

    assert logits.shape == (2, 3), logits.shape
    loss = nn.CrossEntropyLoss()(logits, labels)
    loss.backward()
    assert model.conv1.weight.grad is not None


def main() -> None:
    test_bmif_sequence()
    test_snn_resnet_forward_backward()
    print("所有冒烟测试通过。")


if __name__ == "__main__":
    main()
