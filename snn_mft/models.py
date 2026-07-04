"""用于 MFT 复现的 ANN 和 SNN ResNet 模型。

SNN 模型尽量沿用 torchvision ResNet 的参数命名。
这样 ANN -> SNN 转换会更简单：卷积层、批归一化层和分类器中形状匹配的权重可以按键名复制。
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, ResNet34_Weights, resnet18, resnet34

from .bmif import BmIfActivation, reset_spiking_state


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SpikingBasicBlock(nn.Module):
    """带 BM-IF 激活的 ResNet BasicBlock。"""

    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        threshold: float = 1.0,
        alpha: float = 1.0,
        transmit_negative: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.spike1 = BmIfActivation(threshold=threshold, alpha=alpha, transmit_negative=transmit_negative)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.spike2 = BmIfActivation(threshold=threshold, alpha=alpha, transmit_negative=transmit_negative)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.spike1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.spike2(out)
        return out


class SNNResNet(nn.Module):
    """带 Repeat Encoding 和 BM-IF 神经元的 ResNet-18/34 风格 SNN。"""

    def __init__(
        self,
        layers: list[int],
        num_classes: int,
        time_steps: int = 4,
        threshold: float = 1.0,
        alpha: float = 1.0,
        transmit_negative: bool = False,
    ) -> None:
        super().__init__()
        self.inplanes = 64
        self.time_steps = int(time_steps)
        self.threshold = float(threshold)
        self.alpha = float(alpha)

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.spike1 = BmIfActivation(threshold=threshold, alpha=alpha, transmit_negative=transmit_negative)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0], threshold, alpha, transmit_negative)
        self.layer2 = self._make_layer(128, layers[1], threshold, alpha, transmit_negative, stride=2)
        self.layer3 = self._make_layer(256, layers[2], threshold, alpha, transmit_negative, stride=2)
        self.layer4 = self._make_layer(512, layers[3], threshold, alpha, transmit_negative, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * SpikingBasicBlock.expansion, num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(
        self,
        planes: int,
        blocks: int,
        threshold: float,
        alpha: float,
        transmit_negative: bool,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * SpikingBasicBlock.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * SpikingBasicBlock.expansion, stride),
                nn.BatchNorm2d(planes * SpikingBasicBlock.expansion),
            )

        layers = [
            SpikingBasicBlock(
                self.inplanes,
                planes,
                stride=stride,
                downsample=downsample,
                threshold=threshold,
                alpha=alpha,
                transmit_negative=transmit_negative,
            )
        ]
        self.inplanes = planes * SpikingBasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(
                SpikingBasicBlock(
                    self.inplanes,
                    planes,
                    threshold=threshold,
                    alpha=alpha,
                    transmit_negative=transmit_negative,
                )
            )

        return nn.Sequential(*layers)

    def _forward_single_step(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.spike1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

    def forward(self, x: Tensor, time_steps: int | None = None) -> Tensor:
        """用 Repeat Encoding 运行 T 个时间步。

        输出层遵循论文设置：不发放脉冲，只累积膜电位/logits，最大值对应预测类别。
        """

        reset_spiking_state(self)
        steps = int(time_steps or self.time_steps)
        output_membrane: Tensor | None = None

        for _ in range(steps):
            current_logits = self._forward_single_step(x)
            if output_membrane is None:
                output_membrane = current_logits
            else:
                output_membrane = output_membrane + current_logits

        if output_membrane is None:
            raise ValueError("time_steps 必须大于等于 1")
        return output_membrane / float(steps)


def build_ann_resnet(depth: int, num_classes: int, pretrained_imagenet: bool = False) -> nn.Module:
    """构建 ANN -> SNN 转换前使用的 ANN ResNet。"""

    if depth == 18:
        weights = ResNet18_Weights.DEFAULT if pretrained_imagenet else None
        model = resnet18(weights=weights)
    elif depth == 34:
        weights = ResNet34_Weights.DEFAULT if pretrained_imagenet else None
        model = resnet34(weights=weights)
    else:
        raise ValueError("论文中只使用 ResNet-18 和 ResNet-34。")

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_snn_resnet(
    depth: int,
    num_classes: int,
    time_steps: int = 4,
    threshold: float = 1.0,
    alpha: float = 1.0,
    transmit_negative: bool = False,
) -> SNNResNet:
    if depth == 18:
        layers = [2, 2, 2, 2]
    elif depth == 34:
        layers = [3, 4, 6, 3]
    else:
        raise ValueError("论文中只使用 S-ResNet18 和 S-ResNet34。")
    return SNNResNet(
        layers=layers,
        num_classes=num_classes,
        time_steps=time_steps,
        threshold=threshold,
        alpha=alpha,
        transmit_negative=transmit_negative,
    )


def _clean_state_key(key: str) -> str:
    for prefix in ("module.", "model."):
        if key.startswith(prefix):
            key = key[len(prefix) :]
    return key


def copy_matching_state(
    source_state: dict[str, Any] | nn.Module,
    target_model: nn.Module,
    include_classifier: bool = False,
) -> tuple[list[str], list[str]]:
    """当参数名和形状匹配时，把 ANN 权重复制到 SNN 模型中。

    返回
    -------
    copied:
        已复制到目标模型中的参数键名。
    skipped:
        存在但没有复制的参数键名。通常表示分类器形状不同，或该键属于 ANN 独有模块。
    """

    if isinstance(source_state, nn.Module):
        raw_state = source_state.state_dict()
    else:
        raw_state = source_state

    source = OrderedDict((_clean_state_key(k), v) for k, v in raw_state.items())
    target = target_model.state_dict()
    copied: list[str] = []
    skipped: list[str] = []

    for key, value in source.items():
        if not include_classifier and key.startswith("fc."):
            skipped.append(key)
            continue
        if key in target and target[key].shape == value.shape:
            target[key] = value.detach().clone()
            copied.append(key)
        else:
            skipped.append(key)

    target_model.load_state_dict(target, strict=True)
    return copied, skipped
