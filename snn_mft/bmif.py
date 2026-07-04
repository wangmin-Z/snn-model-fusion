"""模型融合论文中使用的 BM-IF 脉冲神经元。

论文提出了双侧多强度积分发放神经元。直观理解如下：

1. 膜电位会累加当前时刻输入。
2. 如果膜电位超过正阈值，就发放一个或多个正脉冲，并从膜电位中减去已经发放的部分。
3. 如果膜电位低于负阈值，就发放一个或多个负向复位脉冲，让膜电位回到接近 0 的状态。

前向传播使用整数脉冲强度。反向传播使用论文中的替代梯度窗口：

    H(u) = 1 / alpha * sign(|u - V_thr| < alpha)

在双侧情形下，对 +V_thr 和 -V_thr 附近都使用同样的窗口。
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class BmIfSpike(torch.autograd.Function):
    """带简单替代梯度的整数 BM-IF 脉冲函数。"""

    @staticmethod
    def forward(ctx, membrane: Tensor, threshold: Tensor, alpha: Tensor) -> Tensor:
        threshold = threshold.to(device=membrane.device, dtype=membrane.dtype)
        alpha = alpha.to(device=membrane.device, dtype=membrane.dtype)

        positive_strength = torch.floor(torch.clamp(membrane / threshold, min=0.0))
        negative_strength = -torch.floor(torch.clamp(-membrane / threshold, min=0.0))

        spike = torch.where(
            membrane >= threshold,
            positive_strength,
            torch.where(membrane <= -threshold, negative_strength, torch.zeros_like(membrane)),
        )
        ctx.save_for_backward(membrane, threshold, alpha)
        return spike

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None, None]:
        membrane, threshold, alpha = ctx.saved_tensors
        alpha = torch.clamp(alpha, min=torch.finfo(membrane.dtype).eps)

        near_positive_threshold = (membrane - threshold).abs() < alpha
        near_negative_threshold = (membrane + threshold).abs() < alpha
        surrogate = (near_positive_threshold | near_negative_threshold).to(membrane.dtype) / alpha

        return grad_output * surrogate, None, None


def bmif_spike(membrane: Tensor, threshold: float = 1.0, alpha: float = 1.0) -> Tensor:
    """应用 BM-IF 脉冲函数。"""

    threshold_t = torch.tensor(float(threshold), device=membrane.device, dtype=membrane.dtype)
    alpha_t = torch.tensor(float(alpha), device=membrane.device, dtype=membrane.dtype)
    return BmIfSpike.apply(membrane, threshold_t, alpha_t)


class BmIfActivation(nn.Module):
    """用于时间步仿真的有状态 BM-IF 激活层。

    参数
    ----------
    threshold:
        发放阈值 V_thr。论文推导中，当 ReLU 斜率 k = 1 时，V_thr = 1。
    alpha:
        替代梯度窗口宽度。论文中使用 alpha = 1。
    transmit_negative:
        如果为 False，负脉冲只用于复位内部膜电位，不传给下一层，这对应论文第 3.4 节的描述。
        如果为 True，则返回带符号脉冲，便于单独观察神经元行为。
    """

    def __init__(self, threshold: float = 1.0, alpha: float = 1.0, transmit_negative: bool = False) -> None:
        super().__init__()
        self.threshold = float(threshold)
        self.alpha = float(alpha)
        self.transmit_negative = bool(transmit_negative)
        self.mem: Tensor | None = None

    def reset_state(self) -> None:
        self.mem = None

    def forward(self, current: Tensor) -> Tensor:
        if self.mem is None or self.mem.shape != current.shape or self.mem.device != current.device:
            self.mem = torch.zeros_like(current)

        self.mem = self.mem + current
        signed_spike = bmif_spike(self.mem, threshold=self.threshold, alpha=self.alpha)
        self.mem = self.mem - signed_spike * self.threshold

        if self.transmit_negative:
            return signed_spike
        return signed_spike.clamp_min(0.0)

    def extra_repr(self) -> str:
        return (
            f"threshold={self.threshold}, alpha={self.alpha}, "
            f"transmit_negative={self.transmit_negative}"
        )


def reset_spiking_state(module: nn.Module) -> None:
    """重置模型中所有有状态脉冲激活层。"""

    for submodule in module.modules():
        if hasattr(submodule, "reset_state"):
            submodule.reset_state()
