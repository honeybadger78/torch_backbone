import math
import torch.nn as nn


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(
        self,
        in_channel,
        out_channel,
        kernel=1,
        stride=1,
        padding=None,
        groups=1,
        dilation=1,
        act=True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channel,
            out_channel,
            kernel,
            stride,
            autopad(kernel, padding, dilation),
            groups=groups,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channel)
        self.act = (
            self.default_act
            if act is True
            else act
            if isinstance(act, nn.Module)
            else nn.Identity()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

    def simple_forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class DepthWiseConv(Conv):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel=1,
        stride=1,
        padding=None,
        dilation=1,
        act=True,
    ):
        super().__init__(
            in_channel=in_channel,
            out_channel=out_channel,
            kernel=kernel,
            stride=stride,
            groups=math.gcd(in_channel, out_channel),
            padding=padding,
            dilation=dilation,
            act=act,
        )
