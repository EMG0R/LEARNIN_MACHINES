"""YOLO-flavored CSPDarknet-lite backbone + U-Net-style FPN decoder
for 24-channel semantic segmentation (23 classes + background).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_act(ci, co, k=3, s=1, p=None):
    if p is None:
        p = k // 2
    return nn.Sequential(
        nn.Conv2d(ci, co, k, s, p, bias=False),
        nn.BatchNorm2d(co),
        nn.SiLU(inplace=True),
    )


class Bottleneck(nn.Module):
    """Residual bottleneck used inside C3."""
    def __init__(self, c, shortcut=True):
        super().__init__()
        self.conv1 = conv_bn_act(c, c, k=1)
        self.conv2 = conv_bn_act(c, c, k=3)
        self.add = shortcut

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return x + y if self.add else y


class C3(nn.Module):
    """CSP bottleneck with 3 convolutions (YOLOv5/v8-style).
    Splits via two 1x1 convs, runs n bottlenecks on one branch,
    concats with the other branch, fuses with a final 1x1 conv.
    """
    def __init__(self, ci, co, n=1, shortcut=True):
        super().__init__()
        c_h = co // 2  # hidden channel count per branch
        self.cv1 = conv_bn_act(ci, c_h, k=1)
        self.cv2 = conv_bn_act(ci, c_h, k=1)
        self.m   = nn.Sequential(*[Bottleneck(c_h, shortcut=shortcut) for _ in range(n)])
        self.cv3 = conv_bn_act(2 * c_h, co, k=1)

    def forward(self, x):
        a = self.m(self.cv1(x))
        b = self.cv2(x)
        return self.cv3(torch.cat([a, b], dim=1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (YOLOv5/v8). Three series 5x5
    maxpools form a multi-scale receptive field, concatenated and fused.
    """
    def __init__(self, ci, co, k=5):
        super().__init__()
        c_h = ci // 2
        self.cv1 = conv_bn_act(ci, c_h, k=1)
        self.cv2 = conv_bn_act(c_h * 4, co, k=1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


class Backbone(nn.Module):
    """CSPDarknet-lite. Stem + 4 stages. Returns 4 feature pyramid levels.

    Input:  (B, 3, 320, 320)
    Output: (P2, P3, P4, P5) at strides 4, 8, 16, 32 with channels 64,128,256,512.
    """
    def __init__(self):
        super().__init__()
        self.stem = conv_bn_act(3, 32, k=3, s=2)        # 320 -> 160

        self.s1_down = conv_bn_act(32,  64,  k=3, s=2)  # 160 -> 80
        self.s1_c3   = C3(64,  64,  n=1)

        self.s2_down = conv_bn_act(64,  128, k=3, s=2)  # 80 -> 40
        self.s2_c3   = C3(128, 128, n=2)

        self.s3_down = conv_bn_act(128, 256, k=3, s=2)  # 40 -> 20
        self.s3_c3   = C3(256, 256, n=3)

        self.s4_down = conv_bn_act(256, 512, k=3, s=2)  # 20 -> 10
        self.s4_c3   = C3(512, 512, n=1)
        self.s4_sppf = SPPF(512, 512, k=5)

    def forward(self, x):
        x = self.stem(x)
        p2 = self.s1_c3(self.s1_down(x))            # 80x80
        p3 = self.s2_c3(self.s2_down(p2))           # 40x40
        p4 = self.s3_c3(self.s3_down(p3))           # 20x20
        p5 = self.s4_sppf(self.s4_c3(self.s4_down(p4)))  # 10x10
        return p2, p3, p4, p5
