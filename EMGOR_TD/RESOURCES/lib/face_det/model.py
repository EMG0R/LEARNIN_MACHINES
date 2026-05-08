"""
Anchor-free face detector, FCOS-style.

Backbone: 4 conv blocks (3,32 → 32,64 → 64,128 → 128,256), each with 2×(conv-bn-relu)
and a stride-2 MaxPool. After 3 pools total stride = 8.

Head: shared 3×(conv-bn-relu) stack at 256 channels, then three 1×1 convs producing
objectness (1 channel), bbox regression (4 channels: l, t, r, b in pixels from the
feature-map cell center), and centerness (1 channel).

bbox outputs are passed through exp() at inference; in training we predict raw and
the loss does the exp().
"""
import torch
import torch.nn as nn


def _conv_block(ci, co):
    return nn.Sequential(
        nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
        nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
    )


class FaceDetector(nn.Module):
    STRIDE = 8

    def __init__(self, backbone_ch=(32, 64, 128, 256), head_ch=128):
        super().__init__()
        c0, c1, c2, c3 = backbone_ch
        self.b1 = _conv_block(3,  c0)
        self.b2 = _conv_block(c0, c1)
        self.b3 = _conv_block(c1, c2)
        self.b4 = _conv_block(c2, c3)  # no pool after last block — we need stride 8 total
        self.pool = nn.MaxPool2d(2)

        self.head = nn.Sequential(
            nn.Conv2d(c3, head_ch, 3, padding=1), nn.BatchNorm2d(head_ch), nn.ReLU(inplace=True),
            nn.Conv2d(head_ch, head_ch, 3, padding=1), nn.BatchNorm2d(head_ch), nn.ReLU(inplace=True),
            nn.Conv2d(head_ch, head_ch, 3, padding=1), nn.BatchNorm2d(head_ch), nn.ReLU(inplace=True),
        )
        self.obj  = nn.Conv2d(head_ch, 1, 1)
        self.bbox = nn.Conv2d(head_ch, 4, 1)
        self.ctr  = nn.Conv2d(head_ch, 1, 1)

    def forward(self, x):
        # stride 2, 4, 8 → final feature map is input/8
        x = self.b1(x)
        x = self.b2(self.pool(x))
        x = self.b3(self.pool(x))
        x = self.b4(self.pool(x))
        h = self.head(x)
        return self.obj(h), self.bbox(h), self.ctr(h)


def kaiming_init(model: nn.Module):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
