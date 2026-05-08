"""
5-class face-part U-Net. Same encoder/decoder shape as HAND_JOB/hand_seg's UNet,
with multi-class output (channels=5: background, eye_L, eye_R, mouth, face_skin).
"""
import torch
import torch.nn as nn

NUM_CLASSES = 5  # background, eye_L, eye_R, mouth, face_skin
CLASS_NAMES = ["background", "eye_L", "eye_R", "mouth", "face_skin"]


def _cb(ci, co):
    return nn.Sequential(
        nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
        nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
    )


class FacePartsUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = _cb(3, 32)
        self.d2 = _cb(32, 64)
        self.d3 = _cb(64, 128)
        self.d4 = _cb(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2); self.u3 = _cb(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2);  self.u2 = _cb(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, 2);   self.u1 = _cb(64, 32)
        self.out  = nn.Conv2d(32, NUM_CLASSES, 1)
        self.aux3 = nn.Conv2d(128, NUM_CLASSES, 1)
        self.aux2 = nn.Conv2d(64, NUM_CLASSES, 1)

    def forward(self, x):
        c1 = self.d1(x)
        c2 = self.d2(self.pool(c1))
        c3 = self.d3(self.pool(c2))
        c4 = self.d4(self.pool(c3))
        u3 = self.u3(torch.cat([self.up3(c4), c3], 1))
        u2 = self.u2(torch.cat([self.up2(u3), c2], 1))
        u1 = self.u1(torch.cat([self.up1(u2), c1], 1))
        main = self.out(u1)
        if self.training:
            return main, self.aux3(u3), self.aux2(u2)
        return main


def kaiming_init(model: nn.Module):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
