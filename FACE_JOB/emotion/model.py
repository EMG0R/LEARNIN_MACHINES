"""
Emotion classifier: 4-block wide CNN. Identical shape to HAND_JOB's gesture Wide CNN
with FC final layer resized to 7 classes.
"""
import torch
import torch.nn as nn

NUM_CLASSES = 7  # happy, sad, neutral, surprise, anger, fear, disgust
CLASS_NAMES = ["happy", "sad", "neutral", "surprise", "anger", "fear", "disgust"]


class EmotionWide(nn.Module):
    def __init__(self, n=NUM_CLASSES):
        super().__init__()
        def block(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                nn.Dropout2d(0.1), nn.MaxPool2d(2),
            )
        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 256))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(256, 512), nn.ReLU(inplace=True), nn.Dropout(0.4), nn.Linear(512, n),
        )

    def forward(self, x):
        return self.head(self.features(x))


def kaiming_init(model: nn.Module):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)
