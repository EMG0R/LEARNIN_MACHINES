"""Per-class IoU and macro mIoU via a streaming confusion matrix."""
import torch


class ConfusionAccumulator:
    """Streaming confusion matrix. Rows = ground truth, cols = prediction."""
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.confusion = torch.zeros(num_classes, num_classes, dtype=torch.float64)

    @torch.no_grad()
    def update(self, logits, target, ignore_index: int = 255):
        """logits: (B, C, H, W); target: (B, H, W) long."""
        pred = logits.argmax(dim=1)
        valid = (target != ignore_index)
        t = target[valid].view(-1)
        p = pred[valid].view(-1)
        idx = t * self.num_classes + p
        binc = torch.bincount(idx, minlength=self.num_classes ** 2)
        self.confusion += binc.view(self.num_classes, self.num_classes).double().cpu()

    def reset(self):
        self.confusion.zero_()


def per_class_iou(confusion: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """confusion: (C, C) float. Returns (C,) IoU per class."""
    tp = confusion.diag()
    fp = confusion.sum(dim=0) - tp
    fn = confusion.sum(dim=1) - tp
    return ((tp + eps) / (tp + fp + fn + eps)).float()


def macro_miou(per_class: torch.Tensor, exclude_bg: bool = True) -> float:
    """Macro-averaged mIoU. Excludes class 0 (background) by default."""
    if exclude_bg:
        return per_class[1:].mean().item()
    return per_class.mean().item()
