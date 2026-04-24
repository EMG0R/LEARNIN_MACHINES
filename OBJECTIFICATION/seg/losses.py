"""Cross-entropy + multi-class Dice loss for semantic segmentation.

Combined loss: L = 0.5 * CE + 0.5 * Dice.
CE handles confidence; Dice handles small-object recall and class imbalance.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


IGNORE_INDEX = 255


def dice_loss(logits, target, num_classes, eps=1e-6, ignore_index=IGNORE_INDEX):
    """Multi-class soft Dice loss averaged over classes.

    logits: (B, C, H, W) raw scores
    target: (B, H, W) long, values in [0, C) or = ignore_index
    """
    valid = (target != ignore_index)
    target_clamped = target.clone()
    target_clamped[~valid] = 0
    one_hot = F.one_hot(target_clamped, num_classes=num_classes)  # (B, H, W, C)
    one_hot = one_hot.permute(0, 3, 1, 2).float()                  # (B, C, H, W)
    one_hot = one_hot * valid.unsqueeze(1).float()

    probs = F.softmax(logits, dim=1) * valid.unsqueeze(1).float()

    dims = (0, 2, 3)
    inter = (probs * one_hot).sum(dim=dims)
    denom = probs.sum(dim=dims) + one_hot.sum(dim=dims)
    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def ce_dice_loss(logits, target, num_classes, class_weights=None,
                 ce_weight=0.5, dice_weight=0.5, ignore_index=IGNORE_INDEX):
    """Combined cross-entropy + Dice loss."""
    ce = F.cross_entropy(
        logits, target, weight=class_weights, ignore_index=ignore_index
    )
    dl = dice_loss(logits, target, num_classes=num_classes, ignore_index=ignore_index)
    return ce_weight * ce + dice_weight * dl
