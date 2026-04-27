"""Loss functions for multi-class semantic segmentation.

Two combined losses available:
- ce_dice_loss: standard cross-entropy + Dice (used in v1-v3 — produced
  person IoU 0 across all attempts; person too rare for plain CE).
- focal_dice_loss: focal loss + Dice (Lin et al. 2017, RetinaNet; YOLOv5/v8
  use focal for cls). Down-weights easy / well-classified pixels via
  (1 - p_t)^gamma, focuses gradient on hard / misclassified pixels.
  This is the standard fix for "model ignores rare class" — those classes
  generate large (1 - p_t)^gamma multipliers because the model is
  consistently wrong on them, so their gradient stays loud.
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


def focal_loss(logits, target, alpha=None, gamma=2.0, ignore_index=IGNORE_INDEX):
    """Multi-class focal loss for dense prediction (semantic segmentation).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where p_t is the softmax probability of the true class for that pixel.

    logits:  (B, C, H, W)
    target:  (B, H, W) long
    alpha:   (C,) per-class weight, or None (uniform). Acts like the
             class_weights in CE: higher value -> more loss for that class.
    gamma:   focusing parameter. 0 => regular CE. Standard YOLO / RetinaNet
             value is 2.0.
    """
    valid = (target != ignore_index)
    target_clamped = target.clone()
    target_clamped[~valid] = 0

    log_probs = F.log_softmax(logits, dim=1)              # (B, C, H, W)
    log_p_t = log_probs.gather(1, target_clamped.unsqueeze(1)).squeeze(1)  # (B, H, W)
    p_t = log_p_t.exp()
    focal = (1.0 - p_t) ** gamma                          # (B, H, W)

    if alpha is not None:
        alpha_t = alpha[target_clamped]                   # (B, H, W)
        loss = -alpha_t * focal * log_p_t
    else:
        loss = -focal * log_p_t

    loss = loss * valid.float()
    n = valid.sum().clamp(min=1)
    return loss.sum() / n


def ce_dice_loss(logits, target, num_classes, class_weights=None,
                 ce_weight=0.5, dice_weight=0.5, ignore_index=IGNORE_INDEX):
    """Combined cross-entropy + Dice loss. Kept for backwards compat with v1-v3."""
    ce = F.cross_entropy(
        logits, target, weight=class_weights, ignore_index=ignore_index
    )
    dl = dice_loss(logits, target, num_classes=num_classes, ignore_index=ignore_index)
    return ce_weight * ce + dice_weight * dl


def focal_dice_loss(logits, target, num_classes, class_weights=None,
                    focal_weight=1.0, dice_weight=1.0, gamma=2.0,
                    ignore_index=IGNORE_INDEX):
    """Combined focal + Dice loss. v4+ default.

    Equal-weighted combo: focal handles per-pixel hard-example mining
    and class imbalance, Dice handles region-level overlap directly.
    """
    fl = focal_loss(logits, target, alpha=class_weights, gamma=gamma,
                    ignore_index=ignore_index)
    dl = dice_loss(logits, target, num_classes=num_classes,
                   ignore_index=ignore_index)
    return focal_weight * fl + dice_weight * dl
