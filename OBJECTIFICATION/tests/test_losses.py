import torch
from OBJECTIFICATION.seg.losses import dice_loss, ce_dice_loss


def test_dice_perfect_prediction_is_near_zero():
    target = torch.tensor([[[0, 1], [1, 0]]], dtype=torch.long)
    logits = torch.zeros(1, 2, 2, 2)
    logits[0, 0, 0, 0] = 10; logits[0, 0, 1, 1] = 10
    logits[0, 1, 0, 1] = 10; logits[0, 1, 1, 0] = 10
    loss = dice_loss(logits, target, num_classes=2)
    assert loss.item() < 0.05


def test_dice_inverted_prediction_is_near_one():
    target = torch.tensor([[[0, 1], [1, 0]]], dtype=torch.long)
    logits = torch.zeros(1, 2, 2, 2)
    logits[0, 1, 0, 0] = 10; logits[0, 1, 1, 1] = 10
    logits[0, 0, 0, 1] = 10; logits[0, 0, 1, 0] = 10
    loss = dice_loss(logits, target, num_classes=2)
    assert loss.item() > 0.9


def test_combined_loss_finite_random():
    torch.manual_seed(0)
    logits = torch.randn(2, 24, 32, 32)
    target = torch.randint(0, 24, (2, 32, 32), dtype=torch.long)
    loss = ce_dice_loss(logits, target, num_classes=24)
    assert torch.isfinite(loss).all()


def test_ignore_index_excluded_from_loss():
    """Pixels with ignore_index=255 contribute neither to CE nor Dice."""
    logits = torch.randn(1, 24, 4, 4)
    t_full = torch.zeros(1, 4, 4, dtype=torch.long)
    t_partial = t_full.clone()
    t_partial[0, 0, 0] = 255
    l1 = ce_dice_loss(logits, t_full, num_classes=24)
    l2 = ce_dice_loss(logits, t_partial, num_classes=24)
    assert l1.item() != l2.item()
    assert torch.isfinite(l2).all()


def test_class_weights_scale_loss():
    torch.manual_seed(0)
    logits = torch.randn(1, 24, 8, 8)
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    target[0, 0, 0] = 1
    w_uniform = torch.ones(24)
    w_boosted = torch.ones(24); w_boosted[1] = 100.0
    l_uni  = ce_dice_loss(logits, target, num_classes=24, class_weights=w_uniform)
    l_boost = ce_dice_loss(logits, target, num_classes=24, class_weights=w_boosted)
    assert l_boost.item() > l_uni.item()
