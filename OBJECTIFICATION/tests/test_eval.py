import torch
from OBJECTIFICATION.seg.eval import ConfusionAccumulator, per_class_iou, macro_miou


def test_perfect_prediction_iou_is_one():
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    target[0, :4, :] = 1
    target[0, 4:, :] = 2
    logits = torch.zeros(1, 3, 8, 8)
    logits[0, 0, :, :] = 1.0
    logits[0, 1, :4, :] = 5.0
    logits[0, 2, 4:, :] = 5.0
    acc = ConfusionAccumulator(num_classes=3)
    acc.update(logits, target)
    iou = per_class_iou(acc.confusion)
    assert torch.allclose(iou, torch.tensor([1.0, 1.0, 1.0]))


def test_zero_overlap_iou_is_zero():
    target = torch.zeros(1, 4, 4, dtype=torch.long)
    target[0, :, :2] = 1
    logits = torch.full((1, 2, 4, 4), -1.0)
    logits[0, 1, :, 2:] = 10.0
    logits[0, 0, :, :2] = 10.0
    acc = ConfusionAccumulator(num_classes=2)
    acc.update(logits, target)
    iou = per_class_iou(acc.confusion)
    assert iou[1].item() < 1e-6  # eps=1e-9 in per_class_iou makes it ~6e-11, not exactly 0


def test_macro_miou_excludes_background():
    confusion = torch.tensor([
        [10, 0,  0],
        [0,  5,  5],
        [0,  10, 0],
    ], dtype=torch.float32)
    iou = per_class_iou(confusion)
    miou = macro_miou(iou, exclude_bg=True)
    expected = (0.25 + 0.0) / 2
    assert abs(miou - expected) < 1e-6


def test_accumulator_handles_ignore_index():
    target = torch.zeros(1, 4, 4, dtype=torch.long)
    target[0, 0, 0] = 255
    logits = torch.zeros(1, 2, 4, 4)
    logits[0, 0, :, :] = 5.0
    acc = ConfusionAccumulator(num_classes=2)
    acc.update(logits, target, ignore_index=255)
    assert acc.confusion[0, 0].item() == 15
    assert acc.confusion.sum().item() == 15
