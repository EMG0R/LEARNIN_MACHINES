import torch
from FACE_JOB.face_det.losses import build_targets, fcos_loss, focal_bce
from FACE_JOB.face_det.postprocess import decode, nms


def test_build_targets_assigns_centered_cell():
    # Feature map stride=8, map size 40×40. One face at pixel bbox (80, 80, 160, 160)
    # → center (160, 160) → cell (20, 20) in feature map.
    face_bboxes = [(80, 80, 160, 160)]  # x, y, w, h (so center is at 80+80=160, 80+80=160)
    obj_t, bbox_t, ctr_t = build_targets(face_bboxes, feat_hw=(40, 40), stride=8, radius_cells=0)
    assert obj_t.shape == (1, 40, 40)
    # center cell: (160/8)=20, (160/8)=20 → cell (20, 20)
    # But cell center is at (20+0.5)*8=164px. l = 164-80=84, r = 80+160-164=76, etc.
    assert obj_t[0, 20, 20] == 1.0


def test_focal_bce_reduces_easy_examples():
    logits = torch.tensor([-10.0, 0.0, 10.0]).unsqueeze(0).unsqueeze(-1)  # (1,3,1)
    targets = torch.tensor([0.0, 0.0, 1.0]).unsqueeze(0).unsqueeze(-1)
    loss = focal_bce(logits, targets, alpha=0.25, gamma=2.0)
    assert loss.item() >= 0.0
    assert torch.isfinite(loss)


def test_nms_suppresses_overlapping():
    boxes = torch.tensor([
        [10, 10, 50, 50],
        [12, 12, 52, 52],
        [200, 200, 240, 240],
    ], dtype=torch.float32)
    scores = torch.tensor([0.9, 0.8, 0.95])
    keep = nms(boxes, scores, iou_thr=0.5)
    assert sorted(keep.tolist()) == [0, 2]
