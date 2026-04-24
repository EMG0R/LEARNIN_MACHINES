"""
FCOS-style target assignment + losses for single-class (face) detection.
"""
import torch
import torch.nn.functional as F


def focal_bce(logits, targets, alpha=0.25, gamma=2.0):
    """Sigmoid focal loss, mean over positives (+ small eps for zero-positive batches)."""
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * (1 - p_t) ** gamma * ce
    n_pos = targets.sum().clamp(min=1.0)
    return loss.sum() / n_pos


def giou_loss(pred_ltrb, target_ltrb):
    """pred/target shape (..., 4) = (l, t, r, b), all non-negative."""
    pl, pt, pr, pb = pred_ltrb.unbind(-1)
    tl, tt, tr_, tb = target_ltrb.unbind(-1)
    il = torch.min(pl, tl); it = torch.min(pt, tt)
    ir = torch.min(pr, tr_); ib = torch.min(pb, tb)
    inter_w = (il + ir).clamp(min=0); inter_h = (it + ib).clamp(min=0)
    inter = inter_w * inter_h
    area_p = (pl + pr) * (pt + pb)
    area_t = (tl + tr_) * (tt + tb)
    union = area_p + area_t - inter + 1e-7
    iou = inter / union
    el = torch.max(pl, tl); et_ = torch.max(pt, tt)
    er = torch.max(pr, tr_); eb = torch.max(pb, tb)
    enc = (el + er) * (et_ + eb) + 1e-7
    giou = iou - (enc - union) / enc
    return (1 - giou).mean()


def build_targets(face_bboxes, feat_hw, stride, radius_cells=1):
    """
    face_bboxes: list of (x, y, w, h) in image pixels.
    Returns:
      obj_t (1, H, W)   1.0 where a face-center cell sits (plus optional neighbor radius)
      bbox_t (4, H, W)  l, t, r, b in pixels (valid only where obj_t==1)
      ctr_t  (1, H, W)  centerness target in [0, 1]
    """
    H, W = feat_hw
    obj_t  = torch.zeros((1, H, W))
    bbox_t = torch.zeros((4, H, W))
    ctr_t  = torch.zeros((1, H, W))

    ys = (torch.arange(H) + 0.5) * stride
    xs = (torch.arange(W) + 0.5) * stride

    faces_sorted = sorted(face_bboxes, key=lambda b: -b[2] * b[3])

    for (bx, by, bw, bh) in faces_sorted:
        cx = bx + bw / 2.0
        cy = by + bh / 2.0
        cx_cell = int(cx / stride); cy_cell = int(cy / stride)
        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                iy = cy_cell + dy; ix = cx_cell + dx
                if not (0 <= iy < H and 0 <= ix < W):
                    continue
                px = xs[ix].item(); py = ys[iy].item()
                if not (bx <= px <= bx + bw and by <= py <= by + bh):
                    continue
                l = px - bx; t = py - by; r = (bx + bw) - px; b = (by + bh) - py
                if min(l, t, r, b) <= 0:
                    continue
                obj_t[0, iy, ix] = 1.0
                bbox_t[:, iy, ix] = torch.tensor([l, t, r, b])
                ctr = ((min(l, r) / max(l, r)) * (min(t, b) / max(t, b))) ** 0.5
                ctr_t[0, iy, ix] = ctr
    return obj_t, bbox_t, ctr_t


def fcos_loss(obj_pred, bbox_pred, ctr_pred, obj_t, bbox_t, ctr_t):
    """Batch-wise targets expected with leading batch dim."""
    obj_loss = focal_bce(obj_pred, obj_t)

    pos_mask = obj_t > 0.5  # (B, 1, H, W)
    if pos_mask.sum() == 0:
        zero = obj_pred.new_zeros(())
        return obj_loss, zero, zero, obj_loss

    bbox_pred_exp = bbox_pred.exp()
    B, _, H, W = obj_t.shape
    pm = pos_mask.squeeze(1)  # (B, H, W)
    bp = bbox_pred_exp.permute(0, 2, 3, 1)[pm]   # (N, 4)
    bt = bbox_t.permute(0, 2, 3, 1)[pm]          # (N, 4)
    cp = ctr_pred.permute(0, 2, 3, 1)[pm][:, 0]  # (N,)
    ct = ctr_t.permute(0, 2, 3, 1)[pm][:, 0]     # (N,)

    bbox_loss = giou_loss(bp, bt)
    ctr_loss = F.binary_cross_entropy_with_logits(cp, ct, reduction="mean")
    total = obj_loss + bbox_loss + 0.5 * ctr_loss
    return obj_loss, bbox_loss, ctr_loss, total
