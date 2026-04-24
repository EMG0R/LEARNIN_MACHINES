"""Inference-time decoding + NMS."""
import torch


def decode(obj_logits, bbox_pred, ctr_logits, stride: int, score_thr: float = 0.3):
    """
    obj_logits:  (B, 1, H, W)   raw logits
    bbox_pred:   (B, 4, H, W)   raw (pre-exp) l, t, r, b
    ctr_logits:  (B, 1, H, W)   raw logits
    Returns list (len B) of tuples (boxes_xyxy, scores).
    """
    B, _, H, W = obj_logits.shape
    results = []
    ys = (torch.arange(H, device=obj_logits.device) + 0.5) * stride
    xs = (torch.arange(W, device=obj_logits.device) + 0.5) * stride
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")  # (H, W)
    for b in range(B):
        obj_s = torch.sigmoid(obj_logits[b, 0])
        ctr_s = torch.sigmoid(ctr_logits[b, 0])
        scores = (obj_s * ctr_s).sqrt()
        l, t, r, bt = bbox_pred[b].exp().unbind(0)
        boxes = torch.stack([gx - l, gy - t, gx + r, gy + bt], dim=-1)  # (H, W, 4)
        mask = scores > score_thr
        if mask.sum() == 0:
            results.append((boxes.new_zeros((0, 4)), scores.new_zeros((0,))))
            continue
        results.append((boxes[mask], scores[mask]))
    return results


def nms(boxes, scores, iou_thr=0.5):
    """Pure-PyTorch greedy NMS. boxes: (N, 4) xyxy."""
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        xx1 = torch.max(boxes[i, 0], boxes[rest, 0])
        yy1 = torch.max(boxes[i, 1], boxes[rest, 1])
        xx2 = torch.min(boxes[i, 2], boxes[rest, 2])
        yy2 = torch.min(boxes[i, 3], boxes[rest, 3])
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_rest = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (area_i + area_rest - inter + 1e-7)
        order = rest[iou <= iou_thr]
    return torch.tensor(keep, dtype=torch.long)
