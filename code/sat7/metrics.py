"""Evaluation metrics of paper VIII-E: detection quality and information preservation.

Detection quality (VIII-E.2): precision, recall, F1, mAP@0.5, and small-object recall.
Matching is greedy by confidence, IoU >= 0.5, each ground-truth box used once (the WP1 rule).

Information preservation (VIII-E.3): "a detection is considered preserved when its class and
location remain within predefined tolerances of the reference information". The reference is
what the ground would extract from the full raw image (B0); a reference detection is preserved
when a *delivered* record has the same class and its centre within ``tol_px`` of it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sahi import Detection

SMALL_PX = 32          # "small object": longest side under 32 px (COCO-style)


def match(preds: list[Detection], gts: list[Detection], iou_thr: float = 0.5):
    """Returns (tp flag per prediction in descending-confidence order, found flag per gt)."""
    order = sorted(range(len(preds)), key=lambda i: -preds[i].p)
    used, tp = set(), []
    for i in order:
        best, bj = 0.0, -1
        for j, g in enumerate(gts):
            if j in used or g.c != preds[i].c:
                continue
            v = preds[i].iou(g)
            if v > best:
                best, bj = v, j
        if best >= iou_thr:
            used.add(bj)
            tp.append(True)
        else:
            tp.append(False)
    return [preds[i] for i in order], tp, [j in used for j in range(len(gts))]


@dataclass
class DetAccumulator:
    """Collects predictions over many images, then reports the VIII-E.2 metrics."""
    iou_thr: float = 0.5

    def __post_init__(self):
        self.scores, self.tps = [], []
        self.n_gt = self.n_small = self.found_small = 0

    def add(self, preds: list[Detection], gts: list[Detection]) -> None:
        ordered, tp, found = match(preds, gts, self.iou_thr)
        self.scores += [d.p for d in ordered]
        self.tps += tp
        self.n_gt += len(gts)
        for g, f in zip(gts, found):
            if max(g.w, g.h) < SMALL_PX:
                self.n_small += 1
                self.found_small += int(f)

    def at(self, thr: float) -> dict:
        s, t = np.asarray(self.scores), np.asarray(self.tps, bool)
        k = s >= thr
        tp = int(t[k].sum())
        fp = int(k.sum() - tp)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / self.n_gt if self.n_gt else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        return {"precision": prec, "recall": rec, "F1": f1, "TP": tp, "FP": fp}

    def ap(self) -> float:
        """Average precision at the IoU threshold: all-point interpolated area under P(R)."""
        if not self.n_gt or not self.scores:
            return 0.0
        order = np.argsort(-np.asarray(self.scores), kind="stable")
        t = np.asarray(self.tps, bool)[order]
        ctp, cfp = np.cumsum(t), np.cumsum(~t)
        rec = ctp / self.n_gt
        prec = ctp / np.maximum(ctp + cfp, 1)
        mrec = np.concatenate([[0.0], rec, [1.0]])
        mpre = np.concatenate([[0.0], prec, [0.0]])
        mpre = np.maximum.accumulate(mpre[::-1])[::-1]
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

    def small_recall(self) -> float:
        return self.found_small / self.n_small if self.n_small else float("nan")


def preserved(reference: list[Detection], delivered: list[tuple[int, float, float]],
              tol_px: float = 4.0) -> list[bool]:
    """Per reference detection: is there a delivered (class, cx, cy) within tolerance?"""
    out, used = [], set()
    for r in reference:
        cx, cy = r.centre
        hit = -1
        for k, (c, x, y) in enumerate(delivered):
            if k not in used and c == r.c and (x - cx) ** 2 + (y - cy) ** 2 <= tol_px ** 2:
                hit = k
                break
        if hit >= 0:
            used.add(hit)
        out.append(hit >= 0)
    return out
