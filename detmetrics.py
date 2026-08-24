"""
Detection scoring shared by the robustness and runtime sections.

Deliberately not ultralytics' validator. That one needs a data.yaml, a dataset
on disk and the model's own class indices, none of which survive when the
images are perturbed in memory (occlusion/) or when the predictions come from a
CoreML pipeline rather than a torch model (runtime_bench/). What is needed in
both places is smaller and has to be identical on both sides of the comparison:

  recall          class-agnostic, IoU >= 0.5, greedy by confidence
                  -> "was the fruit found at all"
  stage accuracy  among matched fruit only
                  -> "given it was found, is the ripeness stage right"
  counting-MAE    |predicted - GT| counts per stage, averaged over images,
                  the same definition as evaluate.py and cross_dataset

Splitting detection from staging matters because they fail independently:
a model can keep finding every tomato while its ripeness calls collapse, and
counting-MAE alone cannot tell that from missed fruit.
"""
from collections import defaultdict

import numpy as np

from cross_dataset.taxonomy import STAGES


def iou_matrix(gt, pred):
    """Pairwise IoU, [n_gt, n_pred], both as [x1, y1, x2, y2] in pixels."""
    if not len(gt) or not len(pred):
        return np.zeros((len(gt), len(pred)), dtype=np.float32)
    gt, pred = np.asarray(gt, dtype=np.float32), np.asarray(pred, dtype=np.float32)
    x1 = np.maximum(gt[:, None, 0], pred[None, :, 0])
    y1 = np.maximum(gt[:, None, 1], pred[None, :, 1])
    x2 = np.minimum(gt[:, None, 2], pred[None, :, 2])
    y2 = np.minimum(gt[:, None, 3], pred[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_gt = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    area_pred = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    return inter / (area_gt[:, None] + area_pred[None, :] - inter + 1e-9)


def match(gt_boxes, pred_boxes, confidences, threshold=0.5):
    """Greedy class-agnostic matching, highest confidence first -> {gt: pred}.

    Class-agnostic on purpose: a fruit called half-ripened instead of green is
    a staging error, not a miss, and collapsing the two hides which one moved.
    """
    ious = iou_matrix(gt_boxes, pred_boxes)
    taken, pairs = set(), {}
    order = np.argsort(-np.asarray(confidences)) if len(confidences) else []
    for pred_idx in order:
        column = ious[:, pred_idx]
        for gt_idx in np.argsort(-column):
            if column[gt_idx] < threshold:
                break
            if int(gt_idx) not in taken:
                taken.add(int(gt_idx))
                pairs[int(gt_idx)] = int(pred_idx)
                break
    return pairs


class Score:
    """Accumulates one image at a time, then reports the block above."""

    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold
        self.found = defaultdict(int)
        self.total = defaultdict(int)
        self.confusion = defaultdict(lambda: defaultdict(int))
        self.abs_err = defaultdict(list)
        self.pred_totals = defaultdict(int)
        self.matched_conf, self.matched_iou = [], []
        self.n_images = 0

    def add(self, gt_stages, gt_boxes, pred_stages, pred_boxes, pred_conf):
        self.n_images += 1
        pairs = match(gt_boxes, pred_boxes, pred_conf, self.iou_threshold)
        ious = iou_matrix(gt_boxes, pred_boxes)

        for gt_idx, stage in enumerate(gt_stages):
            self.total[stage] += 1
            if gt_idx in pairs:
                pred_idx = pairs[gt_idx]
                self.found[stage] += 1
                self.confusion[stage][pred_stages[pred_idx]] += 1
                self.matched_conf.append(float(pred_conf[pred_idx]))
                self.matched_iou.append(float(ious[gt_idx, pred_idx]))

        counts_gt, counts_pred = defaultdict(int), defaultdict(int)
        for stage in gt_stages:
            counts_gt[stage] += 1
        for stage in pred_stages:
            counts_pred[stage] += 1
            self.pred_totals[stage] += 1
        for stage in STAGES:
            self.abs_err[stage].append(abs(counts_pred[stage] - counts_gt[stage]))

    def summary(self):
        n_total, n_found = sum(self.total.values()), sum(self.found.values())
        correct = sum(self.confusion[s][s] for s in STAGES)
        return {
            "recall": n_found / n_total if n_total else 0.0,
            "recall_by_stage": {s: (self.found[s] / self.total[s] if self.total[s] else None)
                                for s in STAGES},
            "stage_accuracy": correct / n_found if n_found else 0.0,
            "stage_accuracy_by_stage": {
                s: (self.confusion[s][s] / self.found[s] if self.found[s] else None)
                for s in STAGES},
            "confusion": {s: dict(self.confusion[s]) for s in STAGES},
            "counting_mae": float(np.mean([e for s in STAGES for e in self.abs_err[s]])),
            "counting_mae_by_stage": {s: float(np.mean(self.abs_err[s])) for s in STAGES},
            "pred_totals": dict(self.pred_totals),
            "gt_totals": dict(self.total),
            "mean_conf": float(np.mean(self.matched_conf)) if self.matched_conf else 0.0,
            "mean_iou": float(np.mean(self.matched_iou)) if self.matched_iou else 0.0,
            "n_gt": n_total,
            "n_images": self.n_images,
        }
