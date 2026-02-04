from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

@dataclass
class MetricResult:
    accuracy: float
    precision: float
    recall: float
    f1: float
    auroc: float

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> MetricResult:
    y_pred = (y_prob >= threshold).astype(np.int64)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auroc = float("nan")
    return MetricResult(accuracy=float(acc), precision=float(prec), recall=float(rec), f1=float(f1), auroc=float(auroc))

def as_dict(m: MetricResult) -> Dict[str, float]:
    return {"accuracy": m.accuracy, "precision": m.precision, "recall": m.recall, "f1": m.f1, "auroc": m.auroc}
