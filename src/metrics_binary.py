import numpy as np


def roc_curve_from_scores(y_true: np.ndarray, y_score: np.ndarray):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)

    order = np.argsort(-y_score, kind="mergesort")
    y_true = y_true[order]
    y_score = y_score[order]

    pos = int((y_true == 1).sum())
    neg = int((y_true == 0).sum())
    if pos == 0 or neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    tps = np.cumsum(y_true == 1)
    fps = np.cumsum(y_true == 0)

    distinct = np.where(np.diff(y_score))[0]
    idx = np.r_[distinct, y_true.size - 1]

    tpr = tps[idx] / max(pos, 1)
    fpr = fps[idx] / max(neg, 1)

    tpr = np.r_[0.0, tpr]
    fpr = np.r_[0.0, fpr]
    return fpr, tpr


def _trapezoid_area(y: np.ndarray, x: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if x.size < 2:
        return 0.0
    dx = x[1:] - x[:-1]
    avg_height = (y[1:] + y[:-1]) * 0.5
    return float(np.sum(dx * avg_height))


def binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr = roc_curve_from_scores(y_true, y_score)
    return _trapezoid_area(tpr, fpr)


def binary_eer(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr = roc_curve_from_scores(y_true, y_score)
    fnr = 1.0 - tpr
    idx = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)
