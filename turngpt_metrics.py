"""Binary turn-boundary metrics, independent of model inference libraries."""

from __future__ import annotations

import math

METRIC_NAMES = ("bacc", "precision", "recall", "f1", "accuracy")
COUNT_NAMES = ("tp", "fp", "tn", "fn")


def metrics_from_confusion(counts):
    """Treat <ts> as positive; precision/F1 are zero if no positives are predicted."""
    if any(not isinstance(counts[k], int) or counts[k] < 0 for k in COUNT_NAMES):
        raise ValueError("Confusion counts must be nonnegative integers")
    tp, fp, tn, fn = (counts[k] for k in COUNT_NAMES)
    positives, negatives = tp + fn, tn + fp
    if not positives or not negatives:
        raise ValueError("Balanced accuracy requires both positive and negative labels")
    return {
        "bacc": (tp / positives + tn / negatives) / 2,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / positives,
        "f1": 2 * tp / (2 * tp + fp + fn),
        "accuracy": (tp + tn) / (positives + negatives),
    }


def classification_metrics(scores, labels, threshold):
    """Evaluate all metrics at one fixed threshold, with score >= threshold positive."""
    if len(scores) != len(labels) or not len(scores):
        raise ValueError("Scores and labels must have the same nonzero length")
    if not math.isfinite(threshold):
        raise ValueError("Threshold must be finite")
    counts = dict.fromkeys(COUNT_NAMES, 0)
    for score, label in zip(scores, labels):
        if not math.isfinite(score) or label not in (0, 1):
            raise ValueError("Scores must be finite and labels must be binary")
        positive = score >= threshold
        key = ("tp" if label else "fp") if positive else ("fn" if label else "tn")
        counts[key] += 1
    return metrics_from_confusion(counts), counts


def aggregate_classification_metrics(confusions):
    """Keep corpus means distinct from metrics calculated on pooled predictions."""
    if not confusions:
        raise ValueError("At least one corpus is required")
    metrics = [metrics_from_confusion(counts) for counts in confusions]
    weights = [sum(counts[k] for k in COUNT_NAMES) for counts in confusions]
    total = sum(weights)
    result = {}
    for name in METRIC_NAMES:
        result[f"macro_test_{name}"] = sum(m[name] for m in metrics) / len(metrics)
        result[f"position_weighted_test_{name}"] = sum(
            m[name] * weight for m, weight in zip(metrics, weights)
        ) / total
    pooled = {k: sum(counts[k] for counts in confusions) for k in COUNT_NAMES}
    result["pooled_test_metrics"] = metrics_from_confusion(pooled)
    result["pooled_test_confusion_matrix"] = pooled
    result["test_prediction_positions"] = total
    return result
