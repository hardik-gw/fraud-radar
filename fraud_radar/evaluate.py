"""Evaluation under heavy class imbalance.

Accuracy is not computed anywhere in this module, on purpose. At a fraud rate
under 1%, a model that flags nothing scores over 99% and is worthless. Every
metric here stays sensitive to the rare class.
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
)


def pr_auc(y_true, scores) -> float:
    """Area under the precision-recall curve.

    Preferred over ROC-AUC here. ROC-AUC uses the false positive *rate*, whose
    denominator is the ~99.5% legitimate majority, so thousands of false alarms
    barely move it. Precision uses the flagged set as its denominator, so it
    reflects what an analyst actually experiences.
    """
    return float(average_precision_score(y_true, scores))


def metrics_at_threshold(y_true, scores, threshold: float) -> dict:
    """Precision, recall, F1 and the raw confusion counts at one cut point."""
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        # What the numbers mean operationally, which is how a threshold
        # actually gets argued about in a real team.
        "flagged_per_10k": float(10_000 * (tp + fp) / len(y_true)),
        "fraud_missed": int(fn),
    }


def threshold_for_review_budget(y_true, scores, flagged_per_10k: float) -> float:
    """Pick the threshold a fixed review capacity can afford.

    A fraud team can review so many transactions a day and no more. Rather than
    maximising F1 — which silently assumes precision and recall matter equally,
    and they rarely do — we state the capacity and read off the threshold that
    fills it. The question becomes 'how much fraud does this budget catch?',
    which a business can actually answer.
    """
    n_flagged = max(1, round(len(y_true) * flagged_per_10k / 10_000))
    return float(np.sort(scores)[-n_flagged])


def summarise(name: str, y_true, scores, flagged_per_10k: float = 50.0) -> dict:
    """Full report for one model at a fixed review budget."""
    threshold = threshold_for_review_budget(y_true, scores, flagged_per_10k)
    result = metrics_at_threshold(y_true, scores, threshold)
    result["model"] = name
    result["pr_auc"] = pr_auc(y_true, scores)
    result["review_budget_per_10k"] = flagged_per_10k
    return result


def pr_curve_points(y_true, scores, max_points: int = 400) -> dict:
    """Precision/recall pairs for plotting, thinned to keep the artifact small."""
    precision, recall, _ = precision_recall_curve(y_true, scores)
    step = max(1, len(precision) // max_points)
    return {
        "precision": precision[::step].tolist(),
        "recall": recall[::step].tolist(),
    }
