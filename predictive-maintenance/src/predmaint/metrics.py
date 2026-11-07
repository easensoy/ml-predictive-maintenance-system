"""Evaluation metrics: the standard C-MAPSS ones plus interval, calibration and lead-time checks."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(prediction) - np.asarray(target)) ** 2)))


def mae(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(prediction) - np.asarray(target))))


def nasa_score(prediction: np.ndarray, target: np.ndarray) -> float:
    """PHM08 scoring function: late predictions (d > 0) cost more than early ones."""
    d = np.asarray(prediction, np.float64) - np.asarray(target, np.float64)
    return float(np.where(d < 0, np.expm1(-d / 13.0), np.expm1(d / 10.0)).sum())


def interval_coverage(lower: np.ndarray, upper: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((target >= lower) & (target <= upper)))


def wilson_interval(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval for a proportion (better than the normal approximation near 0
    or 1 and for small n). Treats the n trials as independent: use one sample per engine."""
    if n <= 0:
        raise ValueError("n must be positive")
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def expected_calibration_error(probability: np.ndarray, label: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(probability, edges[1:-1]), 0, bins - 1)
    error = 0.0
    for b in range(bins):
        mask = index == b
        if mask.any():
            error += mask.mean() * abs(probability[mask].mean() - label[mask].mean())
    return float(error)


def reliability_curve(probability: np.ndarray, label: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(probability, edges[1:-1]), 0, bins - 1)
    curve = []
    for b in range(bins):
        mask = index == b
        if mask.any():
            curve.append(
                {
                    "bin_low": float(edges[b]),
                    "bin_high": float(edges[b + 1]),
                    "mean_probability": float(probability[mask].mean()),
                    "observed_rate": float(label[mask].mean()),
                    "count": int(mask.sum()),
                }
            )
    return curve


def alert_metrics(probability: np.ndarray, label: np.ndarray, threshold: float) -> dict:
    label = np.asarray(label).astype(bool)
    predicted = probability >= threshold
    tp = int(np.sum(predicted & label))
    fp = int(np.sum(predicted & ~label))
    fn = int(np.sum(~predicted & label))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    has_both = label.any() and (~label).any()
    return {
        "threshold": threshold,
        "n": len(label),
        "positives": int(label.sum()),
        "average_precision": float(average_precision_score(label, probability))
        if has_both
        else None,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "brier": float(np.mean((probability - label) ** 2)),
        "ece": expected_calibration_error(probability, label.astype(float)),
    }


def lead_times(frame: pd.DataFrame, threshold: float, early_cycles: int) -> dict:
    """When does the alert first fire on engines that run to failure?

    ``frame`` holds per-cycle predictions with columns ``unit``, ``cycle``, ``rul_true`` and
    ``probability``. Lead time is the true number of cycles left when the alert first fires.
    """
    per_unit = []
    for _, group in frame.sort_values(["unit", "cycle"]).groupby("unit"):
        fired = group[group["probability"] >= threshold]
        lead = int(fired["rul_true"].to_numpy()[0]) if len(fired) else None
        per_unit.append({"unit": int(group["unit"].to_numpy()[0]), "lead_time": lead})
    leads = np.array([u["lead_time"] for u in per_unit if u["lead_time"] is not None])
    return {
        "threshold": threshold,
        "units": len(per_unit),
        "missed": sum(u["lead_time"] is None for u in per_unit),
        "early": int(np.sum(leads > early_cycles)) if len(leads) else 0,
        "early_definition_cycles": early_cycles,
        "median": float(np.median(leads)) if len(leads) else None,
        "p25": float(np.percentile(leads, 25)) if len(leads) else None,
        "p75": float(np.percentile(leads, 75)) if len(leads) else None,
        "per_unit": per_unit,
    }
