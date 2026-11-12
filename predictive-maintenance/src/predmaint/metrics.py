"""Evaluation metrics: the standard C-MAPSS ones plus interval, calibration and lead-time checks."""

import numpy as np
import pandas as pd

# The paired bootstrap over test engines: number of draws and seed, shared by every difference.
N_BOOT = 2000
BOOT_SEED = 7


def by_engine(test: pd.DataFrame, model: str, column: str) -> np.ndarray:
    """(repeats, engines) array of one column of a test table, engines in unit order."""
    wide = test[test["model"] == model].pivot(index="repeat", columns="unit", values=column)
    return wide.sort_index().sort_index(axis=1).to_numpy(float)


def rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(prediction) - np.asarray(target)) ** 2)))


def mae(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(prediction) - np.asarray(target))))


def nasa_score(prediction: np.ndarray, target: np.ndarray) -> float:
    """PHM08 scoring function: late predictions (d > 0) cost more than early ones."""
    d = np.asarray(prediction, np.float64) - np.asarray(target, np.float64)
    return float(np.where(d < 0, np.expm1(-d / 13.0), np.expm1(d / 10.0)).sum())


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


def weighted_rmse(prediction: np.ndarray, target: np.ndarray, weights: np.ndarray) -> float:
    """RMSE with per-sample weights, e.g. 1/length so every engine counts equally."""
    d = np.asarray(prediction, np.float64) - np.asarray(target, np.float64)
    return float(np.sqrt(np.sum(weights * d * d) / np.sum(weights)))


def per_engine_weights(units: np.ndarray) -> np.ndarray:
    """Weight 1/(rows of that engine) for every row: each engine then counts once."""
    _, inverse, counts = np.unique(units, return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse]


def alert_behaviour(frame: pd.DataFrame, threshold: float, horizon: int) -> dict:
    """How the alert behaves over the whole lives of engines that run to failure.

    ``frame`` has one row per cycle with ``unit``, ``cycle``, ``rul`` (actual cycles left, 0 at
    the failure cycle) and ``probability``. The alert is on when probability >= threshold.

    * lead time: cycles left when the alert first comes on.
    * missed: the alert never comes on while at least one cycle is left.
    * early: the first alert comes with more than ``horizon`` cycles left, so at that moment it
      is a false alarm by the alert's own definition.
    * false-alert episodes: the alert coming on (from off, or at the first cycle) while more than
      ``horizon`` cycles are left.
    * flaps: the alert switching off again; each on-to-off change counts once.
    """
    per_unit = []
    for _, group in frame.sort_values(["unit", "cycle"]).groupby("unit"):
        on = group["probability"].to_numpy() >= threshold
        rul = group["rul"].to_numpy()
        before = np.r_[False, on[:-1]]
        fired = np.flatnonzero(on & (rul >= 1))
        per_unit.append(
            {
                "unit": int(group["unit"].to_numpy()[0]),
                "lead_time": int(rul[fired[0]]) if len(fired) else None,
                "false_alerts": int((on & ~before & (rul > horizon)).sum()),
                "flaps": int((~on & before).sum()),
            }
        )
    leads = np.array([u["lead_time"] for u in per_unit if u["lead_time"] is not None], float)
    false_alerts = np.array([u["false_alerts"] for u in per_unit])
    flaps = np.array([u["flaps"] for u in per_unit])

    def pct(q: float) -> float | None:
        return float(np.percentile(leads, q)) if len(leads) else None

    return {
        "threshold": threshold,
        "horizon": horizon,
        "units": len(per_unit),
        "missed": int(len(per_unit) - len(leads)),
        "early": int((leads > horizon).sum()),
        "early_twice_horizon": int((leads > 2 * horizon).sum()),
        "lead_median": pct(50),
        "lead_p25": pct(25),
        "lead_p75": pct(75),
        "lead_min": float(leads.min()) if len(leads) else None,
        "lead_max": float(leads.max()) if len(leads) else None,
        "false_alerts_per_engine": float(false_alerts.mean()),
        "engines_with_false_alert": int((false_alerts > 0).sum()),
        "flaps_per_engine": float(flaps.mean()),
        "engines_with_flaps": int((flaps > 0).sum()),
        "per_unit": per_unit,
    }


def binned_by_group(
    probability: np.ndarray, label: np.ndarray, groups: np.ndarray, bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-group reliability sums: (group ids, counts, probability sums, label sums), each count
    and sum of shape (groups, bins). Lets a bootstrap over engines recompute ECE cheaply."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(probability, edges[1:-1]), 0, bins - 1)
    ids, inverse = np.unique(groups, return_inverse=True)
    shape = (len(ids), bins)
    counts, p_sum, y_sum = np.zeros(shape), np.zeros(shape), np.zeros(shape)
    np.add.at(counts, (inverse, index), 1)
    np.add.at(p_sum, (inverse, index), probability)
    np.add.at(y_sum, (inverse, index), label)
    return ids, counts, p_sum, y_sum


def ece_from_bins(counts: np.ndarray, p_sum: np.ndarray, y_sum: np.ndarray) -> float:
    total = counts.sum()
    filled = counts > 0
    gaps = np.abs(p_sum[filled] - y_sum[filled]) / counts[filled]
    return float((counts[filled] / total * gaps).sum())


def bootstrap_indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, n) resampled indices, the same for every statistic computed with one seed, which
    is what makes a bootstrap paired."""
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def percentile_interval(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    tail = 100 * (1 - level) / 2
    low, high = np.percentile(samples, [tail, 100 - tail])
    return float(low), float(high)


def ece_with_interval(
    probability: np.ndarray,
    label: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """ECE (10 equal-width bins) with a 95% interval from resampling whole engines, which keeps
    the correlation between an engine's cycles inside each resample."""
    _, counts, p_sum, y_sum = binned_by_group(probability, label, groups)
    draws = bootstrap_indices(len(counts), n_boot, seed)
    samples = np.array(
        [ece_from_bins(counts[d].sum(0), p_sum[d].sum(0), y_sum[d].sum(0)) for d in draws]
    )
    low, high = percentile_interval(samples)
    return {
        "ece": ece_from_bins(counts.sum(0), p_sum.sum(0), y_sum.sum(0)),
        "ci_low": low,
        "ci_high": high,
        "engines": len(counts),
    }


def mean_sd(values: list[float | None]) -> dict:
    """Mean and sample standard deviation over repeats (sd is None for a single repeat)."""
    clean = np.array([v for v in values if v is not None], float)
    if len(clean) == 0:
        return {"mean": None, "sd": None, "n": 0}
    return {
        "mean": float(clean.mean()),
        "sd": float(clean.std(ddof=1)) if len(clean) > 1 else None,
        "n": len(clean),
    }
