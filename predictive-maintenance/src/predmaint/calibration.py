"""Post-hoc calibration with cross-fitting, so that every calibration score comes from a model
that never saw that engine: not in training, not in early stopping, not in choosing the window.

* **Intervals**: conformalized quantile regression (Romano, Patterson and Candès, 2019) in its
  cross-validated form, CV+ (Barber, Candès, Ramdas and Tibshirani, 2021). The training engines
  are split into K folds and network k is trained without fold k. Each engine contributes one
  conformity score, computed by the network that did not see it at a test-like truncation point.
  A new engine's interval combines every network's bounds with the scores of the engines that
  network did not see. If a new engine and truncation point are exchangeable with the
  calibration ones (one score per engine, folds fixed independently of the data), K-fold CV+
  covers with probability at least 1 - 2*alpha - min(2(1 - 1/K)/(n/K + 1), (1 - K/n)/(K + 1))
  for n engines (Barber et al., Theorem 4; 1 - 2*alpha is the jackknife+ case, K = n), and
  typically close to 1 - alpha in practice.
* **The target cap**: the model predicts RUL capped at ``max_rul``. An interval whose upper end
  reaches the cap is reported as open-ended ("at least ``lower`` cycles"), which makes coverage of
  the capped target and coverage of the actual remaining life the same thing.
* **Alert probability**: temperature scaling (Guo, Pleiss, Sun and Weinberger, 2017), one scalar
  fitted on out-of-fold logits.
"""

import math

import numpy as np


def cqr_scores(lower: np.ndarray, upper: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Conformity score of each calibration point: how far the target lies outside [lower,
    upper] (negative when inside)."""
    return np.maximum(np.asarray(lower) - target, np.asarray(target) - upper)


def _ranks(n: int, alpha: float) -> tuple[int, int]:
    return math.floor(alpha * (n + 1)), math.ceil((1 - alpha) * (n + 1))


def cv_plus_interval(
    lower_by_member: np.ndarray,
    upper_by_member: np.ndarray,
    scores: np.ndarray,
    score_member: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """CV+ interval for N new points.

    ``lower_by_member`` and ``upper_by_member`` have shape (members, N): every member's raw
    bounds for the new points. ``scores[i]`` is calibration engine i's conformity score and
    ``score_member[i]`` the member that never saw engine i. The lower end is the
    floor(alpha (n+1))-th smallest of lower[member(i)] - score[i] over the n engines; the upper
    end is the ceil((1 - alpha)(n+1))-th smallest of upper[member(i)] + score[i]. A rank outside
    1..n gives an unbounded end.
    """
    scores = np.asarray(scores, np.float64)
    members = np.asarray(score_member, np.int64)
    n = len(scores)
    lo_rank, hi_rank = _ranks(n, alpha)
    lows = np.asarray(lower_by_member, np.float64)[members] - scores[:, None]  # (n, N)
    highs = np.asarray(upper_by_member, np.float64)[members] + scores[:, None]
    lower = (
        np.full(lows.shape[1], -np.inf)
        if lo_rank < 1
        else np.partition(lows, lo_rank - 1, axis=0)[lo_rank - 1]
    )
    upper = (
        np.full(highs.shape[1], np.inf)
        if hi_rank > n
        else np.partition(highs, hi_rank - 1, axis=0)[hi_rank - 1]
    )
    return lower, upper


def split_conformal_margin(scores: np.ndarray, alpha: float) -> float:
    """The ceil((n+1)(1-alpha))-th smallest score: the margin for one fixed model."""
    scores = np.sort(np.asarray(scores, np.float64))
    rank = _ranks(len(scores), alpha)[1]
    return float("inf") if rank > len(scores) else float(scores[rank - 1])


def cross_conformal_margins(
    scores: np.ndarray, score_member: np.ndarray, n_members: int, alpha: float
) -> np.ndarray:
    """Margin for each member's own out-of-fold predictions, from the other folds' scores.

    Used for whole-life analysis of training engines, where CV+ itself does not apply (an engine
    cannot be both a calibration point and the point being predicted). This is the usual
    cross-conformal approximation, not CV+'s guarantee.
    """
    members = np.asarray(score_member)
    return np.array(
        [split_conformal_margin(np.asarray(scores)[members != k], alpha) for k in range(n_members)]
    )


def finalize_interval(
    lower: np.ndarray, upper: np.ndarray, max_rul: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interval in cycles for a target capped at ``max_rul``.

    Returns (lower, upper, open_ended). ``open_ended`` marks intervals whose upper end reaches
    the cap; they mean "at least ``lower`` cycles". ``upper`` is then reported as the cap.
    """
    open_ended = np.asarray(upper) >= max_rul
    upper = np.clip(upper, 0.0, max_rul)
    lower = np.minimum(np.clip(lower, 0.0, max_rul), upper)
    return lower, upper, open_ended


def covered(
    lower: np.ndarray, upper: np.ndarray, open_ended: np.ndarray, target: np.ndarray
) -> np.ndarray:
    """Whether each target lies inside its interval, reading open-ended intervals as
    [lower, infinity). Works for the capped and the actual RUL alike."""
    target = np.asarray(target, np.float64)
    return (target >= lower) & (open_ended | (target <= upper))


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray | None = None,
    bounds: tuple[float, float] = (0.05, 20.0),
) -> float:
    """The temperature T in ``bounds`` that minimises weighted cross-entropy of sigmoid(logit/T).

    The loss is convex in 1/T (a one-coefficient logistic regression), so a log-spaced grid
    followed by ternary search finds the minimum without the divergence a gradient method can
    show on badly scaled logits.
    """
    z = np.asarray(logits, np.float64)
    y = np.asarray(labels, np.float64)
    w = np.ones_like(z) if weights is None else np.asarray(weights, np.float64)

    def loss(beta: float) -> float:
        s = beta * z
        return float(np.sum(w * (np.logaddexp(0.0, s) - y * s)))

    betas = np.exp(np.linspace(np.log(1 / bounds[1]), np.log(1 / bounds[0]), 121))
    best = int(np.argmin([loss(b) for b in betas]))
    low, high = betas[max(best - 1, 0)], betas[min(best + 1, len(betas) - 1)]
    for _ in range(80):
        a, b = low + (high - low) / 3, high - (high - low) / 3
        if loss(a) <= loss(b):
            high = b
        else:
            low = a
    return float(1.0 / ((low + high) / 2))


def probability(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Calibrated probability. For (members, N) logits, the members' probabilities are
    averaged."""
    p = 1.0 / (1.0 + np.exp(-np.asarray(logits, np.float64) / temperature))
    return p.mean(axis=0) if p.ndim == 2 else p
