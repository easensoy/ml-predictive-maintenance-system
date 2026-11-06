"""Post-hoc calibration, fitted on validation engines that the model never trained on.

* Conformalized quantile regression (Romano, Patterson and Candès, 2019): widens the model's
  5%/95% bounds by a margin chosen so the interval covers the true RUL with probability
  ``1 - alpha`` on exchangeable data.
* Temperature scaling (Guo et al., 2017): one scalar that makes the alert probability honest
  without changing which engines rank as most at risk.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F


def conformal_margin(
    lower: np.ndarray, upper: np.ndarray, target: np.ndarray, alpha: float
) -> float:
    scores = np.maximum(lower - target, target - upper)
    n = len(scores)
    level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    log_t = torch.zeros(1, requires_grad=True)
    logits_t = torch.as_tensor(logits, dtype=torch.float64)
    labels_t = torch.as_tensor(labels, dtype=torch.float64)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(logits_t / log_t.exp().double(), labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.detach().exp().clamp(0.05, 20.0))


def calibrated_outputs(
    quantiles: np.ndarray, alert_logit: np.ndarray, margin: float, temperature: float, max_rul: int
) -> dict[str, np.ndarray]:
    """Convert raw model outputs (fractions of ``max_rul``) into calibrated values in cycles."""
    lower = np.clip((quantiles[:, 0] - margin) * max_rul, 0, max_rul)
    point = np.clip(quantiles[:, 1] * max_rul, 0, max_rul)
    upper = np.clip((quantiles[:, 2] + margin) * max_rul, 0, max_rul)
    probability = 1.0 / (1.0 + np.exp(-alert_logit / temperature))
    return {"lower": lower, "point": point, "upper": upper, "probability": probability}
