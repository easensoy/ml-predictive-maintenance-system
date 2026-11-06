"""LSTM + attention network with an uncertainty-aware RUL head and an alert head.

Outputs are in units of ``max_rul`` (so RUL targets lie in [0, 1]):

* ``quantiles``: (lower, point, upper). The point estimate is trained with MSE; the bounds are
  trained as the 5th and 95th percentiles with the pinball loss and are kept on either side of
  the point estimate by construction, so they can never cross. Split-conformal calibration
  (see ``calibration.py``) later widens them to a guaranteed coverage level.
* ``alert_logit``: logit of P(failure within ``alert_cycles`` cycles).
* ``attention``: the attention weights over the time steps of the window.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

LOWER_Q, UPPER_Q = 0.05, 0.95


class AttentionPool(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.score = nn.Linear(hidden, 1)

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.score(states).squeeze(-1), dim=1)  # (B, T)
        return (states * weights.unsqueeze(-1)).sum(dim=1), weights


class RULNet(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.config = {
            "n_features": n_features,
            "hidden": hidden,
            "layers": layers,
            "dropout": dropout,
        }
        self.lstm = nn.LSTM(
            n_features, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0
        )
        self.pool = AttentionPool(hidden)
        self.trunk = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.rul_head = nn.Linear(hidden, 3)
        self.alert_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        states, _ = self.lstm(x)
        context, attention = self.pool(states)
        features = self.trunk(torch.cat([states[:, -1], context], dim=1))
        raw = self.rul_head(features)
        point = raw[:, 0]
        lower = point - F.softplus(raw[:, 1])
        upper = point + F.softplus(raw[:, 2])
        return {
            "quantiles": torch.stack([lower, point, upper], dim=1),
            "alert_logit": self.alert_head(features).squeeze(-1),
            "attention": attention,
        }


class Ensemble(nn.Module):
    """Average of independently seeded members (a deep ensemble)."""

    def __init__(self, members: list[RULNet]):
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = [member(x) for member in self.members]
        return {key: torch.stack([o[key] for o in outputs]).mean(dim=0) for key in outputs[0]}


@torch.no_grad()
def predict_raw(
    model: nn.Module, windows: np.ndarray, batch_size: int = 4096
) -> dict[str, np.ndarray]:
    """Run the model in eval mode over (N, T, F) windows and return numpy outputs."""
    model.eval()
    device = next(model.parameters()).device
    chunks: dict[str, list[np.ndarray]] = {"quantiles": [], "alert_logit": [], "attention": []}
    for start in range(0, len(windows), batch_size):
        batch = torch.from_numpy(
            np.ascontiguousarray(windows[start : start + batch_size], np.float32)
        ).to(device)
        out = model(batch)
        for key, values in chunks.items():
            values.append(out[key].cpu().numpy())
    return {key: np.concatenate(values) for key, values in chunks.items()}


def pinball(prediction: torch.Tensor, target: torch.Tensor, quantile: float) -> torch.Tensor:
    diff = target - prediction
    return torch.maximum(quantile * diff, (quantile - 1) * diff).mean()


def multitask_loss(
    out: dict[str, torch.Tensor], rul: torch.Tensor, alert: torch.Tensor, alert_weight: float = 0.5
) -> torch.Tensor:
    lower, point, upper = out["quantiles"].unbind(dim=1)
    return (
        F.mse_loss(point, rul)
        + pinball(lower, rul, LOWER_Q)
        + pinball(upper, rul, UPPER_Q)
        + alert_weight * F.binary_cross_entropy_with_logits(out["alert_logit"], alert)
    )


def integrated_gradients(
    model: nn.Module, window: torch.Tensor, baseline: torch.Tensor | None = None, steps: int = 48
) -> torch.Tensor:
    """Attribute the point RUL estimate to each (time step, sensor) of one window.

    Integrated gradients (Sundararajan et al., 2017) with a midpoint Riemann sum. The default
    baseline is all zeros, which in normalized space is the average training engine in the same
    operating regime. The attributions add up (approximately) to f(window) - f(baseline).
    """
    if baseline is None:
        baseline = torch.zeros_like(window)
    alphas = (torch.arange(steps, dtype=window.dtype) + 0.5) / steps
    path = baseline + alphas.view(-1, 1, 1) * (window - baseline)
    path.requires_grad_(True)
    point = model(path)["quantiles"][:, 1].sum()
    (grads,) = torch.autograd.grad(point, path)
    return (window - baseline) * grads.mean(dim=0)
