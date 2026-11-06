import numpy as np
import pandas as pd
import pytest
import torch

from predmaint.calibration import calibrated_outputs, conformal_margin, fit_temperature
from predmaint.metrics import (
    alert_metrics,
    expected_calibration_error,
    lead_times,
    nasa_score,
    rmse,
)
from predmaint.model import Ensemble, RULNet, integrated_gradients, multitask_loss, predict_raw


@pytest.fixture
def model():
    torch.manual_seed(0)
    return RULNet(n_features=5, hidden=8, layers=2).eval()


def test_outputs_have_expected_shapes_and_ordered_bounds(model):
    out = model(torch.randn(7, 12, 5))
    assert out["quantiles"].shape == (7, 3)
    assert out["alert_logit"].shape == (7,)
    assert out["attention"].shape == (7, 12)
    lower, point, upper = out["quantiles"].unbind(dim=1)
    assert torch.all(lower <= point)
    assert torch.all(point <= upper)
    torch.testing.assert_close(out["attention"].sum(dim=1), torch.ones(7))


def test_loss_is_finite_and_backpropagates(model):
    model.train()
    out = model(torch.randn(4, 12, 5))
    loss = multitask_loss(out, torch.rand(4), torch.tensor([0.0, 1.0, 0.0, 1.0]))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None for p in model.parameters())


def test_ensemble_averages_members():
    torch.manual_seed(1)
    members = [RULNet(5, 8, 1).eval() for _ in range(3)]
    x = torch.randn(2, 6, 5)
    expected = torch.stack([m(x)["quantiles"] for m in members]).mean(dim=0)
    torch.testing.assert_close(Ensemble(members)(x)["quantiles"], expected)


def test_integrated_gradients_satisfy_completeness(model):
    window = torch.randn(12, 5)
    attributions = integrated_gradients(model, window, steps=256)
    with torch.no_grad():
        delta = (
            model(window[None])["quantiles"][0, 1] - model(torch.zeros(1, 12, 5))["quantiles"][0, 1]
        )
    assert attributions.shape == (12, 5)
    assert abs(float(attributions.sum() - delta)) < 0.02 * max(1.0, abs(float(delta)))


def test_predict_raw_matches_forward(model):
    windows = np.random.default_rng(0).normal(size=(9, 12, 5)).astype(np.float32)
    raw = predict_raw(model, windows, batch_size=4)
    with torch.no_grad():
        direct = model(torch.from_numpy(windows))["quantiles"].numpy()
    np.testing.assert_allclose(raw["quantiles"], direct, rtol=1e-5, atol=1e-6)


def test_conformal_margin_reaches_target_coverage():
    rng = np.random.default_rng(0)
    target = rng.normal(size=5000)
    lower, upper = np.full(5000, -0.5), np.full(5000, 0.5)
    margin = conformal_margin(lower, upper, target, alpha=0.1)
    fresh = rng.normal(size=5000)
    coverage = np.mean((fresh >= lower - margin) & (fresh <= upper + margin))
    assert 0.88 <= coverage <= 0.92


def test_temperature_scaling_fixes_overconfidence():
    rng = np.random.default_rng(0)
    true_logit = rng.normal(0, 2, 20000)
    labels = (rng.uniform(size=20000) < 1 / (1 + np.exp(-true_logit))).astype(float)
    overconfident = 3.0 * true_logit
    temperature = fit_temperature(overconfident, labels)
    assert 2.6 < temperature < 3.4
    before = expected_calibration_error(1 / (1 + np.exp(-overconfident)), labels)
    after = expected_calibration_error(1 / (1 + np.exp(-overconfident / temperature)), labels)
    assert after < before


def test_calibrated_outputs_are_clipped_to_valid_range():
    q = np.array([[-0.2, 0.1, 1.4]])
    out = calibrated_outputs(q, np.array([0.0]), margin=0.05, temperature=1.0, max_rul=125)
    assert out["lower"][0] == 0
    assert out["upper"][0] == 125
    assert out["point"][0] == pytest.approx(12.5)
    assert out["probability"][0] == pytest.approx(0.5)


def test_nasa_score_penalises_late_predictions_more():
    assert nasa_score(np.array([10.0]), np.array([0.0])) == pytest.approx(np.e - 1)
    assert nasa_score(np.array([0.0]), np.array([13.0])) == pytest.approx(np.e - 1)
    assert nasa_score(np.array([20.0]), np.array([10.0])) > nasa_score(
        np.array([0.0]), np.array([10.0])
    )
    assert rmse(np.array([1.0, 3.0]), np.array([1.0, 1.0])) == pytest.approx(np.sqrt(2))


def test_alert_metrics_and_lead_times():
    probability = np.array([0.1, 0.6, 0.9, 0.2])
    label = np.array([0, 1, 1, 1])
    m = alert_metrics(probability, label, 0.5)
    assert (m["precision"], m["recall"], m["positives"]) == (1.0, pytest.approx(2 / 3), 3)

    frame = pd.DataFrame(
        {
            "unit": [1, 1, 1, 2, 2],
            "cycle": [1, 2, 3, 1, 2],
            "rul_true": [40, 20, 0, 5, 0],
            "probability": [0.1, 0.7, 0.9, 0.1, 0.2],
        }
    )
    lead = lead_times(frame, threshold=0.5, early_cycles=60)
    assert lead["per_unit"] == [{"unit": 1, "lead_time": 20}, {"unit": 2, "lead_time": None}]
    assert (lead["missed"], lead["median"]) == (1, 20.0)
