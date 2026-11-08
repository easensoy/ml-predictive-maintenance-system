import numpy as np
import pandas as pd
import pytest
import torch

from predmaint.calibration import (
    covered,
    cqr_scores,
    cross_conformal_margins,
    cv_plus_interval,
    finalize_interval,
    fit_temperature,
    probability,
    split_conformal_margin,
)
from predmaint.metrics import (
    alert_behaviour,
    ece_with_interval,
    expected_calibration_error,
    mean_sd,
    nasa_score,
    per_engine_weights,
    rmse,
    weighted_rmse,
    wilson_interval,
)
from predmaint.model import Ensemble, RULNet, integrated_gradients, multitask_loss, predict_raw


@pytest.fixture
def model():
    torch.manual_seed(0)
    return RULNet(n_features=5, hidden=8, layers=2, window=12).eval()


def test_outputs_have_expected_shapes_and_ordered_bounds(model):
    out = model(torch.randn(7, 12, 5))
    assert out["quantiles"].shape == (7, 3)
    assert out["alert_logit"].shape == (7,)
    assert out["attention"].shape == (7, 12)
    lower, point, upper = out["quantiles"].unbind(dim=1)
    assert torch.all(lower <= point)
    assert torch.all(point <= upper)
    torch.testing.assert_close(out["attention"].sum(dim=1), torch.ones(7))


def test_network_reads_only_its_own_window(model):
    x = torch.randn(3, 20, 5)
    torch.testing.assert_close(model(x)["quantiles"], model(x[:, -12:])["quantiles"])
    with pytest.raises(ValueError, match="needs 12"):
        model(torch.randn(3, 8, 5))


def test_loss_is_finite_and_backpropagates(model):
    model.train()
    out = model(torch.randn(4, 12, 5))
    loss = multitask_loss(out, torch.rand(4), torch.tensor([0.0, 1.0, 0.0, 1.0]))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None for p in model.parameters())


def test_ensemble_keeps_members_separate_and_averages_the_point():
    torch.manual_seed(1)
    members = [RULNet(5, 8, 1, window=w).eval() for w in (4, 6, 6)]
    ensemble = Ensemble(members)
    x = torch.randn(2, 6, 5)
    out = ensemble(x)
    assert out["quantiles"].shape == (3, 2, 3)
    assert out["alert_logit"].shape == (3, 2)
    assert ensemble.window == 6
    expected = torch.stack([m(x)["quantiles"][:, 1] for m in members]).mean(dim=0)
    torch.testing.assert_close(ensemble.point(x), expected)


def test_integrated_gradients_satisfy_completeness():
    torch.manual_seed(2)
    ensemble = Ensemble([RULNet(5, 8, 1, window=w).eval() for w in (8, 12)])
    window = torch.randn(12, 5)
    attributions = integrated_gradients(ensemble.point, window, steps=256)
    with torch.no_grad():
        delta = ensemble.point(window[None])[0] - ensemble.point(torch.zeros(1, 12, 5))[0]
    assert attributions.shape == (12, 5)
    assert abs(float(attributions.sum() - delta)) < 0.02 * max(1.0, abs(float(delta)))


def test_shorter_member_attributes_nothing_to_rows_it_cannot_see():
    torch.manual_seed(3)
    ensemble = Ensemble([RULNet(5, 8, 1, window=4).eval()])
    attributions = integrated_gradients(ensemble.point, torch.randn(10, 5))
    assert torch.all(attributions[:6] == 0)


def test_predict_raw_matches_forward(model):
    windows = np.random.default_rng(0).normal(size=(9, 12, 5)).astype(np.float32)
    raw = predict_raw(model, windows, batch_size=4)
    with torch.no_grad():
        direct = model(torch.from_numpy(windows))["quantiles"].numpy()
    np.testing.assert_allclose(raw["quantiles"], direct, rtol=1e-5, atol=1e-6)
    ensemble = Ensemble([model, model])
    assert predict_raw(ensemble, windows, batch_size=4)["quantiles"].shape == (2, 9, 3)


def _toy_cv_plus(n: int, members: int, rng: np.random.Generator, alpha: float = 0.1):
    """A deliberately miscalibrated model (bounds too narrow) on exchangeable data."""
    member = rng.integers(0, members, n)
    noise = rng.normal(size=n)
    scores = cqr_scores(np.full(n, -0.5), np.full(n, 0.5), noise)
    fresh = rng.normal(size=4000)
    lower, upper = cv_plus_interval(
        np.full((members, 4000), -0.5), np.full((members, 4000), 0.5), scores, member, alpha
    )
    return np.mean((fresh >= lower) & (fresh <= upper))


def test_cv_plus_reaches_nominal_coverage_on_exchangeable_data():
    rng = np.random.default_rng(0)
    coverage = np.mean([_toy_cv_plus(200, 5, rng) for _ in range(20)])
    assert 0.87 <= coverage <= 0.93


def test_cv_plus_is_unbounded_when_there_are_too_few_scores():
    lower, upper = cv_plus_interval(
        np.zeros((2, 3)), np.ones((2, 3)), np.array([0.1, 0.2]), np.array([0, 1]), 0.1
    )
    assert np.all(np.isneginf(lower))
    assert np.all(np.isposinf(upper))


def test_split_and_cross_conformal_margins():
    rng = np.random.default_rng(1)
    scores = rng.normal(size=999)
    assert split_conformal_margin(scores, 0.1) == pytest.approx(np.sort(scores)[899])
    margins = cross_conformal_margins(scores, np.arange(999) % 3, 3, 0.1)
    assert margins.shape == (3,)
    assert np.all(np.abs(margins - np.quantile(scores, 0.9)) < 0.2)


def test_open_ended_intervals_cover_any_rul_above_the_lower_end():
    lower, upper, open_ended = finalize_interval(
        np.array([-5.0, 40.0, 130.0]), np.array([20.0, 140.0, 150.0]), 125
    )
    np.testing.assert_array_equal(lower, [0.0, 40.0, 125.0])
    np.testing.assert_array_equal(upper, [20.0, 125.0, 125.0])
    np.testing.assert_array_equal(open_ended, [False, True, True])
    actual = np.array([21.0, 190.0, 300.0])
    np.testing.assert_array_equal(covered(lower, upper, open_ended, actual), [False, True, True])
    capped = np.minimum(actual, 125)
    np.testing.assert_array_equal(
        covered(lower, upper, open_ended, capped), covered(lower, upper, open_ended, actual)
    )


def test_temperature_scaling_fixes_overconfidence_and_honours_weights():
    rng = np.random.default_rng(0)
    true_logit = rng.normal(0, 2, 20000)
    labels = (rng.uniform(size=20000) < 1 / (1 + np.exp(-true_logit))).astype(float)
    overconfident = 3.0 * true_logit
    temperature = fit_temperature(overconfident, labels)
    assert 2.6 < temperature < 3.4
    before = expected_calibration_error(1 / (1 + np.exp(-overconfident)), labels)
    after = expected_calibration_error(probability(overconfident, temperature), labels)
    assert after < before
    doubled = fit_temperature(
        np.r_[overconfident, overconfident], np.r_[labels, labels], np.full(40000, 0.5)
    )
    assert doubled == pytest.approx(temperature, rel=1e-3)


def test_probability_averages_members():
    logits = np.array([[0.0, 2.0], [0.0, -2.0]])
    np.testing.assert_allclose(probability(logits, 1.0), [0.5, 0.5])


def test_nasa_score_penalises_late_predictions_more():
    assert nasa_score(np.array([10.0]), np.array([0.0])) == pytest.approx(np.e - 1)
    assert nasa_score(np.array([0.0]), np.array([13.0])) == pytest.approx(np.e - 1)
    assert nasa_score(np.array([20.0]), np.array([10.0])) > nasa_score(
        np.array([0.0]), np.array([10.0])
    )
    assert rmse(np.array([1.0, 3.0]), np.array([1.0, 1.0])) == pytest.approx(np.sqrt(2))


def test_every_engine_counts_once_in_weighted_rmse():
    units = np.array([1, 1, 1, 1, 2])
    weights = per_engine_weights(units)
    np.testing.assert_allclose(weights, [0.25, 0.25, 0.25, 0.25, 1.0])
    error = weighted_rmse(np.array([1.0, 1, 1, 1, 3]), np.zeros(5), weights)
    assert error == pytest.approx(np.sqrt((1 + 9) / 2))


def test_wilson_interval_matches_known_values():
    low, high = wilson_interval(83, 100)
    assert (round(low, 3), round(high, 3)) == (0.745, 0.891)
    assert wilson_interval(0, 10)[0] == 0.0


def test_alert_behaviour_definitions():
    frame = pd.DataFrame(
        {
            "unit": [1] * 6 + [2] * 3 + [3] * 3,
            "cycle": [1, 2, 3, 4, 5, 6, 1, 2, 3, 1, 2, 3],
            "rul": [50, 40, 31, 20, 10, 0, 40, 1, 0, 20, 10, 0],
            "probability": [0.6, 0.1, 0.2, 0.7, 0.3, 0.9, 0.1, 0.2, 0.9, 0.6, 0.6, 0.6],
        }
    )
    b = alert_behaviour(frame, threshold=0.5, horizon=30)
    per_unit = {u["unit"]: u for u in b["per_unit"]}
    # Engine 1 first alerts at 50 cycles left (early, one false episode), drops out twice.
    assert per_unit[1] == {"unit": 1, "lead_time": 50, "false_alerts": 1, "flaps": 2}
    # Engine 2 only alerts at the failure cycle itself, which counts as missed.
    assert per_unit[2]["lead_time"] is None
    assert per_unit[3] == {"unit": 3, "lead_time": 20, "false_alerts": 0, "flaps": 0}
    assert (b["missed"], b["early"], b["engines_with_false_alert"]) == (1, 1, 1)
    assert b["lead_median"] == pytest.approx(35.0)


def test_ece_interval_resamples_whole_engines():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(50), 40)
    prob = rng.uniform(size=2000)
    label = (rng.uniform(size=2000) < prob).astype(float)
    out = ece_with_interval(prob, label, groups, n_boot=300, seed=1)
    assert out["ci_low"] <= out["ece"] <= out["ci_high"]
    assert out["engines"] == 50
    assert out["ece"] == pytest.approx(expected_calibration_error(prob, label))


def test_mean_sd_over_repeats():
    assert mean_sd([1.0, 3.0]) == {"mean": 2.0, "sd": pytest.approx(np.sqrt(2)), "n": 2}
    assert mean_sd([5.0])["sd"] is None
    assert mean_sd([None])["mean"] is None
