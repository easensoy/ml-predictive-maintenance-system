import json
import shutil

import numpy as np
import pandas as pd
import pytest
import torch

from predmaint import data as cmapss
from predmaint.features import ContractError, FeatureContract
from predmaint.inference import ArtifactsIncompatible, ArtifactsMissing, Predictor
from predmaint.replay import FAILED_HOLD_TICKS, FleetReplay


def test_training_writes_every_artifact(artifacts_dir):
    model_dir = artifacts_dir / "FD001"
    for name in (
        "model.pt",
        "contract.json",
        "calibration.json",
        "metrics.json",
        "history.json",
        "replay.csv",
        "predictions/test.csv",
        "predictions/holdout.csv",
    ):
        assert (model_dir / name).is_file(), name
    metrics = json.loads((model_dir / "metrics.json").read_text())
    for key in ("rmse", "nasa_score", "interval_coverage", "alert"):
        assert key in metrics["test"]
    assert metrics["baseline_lightgbm"]["rmse"] > 0
    assert len(metrics["test_single_models"]) == 4
    split = metrics["data"]["split_engines"]
    assert set(split["train"]).isdisjoint(split["holdout"])


def test_checkpoint_loads_with_weights_only(artifacts_dir):
    bundle = torch.load(artifacts_dir / "FD001" / "model.pt", weights_only=True)
    assert len(bundle["members"]) == 2


def test_predictor_matches_saved_test_predictions(artifacts_dir, cmapss_dir):
    predictor = Predictor(artifacts_dir / "FD001")
    test = cmapss.load_test(cmapss_dir, "FD001")
    windows, units = predictor.contract.last_windows(test)
    out = predictor.predict_windows(windows)
    saved = pd.read_csv(artifacts_dir / "FD001" / "predictions" / "test.csv")
    np.testing.assert_array_equal(saved["unit"], units)
    np.testing.assert_allclose(saved["point"], out["point"], atol=1e-3)


def test_predict_readings_returns_calibrated_explained_output(artifacts_dir, cmapss_dir):
    predictor = Predictor(artifacts_dir / "FD001")
    engine = cmapss.load_test(cmapss_dir, "FD001").query("unit == 1")
    result = predictor.predict_readings(engine)
    rul = result["rul"]
    assert 0 <= rul["lower"] <= rul["point"] <= rul["upper"] <= 125
    assert 0 <= result["failure_within_horizon"]["probability"] <= 1
    assert result["status"] in {"healthy", "watch", "alert"}
    sensors = result["explanation"]["sensors"]
    assert len(sensors) == 14
    assert abs(sensors[0]["contribution_cycles"]) >= abs(sensors[-1]["contribution_cycles"])


def test_missing_artifacts_raise_a_clear_error(tmp_path):
    with pytest.raises(ArtifactsMissing, match="Train first"):
        Predictor(tmp_path)


@pytest.fixture
def run_copy(artifacts_dir, tmp_path):
    """A writable copy of the trained run, for tampering with."""
    return shutil.copytree(artifacts_dir / "FD001", tmp_path / "FD001")


def _edit_json(path, **changes):
    content = json.loads(path.read_text())
    content.update(changes)
    path.write_text(json.dumps(content))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"format": 1}, "format 1 is not supported"),
        ({"window": 0}, "window must be a positive integer"),
        ({"sensor_mean": [[0.0]]}, "sensor_mean must be"),
        ({"feature_min": [0.0] * 3}, "feature_min must hold 14"),
        ({"sensors": ["sensor_99"] * 14}, "without duplicates"),
        ({"surprise": 1}, "unknown keys"),
    ],
)
def test_incompatible_contract_fails_at_startup_with_a_clear_error(run_copy, changes, message):
    _edit_json(run_copy / "contract.json", **changes)
    with pytest.raises(ArtifactsIncompatible, match=message):
        Predictor(run_copy)


def test_contract_without_training_range_is_rejected(run_copy):
    content = json.loads((run_copy / "contract.json").read_text())
    del content["feature_max"]
    with pytest.raises(ContractError, match="missing keys"):
        FeatureContract.from_dict(content)


def test_checkpoint_format_and_shape_are_checked(run_copy):
    path = run_copy / "model.pt"
    bundle = torch.load(path, weights_only=True)
    torch.save({**bundle, "format": 99}, path)
    with pytest.raises(ArtifactsIncompatible, match="checkpoint format 99"):
        Predictor(run_copy)
    torch.save({**bundle, "config": {**bundle["config"], "n_features": 13}}, path)
    with pytest.raises(ArtifactsIncompatible, match="different runs"):
        Predictor(run_copy)
    torch.save({**bundle, "config": {**bundle["config"], "hidden": 32}}, path)
    with pytest.raises(ArtifactsIncompatible, match="does not match its configuration"):
        Predictor(run_copy)


def test_calibration_values_are_checked(run_copy):
    _edit_json(run_copy / "calibration.json", temperature=-1.0)
    with pytest.raises(ArtifactsIncompatible, match="temperature > 0"):
        Predictor(run_copy)


def test_replay_streams_cycles_and_restarts_failed_engines(artifacts_dir):
    predictor = Predictor(artifacts_dir / "FD001")
    replay_rows = pd.read_csv(artifacts_dir / "FD001" / "replay.csv")
    fleet = FleetReplay(predictor, replay_rows, n_engines=2)
    unit = fleet.units[0]
    start = fleet.snapshot()["engines"][0]["cycle"]
    assert fleet.step()["engines"][0]["cycle"] == start + 1

    total = len(fleet.frames[unit])
    for _ in range(total - start - 1):
        fleet.step()
    snapshot = fleet.step()
    engine = next(e for e in snapshot["engines"] if e["unit"] == unit)
    assert engine["status"] == "failed"
    assert engine["rul_true"] == 0
    for _ in range(FAILED_HOLD_TICKS):
        fleet.step()
    engine = next(e for e in fleet.snapshot()["engines"] if e["unit"] == unit)
    assert engine["cycle"] == 1
    assert engine["lap"] == 2

    detail = fleet.detail(unit)
    assert len(detail["attribution"]["values"]) == predictor.contract.window
    assert len(detail["contributions"]) == 14
    assert len(detail["traces"]["sensors"]) == 3
