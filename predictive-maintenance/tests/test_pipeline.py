import json

import numpy as np
import pandas as pd
import pytest
import torch

from predmaint import data as cmapss
from predmaint.inference import ArtifactsMissing, Predictor
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
