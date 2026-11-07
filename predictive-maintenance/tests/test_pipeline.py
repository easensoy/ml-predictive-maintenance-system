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
from predmaint.train import TrainConfig, run


def test_training_writes_every_artifact(artifacts_dir):
    model_dir = artifacts_dir / "FD001"
    for name in (
        "model.pt",
        "contract.json",
        "calibration.json",
        "metrics.json",
        "manifest.json",
        "history.json",
        "replay.csv",
        "predictions/test.csv",
        "predictions/test-members.csv",
        "predictions/calibration-points.csv",
        "predictions/oof.csv.gz",
        "predictions/replay-holdout.csv.gz",
    ):
        assert (model_dir / name).is_file(), name
    metrics = json.loads((model_dir / "metrics.json").read_text())
    for model in ("lstm", "lightgbm"):
        for key in ("rmse", "nasa_score", "coverage", "alert_ap"):
            assert key in metrics["test"]["models"][model]["summary"]
    assert len(metrics["test"]["models"]["lstm"]["per_repeat"]) == 2
    assert "lstm_minus_lightgbm" in metrics["test"]


def test_training_never_reads_the_test_labels(cmapss_dir, tmp_path):
    """Training must work with the RUL file gone: only `predmaint.evaluate` may read it."""
    data = shutil.copytree(cmapss_dir, tmp_path / "data")
    (data / "RUL_FD001.txt").unlink()
    from predmaint.config import Settings

    tiny = TrainConfig(
        subset="FD001",
        repeats=[0],
        folds=2,
        windows=[6],
        epochs=1,
        hidden=8,
        layers=1,
        temperature_draws=1,
        baseline=False,
    )
    metrics = run(tiny, Settings(data_dir=data, artifacts_dir=tmp_path), tmp_path / "r", None)
    assert metrics["provenance"]["test_labels_read"] is False
    assert "rul_true" not in pd.read_csv(tmp_path / "r" / "predictions" / "test.csv")


def test_replay_engines_are_held_out_of_everything(artifacts_dir):
    metrics = json.loads((artifacts_dir / "FD001" / "metrics.json").read_text())
    replay = set(metrics["data"]["replay_engines"])
    for folds in metrics["data"]["folds"].values():
        assert replay.isdisjoint(int(u) for u in folds)
    assert set(pd.read_csv(artifacts_dir / "FD001" / "replay.csv")["unit"]) == replay


def test_every_calibration_score_comes_from_a_network_that_never_saw_the_engine(artifacts_dir):
    metrics = json.loads((artifacts_dir / "FD001" / "metrics.json").read_text())
    points = pd.read_csv(artifacts_dir / "FD001" / "predictions" / "calibration-points.csv")
    for repeat, folds in metrics["data"]["folds"].items():
        fold_of = {int(u): f for u, f in folds.items()}
        rows = points[points["repeat"] == int(repeat)]
        assert (rows["member"] == rows["unit"].map(fold_of)).all()
        assert rows.groupby("model")["unit"].nunique().eq(len(fold_of)).all()  # one per engine
        for k, units in metrics["data"]["inner_validation"][repeat].items():
            assert all(fold_of[u] != int(k) for u in units)


def test_checkpoint_loads_with_weights_only(artifacts_dir):
    bundle = torch.load(artifacts_dir / "FD001" / "model.pt", weights_only=True)
    assert len(bundle["members"]) == 3  # one network per fold


def test_predictor_matches_saved_test_predictions(artifacts_dir, cmapss_dir):
    predictor = Predictor(artifacts_dir / "FD001")
    test = cmapss.load_test_inputs(cmapss_dir, "FD001")
    windows, units = predictor.contract.last_windows(test)
    out = predictor.predict_windows(windows)
    saved = pd.read_csv(artifacts_dir / "FD001" / "predictions" / "test.csv")
    served = saved[(saved["model"] == "lstm") & (saved["repeat"] == 0)].sort_values("unit")
    np.testing.assert_array_equal(served["unit"], units)
    np.testing.assert_allclose(served["point"], out["point"], atol=0.01)
    np.testing.assert_allclose(served["lower"], out["lower"], atol=0.02)
    np.testing.assert_array_equal(served["open_ended"], out["open_ended"])


def test_predict_readings_returns_calibrated_explained_output(artifacts_dir, cmapss_dir):
    predictor = Predictor(artifacts_dir / "FD001")
    engine = cmapss.load_test_inputs(cmapss_dir, "FD001").query("unit == 1")
    result = predictor.predict_readings(engine)
    rul = result["rul"]
    assert 0 <= rul["lower"] <= rul["point"] <= 125
    assert rul["open_ended"] == (rul["upper"] is None)
    assert rul["summary"].startswith("at least") == rul["open_ended"]
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

    def with_config(**changes):
        first = {**bundle["members"][0], "config": {**bundle["members"][0]["config"], **changes}}
        torch.save({**bundle, "members": [first, *bundle["members"][1:]]}, path)

    with_config(n_features=13)
    with pytest.raises(ArtifactsIncompatible, match="different runs"):
        Predictor(run_copy)
    with_config(window=500)
    with pytest.raises(ArtifactsIncompatible, match="500-cycle window"):
        Predictor(run_copy)
    with_config(hidden=32)
    with pytest.raises(ArtifactsIncompatible, match="does not match its configuration"):
        Predictor(run_copy)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"temperature": -1.0}, "temperature > 0"),
        ({"format": 1}, "calibration format 1"),
        ({"score_member": [7]}, "index of one of the 3 networks"),
    ],
)
def test_calibration_values_are_checked(run_copy, changes, message):
    _edit_json(run_copy / "calibration.json", **changes)
    with pytest.raises(ArtifactsIncompatible, match=message):
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
    assert "open_ended" in engine
    for _ in range(FAILED_HOLD_TICKS):
        fleet.step()
    engine = next(e for e in fleet.snapshot()["engines"] if e["unit"] == unit)
    assert engine["cycle"] == 1
    assert engine["lap"] == 2

    detail = fleet.detail(unit)
    assert len(detail["attribution"]["values"]) == predictor.contract.window
    assert len(detail["contributions"]) == 14
    assert len(detail["traces"]["sensors"]) == 3
