"""What the README shows must be the evaluated predictions.

The summary statistics behind the figures must add up to the tabulated metrics. The dashboard and
the API must reproduce the evaluated predictions. The audit must notice when a recorded product
output stops matching them.
"""

import json
import shutil

import pandas as pd
import pytest

from predmaint import report
from predmaint.audit import check_examples
from predmaint.data import SETTINGS, load_test_inputs
from predmaint.inference import Predictor
from predmaint.replay import FleetReplay


def test_five_model_averages_are_the_tabulated_rmse(artifacts_dir):
    run_dir = artifacts_dir / "FD001"
    metrics = json.loads((run_dir / "metrics.json").read_text())
    stats = report.single_models(run_dir)
    folds, repeats = metrics["config"]["folds"], metrics["config"]["repeats"]
    for model, block in metrics["test"]["models"].items():
        tabulated = [block["per_repeat"][str(r)]["rmse"] for r in repeats]
        assert stats[model]["averages"] == pytest.approx(tabulated, abs=1e-9)
        assert len(stats[model]["singles"]) == folds * len(repeats)


def test_coverage_by_remaining_life_adds_up_to_the_overall_coverage(artifacts_dir):
    run_dir = artifacts_dir / "FD001"
    metrics = json.loads((run_dir / "metrics.json").read_text())
    engines = metrics["data"]["test_engines"]
    table = report.coverage_by_rul({"FD001": run_dir})
    for model, rows in table.groupby("model"):
        assert rows["engines"].sum() == engines
        overall = float((rows["coverage"] * rows["engines"]).sum()) / engines
        expected = metrics["test"]["models"][model]["summary"]["coverage"]["mean"]
        assert overall == pytest.approx(expected, abs=1e-9)
        assert ((rows["low"] <= rows["coverage"]) & (rows["coverage"] <= rows["high"])).all()


def test_the_replay_shows_the_evaluated_prediction_at_a_sought_cycle(artifacts_dir):
    run_dir = artifacts_dir / "FD001"
    fleet = FleetReplay(Predictor(run_dir), pd.read_csv(run_dir / "replay.csv"), n_engines=50)
    evaluated = report.served_replay(run_dir)
    unit = fleet.units[0]
    life = evaluated[evaluated["unit"] == unit].sort_values("cycle")
    row = life.iloc[len(life) // 2]
    fleet.seek(unit, int(row["cycle"]))
    detail = fleet.detail(unit)
    shown = detail["history"][-1]
    assert len(detail["history"]) == len(life) // 2 + 1
    assert shown["cycle"] == int(row["cycle"])
    assert shown["rul_true"] == int(row["rul"])
    assert shown["point"] == pytest.approx(row["point"], abs=0.051)
    assert shown["lower"] == pytest.approx(row["lower"], abs=0.051)
    assert shown["probability"] == pytest.approx(row["probability"], abs=2e-4)
    with pytest.raises(ValueError, match="no recorded cycle"):
        fleet.seek(unit, 10_000)


def test_the_api_reproduces_the_test_prediction_of_an_engine(artifacts_dir, cmapss_dir, tmp_path):
    run_dir = artifacts_dir / "FD001"
    inputs = load_test_inputs(cmapss_dir, "FD001")
    unit = int(inputs["unit"].iloc[0])
    engine = inputs[inputs["unit"] == unit].sort_values("cycle")
    columns = [*SETTINGS, *Predictor(run_dir).contract.sensors]
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"readings": engine[columns].to_dict("records")}))
    match = report.matching_test_engine(report.example_response(run_dir, request), run_dir)
    assert match is not None
    assert int(match["unit"]) == unit


@pytest.fixture
def project(artifacts_dir, tmp_path):
    """A copy of the synthetic run laid out like the repository: artifacts/ next to docs/."""
    root = tmp_path / "project"
    shutil.copytree(artifacts_dir, root / "artifacts")
    return root / "artifacts"


def _dashboard_state(artifacts) -> dict:
    hero = report.hero_state(artifacts / "FD001")
    row = hero["row"]
    return {
        "subset": "FD001",
        "unit": hero["unit"],
        "cycle": hero["cycle"],
        "rul_true": int(row["rul"]),
        "point": round(float(row["point"]), 1),
        "lower": round(float(row["lower"]), 1),
        "upper": None if row["open_ended"] else round(float(row["upper"]), 1),
        "open_ended": bool(row["open_ended"]),
        "probability": round(float(row["probability"]), 4),
    }


def test_the_audit_accepts_the_dashboard_state_the_replay_predicts(project):
    path = report.example_files(project)["dashboard"]
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(_dashboard_state(project)))
    assert check_examples(project) == []


def test_the_audit_catches_a_dashboard_screenshot_of_another_cycle(project):
    state = _dashboard_state(project)
    state["cycle"] -= 1
    path = report.example_files(project)["dashboard"]
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(state))
    assert any("cycle" in p for p in check_examples(project))


def test_the_audit_catches_an_api_example_that_no_test_engine_produced(project):
    path = report.example_files(project)["response"]
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "rul": {"point": 999.0, "lower": 998.0, "upper": None, "open_ended": True},
                "failure_within_horizon": {"probability": 0.5},
            }
        )
    )
    assert any("example-response.json" in p for p in check_examples(project))
