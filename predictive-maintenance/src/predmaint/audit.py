"""Recompute the README results from the committed predictions, and check that they agree.

    python -m predmaint.audit          # exit code 1 and a list of problems on any mismatch

Three things must agree: the per-engine predictions committed under
``artifacts/<subset>/predictions/``, the summary numbers in ``artifacts/<subset>/metrics.json``,
and the results table in the README. The test suite runs the same check, so CI fails if any of
the three drifts from the others.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from predmaint.config import PROJECT_ROOT, Settings
from predmaint.metrics import alert_metrics, interval_coverage, lead_times, nasa_score, rmse

SUBSETS = ("FD001", "FD002", "FD003", "FD004")
DOCUMENTS = (PROJECT_ROOT.parent / "README.md", PROJECT_ROOT.parent / "MODEL_CARD.md")

# (name in recompute(), path in metrics.json, absolute tolerance). Predictions are stored with
# four decimals and metrics.json was computed before rounding, hence the small tolerances.
CHECKS = [
    ("rmse", ("test", "rmse"), 1e-3),
    ("nasa_score", ("test", "nasa_score"), 0.05),
    ("rmse_uncapped", ("test", "rmse_uncapped"), 1e-3),
    ("interval_coverage", ("test", "interval_coverage"), 1e-9),
    ("interval_width", ("test", "interval_width"), 1e-3),
    ("alert_ap", ("test", "alert", "average_precision"), 1e-3),
    ("lead_median", ("holdout", "lead_time", "median"), 1e-9),
    ("lead_missed", ("holdout", "lead_time", "missed"), 0),
    ("lead_units", ("holdout", "lead_time", "units"), 0),
]


def recompute(run_dir: Path) -> dict:
    """Every number in the README table that can be derived from the stored predictions."""
    metrics = json.loads((run_dir / "metrics.json").read_text())
    cfg = metrics["config"]
    threshold = metrics["calibration"]["alert_threshold"]
    test = pd.read_csv(run_dir / "predictions" / "test.csv")
    true = test["rul_true"].to_numpy(float)
    capped = np.minimum(true, cfg["max_rul"])
    point, lower, upper = (test[c].to_numpy(float) for c in ("point", "lower", "upper"))
    inside_capped = (capped >= lower) & (capped <= upper)
    inside_actual = (true >= lower) & (true <= upper)
    probability = test["probability"].to_numpy(float)
    alert = alert_metrics(probability, true <= cfg["alert_cycles"], threshold)
    holdout = pd.read_csv(run_dir / "predictions" / "holdout.csv")
    lead = lead_times(holdout, threshold, early_cycles=2 * cfg["alert_cycles"])
    caught = [u["lead_time"] for u in lead["per_unit"] if u["lead_time"] is not None]
    return {
        "test_engines": len(test),
        "rmse": rmse(point, capped),
        "nasa_score": nasa_score(point, capped),
        "rmse_uncapped": rmse(point, true),
        "interval_coverage": interval_coverage(lower, upper, capped),
        "inside_capped": int(inside_capped.sum()),
        "inside_actual": int(inside_actual.sum()),
        "interval_width": float(np.mean(upper - lower)),
        "alert_ap": alert["average_precision"],
        "lead_median": lead["median"],
        "lead_min": min(caught) if caught else None,
        "lead_max": max(caught) if caught else None,
        "lead_missed": lead["missed"],
        "lead_units": lead["units"],
        "lead_per_unit": lead["per_unit"],
    }


def _lookup(metrics: dict, path: tuple[str, ...]) -> object:
    value: object = metrics
    for key in path:
        value = value[key]  # type: ignore[index]
    return value


def check_run(run_dir: Path) -> list[str]:
    """Problems found when comparing one run's predictions with its metrics.json."""
    metrics = json.loads((run_dir / "metrics.json").read_text())
    found = recompute(run_dir)
    problems = []
    for name, path, tolerance in CHECKS:
        stored, value = _lookup(metrics, path), found[name]
        if stored is None or value is None:
            if stored is not value:
                problems.append(f"{run_dir.name} {name}: stored {stored}, recomputed {value}")
        elif abs(float(stored) - float(value)) > tolerance:  # type: ignore[arg-type]
            problems.append(f"{run_dir.name} {name}: stored {stored}, recomputed {value:.6g}")
    if found["lead_per_unit"] != metrics["holdout"]["lead_time"]["per_unit"]:
        problems.append(f"{run_dir.name}: per-engine alert lead times differ from metrics.json")
    if found["test_engines"] != metrics["data"]["test_engines"]:
        problems.append(f"{run_dir.name}: test.csv has {found['test_engines']} engines")
    return problems


def readme_block(readme: Path) -> str:
    from predmaint.report import END, START

    text = readme.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise ValueError(f"{readme} has no {START} ... {END} block")
    return text.split(START, 1)[1].split(END, 1)[0].strip("\n")


def audit(artifacts: Path, *documents: Path) -> list[str]:
    """Problems found in the runs under ``artifacts`` and in each document's results block."""
    from predmaint.report import results_table

    runs = [artifacts / s for s in SUBSETS if (artifacts / s / "metrics.json").is_file()]
    if not runs:
        return [f"no trained runs in {artifacts}"]
    problems = [problem for run in runs for problem in check_run(run)]
    expected = results_table(artifacts)
    for document in documents:
        if readme_block(document) != expected:
            problems.append(
                f"the results table in {document.name} differs from the table generated from "
                "the artifacts; rerun `python -m predmaint.report`"
            )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m predmaint.audit", description=__doc__)
    parser.add_argument(
        "--docs",
        type=Path,
        nargs="*",
        default=list(DOCUMENTS),
        help="files whose results block must match (default: README.md and MODEL_CARD.md)",
    )
    parser.add_argument("--artifacts", type=Path, default=None)
    args = parser.parse_args(argv)
    artifacts = args.artifacts or Settings.from_env().artifacts_dir
    problems = audit(artifacts, *args.docs)
    for problem in problems:
        print(f"MISMATCH  {problem}")
    if problems:
        return 1
    for subset in SUBSETS:
        if (artifacts / subset / "metrics.json").is_file():
            found = recompute(artifacts / subset)
            ap = "n/a" if found["alert_ap"] is None else f"{found['alert_ap']:.3f}"
            print(
                f"{subset}: RMSE {found['rmse']:.2f}, PHM08 {found['nasa_score']:.0f}, "
                f"{found['inside_capped']}/{found['test_engines']} inside the interval, "
                f"alert AP {ap}: predictions, metrics.json and documents agree"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
