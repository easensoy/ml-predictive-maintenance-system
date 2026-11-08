"""Recompute the published results from the committed predictions, and check that they agree.

    python -m predmaint.audit          # exit code 1 and a list of problems on any mismatch

For every run in ``artifacts/``:

1. **Split rules.** No engine is scored by a network that trained on it, and no fold's inner
   validation engines (which chose its window and stopped its training) come from that fold.
2. **Intervals.** The CV+ intervals and alert probabilities in ``predictions/test.csv`` are
   rebuilt from each network's raw test outputs (``test-members.csv``) and the stored
   conformity scores (``calibration-points.csv``), and the scores are rebuilt from their raw
   bounds.
3. **Metrics.** Every test, out-of-fold and replay-holdout metric in ``metrics.json`` is
   recomputed from the prediction files.
4. **Manifest.** Every artifact present has the SHA-256 that ``manifest.json`` recorded when
   the run was evaluated, and every file this audit reads is listed there.
5. **Product outputs.** The example API response in ``docs/`` reproduces a test-set
   prediction, and the state the dashboard screenshots show is the evaluated replay
   prediction for the engine and cycle the hero rule picks.
6. **Documents.** Every generated block in README.md, MODEL_CARD.md, docs/evaluation.md and
   docs/serving.md equals the block generated from the artifacts.

These checks establish that the documents agree with the committed artifacts. They do not
establish that the method is valid: the conditions behind the intervals, and the limits of each
statistic, are argued in docs/evaluation.md.

The test suite runs the same audit, so CI fails if any of these drifts.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from predmaint.calibration import cqr_scores, cv_plus_interval, finalize_interval, probability
from predmaint.config import PROJECT_ROOT, Settings
from predmaint.evaluate import manifest_files, paired_differences, repeat_metrics
from predmaint.train import TrainConfig, whole_life_metrics

SUBSETS = ("FD001", "FD002", "FD003", "FD004")
# The files the audit recomputes from: their hashes must be in the manifest.
AUDITED_FILES = (
    "metrics.json",
    "predictions/test.csv",
    "predictions/test-members.csv",
    "predictions/calibration-points.csv",
    "predictions/oof.csv.gz",
    "predictions/replay-holdout.csv.gz",
)
DOCUMENTS = (
    PROJECT_ROOT.parent / "README.md",
    PROJECT_ROOT.parent / "MODEL_CARD.md",
    PROJECT_ROOT / "docs" / "evaluation.md",
    PROJECT_ROOT / "docs" / "serving.md",
)
# Values are stored rounded (cycles to 0.01, probabilities to 0.0001).
CYCLE_TOLERANCE, PROBABILITY_TOLERANCE = 0.0051, 0.00006


def load_run(run_dir: Path) -> dict[str, Any]:
    predictions = run_dir / "predictions"
    replay = predictions / "replay-holdout.csv.gz"
    return {
        "dir": run_dir,
        "metrics": json.loads((run_dir / "metrics.json").read_text()),
        "test": pd.read_csv(predictions / "test.csv"),
        "members": pd.read_csv(predictions / "test-members.csv"),
        "calibration": pd.read_csv(predictions / "calibration-points.csv"),
        "oof": pd.read_csv(predictions / "oof.csv.gz"),
        "replay": pd.read_csv(replay) if replay.is_file() else None,
    }


def _compare(found: Any, stored: Any, where: str, tolerance: float = 1e-9) -> list[str]:
    """Recursively compare numbers (within tolerance), lists and dicts."""
    if isinstance(stored, dict) and isinstance(found, dict):
        return [
            p for k in stored for p in _compare(found.get(k), stored[k], f"{where}.{k}", tolerance)
        ]
    if isinstance(stored, list) and isinstance(found, list):
        if len(stored) != len(found):
            return [f"{where}: {len(found)} items recomputed, {len(stored)} stored"]
        return [
            p
            for i, (f, s) in enumerate(zip(found, stored, strict=True))
            for p in _compare(f, s, f"{where}[{i}]", tolerance)
        ]
    if isinstance(stored, bool) or stored is None or isinstance(stored, str):
        return [] if found == stored else [f"{where}: recomputed {found!r}, stored {stored!r}"]
    if isinstance(stored, int | float):
        if not isinstance(found, int | float) or isinstance(found, bool):
            return [f"{where}: recomputed {found!r}, stored {stored!r}"]
        if (math.isinf(stored) and found == stored) or (math.isnan(stored) and math.isnan(found)):
            return []
        if abs(found - stored) <= tolerance:
            return []
        return [f"{where}: recomputed {found:.6g}, stored {stored:.6g}"]
    return []


def check_split_rules(run: dict) -> list[str]:
    m = run["metrics"]
    problems = []
    replay = set(m["data"]["replay_engines"])
    for repeat, folds in m["data"]["folds"].items():
        fold_of = {int(u): f for u, f in folds.items()}
        if replay & set(fold_of):
            problems.append(f"repeat {repeat}: replay engines were cross-fitted")
        for k, units in m["data"]["inner_validation"][repeat].items():
            own = [u for u in units if fold_of[u] == int(k)]
            if own:
                problems.append(
                    f"repeat {repeat} fold {k}: inner validation used its own fold's engines {own}"
                )
        cal = run["calibration"][run["calibration"]["repeat"] == int(repeat)]
        wrong = cal[cal["member"] != cal["unit"].map(fold_of)]
        if len(wrong):
            problems.append(
                f"repeat {repeat}: {len(wrong)} calibration scores come from a network that "
                "trained on that engine"
            )
        oof = run["oof"][run["oof"]["repeat"] == int(repeat)]
        if (oof["fold"] != oof["unit"].map(fold_of)).any():
            problems.append(f"repeat {repeat}: out-of-fold predictions came from the wrong network")
    return problems


def check_intervals(run: dict) -> list[str]:
    """Rebuild scores, CV+ intervals and probabilities from the networks' raw outputs."""
    m, cfg = run["metrics"], run["metrics"]["config"]
    problems = []
    cal = run["calibration"]
    capped = np.minimum(cal["rul"].to_numpy(), cfg["max_rul"])
    rebuilt = cqr_scores(cal["lower_raw"].to_numpy(), cal["upper_raw"].to_numpy(), capped)
    if np.abs(rebuilt - cal["score"].to_numpy()).max() > 2 * CYCLE_TOLERANCE:
        problems.append("conformity scores do not match their raw bounds")
    for (repeat, model), members in run["members"].groupby(["repeat", "model"]):
        members = members.sort_values(["member", "unit"])
        shape = (members["member"].nunique(), -1)
        lo, hi, pt, logit = (
            members[c].to_numpy().reshape(shape)
            for c in ("lower_raw", "upper_raw", "point_raw", "logit")
        )
        scores = cal[(cal["repeat"] == repeat) & (cal["model"] == model)]
        lower, upper = cv_plus_interval(
            lo, hi, scores["score"].to_numpy(), scores["member"].to_numpy(), cfg["alpha"]
        )
        lower, upper, open_ended = finalize_interval(lower, upper, cfg["max_rul"])
        temperature = m["calibration"][str(repeat)][model]["temperature"]
        test = run["test"]
        stored = test[(test["repeat"] == repeat) & (test["model"] == model)].sort_values("unit")
        where = f"repeat {repeat} {model}"
        for name, value, tolerance in (
            ("point", np.clip(pt.mean(axis=0), 0, cfg["max_rul"]), 2 * CYCLE_TOLERANCE),
            ("lower", lower, 2 * CYCLE_TOLERANCE),
            ("upper", upper, 2 * CYCLE_TOLERANCE),
            ("probability", probability(logit, temperature), 2 * PROBABILITY_TOLERANCE),
        ):
            gap = float(np.abs(value - stored[name].to_numpy()).max())
            if gap > tolerance:
                problems.append(f"{where}: test {name} differs from the rebuilt value by {gap:.4g}")
        if (open_ended != stored["open_ended"].to_numpy()).any():
            problems.append(f"{where}: open-ended flags differ from the rebuilt intervals")
    return problems


def check_metrics(run: dict) -> list[str]:
    m = run["metrics"]
    cfg = m["config"]
    if m.get("test") is None:
        return ["the test set has not been scored: run `python -m predmaint.evaluate`"]
    problems = []
    test = run["test"]
    for model, block in m["test"]["models"].items():
        for repeat, stored in block["per_repeat"].items():
            rows = test[(test["model"] == model) & (test["repeat"] == int(repeat))]
            found = repeat_metrics(
                rows.sort_values("unit"), cfg["max_rul"], cfg["alert_cycles"], seed=int(repeat)
            )
            problems += _compare(found, stored, f"test.{model}.{repeat}")
    if "lstm_minus_lightgbm" in m["test"]:
        problems += _compare(
            paired_differences(test, cfg["max_rul"]),
            m["test"]["lstm_minus_lightgbm"],
            "test.lstm_minus_lightgbm",
        )
    train_cfg = TrainConfig(
        **{k: v for k, v in cfg.items() if k in TrainConfig.__dataclass_fields__}
    )
    oof = run["oof"]
    for model, block in m["out_of_fold"].items():
        for repeat, stored in block["per_repeat"].items():
            rows = oof[(oof["model"] == model) & (oof["repeat"] == int(repeat))]
            found = whole_life_metrics(rows, train_cfg, seed=int(repeat))
            problems += _compare(found, stored, f"out_of_fold.{model}.{repeat}")
    replay = run.get("replay")
    if replay is not None:
        for model, block in m.get("replay_holdout", {}).items():
            for repeat, stored in block["per_repeat"].items():
                rows = replay[(replay["model"] == model) & (replay["repeat"] == int(repeat))]
                found = whole_life_metrics(rows, train_cfg, seed=int(repeat))
                problems += _compare(found, stored, f"replay_holdout.{model}.{repeat}")
    return problems


def check_manifest(run: dict) -> list[str]:
    """Every artifact present matches the hash recorded in manifest.json when the run was
    evaluated, and every file this audit reads is covered. A listed file may be absent: the
    networks and replay rows of the subsets that are not served are not committed."""
    path = run["dir"] / "manifest.json"
    if not path.is_file():
        return ["no manifest.json"]
    recorded = json.loads(path.read_text())["files_sha256"]
    present = manifest_files(run["dir"])
    problems = [
        f"{name} is not in manifest.json"
        for name in sorted(set(AUDITED_FILES) | set(present))
        if name not in recorded
    ]
    problems += [
        f"{name} differs from its SHA-256 in manifest.json"
        for name, digest in present.items()
        if name in recorded and recorded[name] != digest
    ]
    return problems


def check_examples(artifacts: Path) -> list[str]:
    """The product outputs the README shows must be the evaluated predictions.

    - ``docs/example-response.json`` must reproduce one test-set prediction of the served FD001
      model.
    - ``docs/figures/dashboard-state.json`` must name the engine and cycle that the hero rule picks,
      with the prediction ``replay-holdout.csv.gz`` holds for them.

    The files are looked for next to ``artifacts``: a copy of the artifacts elsewhere has none, and
    nothing is checked.
    """
    from predmaint.report import example_files, hero_state, matching_test_engine

    files = example_files(artifacts)
    problems = []
    if files["response"].is_file() and (artifacts / "FD001" / "metrics.json").is_file():
        response = json.loads(files["response"].read_text())
        if matching_test_engine(response, artifacts / "FD001") is None:
            problems.append(
                "docs/example-response.json matches no test prediction of the served FD001 model; "
                "rerun `python -m predmaint.report`"
            )
    if files["dashboard"].is_file():
        shown = json.loads(files["dashboard"].read_text())
        run_dir = artifacts / shown["subset"]
        if (run_dir / "metrics.json").is_file():
            hero = hero_state(run_dir)
            row = hero["row"]
            expected = {
                "unit": hero["unit"],
                "cycle": hero["cycle"],
                "rul_true": int(row["rul"]),
                "open_ended": bool(row["open_ended"]),
            }
            for key, value in expected.items():
                if shown[key] != value:
                    problems.append(
                        f"the dashboard screenshots show {key} {shown[key]}, but the replay "
                        f"predictions give {value}; rerun `python -m predmaint.screenshots`"
                    )
            keys = ["point", "lower", "probability"] + ([] if row["open_ended"] else ["upper"])
            for key in keys:
                tolerance = PROBABILITY_TOLERANCE * 2 if key == "probability" else 0.051
                if abs(shown[key] - float(row[key])) > tolerance:
                    problems.append(
                        f"the dashboard screenshots show {key} {shown[key]}, but "
                        f"replay-holdout.csv.gz has {float(row[key])}; rerun "
                        "`python -m predmaint.screenshots`"
                    )
    return problems


def audit(artifacts: Path, *documents: Path, complete: bool = False) -> list[str]:
    """Problems found in the runs under ``artifacts`` and in each document's generated blocks.
    With ``complete``, every generated block must also appear in at least one document, so a
    block whose markers were deleted cannot drop out of the check unnoticed."""
    runs = [artifacts / s for s in SUBSETS if (artifacts / s / "metrics.json").is_file()]
    if not runs:
        return [f"no trained runs in {artifacts}"]
    problems = []
    for run_dir in runs:
        run = load_run(run_dir)
        for check in (check_split_rules, check_intervals, check_metrics, check_manifest):
            problems += [f"{run_dir.name}: {p}" for p in check(run)]
    problems += check_examples(artifacts)
    from predmaint.report import BLOCKS, markers

    found_anywhere = set()
    for document in documents:
        text = document.read_text(encoding="utf-8")
        for name, build in BLOCKS.items():
            start, end = markers(name)
            if start in text:
                found_anywhere.add(name)
                found = text.split(start, 1)[1].split(end, 1)[0].strip("\n")
                if found != build(artifacts):
                    problems.append(
                        f"the {name} block in {document.name} differs from the one generated "
                        "from the artifacts; rerun `python -m predmaint.report`"
                    )
    if complete:
        problems += [
            f"no document has the generated {name} block"
            for name in BLOCKS
            if name not in found_anywhere
        ]
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m predmaint.audit", description=__doc__)
    parser.add_argument(
        "--docs",
        type=Path,
        nargs="*",
        default=list(DOCUMENTS),
        help="documents whose generated blocks must match (default: the four documents)",
    )
    parser.add_argument("--artifacts", type=Path, default=None)
    args = parser.parse_args(argv)
    artifacts = args.artifacts or Settings.from_env().artifacts_dir
    problems = audit(artifacts, *args.docs, complete=args.docs == list(DOCUMENTS))
    for problem in problems:
        print(f"MISMATCH  {problem}")
    if problems:
        return 1
    for subset in SUBSETS:
        if (artifacts / subset / "metrics.json").is_file():
            run = load_run(artifacts / subset)
            replay = run["replay"]
            print(
                f"{subset}: {len(run['test'])} test predictions, {len(run['oof'])} out-of-fold "
                f"predictions, {0 if replay is None else len(replay)} replay-holdout predictions, "
                f"{len(run['calibration'])} conformity scores and "
                f"{len(manifest_files(run['dir']))} manifest hashes check out"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
