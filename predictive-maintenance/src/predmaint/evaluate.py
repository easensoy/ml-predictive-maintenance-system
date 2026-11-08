"""Score the official test set, once, after training is final.

    python -m predmaint.evaluate                     # every subset in artifacts/
    python -m predmaint.evaluate --subsets FD001

This is the only code that reads NASA's test labels (RUL_FD00x.txt). For each run it adds the
true RUL to ``predictions/test.csv``, writes the test metrics into ``metrics.json``, writes
``manifest.json`` (hashes of every artifact) and appends a "test viewed" entry to the experiment
ledger, so the ledger shows exactly when the test set was looked at.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from predmaint import data as cmapss
from predmaint.calibration import covered
from predmaint.config import Settings
from predmaint.metrics import (
    bootstrap_indices,
    ece_with_interval,
    mean_sd,
    nasa_score,
    percentile_interval,
    reliability_curve,
    rmse,
    wilson_interval,
)
from predmaint.train import LEDGER, append_ledger, git_state, ledger_path

N_BOOT = 2000
BOOT_SEED = 7
SUMMARY = (
    "rmse",
    "nasa_score",
    "rmse_uncapped",
    "coverage",
    "coverage_actual",
    "coverage_actual_closed_reading",
    "open_ended_share",
    "closed_width",
    "alert_ap",
    "alert_ece",
    "brier",
)


def repeat_metrics(frame: pd.DataFrame, max_rul: int, horizon: int, seed: int) -> dict:
    """Metrics of one model and repeat on the official test set (one row per engine)."""
    true = frame["rul_true"].to_numpy(float)
    capped = np.minimum(true, max_rul)
    point, lower, upper = (frame[c].to_numpy(float) for c in ("point", "lower", "upper"))
    open_ended = frame["open_ended"].to_numpy(bool)
    prob = frame["probability"].to_numpy(float)
    label = true <= horizon
    inside = covered(lower, upper, open_ended, capped)
    closed = ~open_ended
    calibration = ece_with_interval(prob, label.astype(float), frame["unit"].to_numpy(), seed=seed)
    return {
        "engines": len(frame),
        "rmse": rmse(point, capped),
        "nasa_score": nasa_score(point, capped),
        "rmse_uncapped": rmse(point, true),
        "coverage": float(inside.mean()),
        "inside": int(inside.sum()),
        "coverage_actual": float(covered(lower, upper, open_ended, true).mean()),
        # The v2 reading: the interval as a closed range ending at the cap.
        "coverage_actual_closed_reading": float(((true >= lower) & (true <= upper)).mean()),
        "open_ended_share": float(open_ended.mean()),
        "closed_width": float((upper - lower)[closed].mean()) if closed.any() else None,
        "alert_positives": int(label.sum()),
        "alert_ap": float(average_precision_score(label, prob))
        if 0 < label.sum() < len(label)
        else None,
        "alert_ece": calibration["ece"],
        "alert_ece_ci": [calibration["ci_low"], calibration["ci_high"]],
        "brier": float(np.mean((prob - label) ** 2)),
        "reliability": reliability_curve(prob, label.astype(float)),
    }


def _by_engine(test: pd.DataFrame, model: str, column: str) -> np.ndarray:
    """(repeats, engines) array of one column, engines in unit order."""
    wide = test[test["model"] == model].pivot(index="repeat", columns="unit", values=column)
    return wide.sort_index().sort_index(axis=1).to_numpy(float)


def paired_differences(test: pd.DataFrame, max_rul: int, models=("lstm", "lightgbm")) -> dict:
    """LSTM minus LightGBM, averaged over repeats, with 95% intervals from resampling engines.

    Each bootstrap draw resamples test engines with replacement and uses the same engines for
    both models (a paired bootstrap), so the shared difficulty of an engine cancels out.
    """
    first, second = models
    true = _by_engine(test, first, "rul_true")[0]
    capped = np.minimum(true, max_rul)
    draws = bootstrap_indices(len(true), N_BOOT, BOOT_SEED)
    out = {}
    stats = {}
    for model in models:
        point = _by_engine(test, model, "point")
        inside = covered(
            _by_engine(test, model, "lower"),
            _by_engine(test, model, "upper"),
            _by_engine(test, model, "open_ended").astype(bool),
            capped[None, :],
        ).astype(float)
        d = point - capped[None, :]
        stats[model] = {
            "rmse": (d * d, lambda x: np.sqrt(x.mean(axis=-1)).mean(axis=0)),
            "nasa_score": (
                np.where(d < 0, np.expm1(-d / 13.0), np.expm1(d / 10.0)),
                lambda x: x.sum(axis=-1).mean(axis=0),
            ),
            "coverage": (inside, lambda x: x.mean(axis=-1).mean(axis=0)),
        }
    for metric in ("rmse", "nasa_score", "coverage"):
        values = {}
        samples = {}
        for model in models:
            per_engine, stat = stats[model][metric]
            values[model] = float(stat(per_engine))
            samples[model] = stat(per_engine[:, draws])  # (n_boot,)
        low, high = percentile_interval(samples[first] - samples[second])
        out[metric] = {
            "difference": values[first] - values[second],
            "ci_low": low,
            "ci_high": high,
            "first": first,
            "second": second,
        }
    return out


def evaluate_run(run_dir: Path, settings: Settings, ledger: Path | None) -> dict:
    metrics = json.loads((run_dir / "metrics.json").read_text())
    cfg = metrics["config"]
    subset = metrics["subset"]
    labels = cmapss.load_test_labels(settings.data_dir, subset)
    path = run_dir / "predictions" / "test.csv"
    test = pd.read_csv(path).drop(columns=["rul_true"], errors="ignore")
    test["rul_true"] = test["unit"].map(labels).astype("int64")
    test.to_csv(path, index=False)

    result: dict = {"models": {}}
    for model, frame in test.groupby("model", sort=False):
        per_repeat = {}
        for _, g in frame.groupby("repeat"):
            repeat = int(g["repeat"].to_numpy()[0])
            per_repeat[str(repeat)] = repeat_metrics(
                g.sort_values("unit"), cfg["max_rul"], cfg["alert_cycles"], seed=repeat
            )
        summary: dict[str, Any] = {k: mean_sd([m[k] for m in per_repeat.values()]) for k in SUMMARY}
        n = next(iter(per_repeat.values()))["engines"]
        mean_coverage = summary["coverage"]["mean"]
        summary["coverage_wilson_at_mean"] = list(wilson_interval(round(mean_coverage * n), n))
        result["models"][model] = {"per_repeat": per_repeat, "summary": summary}
    if {"lstm", "lightgbm"} <= set(result["models"]):
        result["lstm_minus_lightgbm"] = paired_differences(test, cfg["max_rul"])
    git = git_state()
    result["labels_sha256"] = cmapss.file_digests(settings.data_dir, [f"RUL_{subset}.txt"])
    result["evaluated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    result["git"] = git
    result["bootstrap"] = {"draws": N_BOOT, "seed": BOOT_SEED, "unit": "test engine"}
    metrics["test"] = result
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    write_manifest(run_dir, metrics)
    if ledger is not None:
        append_ledger(
            ledger,
            {
                "event": "evaluate",
                "at": result["evaluated_at"],
                "subset": subset,
                "out": ledger_path(run_dir),
                "git_commit": git["commit"] if git else None,
                "test_viewed": True,
                "summary": {
                    model: {
                        k: v["mean"] for k, v in block["summary"].items() if isinstance(v, dict)
                    }
                    for model, block in result["models"].items()
                },
            },
        )
    return result


def write_manifest(run_dir: Path, metrics: dict) -> None:
    """Hashes of every artifact, with the code, data and environment that produced them."""
    files = {
        p.relative_to(run_dir).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(run_dir.rglob("*"))
        if p.is_file() and p.name != "manifest.json"
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "subset": metrics["subset"],
                "provenance": metrics["provenance"],
                "evaluated_with": metrics["test"]["git"] if metrics.get("test") else None,
                "files_sha256": files,
            },
            indent=1,
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m predmaint.evaluate", description=__doc__)
    parser.add_argument("--subsets", nargs="+", default=list(cmapss.SUBSETS))
    parser.add_argument("--artifacts", type=Path, default=None)
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    args = parser.parse_args(argv)
    settings = Settings.from_env()
    artifacts = args.artifacts or settings.artifacts_dir
    for subset in args.subsets:
        run_dir = artifacts / subset
        if not (run_dir / "metrics.json").is_file():
            continue
        result = evaluate_run(run_dir, settings, args.ledger)
        for model, block in result["models"].items():
            s = block["summary"]
            print(
                f"{subset} {model}: RMSE {s['rmse']['mean']:.2f}, coverage "
                f"{s['coverage']['mean']:.1%}, alert AP {s['alert_ap']['mean']:.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
