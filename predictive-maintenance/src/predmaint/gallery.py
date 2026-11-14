"""Failure cases: where, and how, the models are wrong.

    python -m predmaint.gallery        # writes docs/failure-cases.md and its figure

Everything comes from the committed predictions, for the served repeat:

* the worst interval misses on the official test sets;
* the earliest and latest first alerts, and the missed engines, out of fold;
* test errors sliced by remaining life, observed trajectory length (short histories are padded
  to the network's window), and by subset (one or six operating conditions, one or two faults).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from predmaint.calibration import covered
from predmaint.config import PROJECT_ROOT, SUBSETS, Settings

NAMES = {"lstm": "LSTM", "lightgbm": "LightGBM"}
RUL_BINS = [-1, 25, 50, 75, 100, 124, 10_000]
RUL_LABELS = ["0–25", "26–50", "51–75", "76–100", "101–124", "125 or more"]
LENGTH_BINS = [0, 29, 59, 119, 10_000]
LENGTH_LABELS = ["under 30", "30–59", "60–119", "120 or more"]
CONDITIONS = {
    "FD001": "1 condition, 1 fault",
    "FD002": "6 conditions, 1 fault",
    "FD003": "1 condition, 2 faults",
    "FD004": "6 conditions, 2 faults",
}


def load(artifacts: Path) -> dict[str, dict]:
    runs = {}
    for subset in SUBSETS:
        run_dir = artifacts / subset
        if not (run_dir / "metrics.json").is_file():
            continue
        metrics = json.loads((run_dir / "metrics.json").read_text())
        if not (metrics.get("test") and metrics.get("served")):
            continue
        repeat = metrics["served"]["repeat"]
        test = pd.read_csv(run_dir / "predictions" / "test.csv")
        oof = pd.read_csv(run_dir / "predictions" / "oof.csv.gz")
        runs[subset] = {
            "metrics": metrics,
            "test": test[test["repeat"] == repeat].copy(),
            "oof": oof[oof["repeat"] == repeat].copy(),
        }
    return runs


def _miss(frame: pd.DataFrame, cap: int) -> pd.Series:
    """Cycles by which the capped truth lies outside the interval (0 when inside)."""
    truth = np.minimum(frame["rul_true"].to_numpy(float), cap)
    below = np.maximum(frame["lower"].to_numpy(float) - truth, 0.0)
    above = np.where(frame["open_ended"], 0.0, np.maximum(truth - frame["upper"], 0.0))
    return pd.Series(below + above, frame.index)


def worst_misses(runs: dict, per_subset: int = 3) -> str:
    lines = [
        "| Subset | Engine | Observed cycles | True RUL | Predicted | Interval | Missed by |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for subset, run in runs.items():
        cap = run["metrics"]["config"]["max_rul"]
        test = run["test"][run["test"]["model"] == "lstm"].copy()
        test["miss"] = _miss(test, cap)
        for _, r in test[test["miss"] > 0].nlargest(per_subset, "miss").iterrows():
            interval = (
                f"at least {r['lower']:.0f}"
                if r["open_ended"]
                else f"{r['lower']:.0f} to {r['upper']:.0f}"
            )
            lines.append(
                f"| {subset} | {int(r['unit'])} | {int(r['length'])} | {int(r['rul_true'])} "
                f"| {r['point']:.0f} | {interval} | {r['miss']:.0f} cycles |"
            )
    return "\n".join(lines)


def alert_extremes(runs: dict, n: int = 3) -> str:
    lines = [
        "| Subset | Earliest first alerts (engine: cycles left) "
        "| Latest first alerts (engine: cycles left) | Missed engines |",
        "|---|---|---|---|",
    ]
    for subset, run in runs.items():
        repeat = str(run["metrics"]["served"]["repeat"])
        per_unit = run["metrics"]["out_of_fold"]["lstm"]["per_repeat"][repeat]["alert_per_unit"]
        caught = sorted(
            (u for u in per_unit if u["lead_time"] is not None), key=lambda u: u["lead_time"]
        )
        missed = [str(u["unit"]) for u in per_unit if u["lead_time"] is None]
        earliest = ", ".join(f"{u['unit']}: {u['lead_time']}" for u in caught[::-1][:n])
        latest = ", ".join(f"{u['unit']}: {u['lead_time']}" for u in caught[:n])
        lines.append(f"| {subset} | {earliest} | {latest} | {', '.join(missed) or 'none'} |")
    return "\n".join(lines)


def _slice_table(frame: pd.DataFrame, by: str, labels: list[str], cap: int) -> list[str]:
    rows = []
    for label in labels:
        part = frame[frame[by] == label]
        if part.empty:
            continue
        cells = []
        for model in ("lstm", "lightgbm"):
            m = part[part["model"] == model]
            if m.empty:
                cells.append("–")
                continue
            truth = np.minimum(m["rul_true"].to_numpy(float), cap)
            error = m["point"].to_numpy(float) - truth
            inside = covered(
                m["lower"].to_numpy(float),
                m["upper"].to_numpy(float),
                m["open_ended"].to_numpy(bool),
                truth,
            ).mean()
            cells.append(f"{np.sqrt(np.mean(error**2)):.1f} / {error.mean():+.1f} / {inside:.0%}")
        engines = int((part["model"] == "lstm").sum())
        rows.append(f"| {label} | {engines} | {cells[0]} | {cells[1]} |")
    return rows


def slices(runs: dict) -> str:
    cap = next(iter(runs.values()))["metrics"]["config"]["max_rul"]
    frames = []
    for subset, run in runs.items():
        t = run["test"].copy()
        t["subset"] = f"{subset} ({CONDITIONS[subset]})"
        frames.append(t)
    test = pd.concat(frames, ignore_index=True)
    test["rul_bin"] = pd.cut(test["rul_true"], RUL_BINS, labels=RUL_LABELS).astype(str)
    test["length_bin"] = pd.cut(test["length"], LENGTH_BINS, labels=LENGTH_LABELS).astype(str)
    header = (
        "| {what} | Test engines | LSTM: RMSE / bias / coverage "
        "| LightGBM: RMSE / bias / coverage |\n|---|---:|---|---|"
    )
    parts = [
        header.format(what="True RUL (cycles)"),
        *_slice_table(test, "rul_bin", RUL_LABELS, cap),
        "",
        header.format(what="Observed cycles (shorter than the window: padded)"),
        *_slice_table(test, "length_bin", LENGTH_LABELS, cap),
        "",
        header.format(what="Subset"),
        *_slice_table(test, "subset", sorted(test["subset"].unique()), cap),
    ]
    return "\n".join(parts)


def write(artifacts: Path, out: Path) -> None:
    runs = load(artifacts)
    if not runs:
        raise SystemExit(f"No evaluated runs in {artifacts}")
    windows = sorted({w for run in runs.values() for w in run["metrics"]["served"]["windows"]})
    text = f"""# Failure cases

Generated by `python -m predmaint.gallery` from the committed predictions of the served repeat.
Errors are against the truth capped at 125 cycles, like the training target. "Bias" is mean
predicted minus true RUL: positive means the model thinks the engine has more life left than it
has, which is the dangerous direction.

## Worst interval misses on the official test sets

The test engines whose capped true RUL lies furthest outside the served LSTM's nominal 90%
interval.

{worst_misses(runs)}

## Earliest, latest and missed first alerts, out of fold

Over every cross-fitted training engine, run to failure. An alert counts when the probability of
failure within 30 cycles reaches 0.5 while at least one cycle is left.

{alert_extremes(runs)}

## Where the errors are

All four test sets pooled, one prediction per engine. Coverage reads open-ended intervals as
"at least the lower end". The networks read the last {", ".join(map(str, windows))} cycles;
shorter histories are padded by repeating the first cycle.

{slices(runs)}
"""
    out.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m predmaint.gallery", description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "docs" / "failure-cases.md")
    args = parser.parse_args(argv)
    write(args.artifacts or Settings.from_env().artifacts_dir, args.out)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
