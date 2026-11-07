"""Write the results tables in README.md and MODEL_CARD.md from ``artifacts/<subset>/``.

    python -m predmaint.report          # rewrites the results block in both documents

The documents never contain a hand-typed metric: rerun this after training and evaluation.
``python -m predmaint.audit`` recomputes every number from the committed predictions and checks
that the block matches.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from predmaint.audit import DOCUMENTS, SUBSETS
from predmaint.config import Settings

START, END = "<!-- results:start -->", "<!-- results:end -->"
DASH = "–"
NAMES = {"lstm": "LSTM ensemble", "lightgbm": "LightGBM"}


def _ms(stat: dict | None, spec: str, scale: float = 1.0, unit: str = "") -> str:
    """'mean ± sd' (sd omitted for one repeat)."""
    if not stat or stat.get("mean") is None:
        return DASH
    mean = format(stat["mean"] * scale, spec)
    sd = "" if stat.get("sd") is None else f" ± {format(stat['sd'] * scale, spec)}"
    return f"{mean}{sd}{unit}"


def _ci(diff: dict, spec: str, scale: float = 1.0) -> str:
    signed = "+" + spec
    return (
        f"{format(diff['difference'] * scale, signed)} "
        f"({format(diff['ci_low'] * scale, signed)} to "
        f"{format(diff['ci_high'] * scale, signed)})"
    )


def _bold_best(
    cells: dict[str, str], values: dict[str, float | None], lower_is_better: bool
) -> dict[str, str]:
    present = {m: v for m, v in values.items() if v is not None}
    if len(present) < 2:
        return cells
    best = min(present.values()) if lower_is_better else max(present.values())
    return {m: f"**{c}**" if values.get(m) == best else c for m, c in cells.items()}


def _runs(artifacts: Path) -> dict[str, dict[str, Any]]:
    runs = {}
    for subset in SUBSETS:
        path = artifacts / subset / "metrics.json"
        if path.is_file():
            metrics = json.loads(path.read_text())
            if metrics.get("test"):
                runs[subset] = metrics
    return runs


def test_table(runs: dict[str, dict]) -> str:
    header = (
        "| Subset | Model | RMSE | PHM08 score | RMSE vs actual RUL "
        "| Inside the nominal 90% interval (95% CI) | Open-ended intervals "
        "| Width of closed intervals | Alert AP | Alert ECE |"
    )
    lines = [header, "|---|---|" + "---:|" * 8]
    spec = {
        "rmse": (".2f", True),
        "nasa_score": (".0f", True),
        "rmse_uncapped": (".2f", True),
        "alert_ap": (".3f", False),
    }
    for subset, m in runs.items():
        models = m["test"]["models"]
        n = m["data"]["test_engines"]
        columns: dict[str, dict[str, str]] = {}
        for key, (fmt, lower_better) in spec.items():
            cells = {name: _ms(block["summary"][key], fmt) for name, block in models.items()}
            means = {name: block["summary"][key]["mean"] for name, block in models.items()}
            columns[key] = _bold_best(cells, means, lower_better)
        for name, block in models.items():
            s = block["summary"]
            low, high = s["coverage_wilson_at_mean"]
            coverage = f"{_ms(s['coverage'], '.1f', 100)}% ({100 * low:.0f}–{100 * high:.0f}%)"
            lines.append(
                f"| {subset} ({n} engines) | {NAMES[name]} | {columns['rmse'][name]} "
                f"| {columns['nasa_score'][name]} | {columns['rmse_uncapped'][name]} "
                f"| {coverage} | {_ms(s['open_ended_share'], '.0f', 100)}% "
                f"| {_ms(s['closed_width'], '.0f')} cycles | {columns['alert_ap'][name]} "
                f"| {_ms(s['alert_ece'], '.3f')} |"
            )
    return "\n".join(lines)


def difference_table(runs: dict[str, dict]) -> str:
    lines = [
        "| Subset | RMSE, LSTM minus LightGBM | PHM08 score, LSTM minus LightGBM "
        "| Coverage, LSTM minus LightGBM (points) |",
        "|---|---:|---:|---:|",
    ]
    for subset, m in runs.items():
        d = m["test"].get("lstm_minus_lightgbm")
        if d:
            lines.append(
                f"| {subset} | {_ci(d['rmse'], '.2f')} | {_ci(d['nasa_score'], '.0f')} "
                f"| {_ci(d['coverage'], '.1f', 100)} |"
            )
    return "\n".join(lines)


def alert_table(runs: dict[str, dict]) -> str:
    lines = [
        "| Subset | Model | Engines | First alert, median cycles before failure | Missed "
        "| First alert more than 30 cycles early | False-alert episodes per engine "
        "| Flaps per engine | ECE over all cycles (95% CI) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for subset, m in runs.items():
        for name, block in m["out_of_fold"].items():
            s = block["summary"]
            repeats = list(block["per_repeat"].values())
            low = sum(r["calibration"]["ci_low"] for r in repeats) / len(repeats)
            high = sum(r["calibration"]["ci_high"] for r in repeats) / len(repeats)
            lines.append(
                f"| {subset} | {NAMES[name]} | {repeats[0]['engines']} "
                f"| {_ms(s['alert.lead_median'], '.0f')} | {_ms(s['alert.missed'], '.1f')} "
                f"| {_ms(s['alert.early'], '.1f')} "
                f"| {_ms(s['alert.false_alerts_per_engine'], '.2f')} "
                f"| {_ms(s['alert.flaps_per_engine'], '.2f')} "
                f"| {_ms(s['calibration.ece'], '.3f')} ({low:.3f}–{high:.3f}) |"
            )
    return "\n".join(lines)


def results_block(artifacts: Path) -> str:
    runs = _runs(artifacts)
    if not runs:
        return "_No evaluated runs yet: run `predmaint.train`, then `predmaint.evaluate`._"
    repeats = len(next(iter(runs.values()))["config"]["repeats"])
    return "\n\n".join(
        [
            f"**Official test sets.** Mean ± sd over {repeats} repeats of the whole pipeline; "
            "bold marks the better model in each subset. The coverage interval is a 95% Wilson "
            "interval over test engines at the mean rate.",
            test_table(runs),
            "**LSTM minus LightGBM on the same test engines.** Repeat-averaged difference, with a "
            "95% interval from a paired bootstrap over test engines (2,000 draws). An interval "
            "that excludes 0 means the difference is larger than engine-to-engine noise.",
            difference_table(runs),
            "**The alert over whole lives, out of fold.** Every cycle of every cross-fitted "
            "training engine, each predicted by a model that never saw it. Mean ± sd over "
            "repeats; the ECE interval is the repeat-averaged 95% bootstrap interval over "
            "engines.",
            alert_table(runs),
        ]
    )


def update_readme(document: Path, artifacts: Path) -> None:
    text = document.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise SystemExit(f"{document} has no {START} ... {END} block")
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    document.write_text(f"{head}{START}\n{results_block(artifacts)}\n{END}{tail}", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m predmaint.report", description=__doc__)
    parser.add_argument("--docs", type=Path, nargs="*", default=list(DOCUMENTS))
    parser.add_argument("--artifacts", type=Path, default=None)
    args = parser.parse_args(argv)
    artifacts = args.artifacts or Settings.from_env().artifacts_dir
    for document in args.docs:
        update_readme(document, artifacts)
        print(f"Updated results block in {document}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
