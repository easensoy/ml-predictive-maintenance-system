"""Write the README results table from ``artifacts/<subset>/``.

    python -m predmaint.report          # rewrites the results block in README.md and MODEL_CARD.md

The README never contains a hand-typed metric: rerun this after training. Summary metrics come
from each run's ``metrics.json``; the few derived numbers (coverage of the actual remaining life,
Wilson intervals, the range of alert lead times) are recomputed from the committed predictions.
``python -m predmaint.audit`` checks that all three agree.
"""

import argparse
import json
from pathlib import Path

from predmaint.audit import DOCUMENTS, recompute
from predmaint.config import Settings
from predmaint.metrics import wilson_interval

START, END = "<!-- results:start -->", "<!-- results:end -->"
SUBSETS = ("FD001", "FD002", "FD003", "FD004")
HEADER = [
    "Subset",
    "Model",
    "RMSE",
    "PHM08 score",
    "RMSE vs actual RUL",
    "Inside nominal 90% interval, capped truth (95% CI)",
    "Inside, actual RUL (95% CI)",
    "Mean interval width",
    "Alert AP",
    "First alert, cycles before failure: median (range)",
]
# Columns where the best model in each subset is bolded; lower is better in all of them.
BOLD_COLUMNS = ("rmse", "nasa", "rmse_uncapped")
DASH = "–"


def _coverage(inside: int, n: int) -> str:
    low, high = wilson_interval(inside, n)
    return f"{inside / n:.0%} ({100 * low:.0f}–{100 * high:.0f}%)"


def _rows(run_dir: Path) -> list[dict]:
    """One dict per model: display strings plus the raw values used for bolding."""
    m = json.loads((run_dir / "metrics.json").read_text())
    found = recompute(run_dir)
    t, n = m["test"], found["test_engines"]
    members = len(m["config"]["seeds"])
    lead = m["holdout"]["lead_time"]
    ap = t["alert"]["average_precision"]
    lead_text = (
        f"{lead['median']:.0f} ({found['lead_min']}–{found['lead_max']}); "
        f"{lead['missed']}/{lead['units']} holdout engines missed"
        if lead["median"] is not None
        else f"{lead['missed']}/{lead['units']} missed"
    )
    rows = [
        {
            "model": f"LSTM ensemble ({members} networks)",
            "rmse": t["rmse"],
            "nasa": t["nasa_score"],
            "rmse_uncapped": t["rmse_uncapped"],
            "cells": [
                _coverage(found["inside_capped"], n),
                _coverage(found["inside_actual"], n),
                f"{t['interval_width']:.0f} cycles",
                DASH if ap is None else f"{ap:.3f}",
                lead_text,
            ],
        }
    ]
    single = m["test_single_models"]
    rows.append(
        {
            "model": f"One LSTM network (mean ± sd over the {members} networks of this ensemble)",
            "rmse": single["rmse_mean"],
            "nasa": single["nasa_score_mean"],
            "rmse_uncapped": None,
            "text": {
                "rmse": f"{single['rmse_mean']:.2f} ± {single['rmse_std']:.2f}",
                "nasa": f"{single['nasa_score_mean']:.0f} ± {single['nasa_score_std']:.0f}",
            },
            "cells": [DASH] * 5,
        }
    )
    if base := m.get("baseline_lightgbm"):
        rows.append(
            {
                "model": "LightGBM on window statistics (one seed)",
                "rmse": base["rmse"],
                "nasa": base["nasa_score"],
                "rmse_uncapped": base["rmse_uncapped"],
                "cells": [DASH] * 5,
            }
        )
    return rows


def results_table(artifacts: Path) -> str:
    lines = [
        "| " + " | ".join(HEADER) + " |",
        "|---|---|" + "---:|" * (len(HEADER) - 2),
    ]
    for subset in SUBSETS:
        run_dir = artifacts / subset
        if not (run_dir / "metrics.json").is_file():
            continue
        rows = _rows(run_dir)
        best = {
            column: min(r[column] for r in rows if r[column] is not None) for column in BOLD_COLUMNS
        }
        engines = json.loads((run_dir / "metrics.json").read_text())["data"]["test_engines"]
        for row in rows:
            cells = []
            for column, spec in (("rmse", ".2f"), ("nasa", ".0f"), ("rmse_uncapped", ".2f")):
                value = row[column]
                text = row.get("text", {}).get(column) or (
                    DASH if value is None else format(value, spec)
                )
                bold = value is not None and value == best[column] and len(rows) > 1
                cells.append(f"**{text}**" if bold else text)
            lines.append(
                f"| {subset} ({engines} engines) | {row['model']} | "
                + " | ".join(cells + row["cells"])
                + " |"
            )
    return "\n".join(lines)


def update_readme(readme: Path, artifacts: Path) -> None:
    text = readme.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise SystemExit(f"{readme} has no {START} ... {END} block")
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    readme.write_text(f"{head}{START}\n{results_table(artifacts)}\n{END}{tail}", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m predmaint.report", description=__doc__)
    parser.add_argument("--docs", type=Path, nargs="*", default=list(DOCUMENTS))
    parser.add_argument("--artifacts", type=Path, default=None)
    args = parser.parse_args(argv)
    for document in args.docs:
        update_readme(document, args.artifacts or Settings.from_env().artifacts_dir)
        print(f"Updated results table in {document}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
