"""Write the README results table from ``artifacts/<subset>/metrics.json``.

    python -m predmaint.report          # rewrites the block between the results markers

The README never contains a hand-typed metric: rerun this after training.
"""

import argparse
import json
from pathlib import Path

from predmaint.config import PROJECT_ROOT, Settings

START, END = "<!-- results:start -->", "<!-- results:end -->"
SUBSETS = ("FD001", "FD002", "FD003", "FD004")


def _fmt(value, spec: str) -> str:
    return "–" if value is None else format(value, spec)


def results_table(artifacts: Path) -> str:
    rows = [
        "| Subset | Window | Test engines | RMSE | PHM08 score | RMSE, uncapped truth "
        "| Inside 90% interval | Alert PR-AUC | Median alert lead time | Single LSTM RMSE "
        "| LightGBM RMSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for subset in SUBSETS:
        path = artifacts / subset / "metrics.json"
        if not path.is_file():
            continue
        m = json.loads(path.read_text())
        t, single, base = m["test"], m["test_single_models"], m["baseline_lightgbm"] or {}
        lead = m["holdout"]["lead_time"]
        n_members = len(m["config"]["seeds"])
        rows.append(
            f"| {subset} | {m['config']['window']} | {m['data']['test_engines']} "
            f"| **{t['rmse']:.2f}** | {t['nasa_score']:.0f} | {t['rmse_uncapped']:.2f} "
            f"| {t['interval_coverage']:.0%} (±{t['interval_width'] / 2:.0f} cycles) "
            f"| {_fmt(t['alert']['average_precision'], '.3f')} "
            f"| {_fmt(lead['median'], '.0f')} cycles ({lead['missed']}/{lead['units']} missed) "
            f"| {single['rmse_mean']:.2f} ± {single['rmse_std']:.2f} (n={n_members}) "
            f"| {_fmt(base.get('rmse'), '.2f')} |"
        )
    return "\n".join(rows)


def update_readme(readme: Path, artifacts: Path) -> None:
    text = readme.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise SystemExit(f"{readme} has no {START} ... {END} block")
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    readme.write_text(f"{head}{START}\n{results_table(artifacts)}\n{END}{tail}", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m predmaint.report", description=__doc__)
    parser.add_argument("--readme", type=Path, default=PROJECT_ROOT.parent / "README.md")
    parser.add_argument("--artifacts", type=Path, default=None)
    args = parser.parse_args(argv)
    update_readme(args.readme, args.artifacts or Settings.from_env().artifacts_dir)
    print(f"Updated results table in {args.readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
