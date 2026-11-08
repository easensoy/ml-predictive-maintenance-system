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

import pandas as pd

from predmaint.audit import DOCUMENTS, SUBSETS
from predmaint.config import PROJECT_ROOT, Settings
from predmaint.evaluate import BOOT_SEED, N_BOOT, _by_engine
from predmaint.metrics import bootstrap_indices, mean_sd, percentile_interval

START, END = "<!-- results:start -->", "<!-- results:end -->"
LEDGER_START, LEDGER_END = "<!-- ledger:start -->", "<!-- ledger:end -->"
LEDGER_JSONL = PROJECT_ROOT / "docs" / "experiment-ledger.jsonl"
LEDGER_MD = PROJECT_ROOT / "docs" / "experiment-ledger.md"
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
            if metrics.get("test") and metrics.get("served"):
                metrics["width"] = width_stats(artifacts / subset)
                runs[subset] = metrics
    return runs


def width_stats(run_dir: Path) -> dict[str, Any]:
    """Mean interval width on the test engines, open-ended intervals counted up to the cap, so
    both models are measured on the same engines; plus a paired bootstrap of the difference."""
    test = pd.read_csv(run_dir / "predictions" / "test.csv")
    widths = {
        model: _by_engine(test, model, "upper") - _by_engine(test, model, "lower")
        for model in test["model"].unique()
    }
    out: dict[str, Any] = {m: mean_sd(w.mean(axis=1).tolist()) for m, w in widths.items()}
    if {"lstm", "lightgbm"} <= set(widths):
        lstm, lgbm = widths["lstm"], widths["lightgbm"]
        draws = bootstrap_indices(lstm.shape[1], N_BOOT, BOOT_SEED)
        samples = lstm[:, draws].mean(-1).mean(0) - lgbm[:, draws].mean(-1).mean(0)
        low, high = percentile_interval(samples)
        out["difference"] = {
            "difference": float(lstm.mean() - lgbm.mean()),
            "ci_low": low,
            "ci_high": high,
        }
    return out


def test_table(runs: dict[str, dict]) -> str:
    header = (
        "| Subset | Model | RMSE | PHM08 score | RMSE vs actual RUL "
        "| Inside the nominal 90% interval (95% CI) | Open-ended intervals "
        "| Mean interval width | Alert AP | Alert ECE |"
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
                f"| {_ms(m['width'].get(name), '.1f')} cycles | {columns['alert_ap'][name]} "
                f"| {_ms(s['alert_ece'], '.3f')} |"
            )
    return "\n".join(lines)


def difference_table(runs: dict[str, dict]) -> str:
    lines = [
        "| Subset | RMSE, LSTM minus LightGBM | PHM08 score, LSTM minus LightGBM "
        "| Coverage, LSTM minus LightGBM (points) | Interval width, LSTM minus LightGBM |",
        "|---|---:|---:|---:|---:|",
    ]
    for subset, m in runs.items():
        d = m["test"].get("lstm_minus_lightgbm")
        if d:
            width = m["width"].get("difference")
            lines.append(
                f"| {subset} | {_ci(d['rmse'], '.2f')} | {_ci(d['nasa_score'], '.0f')} "
                f"| {_ci(d['coverage'], '.1f', 100)} "
                f"| {_ci(width, '.1f') if width else DASH} |"
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
            "interval over test engines at the mean rate. Width counts an open-ended interval up "
            "to the 125-cycle cap, so both models are measured on the same engines.",
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


def _span(values: list[float], spec: str) -> str:
    low, high = min(values), max(values)
    return (
        format(low, spec)
        if format(low, spec) == format(high, spec)
        else (f"{format(low, spec)} to {format(high, spec)}")
    )


def _significant(diff: dict) -> bool:
    return diff["ci_low"] > 0 or diff["ci_high"] < 0


def summary_block(artifacts: Path) -> str:
    """The headline findings, computed from the same artifacts as the tables (so no number in
    the README is typed by hand)."""
    runs = _runs(artifacts)
    complete = {s: m for s, m in runs.items() if "lstm_minus_lightgbm" in m["test"]}
    if not complete:
        return "_No evaluated runs yet._"
    n = len(complete)
    models = {s: m["test"]["models"] for s, m in complete.items()}
    rmse_diffs = [m["test"]["lstm_minus_lightgbm"]["rmse"] for m in complete.values()]
    lgbm_better = sum(d["difference"] > 0 for d in rmse_diffs)
    rmse_significant = sum(_significant(d) for d in rmse_diffs)
    coverage = [
        100 * models[s][name]["summary"]["coverage"]["mean"]
        for s in complete
        for name in ("lstm", "lightgbm")
    ]
    widths = {s: m["width"] for s, m in complete.items()}
    narrower = [
        100 * (1 - widths[s]["lstm"]["mean"] / widths[s]["lightgbm"]["mean"])
        for s in complete
        if widths[s]["lstm"]["mean"] < widths[s]["lightgbm"]["mean"]
    ]
    width_significant = sum(
        _significant(widths[s]["difference"]) and widths[s]["difference"]["difference"] < 0
        for s in complete
    )
    flaps = {
        name: [
            m["out_of_fold"][name]["summary"]["alert.flaps_per_engine"]["mean"]
            for m in complete.values()
        ]
        for name in ("lstm", "lightgbm")
    }
    ap = [models[s][name]["summary"]["alert_ap"]["mean"] for s in complete for name in models[s]]
    lead = [
        m["out_of_fold"][name]["summary"]["alert.lead_median"]["mean"]
        for m in complete.values()
        for name in ("lstm", "lightgbm")
    ]
    accuracy = (
        f"LightGBM's test RMSE is lower on {lgbm_better} of {n} subsets, and the paired "
        f"difference is significant on {rmse_significant} of {n}"
        + (" (every 95% interval includes zero)" if rmse_significant == 0 else "")
        + "."
    )
    intervals = (
        f"Both models cover {_span(coverage, '.0f')}% of test engines with a nominal 90% "
        "interval, so the intervals are conservative rather than short. "
        + (
            f"The LSTM's intervals are narrower on {len(narrower)} of {n} subsets, by "
            f"{_span(narrower, '.0f')}%, significantly on {width_significant}."
            if narrower
            else "The LSTM's intervals are not narrower than LightGBM's."
        )
    )
    alerts = (
        f"The alert ranks engines equally well for both models (alert AP {_span(ap, '.3f')}) "
        f"and first fires a median of {_span(lead, '.0f')} cycles before failure. The LSTM's "
        f"alert switches off again {_span(flaps['lstm'], '.2f')} times per engine over a whole "
        f"life; LightGBM's, {_span(flaps['lightgbm'], '.2f')} times."
    )
    return "\n".join(
        f"- **{head}** {text}"
        for head, text in (
            ("Accuracy.", accuracy),
            ("Uncertainty.", intervals),
            ("Alerting.", alerts),
        )
    )


def attribution_block(artifacts: Path) -> str:
    """The attribution checks on the served model, in words, from attribution-checks.json."""
    path = artifacts / "FD001" / "attribution-checks.json"
    if not path.is_file():
        return "_Attribution checks have not been run: `python -m predmaint.attribution_checks`._"
    r = json.loads(path.read_text())
    s, c, n = r["summary"], r["comparisons"], r["curve_windows"]

    def med(key: str, spec: str = ".2f") -> str:
        return format(s[key]["median"], spec)

    return "\n".join(
        [
            f"On {r['windows']} windows from {r['engines']} FD001 replay engines (at "
            f"{', '.join(map(str, r['stages_cycles_left']))} cycles before failure), with the "
            "served five-network ensemble:",
            "",
            f"- **Completeness holds.** The attributions add up to the change in prediction to "
            f"within {med('completeness_gap_cycles')} cycles (median), against a median change "
            f"of {abs(s['effect_cycles']['median']):.0f} cycles.",
            f"- **They find readings the output depends on.** Removing readings in attribution "
            f"order moves the prediction to the baseline far faster than random order: median "
            f"deletion area {med('deletion_auc_ig', '.3f')} against "
            f"{med('deletion_auc_random', '.3f')} (lower is better), better in "
            f"{c['ig_beats_random_on_deletion']} of {n} windows.",
            f"- **But much of that is input deviation.** Ranking readings simply by how far they "
            f"are from the baseline does almost as well (deletion area "
            f"{med('deletion_auc_input_deviation', '.3f')}; attributions win in "
            f"{c['ig_beats_input_deviation_on_deletion']} of {n} windows, and in "
            f"{c['ig_beats_input_deviation_on_insertion']} of {n} on insertion).",
            f"- **The trained weights matter, moderately.** A randomly re-initialised network "
            f"gives maps that rank-correlate {med('spearman_vs_full_randomisation')} with the "
            f"trained ones and delete less well ({med('deletion_auc_randomised_ig', '.3f')}).",
            f"- **The baseline matters.** Using each engine's own first cycles as the baseline "
            f"instead of the regime average changes the maps (rank correlation "
            f"{med('spearman_vs_own_baseline')}; the top three sensors keep "
            f"{s['top3_sensors_kept_with_own_baseline']['median']:.0f} of 3 in the median "
            "window).",
        ]
    )


def latency_block(benchmark: Path) -> str:
    """Serving latency, in words, from docs/benchmark.json."""
    if not benchmark.is_file():
        return "_The latency benchmark has not been run: `python -m predmaint.benchmark`._"
    b = json.loads(benchmark.read_text())
    lat, hw = b["latency"], b["hardware"]
    defaults = Settings()

    def ms(key: str) -> str:
        return f"{lat[key]['median_ms']:.0f} ms (p95 {lat[key]['p95_ms']:.0f} ms)"

    def tick_p95(refreshes: int) -> float:
        fleet = lat["replay_tick_whole_fleet"]["p95_ms"]
        return fleet + refreshes * lat["engine_detail_with_attribution"]["p95_ms"]

    per_tick = defaults.explain_per_tick
    processor = " ".join(hw["processor"].replace("(R)", "").replace("(TM)", "").split())
    return (
        f"Measured in process on an {processor} with {hw['torch_threads']} PyTorch "
        f"threads, CPU only, for the served {b['networks']}-network FD001 model:\n\n"
        f"| Operation | Median (p95) |\n|---|---|\n"
        f"| Prediction | {ms('predict_without_explanation')} |\n"
        f"| Prediction with integrated gradients | "
        f"{ms('predict_with_integrated_gradients')} |\n"
        f"| Replay tick, whole {b['fleet_engines']}-engine fleet, no attributions | "
        f"{ms('replay_tick_whole_fleet')} |\n"
        f"| Attribution refresh for one watched engine | "
        f"{ms('engine_detail_with_attribution')} |\n\n"
        f"The replay ticks every {defaults.replay_interval_s * 1000:.0f} ms. At the 95th "
        f"percentile, the fleet step plus {per_tick} attribution refreshes takes "
        f"{tick_p95(per_tick):.0f} ms, and one more refresh would take it to "
        f"{tick_p95(per_tick + 1):.0f} ms. So each tick refreshes at most `PM_EXPLAIN_PER_TICK` "
        f"watched engines ({per_tick} by default), taking turns when more are watched."
    )


BLOCKS = {
    "summary": lambda artifacts: summary_block(artifacts),
    "attribution": lambda artifacts: attribution_block(artifacts),
    "latency": lambda artifacts: latency_block(PROJECT_ROOT / "docs" / "benchmark.json"),
}


def markers(name: str) -> tuple[str, str]:
    return f"<!-- {name}:start -->", f"<!-- {name}:end -->"


def short_path(path: str) -> str:
    """Where a run wrote its artifacts, without local directory names: artifacts/FD001 for
    published runs, scratch/<name> for anything outside the repository."""
    parts = Path(path.replace("\\", "/")).parts
    if "artifacts" in parts:
        return "/".join(parts[parts.index("artifacts") :][:2])
    return f"scratch/{parts[-1]}" if parts else path


def ledger_section(jsonl: Path) -> str:
    """The cross-fitted part of the experiment ledger, generated from its JSON-lines log."""
    events = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line]
    lines = [
        "| When | Event | Subset | Code commit | Sources committed | Output | Test set viewed |",
        "|---|---|---|---|---|---|---|",
    ]
    for e in events:
        commit = (e.get("git_commit") or "?")[:7]
        clean = {True: "no", False: "yes", None: "–"}[e.get("git_dirty")]
        if e["event"] == "evaluate":
            clean = "–"
        lines.append(
            f"| {e['at'][:16].replace('T', ' ')} | {e['event']} | {e['subset']} | `{commit}` "
            f"| {clean} | {short_path(e.get('out', ''))} | {'yes' if e['test_viewed'] else 'no'} |"
        )
    finals = [
        e
        for e in events
        if e["event"] == "train"
        and short_path(e.get("out", "")).startswith("artifacts/")
        and e.get("git_dirty") is False
    ]
    summary = [
        "",
        "Window choice in the published runs, over every repeat and fold: how often each window "
        "won, and its mean inner-validation RMSE (cycles, every engine weighted equally).",
        "",
        "| Subset | Model | Times chosen (30 / 40 / 50 / 60) "
        "| Mean inner-validation RMSE (30 / 40 / 50 / 60) |",
        "|---|---|---|---|",
    ]
    # The last clean training run of each subset is the one whose artifacts are published.
    published = {e["subset"]: e for e in finals}
    for e in (published[s] for s in sorted(published)):
        windows = e["config"]["windows"]
        for model in ("lstm", "lightgbm"):
            rows = [r for r in e["selection"] if r["model"] == model]
            if not rows:
                continue
            chosen = [sum(r["chosen"] for r in rows if r["window"] == w) for w in windows]
            mean = [
                sum(r["val_rmse"] for r in rows if r["window"] == w)
                / max(1, sum(r["window"] == w for r in rows))
                for w in windows
            ]
            summary.append(
                f"| {e['subset']} | {NAMES[model]} | {' / '.join(map(str, chosen))} "
                f"| {' / '.join(f'{v:.2f}' for v in mean)} |"
            )
    return "\n".join(lines + (summary if finals else []))


def update_ledger(markdown: Path = LEDGER_MD, jsonl: Path = LEDGER_JSONL) -> bool:
    if not (markdown.is_file() and jsonl.is_file()):
        return False
    text = markdown.read_text(encoding="utf-8")
    if LEDGER_START not in text or LEDGER_END not in text:
        return False
    head, rest = text.split(LEDGER_START, 1)
    _, tail = rest.split(LEDGER_END, 1)
    section = ledger_section(jsonl)
    markdown.write_text(f"{head}{LEDGER_START}\n{section}\n{LEDGER_END}{tail}", encoding="utf-8")
    return True


def _replace_block(text: str, start: str, end: str, body: str) -> str:
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    return f"{head}{start}\n{body}\n{end}{tail}"


def update_readme(document: Path, artifacts: Path) -> None:
    """Rewrite every generated block the document has: always the results, and the findings,
    attribution and latency blocks where their markers appear."""
    text = document.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise SystemExit(f"{document} has no {START} ... {END} block")
    text = _replace_block(text, START, END, results_block(artifacts))
    for name, build in BLOCKS.items():
        start, end = markers(name)
        if start in text and end in text:
            text = _replace_block(text, start, end, build(artifacts))
    document.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m predmaint.report", description=__doc__)
    parser.add_argument("--docs", type=Path, nargs="*", default=list(DOCUMENTS))
    parser.add_argument("--artifacts", type=Path, default=None)
    args = parser.parse_args(argv)
    artifacts = args.artifacts or Settings.from_env().artifacts_dir
    for document in args.docs:
        update_readme(document, artifacts)
        print(f"Updated results block in {document}")
    if update_ledger():
        print(f"Updated the generated section of {LEDGER_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
