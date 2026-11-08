"""Write the generated parts of the documents from ``artifacts/<subset>/``.

    python -m predmaint.report          # rewrites every generated block of every document

The documents are README.md, MODEL_CARD.md, docs/evaluation.md and docs/serving.md.

It fills each block whose markers a document has: the headline, the findings, the summary
table, the full results tables, the attribution checks, the latency table, the dashboard state
and the API example. It first
writes ``docs/example-response.json`` by posting ``docs/example-request.json`` to the real
``/api/predict`` route. It also rewrites the generated section of the experiment ledger.

The documents never contain a hand-typed result: rerun this after training and evaluation.
``python -m predmaint.audit`` recomputes every number from the committed predictions and checks
that each block matches.
"""

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from predmaint.audit import DOCUMENTS, SUBSETS
from predmaint.calibration import covered
from predmaint.config import PROJECT_ROOT, Settings
from predmaint.evaluate import BOOT_SEED, N_BOOT, _by_engine
from predmaint.metrics import bootstrap_indices, mean_sd, percentile_interval, wilson_interval

START, END = "<!-- results:start -->", "<!-- results:end -->"
LEDGER_START, LEDGER_END = "<!-- ledger:start -->", "<!-- ledger:end -->"
LEDGER_JSONL = PROJECT_ROOT / "docs" / "experiment-ledger.jsonl"
LEDGER_MD = PROJECT_ROOT / "docs" / "experiment-ledger.md"
DASH = "–"
# Both are served as the average of their five cross-fitted fold models, so neither name says
# "ensemble"; the tables say so once instead.
NAMES = {"lstm": "LSTM", "lightgbm": "LightGBM"}


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
    cells: dict[str, str], values: Mapping[str, float | None], lower_is_better: bool
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
        "| First alert with more than 30 cycles left | False-alert episodes per engine "
        "| Switch-offs before failure per engine | ECE over all cycles (95% CI) |",
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
            "**Official test sets.** Each repeat of each model is evaluated as its five-fold "
            "ensemble: the point prediction is the mean of the five fold models and the interval "
            "is the CV+ interval over them. The API and dashboard serve the first repeat's LSTM. "
            f"Mean ± sd over {repeats} repeats of the whole pipeline on the same test engines, so "
            "the ± is training-seed spread, not the precision of the estimate; bold marks the "
            "better model in each subset. The coverage interval is a 95% Wilson interval over "
            "test engines at the mean rate, which treats engines as independent and covers "
            "engine sampling only. Width counts an open-ended interval up to the 125-cycle cap, "
            "so both models are measured on the same engines; that summary was defined after the "
            "test labels were read.",
            test_table(runs),
            "**LSTM minus LightGBM on the same test engines.** Repeat-averaged difference, with a "
            "95% interval from a paired bootstrap over test engines (2,000 draws, each applied to "
            "both models and every repeat). An interval that excludes 0 means the difference is "
            "larger than the noise from which engines are in the test set. It does not include "
            "training-seed variation, and the 16 intervals are not corrected for multiple "
            "comparisons.",
            difference_table(runs),
            "**The alert over whole lives, out of fold.** Every cycle of every cross-fitted "
            "training engine, each predicted by the single fold model that never saw it, not by "
            "the five-fold ensemble. Engine counts are means over repeats, ± sd; the ECE interval "
            "is the repeat-averaged 95% bootstrap interval over engines.",
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


def first_alert(group: pd.DataFrame, threshold: float) -> pd.Series | None:
    """The first cycle of one engine's life at which the alert fires (at least one cycle left)."""
    fired = group[(group["probability"] >= threshold) & (group["rul"] >= 1)]
    return fired.iloc[0] if len(fired) else None


def hero_engine(replay: pd.DataFrame, threshold: float, cap: int) -> int:
    """The replay engine shown in the hero figure, the dashboard screenshots and the attribution
    example, picked by a fixed rule: among engines whose interval misses the capped truth at
    least once, the one with the median first-alert lead time. That favours a typical engine
    with a visible failure over the best-looking one."""
    rows = []
    for _, g in replay.groupby("unit"):
        truth = np.minimum(g["rul"].to_numpy(), cap)
        inside = covered(
            g["lower"].to_numpy(), g["upper"].to_numpy(), g["open_ended"].to_numpy(bool), truth
        )
        alert = first_alert(g, threshold)
        lead = int(alert["rul"]) if alert is not None else -1
        rows.append((int(g["unit"].to_numpy()[0]), int((~inside).sum()), lead))
    table = pd.DataFrame(rows, columns=["unit", "misses", "lead"])
    pool = table[table["misses"] > 0] if (table["misses"] > 0).any() else table
    pool = pool.sort_values(["lead", "unit"]).reset_index(drop=True)
    return int(pool["unit"].to_numpy()[len(pool) // 2])


def served_replay(run_dir: Path) -> pd.DataFrame:
    """The served networks' predictions for every cycle of the replay engines."""
    metrics = json.loads((run_dir / "metrics.json").read_text())
    replay = pd.read_csv(run_dir / "predictions" / "replay-holdout.csv.gz")
    repeat = metrics["served"]["repeat"]
    return replay[(replay["model"] == "lstm") & (replay["repeat"] == repeat)]


def hero_state(run_dir: Path) -> dict[str, Any]:
    """The engine and cycle the README shows in three views (hero, dashboard, attribution): the
    hero engine at the cycle its alert first fires."""
    metrics = json.loads((run_dir / "metrics.json").read_text())
    cfg = metrics["config"]
    replay = served_replay(run_dir)
    unit = hero_engine(replay, cfg["alert_threshold"], cfg["max_rul"])
    life = replay[replay["unit"] == unit].sort_values("cycle")
    alert = first_alert(life, cfg["alert_threshold"])
    row = alert if alert is not None else life.iloc[-1]
    return {"unit": unit, "cycle": int(row["cycle"]), "row": row, "life": life}


def single_models(run_dir: Path) -> dict[str, dict[str, list[float]]]:
    """Test RMSE against the capped truth of every single fold model (one per fold and repeat)
    and of each repeat's five-fold ensemble, which is what is tabulated (and, for the first LSTM
    repeat, served)."""
    cap = json.loads((run_dir / "metrics.json").read_text())["config"]["max_rul"]
    test = pd.read_csv(run_dir / "predictions" / "test.csv")
    members = pd.read_csv(run_dir / "predictions" / "test-members.csv")
    truth = test.drop_duplicates("unit").set_index("unit")["rul_true"].clip(upper=cap)
    out = {}
    for model, g in members.groupby("model", sort=False):
        error = g["point_raw"].clip(0, cap) - g["unit"].map(truth)
        singles = (error**2).groupby([g["repeat"], g["member"]]).mean() ** 0.5
        t = test[test["model"] == model]
        averages = ((t["point"] - t["unit"].map(truth)) ** 2).groupby(t["repeat"]).mean() ** 0.5
        out[str(model)] = {"singles": singles.tolist(), "averages": averages.tolist()}
    return out


def coverage_by_rul(run_dirs: dict[str, Path]) -> pd.DataFrame:
    """Test coverage by true-RUL range, all subsets pooled: for each model and range, the mean
    over repeats, the number of test engines and a 95% Wilson interval at the mean rate."""
    from predmaint.gallery import RUL_BINS, RUL_LABELS

    frames = []
    for run_dir in run_dirs.values():
        cap = json.loads((run_dir / "metrics.json").read_text())["config"]["max_rul"]
        t = pd.read_csv(run_dir / "predictions" / "test.csv")
        t["inside"] = covered(
            t["lower"].to_numpy(),
            t["upper"].to_numpy(),
            t["open_ended"].to_numpy(bool),
            np.minimum(t["rul_true"].to_numpy(), cap),
        )
        frames.append(t)
    test = pd.concat(frames, ignore_index=True)
    test["bin"] = pd.cut(test["rul_true"], RUL_BINS, labels=RUL_LABELS).astype(str)
    rows = []
    for model in test["model"].unique():
        for label in RUL_LABELS:
            g = test[(test["model"] == model) & (test["bin"] == label)]
            if not len(g):
                continue
            rate = float(g.groupby("repeat")["inside"].mean().mean())
            engines = int(len(g) / g["repeat"].nunique())
            low, high = wilson_interval(round(rate * engines), engines)
            rows.append((model, label, engines, rate, low, high))
    return pd.DataFrame(rows, columns=["model", "bin", "engines", "coverage", "low", "high"])


def closed_widths(run_dir: Path) -> dict[str, float]:
    """Mean test interval width of each model on the engines where neither model's interval is
    open-ended, so that no width depends on how an unbounded interval is counted. Computed per
    repeat, then averaged; ``engines`` is the mean number of such engines."""
    test = pd.read_csv(run_dir / "predictions" / "test.csv")
    closed = (_by_engine(test, "lstm", "open_ended") == 0) & (
        _by_engine(test, "lightgbm", "open_ended") == 0
    )
    out = {"engines": float(closed.sum(axis=1).mean())}
    for model in ("lstm", "lightgbm"):
        width = _by_engine(test, model, "upper") - _by_engine(test, model, "lower")
        out[model] = float(np.mean([w[c].mean() for w, c in zip(width, closed, strict=True)]))
    return out


def late_switch_offs(run_dir: Path) -> dict[str, float]:
    """Out of fold, per engine: how often the alert switches off at a cycle with at most the
    alert horizon left, when it should stay on. ``metrics.json`` counts every switch-off; this
    stricter count was added after the test labels were read, to check that premature alerts
    turning off do not explain the difference between the models."""
    cfg = json.loads((run_dir / "metrics.json").read_text())["config"]
    oof = pd.read_csv(run_dir / "predictions" / "oof.csv.gz").sort_values(
        ["model", "repeat", "unit", "cycle"], kind="stable"
    )
    keys = oof[["model", "repeat", "unit"]]
    same_engine = (keys.shift(-1) == keys).all(axis=1).to_numpy()
    on = (oof["probability"] >= cfg["alert_threshold"]).to_numpy()
    next_on = np.r_[on[1:], False]
    next_rul = oof["rul"].shift(-1).to_numpy()
    oof["late_off"] = on & ~next_on & same_engine & (next_rul <= cfg["alert_cycles"])
    per_repeat = oof.groupby(["model", "repeat"]).agg(
        off=("late_off", "sum"), engines=("unit", "nunique")
    )
    rate = (per_repeat["off"] / per_repeat["engines"]).groupby(level="model").mean()
    return {str(model): float(value) for model, value in rate.items()}


def cv_plus_bound(n: int, folds: int, alpha: float) -> float:
    """The worst-case marginal coverage of K-fold CV+ with n calibration points (Barber, Candès,
    Ramdas and Tibshirani, 2021, Theorem 4). With K = n it is the jackknife+ bound, 1 - 2 alpha."""
    return 1 - 2 * alpha - min(2 * (1 - 1 / folds) / (n / folds + 1), (1 - folds / n) / (folds + 1))


def pooled_coverage(run_dirs: dict[str, Path]) -> dict[str, dict[str, float]]:
    """Coverage of every test engine of every subset pooled, per model: the mean over repeats,
    with a 95% Wilson interval over the pooled engines at that rate."""
    by_rul = coverage_by_rul(run_dirs)
    out = {}
    for model, g in by_rul.groupby("model", sort=False):
        engines = int(g["engines"].sum())
        rate = float((g["coverage"] * g["engines"]).sum() / engines)
        low, high = wilson_interval(round(rate * engines), engines)
        out[str(model)] = {"coverage": rate, "low": low, "high": high, "engines": engines}
    return out


def _findings(artifacts: Path) -> dict[str, Any] | None:
    """Every number the headline and the findings quote, computed once."""
    runs = _runs(artifacts)
    complete = {s: m for s, m in runs.items() if "lstm_minus_lightgbm" in m["test"]}
    if not complete:
        return None
    models = {s: m["test"]["models"] for s, m in complete.items()}
    rmse = {s: m["test"]["lstm_minus_lightgbm"]["rmse"] for s, m in complete.items()}
    widths = {s: m["width"] for s, m in complete.items()}
    singles = {s: single_models(artifacts / s) for s in complete}
    by_rul = coverage_by_rul({s: artifacts / s for s in complete})
    nominal = 100 * (1 - next(iter(complete.values()))["config"]["alpha"])
    late = {s: late_switch_offs(artifacts / s) for s in complete}
    return {
        "n": len(complete),
        "late_flaps": {name: [late[s][name] for s in complete] for name in ("lstm", "lightgbm")},
        "lgbm_lower": [s for s, d in rmse.items() if d["difference"] > 0],
        "significant": [s for s, d in rmse.items() if _significant(d)],
        "rmse": rmse,
        "gain": [
            sum(v["singles"]) / len(v["singles"]) - sum(v["averages"]) / len(v["averages"])
            for per in singles.values()
            for v in per.values()
        ],
        "coverage": [
            100 * models[s][name]["summary"]["coverage"]["mean"]
            for s in complete
            for name in ("lstm", "lightgbm")
        ],
        "nominal": nominal,
        "worst_bin": {
            name: g.iloc[int(np.argmin(g["coverage"].to_numpy()))]
            for name, g in by_rul.groupby("model", sort=False)
        },
        "oof": {
            (s, name): 100
            * m["out_of_fold"][name]["summary"]["coverage_per_engine_weighted"]["mean"]
            for s, m in complete.items()
            for name in ("lstm", "lightgbm")
        },
        "narrower": [
            100 * (1 - widths[s]["lstm"]["mean"] / widths[s]["lightgbm"]["mean"])
            for s in complete
            if widths[s]["lstm"]["mean"] < widths[s]["lightgbm"]["mean"]
        ],
        "width_significant": sum(
            _significant(widths[s]["difference"]) and widths[s]["difference"]["difference"] < 0
            for s in complete
        ),
        "flaps": {
            name: [
                m["out_of_fold"][name]["summary"]["alert.flaps_per_engine"]["mean"]
                for m in complete.values()
            ]
            for name in ("lstm", "lightgbm")
        },
        "ap": [
            models[s][name]["summary"]["alert_ap"]["mean"] for s in complete for name in models[s]
        ],
        "base_rate": [
            100
            * models[s]["lstm"]["per_repeat"]["0"]["alert_positives"]
            / m["data"]["test_engines"]
            for s, m in complete.items()
        ],
        "lead": [
            m["out_of_fold"][name]["summary"]["alert.lead_median"]["mean"]
            for m in complete.values()
            for name in ("lstm", "lightgbm")
        ],
        "subsets": list(complete),
        "repeats": len(next(iter(complete.values()))["config"]["repeats"]),
        "cap": next(iter(complete.values()))["config"]["max_rul"],
        "horizon": next(iter(complete.values()))["config"]["alert_cycles"],
        "test_engines": {s: m["data"]["test_engines"] for s, m in complete.items()},
        "mean": {
            (s, name, key): models[s][name]["summary"][key]["mean"]
            for s in complete
            for name in ("lstm", "lightgbm")
            for key in (
                "rmse",
                "nasa_score",
                "rmse_uncapped",
                "coverage",
                "open_ended_share",
                "closed_width",
            )
        },
        "differences": {s: m["test"]["lstm_minus_lightgbm"] for s, m in complete.items()},
        "widths": widths,
        "closed": {s: closed_widths(artifacts / s) for s in complete},
        "pooled": pooled_coverage({s: artifacts / s for s in complete}),
        "oof_engines": {
            s: next(iter(m["out_of_fold"]["lstm"]["per_repeat"].values()))["engines"]
            for s, m in complete.items()
        },
        "alerts": {
            (s, name): m["out_of_fold"][name]["summary"]
            for s, m in complete.items()
            for name in ("lstm", "lightgbm")
        },
    }


def _difference(diff: dict) -> str:
    """'+1.06 cycles (95% CI +0.26 to +1.93)' for a paired LSTM-minus-LightGBM difference."""
    return (
        f"{diff['difference']:+.2f} cycles (95% CI {diff['ci_low']:+.2f} to {diff['ci_high']:+.2f})"
    )


def _by_model(pairs: list[tuple[str, str]]) -> str:
    """'LightGBM on FD001 and FD002 and the LSTM on FD003' from (subset, model) pairs."""
    groups: dict[str, list[str]] = {}
    for subset, model in pairs:
        groups.setdefault(model, []).append(subset)
    article = {"lstm": "the ", "lightgbm": ""}
    return " and ".join(
        f"{article[m]}{NAMES[m]} on {' and '.join(subsets)}" for m, subsets in groups.items()
    )


def _and(items: list[str]) -> str:
    """'FD001', 'FD001 and FD002', 'FD001, FD002 and FD003'."""
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


def _on(subsets: list[str], n: int) -> str:
    """Where something holds: 'all 4 subsets', or the subsets by name."""
    return f"all {n} subsets" if len(subsets) == n else _and(subsets)


def headline_block(artifacts: Path) -> str:
    """The two sentences under the README title: the task, then the finding."""
    f = _findings(artifacts)
    if f is None:
        return "_No evaluated runs yet._"
    n, k, clear = f["n"], len(f["lgbm_lower"]), f["significant"]
    how = (
        f"clearly only on {clear[0]}"
        if len(clear) == 1
        else f"clearly on {_and(clear)}"
        if clear
        else "never clearly"
    )
    if 2 * k >= n:
        accuracy = (
            f"LightGBM is at least as accurate (lower test RMSE on {k} of {n} subsets, {how})"
        )
    else:
        accuracy = f"The LSTM is more accurate (lower test RMSE on {n - k} of {n} subsets, {how})"
    lstm = []
    if len(f["narrower"]) == n:
        similar = not any(_significant(d["coverage"]) for d in f["differences"].values())
        lstm.append("narrower intervals" + (" at similar coverage" if similar else ""))
    if all(a < b for a, b in zip(f["flaps"]["lstm"], f["flaps"]["lightgbm"], strict=True)):
        lstm.append("an alert that switches off before failure less often")
    return (
        "This project predicts how many cycles a simulated turbofan engine has left, with a "
        f"nominal {f['nominal']:.0f}% prediction interval and an alert for failure within "
        f"{f['horizon']} cycles, and compares a five-network LSTM with a LightGBM baseline on "
        f"NASA's C-MAPSS data under one cross-fitted protocol. {accuracy}"
        + (f", while the LSTM gives {' and '.join(lstm)}." if lstm else ".")
    )


def summary_block(artifacts: Path) -> str:
    """The findings, computed from the same artifacts as the tables (so no number in the README
    is typed by hand)."""
    f = _findings(artifacts)
    if f is None:
        return "_No evaluated runs yet._"
    n = f["n"]
    significant = f["significant"]
    accuracy = (
        f"LightGBM's test RMSE is lower on {len(f['lgbm_lower'])} of {n} subsets; the paired "
        + (
            "difference (LSTM minus LightGBM) is significant only on "
            + ", ".join(f"{s}, {_difference(f['rmse'][s])}" for s in significant)
            if significant
            else "difference is not significant on any subset"
        )
        + f". Averaging five fold models lowers RMSE by {_span(f['gain'], '.1f')} cycles against "
        "the mean single fold model, for both families."
    )
    worst = f["worst_bin"]
    below = [(s, name) for (s, name), v in f["oof"].items() if v < f["nominal"]]
    article = {"lstm": "the ", "lightgbm": ""}
    uncertainty = (
        f"Both models cover {_span(f['coverage'], '.0f')}% of test engines with a nominal "
        f"{f['nominal']:.0f}% interval, so the intervals are conservative on the test sets. "
        "Coverage is not uniform: by true remaining life it falls to "
        + " and ".join(
            f"{100 * row['coverage']:.0f}% for {article[name]}{NAMES[name]} "
            f"({row['bin']} cycles left)"
            for name, row in worst.items()
        )
        + ". Out of fold, over whole lives, it is "
        + f"{_span(list(f['oof'].values()), '.0f')}%"
        + (f", below nominal for {_by_model(below)}" if below else "")
        + ". "
        + (
            f"The LSTM's intervals are narrower on {len(f['narrower'])} of {n} subsets, by "
            f"{_span(f['narrower'], '.0f')}%, significantly on {f['width_significant']}."
            if f["narrower"]
            else "The LSTM's intervals are not narrower than LightGBM's."
        )
    )
    alerts = (
        f"Both alerts rank test engines by risk almost perfectly: average precision "
        f"{_span(f['ap'], '.3f')}, where a random score would get the base rate of "
        f"{_span(f['base_rate'], '.0f')}%. Ranking says nothing about calibration, which is "
        f"measured separately. Over whole lives the first alert comes a median of "
        f"{_span(f['lead'], '.0f')} cycles before failure; the LSTM's alert switches off again "
        f"{_span(f['flaps']['lstm'], '.2f')} times per engine, LightGBM's "
        f"{_span(f['flaps']['lightgbm'], '.2f')} times."
    )
    return "\n".join(
        f"- **{head}** {text}"
        for head, text in (
            ("Accuracy.", accuracy),
            ("Uncertainty.", uncertainty),
            ("Alerting.", alerts),
        )
    )


def findings_block(artifacts: Path) -> str:
    """The three findings under the README hero: accuracy, uncertainty, alerts."""
    f = _findings(artifacts)
    if f is None:
        return "_No evaluated runs yet._"
    n, subsets, cap = f["n"], f["subsets"], f["cap"]
    rmse, clear = f["rmse"], f["significant"]
    lstm_lower = [s for s in subsets if rmse[s]["difference"] < 0]
    accuracy = f"LightGBM's test RMSE is lower on {len(f['lgbm_lower'])} of {n} subsets" + (
        ", but the paired LSTM − LightGBM difference excludes 0 only on "
        + "; ".join(f"{s}: {_difference(rmse[s])}" for s in clear)
        if clear
        else "; no paired LSTM − LightGBM difference excludes 0"
    )
    accuracy += "." + (
        " The LSTM is lower on "
        + "; ".join(f"{s}: {_difference(rmse[s])}" for s in lstm_lower)
        + "."
        if lstm_lower
        else ""
    )

    pooled, worst = f["pooled"], f["worst_bin"]
    engines = next(iter(pooled.values()))["engines"]
    above = all(p["low"] > f["nominal"] / 100 for p in pooled.values())
    open_share = [100 * f["mean"][(s, m, "open_ended_share")] for s in subsets for m in NAMES]
    capped = [f["widths"][s]["lightgbm"]["mean"] - f["widths"][s]["lstm"]["mean"] for s in subsets]
    closed = [f["closed"][s]["lightgbm"] - f["closed"][s]["lstm"] for s in subsets]
    own = [
        f["mean"][(s, "lightgbm", "closed_width")] - f["mean"][(s, "lstm", "closed_width")]
        for s in subsets
    ]
    uncertainty = (
        f"Pooled over all {engines} test engines, the nominal {f['nominal']:.0f}% intervals "
        "cover "
        + " and ".join(f"{100 * p['coverage']:.1f}% ({NAMES[m]})" for m, p in pooled.items())
        + (f", both 95% CIs above {f['nominal']:.0f}%" if above else "")
        + ", but only "
        + " and ".join(
            f"{100 * row['coverage']:.0f}% ({NAMES[m]}, {row['bin']} cycles left)"
            for m, row in worst.items()
        )
        + f" in their weakest band of remaining life. {_span(open_share, '.0f')}% of intervals "
        "are open-ended (“at least N cycles”)."
    )
    if all(c > 0 for c in capped + closed + own):
        uncertainty += (
            f" The LSTM's are narrower on all {n} subsets under each of three width definitions: "
            f"by {_span(capped, '.0f')} cycles with open-ended intervals counted up to the "
            f"{cap}-cycle cap, by {_span(closed, '.0f')} on the engines where both are closed, "
            f"and by {_span(own, '.0f')} over each model's own closed intervals, the only one of "
            "the three fixed before the test labels were read."
        )

    alerts = f["alerts"]
    missed = {m: [s for s in subsets if alerts[(s, m)]["alert.missed"]["mean"] > 0] for m in NAMES}
    misses = [
        f"The {NAMES[m]} misses "
        + ", ".join(
            f"{alerts[(s, m)]['alert.missed']['mean']:.1f} of {f['oof_engines'][s]} {s} engines"
            for s in missed[m]
        )
        + " per repeat"
        for m in NAMES
        if missed[m]
    ]
    none = [NAMES[m] for m in NAMES if not missed[m]]
    alert_text = (
        "Out of fold, over whole lives, both alerts first fire a median "
        f"{_span(f['lead'], '.0f')} cycles before failure. LightGBM's switches off again "
        f"before failure {_span(f['flaps']['lightgbm'], '.2f')} times per engine and the "
        f"LSTM's {_span(f['flaps']['lstm'], '.2f')}, or "
        f"{_span(f['late_flaps']['lightgbm'], '.2f')} against "
        f"{_span(f['late_flaps']['lstm'], '.2f')} counting only switch-offs in the last "
        f"{f['horizon']} cycles, when the alert should stay on. "
        + (
            "; ".join(misses) + (f"; {_and(none)} misses none." if none else ".")
            if misses
            else "Neither misses an engine."
        )
    )
    return "\n".join(
        f"- **{head}** {text}"
        for head, text in (
            ("Accuracy.", accuracy),
            ("Uncertainty.", uncertainty),
            ("Alerts.", alert_text),
        )
    )


def summary_table_block(artifacts: Path) -> str:
    """The compact README table, one row per subset, and where each model is better."""
    f = _findings(artifacts)
    if f is None:
        return "_No evaluated runs yet._"
    n, subsets, mean, diffs = f["n"], f["subsets"], f["mean"], f["differences"]

    def pair(values: dict[str, float], spec: str, lower_better: bool | None, unit: str = "") -> str:
        cells = {m: format(v, spec) + unit for m, v in values.items()}
        if lower_better is not None:
            cells = _bold_best(cells, values, lower_better)
        # Non-breaking spaces keep each pair on one line in a narrow table column.
        return f"{cells['lstm']}&nbsp;/&nbsp;{cells['lightgbm']}"

    lines = [
        "| Subset (test engines) | Test RMSE | RMSE difference, LSTM − LightGBM (95% CI) "
        f"| Inside the nominal {f['nominal']:.0f}% interval | Mean interval width (cycles) "
        "| Open-ended intervals | Alert switch-offs per engine |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in subsets:
        models = list(NAMES)
        flaps = {m: f["alerts"][(s, m)]["alert.flaps_per_engine"]["mean"] for m in models}
        open_share = {m: 100 * mean[(s, m, "open_ended_share")] for m in models}
        lines.append(
            f"| {s} ({f['test_engines'][s]}) "
            f"| {pair({m: mean[(s, m, 'rmse')] for m in models}, '.2f', True)} "
            f"| {_ci(diffs[s]['rmse'], '.2f')} "
            f"| {pair({m: 100 * mean[(s, m, 'coverage')] for m in models}, '.0f', None, '%')} "
            f"| {pair({m: f['widths'][s][m]['mean'] for m in models}, '.1f', True)} "
            f"| {pair(open_share, '.0f', None, '%')} "
            f"| {pair(flaps, '.2f', True)} |"
        )
    notes = (
        f"Each cell is LSTM / LightGBM, the mean over {f['repeats']} repeats of the whole "
        "pipeline; bold marks the lower value where lower is better. The test columns score one "
        "prediction per official test engine, at its last observed cycle, by each repeat's "
        "five-fold ensemble (the mean of the five fold models, with the CV+ interval over "
        f"them). RMSE is against the true RUL capped at {f['cap']} cycles, and coverage is the "
        "share of test engines inside the interval. Width counts an open-ended interval up to "
        "the cap, a summary defined after the test labels were read. The difference is the "
        "paired one, with a 95% bootstrap interval over test engines. Switch-offs are counted "
        "out of fold, over every cycle of every cross-fitted training engine, for single fold "
        "models."
    )

    def better(key: str, lgbm_wins: bool) -> list[str]:
        return [
            s for s in subsets if (mean[(s, "lightgbm", key)] < mean[(s, "lstm", key)]) == lgbm_wins
        ]

    def clearly(key: str, chosen: list[str]) -> str:
        sig = [s for s in chosen if _significant(diffs[s][key])]
        if not sig:
            return " (none clearly)" if len(chosen) > 1 else " (not clearly)"
        if len(sig) == len(chosen):
            return " (all clearly)" if len(chosen) > 1 else " (clearly)"
        return f" (clearly only on {_and(sig)})"

    lgbm: list[str] = []
    lstm: list[str] = []
    for key, label in (("rmse", "test RMSE"), ("nasa_score", "PHM08 score")):
        for lgbm_wins, out in ((True, lgbm), (False, lstm)):
            chosen = better(key, lgbm_wins)
            if chosen:
                out.append(f"lower {label} on {_on(chosen, n)}{clearly(key, chosen)}")
    uncapped = better("rmse_uncapped", True)
    if uncapped:
        lgbm.append(f"lower RMSE against the actual, uncapped remaining life on {_on(uncapped, n)}")
    narrower = [
        s for s in subsets if f["widths"][s]["lstm"]["mean"] < f["widths"][s]["lightgbm"]["mean"]
    ]
    if narrower:
        sig = sum(_significant(f["widths"][s]["difference"]) for s in narrower)
        lstm.append(f"narrower intervals on {_on(narrower, n)} (clearly on {sig})")
    calmer = [
        s
        for s in subsets
        if f["alerts"][(s, "lstm")]["alert.flaps_per_engine"]["mean"]
        < f["alerts"][(s, "lightgbm")]["alert.flaps_per_engine"]["mean"]
    ]
    if calmer:
        lstm.append(f"fewer alert switch-offs on {_on(calmer, n)}")
    coverage_clear = [s for s in subsets if _significant(diffs[s]["coverage"])]
    neither = (
        "no coverage difference excludes 0 on any subset"
        if not coverage_clear
        else f"coverage differs clearly only on {_and(coverage_clear)}"
    ) + f", and alert average precision is {_span(f['ap'], '.3f')} for both."
    bullets = [
        f"- **LightGBM:** {_and(lgbm)}." if lgbm else "",
        f"- **LSTM:** {_and(lstm)}." if lstm else "",
        f"- **Neither:** {neither}",
    ]
    return (
        "\n".join(lines)
        + "\n\n"
        + notes
        + "\n\n**Where each model is better**\n\n"
        + "\n".join(b for b in bullets if b)
    )


def guarantee_block(artifacts: Path) -> str:
    """What the CV+ guarantee is for these subsets, and why the test sets do not meet its
    assumptions: the worst-case bound and the remaining-life mix of calibration and test points."""
    runs = _runs(artifacts)
    if not runs:
        return "_No evaluated runs yet._"
    first = next(iter(runs.values()))["config"]
    cap, folds, alpha = first["max_rul"], first["folds"], first["alpha"]
    lines = [
        "| Subset | Calibration engines | Worst-case CV+ coverage "
        f"| Calibration points with {cap} or more cycles left "
        f"| Test engines with {cap} or more cycles left |",
        "|---|---:|---:|---:|---:|",
    ]
    for subset, m in runs.items():
        n = m["data"]["crossfit_engines"]
        points = pd.read_csv(artifacts / subset / "predictions" / "calibration-points.csv")
        test = pd.read_csv(artifacts / subset / "predictions" / "test.csv").drop_duplicates("unit")
        lines.append(
            f"| {subset} | {n} | {100 * cv_plus_bound(n, folds, alpha):.1f}% "
            f"| {100 * (points['rul'] >= cap).mean():.0f}% "
            f"| {100 * (test['rul_true'] >= cap).mean():.0f}% |"
        )
    a = chr(0x3B1)  # the Greek letter alpha, spelled out so the source stays ASCII-safe
    return (
        f"{folds}-fold CV+ (Barber et al., 2021, Theorem 4) guarantees marginal coverage of at "
        f"least 1 − 2{a} − min{{2(1 − 1/K)/(n/K + 1), (1 − K/n)/(K + 1)}} for n calibration "
        f"engines and K folds. At the nominal {100 * (1 - alpha):.0f}% ({a} = {alpha:g}) and "
        f"K = {folds}:"
        "\n\n" + "\n".join(lines) + "\n\n"
        f"The {100 * (1 - 2 * alpha):.0f}% often quoted is the jackknife+ bound, K = n. The "
        "guarantee needs each new pair of engine and truncation point to be exchangeable with "
        "the calibration pairs, one score per engine, folds fixed independently of the data and "
        "training that treats every engine alike. The test sets do not meet that: their "
        "engines are less often far from failure than the calibration points (table), the "
        "folds are stratified by life length, and the feature scaling and operating-regime "
        "clusters were fitted on the sensor readings of every calibration engine. The guarantee "
        "is also marginal, so it says nothing about coverage at a given remaining life. The "
        "evidence for these intervals is their measured coverage, not the bound."
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
            "served five-network average:",
            "",
            f"- **Completeness holds numerically.** Integrated gradients satisfy it by "
            f"construction, so this checks the numerical integration: the attributions add up to "
            f"the change in prediction to within {med('completeness_gap_cycles')} cycles "
            f"(median), against a median change of {abs(s['effect_cycles']['median']):.0f} cycles.",
            f"- **In the median window they find readings the output depends on.** Removing "
            f"readings in attribution order moves the prediction to the baseline faster than "
            f"random order: median deletion area {med('deletion_auc_ig', '.3f')} against "
            f"{med('deletion_auc_random', '.3f')} (lower is better). It is not better in every "
            f"window: it beats random order in {c['ig_beats_random_on_deletion']} of {n}, and "
            f"in the worst tenth of windows its deletion area exceeds "
            f"{s['deletion_auc_ig']['p90']:.3f}.",
            f"- **Much of that is input deviation.** Ranking readings simply by how far they are "
            f"from the baseline does about as well: median deletion area "
            f"{med('deletion_auc_input_deviation', '.3f')}, and a worst tenth above only "
            f"{s['deletion_auc_input_deviation']['p90']:.3f}. Attributions beat it in "
            f"{c['ig_beats_input_deviation_on_deletion']} of {n} windows on deletion and in "
            f"{c['ig_beats_input_deviation_on_insertion']} of {n} on insertion.",
            f"- **The trained weights matter, moderately.** A randomly re-initialised network "
            f"gives maps that rank-correlate {med('spearman_vs_full_randomisation')} with the "
            f"trained ones and delete less well ({med('deletion_auc_randomised_ig', '.3f')}).",
            f"- **The baseline matters.** Using each engine's own first cycles as the baseline "
            f"instead of the regime average changes the maps (rank correlation "
            f"{med('spearman_vs_own_baseline')}; the top three sensors keep "
            f"{s['top3_sensors_kept_with_own_baseline']['median']:.0f} of 3 in the median "
            "window).",
            f"- **Scope.** {r['windows']} windows from {r['engines']} engines of one subset, for "
            "the served network's point prediction, with one seed and no confidence intervals. "
            "There is no ground truth for which readings matter.",
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


def example_files(artifacts: Path) -> dict[str, Path]:
    """The example request and response, and the recorded dashboard state, of the project
    that holds ``artifacts`` (in this repository: ``docs/``)."""
    docs = artifacts.resolve().parent / "docs"
    return {
        "request": docs / "example-request.json",
        "response": docs / "example-response.json",
        "dashboard": docs / "figures" / "dashboard-state.json",
    }


def example_response(model_dir: Path, request: Path) -> dict:
    """The /api/predict response to the example request, through the real Flask route."""
    from predmaint.api import create_app

    settings = Settings(
        artifacts_dir=model_dir.parent,
        subset=model_dir.name,
        replay_autostart=False,
        replay_engines=1,
    )
    client = create_app(settings).test_client()
    response = client.post("/api/predict", json=json.loads(request.read_text()))
    if response.status_code != 200:
        raise SystemExit(f"/api/predict answered {response.status_code}: {response.get_data()!r}")
    return response.get_json()


def matching_test_engine(response: dict, run_dir: Path) -> pd.Series | None:
    """The served model's test prediction (LSTM, served repeat) that the response reproduces,
    if exactly one test engine matches it to the response's rounding."""
    metrics = json.loads((run_dir / "metrics.json").read_text())
    test = pd.read_csv(run_dir / "predictions" / "test.csv")
    test = test[(test["model"] == "lstm") & (test["repeat"] == metrics["served"]["repeat"])]
    rul, alert = response["rul"], response["failure_within_horizon"]
    match = test[
        ((test["point"] - rul["point"]).abs() <= 0.051)
        & ((test["lower"] - rul["lower"]).abs() <= 0.051)
        & (test["open_ended"].astype(bool) == rul["open_ended"])
        & ((test["probability"] - alert["probability"]).abs() <= 0.00011)
    ]
    return match.iloc[0] if len(match) == 1 else None


def _compact_json(value: Any, indent: str = "  ") -> str:
    """JSON with one top-level field per line and each value on one line."""
    lines = [
        f'{indent}"{key}": {json.dumps(item, separators=(", ", ": "), ensure_ascii=False)}'
        for key, item in value.items()
    ]
    return "{\n" + ",\n".join(lines) + "\n}"


def api_example_block(artifacts: Path) -> str:
    """The stored example response, trimmed for the README, and the test engine it matches."""
    path = example_files(artifacts)["response"]
    if not path.is_file():
        return "_No example response yet: run `python -m predmaint.report`._"
    response = json.loads(path.read_text())
    shown = dict(response)
    explanation = response.get("explanation")
    more = 0
    if explanation:
        sensors = explanation["sensors"]
        more = len(sensors) - 3
        shown["explanation"] = {
            "method": explanation["method"],
            "baseline_rul_cycles": explanation["baseline_rul_cycles"],
            "sensors": [
                {k: s[k] for k in ("symbol", "description", "contribution_cycles")}
                for s in sensors[:3]
            ],
        }
    lines = [
        "```json",
        _compact_json(shown),
        "```",
        "",
    ]
    rul = response["rul"]
    match = matching_test_engine(response, artifacts / "FD001")
    trimmed = (
        f"The explanation is trimmed to its three largest sensor effects ({more} more in the "
        "full response, [`docs/example-response.json`](example-response.json))."
        if explanation
        else ""
    )
    if match is not None:
        truth = int(match["rul_true"])
        inside = rul["lower"] <= truth and (rul["open_ended"] or truth <= rul["upper"])
        lines.append(
            f"The request holds {response['cycles_received']} cycles of FD001 test engine "
            f"{int(match['unit'])}, ending at the last of its {int(match['length'])} recorded "
            "cycles, and the response matches, to its rounding, the prediction scored for that "
            f"engine in the test table (LSTM, served repeat). NASA's true RUL for it is {truth} "
            "cycles: "
            + ("inside" if inside else "outside")
            + f" the interval, {abs(truth - rul['point']):.0f} cycles "
            + ("above" if truth > rul["point"] else "below")
            + f" the point estimate. {trimmed}"
        )
    else:
        lines.append(
            "The response matches no single test-set prediction of the served model, so it is "
            f"shown without a label. {trimmed}"
        )
    return "\n".join(lines).rstrip()


def dashboard_block(artifacts: Path) -> str:
    """What the dashboard screenshots show, from the state the screenshot script recorded."""
    path = example_files(artifacts)["dashboard"]
    if not path.is_file():
        return "_No screenshot state yet: run `python -m predmaint.screenshots`._"
    s = json.loads(path.read_text())
    interval = (
        f"at least {s['lower']:.0f}" if s["open_ended"] else f"{s['lower']:.0f} to {s['upper']:.0f}"
    )
    return (
        f"Both captures show the same replay state: {s['subset']} replay engine {s['unit']} at "
        f"cycle {s['cycle']} of {s['total_cycles']}, the cycle at which its alert first fires "
        f"(the hero figure shows its whole life). The dashboard predicts {s['point']:.0f} cycles "
        f"(nominal {100 * s['interval']:.0f}% interval {interval}) and a {s['probability']:.0%} "
        f"probability of failure within {s['horizon_cycles']} cycles; the true RUL, known because "
        f"the simulated engine ran to failure, is {s['rul_true']}. These are the numbers in that "
        "engine's row of `replay-holdout.csv.gz`, which `predmaint.audit` checks."
    )


BLOCKS = {
    "headline": lambda artifacts: headline_block(artifacts),
    "findings": lambda artifacts: findings_block(artifacts),
    "summary-table": lambda artifacts: summary_table_block(artifacts),
    "results": lambda artifacts: results_block(artifacts),
    "summary": lambda artifacts: summary_block(artifacts),
    "guarantee": lambda artifacts: guarantee_block(artifacts),
    "attribution": lambda artifacts: attribution_block(artifacts),
    "latency": lambda artifacts: latency_block(PROJECT_ROOT / "docs" / "benchmark.json"),
    "api-example": lambda artifacts: api_example_block(artifacts),
    "dashboard": lambda artifacts: dashboard_block(artifacts),
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
    """Rewrite every generated block whose markers the document has."""
    text = document.read_text(encoding="utf-8")
    if not any(markers(name)[0] in text for name in BLOCKS):
        raise SystemExit(f"{document} has no generated block")
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
    files = example_files(artifacts)
    if (artifacts / "FD001" / "model.pt").is_file() and files["request"].is_file():
        response = example_response(artifacts / "FD001", files["request"])
        text = json.dumps(response, indent=2, ensure_ascii=False) + "\n"
        files["response"].write_text(text, encoding="utf-8")
        print(f"Wrote {files['response']}")
    for document in args.docs:
        update_readme(document, artifacts)
        print(f"Updated results block in {document}")
    if update_ledger():
        print(f"Updated the generated section of {LEDGER_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
