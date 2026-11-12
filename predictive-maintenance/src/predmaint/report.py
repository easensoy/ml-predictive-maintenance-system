"""Write the generated parts of the documents from ``artifacts/<subset>/``.

    python -m predmaint.report          # rewrites every generated block of every document

The documents are README.md, MODEL_CARD.md, docs/evaluation.md and docs/serving.md.

It fills each block whose markers a document has: the abstract ("headline"), the findings, the
summary table, the full results tables, the interval-width statistics, the conformal-guarantee
conditions, the attribution checks, the latency table, the dashboard state and the API example.
It first
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

from predmaint.calibration import _ranks, covered
from predmaint.config import DOCUMENTS, PROJECT_ROOT, SUBSETS, Settings
from predmaint.metrics import (
    BOOT_SEED,
    N_BOOT,
    bootstrap_indices,
    mean_sd,
    percentile_interval,
    wilson_interval,
)
from predmaint.metrics import by_engine as _by_engine

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
        "| Subset | Model | RMSE | PHM08 score | RMSE vs uncapped RUL "
        "| Empirical coverage, nominal 90% (95% CI) | Open-ended intervals "
        "| Mean width, open-ended truncated at the cap | Alert AP | Alert ECE |"
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


def replay_table(runs: dict[str, dict]) -> str:
    lines = [
        "| Subset | Model | Engines | RMSE, every cycle | Empirical coverage, every cycle "
        "| Open-ended intervals | First alert, median cycles before failure | Missed "
        "| First alert with more than 30 cycles left | Switch-offs before failure per engine |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for subset, m in runs.items():
        for name, block in m.get("replay_holdout", {}).items():
            s = block["summary"]
            engines = next(iter(block["per_repeat"].values()))["engines"]
            lines.append(
                f"| {subset} | {NAMES[name]} | {engines} "
                f"| {_ms(s['rmse_per_engine_weighted'], '.2f')} "
                f"| {_ms(s['coverage_per_engine_weighted'], '.1f', 100)}% "
                f"| {_ms(s['open_ended_share'], '.0f', 100)}% "
                f"| {_ms(s['alert.lead_median'], '.0f')} | {_ms(s['alert.missed'], '.1f')} "
                f"| {_ms(s['alert.early'], '.1f')} | {_ms(s['alert.flaps_per_engine'], '.2f')} |"
            )
    return "\n".join(lines)


def results_block(artifacts: Path) -> str:
    runs = _runs(artifacts)
    if not runs:
        return "_No evaluated runs yet: run `predmaint.train`, then `predmaint.evaluate`._"
    first = next(iter(runs.values()))
    repeats, cap = len(first["config"]["repeats"]), first["config"]["max_rul"]
    replay = [
        next(iter(m["replay_holdout"]["lstm"]["per_repeat"].values()))["engines"]
        for m in runs.values()
        if "lstm" in m.get("replay_holdout", {})
    ]
    parts = [
        "**Official test endpoints.** One prediction per test engine, at its last observed "
        "cycle. Each repeat of each model is evaluated as its five-fold ensemble: the point "
        "prediction is the mean of the five fold models and the interval is their CV+ interval. "
        "The API and dashboard serve the first repeat's LSTM ensemble. Entries are mean ± sample "
        f"sd over {repeats} repeats on the same test engines; the ± is repeat variability (fold "
        "assignment, inner-validation splits, initialisation and calibration truncation; the "
        "replay holdout and the preprocessing are fixed), not the precision of the estimate. "
        "Bold marks the better model in each subset. RMSE is against the true RUL capped at "
        f"{cap} cycles; RMSE against the uncapped RUL scores the same capped predictions, so it "
        f"includes the error the cap imposes on engines with more than {cap} cycles left. "
        "Coverage is the empirical, marginal share of test engines whose capped "
        "RUL lies in the interval, an open-ended interval counting as [lower, ∞); its 95% "
        "Wilson interval takes n = test engines at the repeat-mean rate and covers engine "
        "sampling only. Width truncates open-ended intervals at the cap, a summary defined "
        "after the test labels were read; the evaluation notes give two other width "
        "statistics.",
        test_table(runs),
        "**Paired differences, LSTM minus LightGBM, on the same test engines.** Repeat-averaged "
        "difference with a 95% percentile interval from a paired engine-level bootstrap: 2,000 "
        "resamples of the test engines, each applied to both models and every repeat. It "
        "reflects which engines are in the test set, for the trained models; it excludes "
        f"repeat variability, and the {4 * len(runs)} intervals are not corrected for multiple "
        "comparisons.",
        difference_table(runs),
        "**Alerts over whole lives, out of fold.** Every cycle of every cross-fitted training "
        "engine, each predicted by the single fold model that never saw it, not by an ensemble. "
        "The alert fires when the probability of failure within 30 cycles reaches 0.5. Engine "
        "counts are means ± sd over repeats; the ECE interval is the repeat-averaged 95% "
        "bootstrap interval over engines.",
        alert_table(runs),
    ]
    if replay:
        parts += [
            "**The ensembles over whole lives: replay holdout.** Every cycle of the held-out "
            "replay engines, which no network, preprocessing or calibration step used, predicted "
            "by each repeat's five-fold ensemble, the form the API serves (only the first "
            "repeat's LSTM is deployed). Definitions as above; RMSE and coverage weight engines "
            f"equally. With {_span([float(r) for r in replay], '.0f')} engines per subset, the "
            "alert counts are small.",
            replay_table(runs),
        ]
    return "\n\n".join(parts)


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
                "coverage_actual_closed_reading",
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
        "replay": {
            (s, name): m["replay_holdout"][name]["summary"]
            for s, m in complete.items()
            for name in ("lstm", "lightgbm")
            if name in m.get("replay_holdout", {})
        },
    }


def _width_margins(f: dict[str, Any]) -> dict[str, list[float]]:
    """LightGBM's mean width minus the LSTM's, per subset, under each width statistic: open-ended
    intervals truncated at the cap, engines where both intervals are closed, and each model's
    own closed intervals (the one defined before the test labels were read)."""
    subsets = f["subsets"]
    return {
        "truncated": [
            f["widths"][s]["lightgbm"]["mean"] - f["widths"][s]["lstm"]["mean"] for s in subsets
        ],
        "both_closed": [f["closed"][s]["lightgbm"] - f["closed"][s]["lstm"] for s in subsets],
        "own_closed": [
            f["mean"][(s, "lightgbm", "closed_width")] - f["mean"][(s, "lstm", "closed_width")]
            for s in subsets
        ],
    }


def _early_share(f: dict[str, Any], model: str) -> float:
    """Out of fold, all subsets pooled: the share of engines whose first alert came with more
    than the alert horizon left (mean over repeats)."""
    early = sum(f["alerts"][(s, model)]["alert.early"]["mean"] for s in f["subsets"])
    return 100 * early / sum(f["oof_engines"][s] for s in f["subsets"])


def _early_within_twice(f: dict[str, Any], model: str) -> float:
    """Of those early first alerts, the share that came with at most twice the horizon left."""
    early = sum(f["alerts"][(s, model)]["alert.early"]["mean"] for s in f["subsets"])
    twice = sum(f["alerts"][(s, model)]["alert.early_twice_horizon"]["mean"] for s in f["subsets"])
    return 100 * (1 - twice / early) if early else 100.0


def _missing(f: dict[str, Any]) -> dict[str, list[str]]:
    """Per model, the subsets on which its out-of-fold alert misses at least one engine."""
    return {
        m: [s for s in f["subsets"] if f["alerts"][(s, m)]["alert.missed"]["mean"] > 0]
        for m in NAMES
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


def _dash(values: list[float], spec: str) -> str:
    """'30–31' or '30': a compact range for the opening, where every word counts."""
    low, high = format(min(values), spec), format(max(values), spec)
    return low if low == high else f"{low}–{high}"


def headline_block(artifacts: Path) -> str:
    """The README abstract, at most 80 words: the controlled comparison, then the principal
    result. Definitions and qualifications are in the reading block below the table."""
    f = _findings(artifacts)
    if f is None:
        return "_No evaluated runs yet._"
    n, k, clear = f["n"], len(f["lgbm_lower"]), f["significant"]
    task = (
        "An LSTM on raw sensor windows and LightGBM on per-sensor window statistics predict "
        "remaining useful life on NASA's simulated C-MAPSS turbofans under one engine-disjoint "
        f"cross-fitting protocol, scored separately on point error, nominal {f['nominal']:.0f}% "
        f"intervals and a {f['horizon']}-cycle alert."
    )
    where = (
        f"paired 95% interval excludes zero only on {clear[0]}"
        if len(clear) == 1
        else f"paired 95% intervals exclude zero on {_and(clear)}"
        if clear
        else "no paired 95% interval excludes zero"
    )
    if 2 * k >= n:
        accuracy = (
            f"LightGBM is at least as accurate (lower test RMSE on {k} of {n} subsets; {where})"
        )
    else:
        accuracy = f"The LSTM is more accurate (lower test RMSE on {n - k} of {n} subsets; {where})"
    lstm = []
    if all(v > 0 for margins in _width_margins(f).values() for v in margins):
        similar = not any(_significant(d["coverage"]) for d in f["differences"].values())
        lstm.append("narrower intervals" + (" at similar coverage" if similar else ""))
    if all(a < b for a, b in zip(f["flaps"]["lstm"], f["flaps"]["lightgbm"], strict=True)):
        missing = _missing(f)
        only = bool(missing["lstm"]) and not missing["lightgbm"]
        lstm.append(
            "an out-of-fold alert that switches off less often"
            + (" but misses engines" if only else "")
        )
    return f"{task} {accuracy}" + (f"; the LSTM gives {' and '.join(lstm)}." if lstm else ".")


def _miss_counts(f: dict[str, Any], model: str) -> str:
    """'1.7 of 224 FD004 engines per repeat' for every subset on which the model misses."""
    return ", ".join(
        f"{f['alerts'][(s, model)]['alert.missed']['mean']:.1f} of {f['oof_engines'][s]} {s} "
        "engines per repeat"
        for s in _missing(f)[model]
    )


def _misses(f: dict[str, Any]) -> str:
    """Which model's out-of-fold alert misses engines, and how many."""
    missing = _missing(f)
    if missing["lstm"] and not missing["lightgbm"]:
        return f"the LSTM alone misses engines ({_miss_counts(f, 'lstm')})"
    if missing["lightgbm"] and not missing["lstm"]:
        return f"LightGBM alone misses engines ({_miss_counts(f, 'lightgbm')})"
    if missing["lstm"]:
        return (
            f"both miss engines (LSTM {_miss_counts(f, 'lstm')}; LightGBM "
            f"{_miss_counts(f, 'lightgbm')})"
        )
    return "neither misses an engine"


def _replay_flaps(f: dict[str, Any]) -> dict[str, list[float]]:
    """Per model, the five-fold ensembles' switch-offs per engine on each replay holdout."""
    return {
        m: [
            f["replay"][(s, m)]["alert.flaps_per_engine"]["mean"]
            for s in f["subsets"]
            if (s, m) in f["replay"]
        ]
        for m in NAMES
    }


def summary_block(artifacts: Path) -> str:
    """The findings in full, for the evaluation notes: the subsidiary statistics that the
    README's three findings leave out, each with the population it is measured on."""
    f = _findings(artifacts)
    if f is None:
        return "_No evaluated runs yet._"
    n, subsets, horizon = f["n"], f["subsets"], f["horizon"]
    clear = f["significant"]
    lstm_lower = [s for s in subsets if f["rmse"][s]["difference"] < 0]
    accuracy = (
        f"LightGBM's test RMSE is lower on {len(f['lgbm_lower'])} of {n} subsets; the paired "
        "difference (LSTM minus LightGBM) "
        + (
            ("excludes zero only on " if len(clear) < n else "excludes zero on ")
            + "; ".join(f"{s}, {_difference(f['rmse'][s])}" for s in clear)
            if clear
            else "excludes zero on no subset"
        )
        + "."
        + (
            " The LSTM is lower on "
            + "; ".join(f"{s}, {_difference(f['rmse'][s])}" for s in lstm_lower)
            + "."
            if lstm_lower
            else ""
        )
        + f" Averaging the five fold models lowers RMSE by {_span(f['gain'], '.1f')} cycles "
        "relative to the mean single fold model, for both families."
    )
    worst = f["worst_bin"]
    below = [(s, name) for (s, name), v in f["oof"].items() if v < f["nominal"]]
    article = {"lstm": "the ", "lightgbm": ""}
    open_share = [100 * f["mean"][(s, m, "open_ended_share")] for s in subsets for m in NAMES]
    narrower = all(v > 0 for margins in _width_margins(f).values() for v in margins)
    intervals = (
        f"Empirical coverage of the nominal {f['nominal']:.0f}% intervals is "
        f"{_span(f['coverage'], '.0f')}% per subset and model, with "
        f"{_span(open_share, '.0f')}% of intervals open-ended. It is marginal coverage: by true "
        "remaining life it falls to "
        + " and ".join(
            f"{100 * row['coverage']:.0f}% for {article[name]}{NAMES[name]} "
            f"({row['bin']} cycles left)"
            for name, row in worst.items()
        )
        + ". Out of fold, over whole lives, where each single fold model's quantiles are widened "
        "by the conformal quantile of the other folds' scores, it is "
        f"{_span(list(f['oof'].values()), '.0f')}%"
        + (f", below nominal for {_by_model(below)}" if below else "")
        + ". "
        + (
            "The LSTM's mean width is lower under all three width statistics on every subset "
            "([Interval width](#interval-width))."
            if narrower
            else "The width statistics do not all favour one model "
            "([Interval width](#interval-width))."
        )
    )
    late, replay = f["late_flaps"], _replay_flaps(f)
    alerts = (
        f"*Ranking* (test endpoints): average precision {_span(f['ap'], '.3f')} against base "
        f"rates of {_span(f['base_rate'], '.0f')}%. *Threshold decisions* (out of fold): the "
        f"first alert comes a median {_span(f['lead'], '.0f')} cycles before failure; "
        f"{_early_share(f, 'lstm'):.0f}% (LSTM) and {_early_share(f, 'lightgbm'):.0f}% (LightGBM) "
        f"of first alerts come with more than {horizon} cycles left, false alarms at that "
        "moment by the label's definition, and "
        f"{_early_within_twice(f, 'lstm'):.0f}% and {_early_within_twice(f, 'lightgbm'):.0f}% of "
        f"those with at most {2 * horizon}; {_misses(f)}. *Temporal "
        "stability* (out of fold): the alert switches off before failure "
        f"{_span(f['flaps']['lstm'], '.2f')} times per engine for the LSTM and "
        f"{_span(f['flaps']['lightgbm'], '.2f')} for LightGBM, or "
        f"{_span(late['lstm'], '.2f')} against {_span(late['lightgbm'], '.2f')} counting only "
        f"switch-offs in the last {horizon} cycles."
        + (
            " With the five-fold ensembles on the replay holdout, switch-offs are "
            f"{_span(replay['lstm'], '.2f')} against {_span(replay['lightgbm'], '.2f')} per engine."
            if replay["lstm"] and replay["lightgbm"]
            else ""
        )
        + " *Calibration* is reported separately, as ECE in the tables and as reliability "
        "diagrams below."
    )
    return "\n".join(
        f"- **{head}** {text}"
        for head, text in (
            ("Point accuracy.", accuracy),
            ("Intervals.", intervals),
            ("Alerts.", alerts),
        )
    )


def findings_block(artifacts: Path) -> str:
    """The three findings under the README hero, at most 35 words each, one principal result
    apiece. Their populations, intervals and caveats are in the reading block."""
    f = _findings(artifacts)
    if f is None:
        return "_No evaluated runs yet._"
    n, clear, rmse = f["n"], f["significant"], f["rmse"]
    lower = len(f["lgbm_lower"])
    if clear:
        detail = "; ".join(f"{s}: {_difference(rmse[s])}" for s in clear)
        only = "only " if len(clear) < n else ""
        accuracy = (
            f"LightGBM's test RMSE is lower on {lower} of {n} subsets, but the paired difference "
            f"excludes zero {only}on {detail}."
        )
    else:
        accuracy = (
            f"LightGBM's test RMSE is lower on {lower} of {n} subsets; no paired difference "
            "excludes zero."
        )
    pooled = f["pooled"]
    engines = next(iter(pooled.values()))["engines"]
    shares = [100 * f["mean"][(s, m, "open_ended_share")] for s in f["subsets"] for m in NAMES]
    narrower = all(v > 0 for values in _width_margins(f).values() for v in values)
    intervals = (
        f"Pooled over {engines} test engines, coverage of the nominal {f['nominal']:.0f}% "
        "intervals is "
        + " and ".join(f"{100 * p['coverage']:.1f}% ({NAMES[m]})" for m, p in pooled.items())
        + f", with {_dash(shares, '.0f')}% open-ended"
        + (
            "; the LSTM's intervals are narrower on every subset under all three width statistics."
            if narrower
            else "."
        )
    )
    flaps, missing = f["flaps"], _missing(f)
    if missing["lstm"] and not missing["lightgbm"]:
        ending = ", but only the LSTM misses engines."
    elif missing["lightgbm"] and not missing["lstm"]:
        ending = ", but only LightGBM misses engines."
    else:
        ending = "."
    alerts = (
        f"Out of fold, both alerts first fire a median {_dash(f['lead'], '.0f')} cycles before "
        f"failure; LightGBM's switches off again {_dash(flaps['lightgbm'], '.2f')} times per "
        f"engine against the LSTM's {_dash(flaps['lstm'], '.2f')}{ending}"
    )
    return "\n".join(
        f"- **{head}** {text}"
        for head, text in (
            ("Point accuracy.", accuracy),
            ("Intervals.", intervals),
            ("Alert dynamics.", alerts),
        )
    )


def reading_block(artifacts: Path) -> str:
    """How to read the README's findings: the populations, what each interval covers, the
    width statistics and the alert evidence, with the numbers the findings leave out."""
    f = _findings(artifacts)
    if f is None:
        return "_No evaluated runs yet._"
    n, cap, horizon = f["n"], f["cap"], f["horizon"]
    worst, late, replay = f["worst_bin"], f["late_flaps"], _replay_flaps(f)
    margins = _width_margins(f)
    misses = _misses(f)
    bullets = [
        "**Test endpoints.** Accuracy and intervals score one prediction per official test "
        "engine, at its last observed cycle, from each repeat's five-fold ensemble, averaged "
        f"over {f['repeats']} repeats. RMSE is against the true RUL capped at {cap} cycles.",
        "**Paired differences.** Each 95% CI is a paired engine-level bootstrap interval: it "
        "reflects which engines are in the test set, excludes repeat variability and is not "
        f"corrected for the {4 * n} comparisons.",
        "**Coverage** is marginal: by true RUL it falls to "
        + " and ".join(
            f"{100 * row['coverage']:.0f}% ({NAMES[m]}, {row['bin']} cycles left)"
            for m, row in worst.items()
        )
        + f". An open-ended interval, one whose upper end reached the {cap}-cycle cap, means "
        "[lower, ∞).",
    ]
    if all(v > 0 for values in margins.values() for v in values):
        bullets.append(
            "**Width.** The LSTM's intervals are narrower by "
            f"{_span(margins['truncated'], '.0f')} cycles with open-ended intervals truncated at "
            f"the cap, by {_span(margins['both_closed'], '.0f')} on engines where both "
            f"intervals are closed, and by {_span(margins['own_closed'], '.0f')} over each "
            "model's own closed intervals, the only one of the three defined before the test "
            "labels were read."
        )
    replay_text = (
        "; the five-fold ensembles on the held-out replay engines give "
        f"{_span(replay['lightgbm'], '.2f')} against {_span(replay['lstm'], '.2f')}"
        if replay["lstm"] and replay["lightgbm"]
        else ""
    )
    bullets.append(
        "**Alerts** are scored out of fold: every cycle of every cross-fitted training engine, "
        "predicted by the single fold model that never saw it. "
        f"{_early_share(f, 'lstm'):.0f}% (LSTM) and {_early_share(f, 'lightgbm'):.0f}% "
        f"(LightGBM) of first alerts come with more than {horizon} cycles left, false alarms by "
        "the label's definition when they fire. Counting only switch-offs in the last "
        f"{horizon} cycles gives {_span(late['lightgbm'], '.2f')} (LightGBM) against "
        f"{_span(late['lstm'], '.2f')} (LSTM){replay_text}. {misses[0].upper()}{misses[1:]}."
    )
    return "\n".join(f"- {b}" for b in bullets)


def _pair(values: Mapping[str, float], spec: str, lower_better: bool | None, unit: str = "") -> str:
    """'LSTM&nbsp;/&nbsp;LightGBM' cells, the lower one bold where lower is better."""
    cells = {m: format(v, spec) + unit for m, v in values.items()}
    if lower_better is not None:
        cells = _bold_best(cells, values, lower_better)
    # Non-breaking spaces keep each pair on one line in a narrow table column.
    return f"{cells['lstm']}&nbsp;/&nbsp;{cells['lightgbm']}"


def summary_table_block(artifacts: Path) -> str:
    """The compact README table, one row per subset, and where each model is better."""
    f = _findings(artifacts)
    if f is None:
        return "_No evaluated runs yet._"
    n, subsets, mean, diffs = f["n"], f["subsets"], f["mean"], f["differences"]
    pair = _pair

    lines = [
        "| Subset (test engines) | Test RMSE | LSTM − LightGBM RMSE (95% CI) "
        f"| Coverage at nominal {f['nominal']:.0f}% | Open-ended intervals "
        f"| Mean width (truncated at {f['cap']}) | Switch-offs per engine (out of fold) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in subsets:
        models = list(NAMES)
        flaps = {m: f["alerts"][(s, m)]["alert.flaps_per_engine"]["mean"] for m in models}
        open_share = {m: 100 * mean[(s, m, "open_ended_share")] for m in models}
        lines.append(
            f"| {s}&nbsp;({f['test_engines'][s]}) "
            f"| {pair({m: mean[(s, m, 'rmse')] for m in models}, '.2f', True)} "
            f"| {_ci(diffs[s]['rmse'], '.2f').replace(' ', '&nbsp;')} "
            f"| {pair({m: 100 * mean[(s, m, 'coverage')] for m in models}, '.0f', None, '%')} "
            f"| {pair(open_share, '.0f', None, '%')} "
            f"| {pair({m: f['widths'][s][m]['mean'] for m in models}, '.1f', True)} "
            f"| {pair(flaps, '.2f', True)} |"
        )
    notes = (
        f"Each cell is LSTM / LightGBM, the mean over {f['repeats']} repeats; bold marks the "
        "lower value where lower is better. The test columns score the official test endpoints: "
        "one prediction per test engine, at its last observed cycle, from each repeat's "
        f"five-fold ensemble. RMSE is against the true RUL capped at {f['cap']} cycles. Coverage "
        "is the empirical, marginal share of test engines whose capped RUL lies in the nominal "
        f"{f['nominal']:.0f}% interval, an open-ended interval counting as [lower, ∞); the next "
        "column gives the share of such intervals. Width truncates open-ended intervals at the "
        "cap, a summary defined after the test labels were read; the evaluation notes give two "
        "other width statistics. The CI is a paired engine-level bootstrap interval. "
        "Switch-offs are counted out of fold, over every cycle of every cross-fitted training "
        "engine, for single fold models."
    )

    def better(key: str, lgbm_wins: bool) -> list[str]:
        return [
            s for s in subsets if (mean[(s, "lightgbm", key)] < mean[(s, "lstm", key)]) == lgbm_wins
        ]

    def clearly(key: str, chosen: list[str]) -> str:
        sig = [s for s in chosen if _significant(diffs[s][key])]
        if not sig:
            return " (no CI excludes zero)" if len(chosen) > 1 else " (the CI includes zero)"
        if len(sig) == len(chosen):
            return " (every CI excludes zero)" if len(chosen) > 1 else " (the CI excludes zero)"
        return f" (the CI excludes zero only on {_and(sig)})"

    lgbm: list[str] = []
    lstm: list[str] = []
    for key, label in (("rmse", "test RMSE"), ("nasa_score", "PHM08 score")):
        for lgbm_wins, out in ((True, lgbm), (False, lstm)):
            chosen = better(key, lgbm_wins)
            if chosen:
                out.append(f"lower {label} on {_on(chosen, n)}{clearly(key, chosen)}")
    uncapped = better("rmse_uncapped", True)
    if uncapped:
        lgbm.append(f"lower RMSE against the uncapped remaining life on {_on(uncapped, n)}")
    narrower = [
        s for s in subsets if f["widths"][s]["lstm"]["mean"] < f["widths"][s]["lightgbm"]["mean"]
    ]
    if narrower:
        sig = [s for s in narrower if _significant(f["widths"][s]["difference"])]
        where = (
            "every CI excludes zero"
            if len(sig) == len(narrower)
            else f"the CI excludes zero on {_and(sig)}"
            if sig
            else "no CI excludes zero"
        )
        lstm.append(f"narrower truncated widths on {_on(narrower, n)} ({where})")
    calmer = [
        s
        for s in subsets
        if f["alerts"][(s, "lstm")]["alert.flaps_per_engine"]["mean"]
        < f["alerts"][(s, "lightgbm")]["alert.flaps_per_engine"]["mean"]
    ]
    if calmer:
        lstm.append(f"fewer out-of-fold alert switch-offs on {_on(calmer, n)}")
    missing = _missing(f)
    article = {"lstm": "the ", "lightgbm": ""}
    for winner, loser, out in (("lightgbm", "lstm", lgbm), ("lstm", "lightgbm", lstm)):
        if missing[loser] and not missing[winner]:
            out.append(
                f"no missed engines out of fold ({article[loser]}{NAMES[loser]} misses engines "
                f"on {_and(missing[loser])})"
            )
    coverage_clear = [s for s in subsets if _significant(diffs[s]["coverage"])]
    neither = (
        "no coverage difference excludes zero on any subset"
        if not coverage_clear
        else f"the coverage difference excludes zero only on {_and(coverage_clear)}"
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


def widths_block(artifacts: Path) -> str:
    """The three interval-width statistics side by side, so that every width comparison names
    its population and how it treats open-ended intervals."""
    f = _findings(artifacts)
    if f is None:
        return "_No evaluated runs yet._"
    cap, mean = f["cap"], f["mean"]
    lines = [
        "| Subset | Open-ended intervals | Truncated width, every test engine "
        "| Truncated width, LSTM − LightGBM (95% CI) | Engines with both intervals closed "
        "| Width on those engines | Width over each model's own closed intervals |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in f["subsets"]:
        closed = f["closed"][s]
        open_share = {m: 100 * mean[(s, m, "open_ended_share")] for m in NAMES}
        lines.append(
            f"| {s} "
            f"| {_pair(open_share, '.0f', None, '%')} "
            f"| {_pair({m: f['widths'][s][m]['mean'] for m in NAMES}, '.1f', True)} "
            f"| {_ci(f['widths'][s]['difference'], '.1f')} "
            f"| {closed['engines']:.0f} of {f['test_engines'][s]} "
            f"| {_pair({m: closed[m] for m in NAMES}, '.1f', True)} "
            f"| {_pair({m: mean[(s, m, 'closed_width')] for m in NAMES}, '.1f', True)} |"
        )
    margins = _width_margins(f)
    verdict = (
        "The LSTM's mean width is lower under all three statistics on every subset."
        if all(v > 0 for values in margins.values() for v in values)
        else "The three statistics do not all rank the models the same way on every subset."
    )
    closed_reading = [
        100 * mean[(s, m, "coverage_actual_closed_reading")] for s in f["subsets"] for m in NAMES
    ]
    return (
        "\n".join(lines)
        + "\n\n"
        + f"Widths in cycles; each cell is LSTM / LightGBM, the mean over {f['repeats']} "
        "repeats, on the official test endpoints. An open-ended interval is [lower, ∞): its "
        f"upper end reached the {cap}-cycle cap, so it has no finite width. Read instead as "
        f"closed at the cap, [lower, {cap}], the same intervals would contain the actual, "
        f"uncapped RUL of only {_span(closed_reading, '.0f')}% of test engines (a statistic "
        "defined before the test labels were read), so every coverage figure in these notes "
        "depends on the open-ended reading.\n\n"
        f"- **Truncated width** counts an open-ended interval as {cap} minus its lower end. It "
        "measures both models on every engine, but it summarises the truncated interval, not "
        "the prediction set. It was defined after the test labels were read, with a paired "
        "engine-level bootstrap interval for the difference.\n"
        "- **Width with both intervals closed** averages upper minus lower over the engines on "
        "which neither model's interval is open-ended, per repeat. It compares the models on "
        "the same engines, but only on engines whose predicted upper end stays below the cap. It "
        "was also defined after the test labels were read.\n"
        "- **Width over each model's own closed intervals** (`closed_width` in `metrics.json`) "
        "was defined in the evaluated code before the test labels were read. It averages over "
        "a different set of engines for each model.\n\n" + verdict
    )


def guarantee_block(artifacts: Path) -> str:
    """The CV+ construction as implemented, the finite-sample guarantee it would carry, and
    which of that guarantee's conditions the official test sets do not meet."""
    from predmaint.gallery import RUL_LABELS

    runs = _runs(artifacts)
    if not runs:
        return "_No evaluated runs yet._"
    first = next(iter(runs.values()))["config"]
    cap, folds, alpha = first["max_rul"], first["folds"], first["alpha"]
    lines = [
        "| Subset | Calibration engines n | Order statistics, lower and upper end "
        f"| Worst-case CV+ coverage | Calibration points with {cap} or more cycles left "
        f"| Test engines with {cap} or more cycles left "
        "| Correlation of observed length with total life, calibration / test |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    p90: dict[str, dict[str, list[float]]] = {m: {"cap": [], "below": []} for m in NAMES}
    for subset, m in runs.items():
        n = m["data"]["crossfit_engines"]
        points = pd.read_csv(artifacts / subset / "predictions" / "calibration-points.csv")
        test = pd.read_csv(artifacts / subset / "predictions" / "test.csv").drop_duplicates("unit")
        lstm = points[points["model"] == "lstm"]
        r_cal = [
            float(np.corrcoef(g["cycle"], g["cycle"] + g["rul"])[0, 1])
            for _, g in lstm.groupby("repeat")
        ]
        r_test = float(np.corrcoef(test["length"], test["length"] + test["rul_true"])[0, 1])
        for model in NAMES:
            own = points[points["model"] == model]
            p90[model]["cap"].append(float(np.percentile(own["score"][own["rul"] >= cap], 90)))
            p90[model]["below"].append(float(np.percentile(own["score"][own["rul"] < cap], 90)))
        lo_rank, hi_rank = _ranks(n, alpha)
        lines.append(
            f"| {subset} | {n} | {lo_rank}, {hi_rank} "
            f"| {100 * cv_plus_bound(n, folds, alpha):.1f}% "
            f"| {100 * (points['rul'] >= cap).mean():.0f}% "
            f"| {100 * (test['rul_true'] >= cap).mean():.0f}% "
            f"| {_span(r_cal, '.2f')} / {r_test:.2f} |"
        )
    by_rul = coverage_by_rul({s: artifacts / s for s in runs})
    below = by_rul[by_rul["bin"] != RUL_LABELS[-1]]
    below_cap = {
        str(model): 100 * float((g["coverage"] * g["engines"]).sum() / g["engines"].sum())
        for model, g in below.groupby("model", sort=False)
    }
    lower_scores = all(max(p90[m]["cap"]) < min(p90[m]["below"]) for m in NAMES if p90[m]["cap"])
    a = chr(0x3B1)  # the Greek letter alpha, spelled out so the source stays ASCII-safe
    return (
        "**Procedure.** Each cross-fitted engine i contributes one conformity score, at one "
        "truncation point: the conformalized-quantile-regression score "
        "`s_i = max(lo(x_i) − y_i, y_i − hi(x_i))`, where `lo` and `hi` are the 5th and 95th "
        "percentile outputs of the fold model trained without engine i's fold and `y_i` is its "
        f"RUL capped at {cap}. For a new window x, the lower end is the `⌊{a}(n + 1)⌋`-th "
        "smallest of `lo_k(i)(x) − s_i` over the n calibration engines and the upper end the "
        f"`⌈(1 − {a})(n + 1)⌉`-th smallest of `hi_k(i)(x) + s_i`, where `k(i)` is the fold model "
        f"that scored engine i and {a} = {alpha:g}. An upper end at or above the cap is reported "
        "as open-ended, [lower, ∞). "
        "`predmaint.audit` rebuilds every test interval from the stored scores and raw quantile "
        "outputs; that verifies the implementation, not a coverage guarantee.\n\n"
        "**The guarantee.** If the n calibration pairs (engine, truncation point) and the test "
        "pair are exchangeable, training treats its engines symmetrically and the fold "
        f"partition is fixed independently of the data, {folds}-fold CV+ guarantees marginal "
        f"coverage of at least 1 − 2{a} − min{{2(1 − 1/K)/(n/K + 1), (1 − K/n)/(K + 1)}} "
        f"(Barber et al., 2021, Theorem 4). At {a} = {alpha:g} and K = {folds} that is below the "
        f"nominal {100 * (1 - alpha):.0f}% even when the conditions hold; the "
        f"{100 * (1 - 2 * alpha):.0f}% often quoted is the jackknife+ bound, K = n."
        "\n\n" + "\n".join(lines) + "\n\n"
        "**Conditions this setup does not meet.**\n\n"
        "- *Exchangeability.* Calibration truncation points are drawn uniformly from the test "
        "trajectory lengths shorter than each engine's life. In NASA's test sets the observed "
        "length tracks total life, and in calibration it barely does (last column), so "
        f"calibration over-represents engines with {cap} or more cycles left. The last two "
        "columns use the test labels and were computed after they were read.\n"
        "- *Symmetric training.* The feature scaling and operating-regime clustering are fitted "
        "once on the sensor readings of all n cross-fitted engines, so fold model k depends on "
        "the covariates of the engines it scores, but not on a test engine's. Training is also "
        "symmetric only up to fixed seeds and engine order.\n"
        "- *Fixed folds.* Folds are formed by sorting engines by life and dealing each block "
        "of five at random, so the partition depends on the training data.\n"
        "- *Marginal coverage only.* Even when it holds, the guarantee averages over engines; "
        "it implies nothing about coverage at a given remaining life.\n\n"
        "**Consequence.** These are CV+ intervals on CQR scores with nominal "
        f"{100 * (1 - alpha):.0f}% coverage. No finite-sample guarantee is claimed for the test "
        "sets: the evidence is their empirical coverage, reported with the open-ended share and "
        "by remaining life. The calibration mismatch does not explain the over-coverage"
        + (
            ": in every subset the 90th percentile of the conformity scores is lower at the cap "
            "than below it, so over-representing engines at the cap tends to narrow the "
            "intervals, the opposite direction"
            if lower_scores
            else ""
        )
        + f"; and on test engines with fewer than {cap} cycles left, coverage is still "
        + " and ".join(f"{v:.1f}% ({NAMES[m]})" for m, v in below_cap.items())
        + ". The cause of the over-coverage has not been established."
    )


def attribution_block(artifacts: Path) -> str:
    """The attribution checks on the served model, in words, from attribution-checks.json."""
    path = artifacts / "FD001" / "attribution-checks.json"
    if not path.is_file():
        return "_Attribution checks have not been run: `python -m predmaint.attribution_checks`._"
    r = json.loads(path.read_text())
    s, c, n = r["summary"], r["comparisons"], r["curve_windows"]
    windows = pd.DataFrame(r["per_window"])
    far_rul = max(r["stages_cycles_left"])
    far = windows[windows["rul"] == far_rul]
    above_one = windows["deletion_auc_ig"] > 1

    def med(key: str, spec: str = ".2f") -> str:
        return format(s[key]["median"], spec)

    return "\n".join(
        [
            "**Scope.** Integrated gradients (IG) of the served FD001 ensemble's point "
            "prediction, relative to a reference window in which every scaled reading equals its "
            f"operating regime's training mean, with {r['integration_steps_served']} integration "
            f"steps. {r['windows']} windows: the {r['engines']} FD001 replay engines at "
            f"{_and([str(v) for v in r['stages_cycles_left']])} cycles before failure. One "
            "trained ensemble from one seed; each statistic is a median or a count over windows, "
            "with no confidence interval, and there is no ground truth for which readings "
            "matter.",
            "",
            "- **Completeness, a numerical check.** IG satisfies completeness by construction, so "
            "the gap between the attributions' sum and the prediction's change from the reference "
            "tests only the path-integral approximation: median "
            f"{med('completeness_gap_cycles')} cycles against a median change of "
            f"{abs(s['effect_cycles']['median']):.0f} cycles, with a worst window of "
            f"{windows['completeness_gap_cycles'].abs().max():.2f} cycles "
            f"({windows['completeness_gap_256_steps'].abs().max():.2f} with 256 steps).",
            "- **Deletion and insertion, perturbation sensitivity.** Readings are replaced by "
            "their reference values in order of attribution magnitude, and the prediction's "
            "normalised departure from the reference is averaged over the replacement fractions "
            "(lower is better for deletion, higher for insertion). In the median window IG beats "
            f"random order (deletion area {med('deletion_auc_ig', '.3f')} against "
            f"{med('deletion_auc_random', '.3f')}; better in {c['ig_beats_random_on_deletion']} "
            f"of {n} windows) and a re-initialised network's IG "
            f"({med('deletion_auc_randomised_ig', '.3f')}; better in "
            f"{c['ig_beats_randomised_ig_on_deletion']} of {n}).",
            f"- **The tail.** The medians hide poor windows: {int(above_one.sum())} of {n} have a "
            "deletion area above 1, where partial replacement moves the prediction further from "
            "the reference than the full window does, and the mean deletion area is "
            f"{windows['deletion_auc_ig'].mean():.3f} against "
            f"{windows['deletion_auc_random'].mean():.3f} for random order. At {far_rul} cycles "
            f"before failure, where the median change is {far['effect_cycles'].abs().median():.1f} "
            "cycles, IG loses to random order in "
            f"{int((far['deletion_auc_ig'] > far['deletion_auc_random']).sum())} of {len(far)} "
            "windows.",
            "- **Input deviation, a negative comparison.** Ranking readings by their distance "
            "from the reference alone gives a median deletion area of "
            f"{med('deletion_auc_input_deviation', '.3f')} and insertion area of "
            f"{med('insertion_auc_input_deviation', '.3f')}, against IG's "
            f"{med('deletion_auc_ig', '.3f')} and {med('insertion_auc_ig', '.3f')}. IG beats it "
            f"in {c['ig_beats_input_deviation_on_deletion']} of {n} windows on deletion and "
            f"{c['ig_beats_input_deviation_on_insertion']} of {n} on insertion, and the two maps "
            f"have a median rank correlation of {med('spearman_vs_input_deviation')}.",
            "- **Weight randomisation.** Re-initialising every weight, or only the output heads, "
            "gives maps with median rank correlations of "
            f"{med('spearman_vs_full_randomisation')} and {med('spearman_vs_head_randomisation')} "
            "to the trained maps.",
            "- **Reference sensitivity.** With each engine's own mean over its first cycles as "
            "the reference instead of the regime mean, the maps change: median rank correlation "
            f"{med('spearman_vs_own_baseline')}, and "
            f"{s['top3_sensors_kept_with_own_baseline']['median']:.0f} of the top 3 sensors kept.",
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
    fleet = s.get("fleet") or []
    counts = {
        status: sum(e["status"] == status for e in fleet)
        for status in ("alert", "watch", "healthy", "failed")
    }
    words = {"alert": "in alert", "watch": "on watch", "healthy": "healthy", "failed": "failed"}
    mix = _and(
        [f"{n} {'is' if n == 1 else 'are'} {words[status]}" for status, n in counts.items() if n]
    )
    return (
        f"Both captures show one replay tick of the {s['subset']} fleet: the tick at which engine "
        f"{s['unit']}'s alert first fires, at cycle {s['cycle']} of {s['total_cycles']} (its "
        "row is highlighted; the hero figure shows its whole life)"
        + (f". Of the {len(fleet)} replayed engines, {mix}" if fleet else "")
        + f". Engine {s['unit']} is predicted at {s['point']:.0f} cycles (nominal "
        f"{100 * s['interval']:.0f}% interval {interval}) with a {s['probability']:.0%} "
        f"probability of failure within {s['horizon_cycles']} cycles; its true RUL, known "
        f"because the simulated engine ran to failure, is {s['rul_true']}. Every row is the "
        "evaluated prediction in `replay-holdout.csv.gz` for that engine and cycle, which "
        "`predmaint.audit` checks."
    )


BLOCKS = {
    "headline": lambda artifacts: headline_block(artifacts),
    "findings": lambda artifacts: findings_block(artifacts),
    "summary-table": lambda artifacts: summary_table_block(artifacts),
    "results": lambda artifacts: results_block(artifacts),
    "summary": lambda artifacts: summary_block(artifacts),
    "widths": lambda artifacts: widths_block(artifacts),
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
