"""Render the README figures from trained artifacts, in a light and a dark variant.

    python -m predmaint.figures                     # artifacts/ -> docs/figures/
    python -m predmaint.figures --subsets FD001 FD003

Every number in every figure is read from ``artifacts/<subset>/``: nothing is typed in by hand.
Colours follow the validated reference palette (categorical slots 1-3, the blue/red diverging
pair with a grey midpoint, and the fixed status colours).
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)  # quiet font fallback
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from predmaint.config import PROJECT_ROOT, Settings  # noqa: E402
from predmaint.data import SENSOR_INFO  # noqa: E402
from predmaint.inference import Predictor  # noqa: E402

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "text2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a"],
        "truth": "#52514e",
        "div": ["#e34948", "#f0efec", "#2a78d6"],
        "zone": "#ec835a",
        "serious": "#ec835a",
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#f0efec",
        "text2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70"],
        "truth": "#c3c2b7",
        "div": ["#e66767", "#383835", "#3987e5"],
        "zone": "#ec835a",
        "serious": "#ec835a",
    },
}
DPI = 200
MODEL_LABEL = {"lstm": "LSTM ensemble", "lightgbm": "LightGBM"}
MODEL_SLOT = {"lstm": 0, "lightgbm": 2}


def _style(t: dict) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": t["surface"],
            "axes.facecolor": t["surface"],
            "savefig.facecolor": t["surface"],
            "font.family": ["Segoe UI", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "text.color": t["text"],
            "axes.labelcolor": t["text2"],
            "axes.edgecolor": t["axis"],
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": t["grid"],
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "xtick.color": t["muted"],
            "ytick.color": t["muted"],
            "xtick.labelcolor": t["muted"],
            "ytick.labelcolor": t["muted"],
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 10,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlecolor": t["text"],
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "legend.labelcolor": t["text2"],
            "lines.solid_capstyle": "round",
        }
    )


def _title(fig, t: dict, title: str, subtitle: str) -> None:
    height = fig.get_figheight()
    fig.text(
        0.012,
        1 - 0.08 / height,
        title,
        ha="left",
        va="top",
        fontsize=12.5,
        weight="semibold",
        color=t["text"],
    )
    fig.text(0.012, 1 - 0.34 / height, subtitle, ha="left", va="top", fontsize=9, color=t["text2"])


def _save(fig, out_dir: Path, name: str, theme: str) -> Path:
    path = out_dir / f"{name}-{theme}.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def _load(artifacts: Path, subset: str) -> dict:
    model_dir = artifacts / subset
    metrics = json.loads((model_dir / "metrics.json").read_text())
    return {
        "metrics": metrics,
        "served_repeat": metrics["served"]["repeat"],
        "test": pd.read_csv(model_dir / "predictions" / "test.csv"),
        "oof": pd.read_csv(model_dir / "predictions" / "oof.csv.gz"),
        "dir": model_dir,
    }


def _served(frame: pd.DataFrame, run: dict, model: str = "lstm") -> pd.DataFrame:
    return frame[(frame["model"] == model) & (frame["repeat"] == run["served_repeat"])]


def _intervals(ax, x, lower, upper, open_ended, top: float, t: dict, width: float = 1.6) -> None:
    """Closed intervals as solid bars; open-ended ones ("at least lower") run off the top."""
    closed = ~open_ended
    ax.vlines(x[closed], lower[closed], upper[closed], color=t["series"][0], alpha=0.3, lw=width)
    ax.vlines(x[open_ended], lower[open_ended], top, color=t["series"][0], alpha=0.13, lw=width)
    ax.scatter(
        x[open_ended],
        np.full(open_ended.sum(), top - 1.5),
        marker="^",
        s=7,
        linewidths=0,
        color=t["series"][0],
        alpha=0.5,
    )


def fig_test_sets(runs: dict, t: dict, out_dir: Path, theme: str) -> Path:
    subsets = list(runs)
    cols = 2 if len(subsets) > 1 else 1
    rows = int(np.ceil(len(subsets) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 3.1 * rows + 0.9), squeeze=False)
    top = 140.0
    for ax, subset in zip(axes.flat, subsets, strict=False):
        run = runs[subset]
        cap = run["metrics"]["config"]["max_rul"]
        test = _served(run["test"], run).copy()
        test["capped"] = np.minimum(test["rul_true"], cap)
        # Ties at the cap keep engine order, so the plateau shows the predictions' real scatter.
        test = test.sort_values(["capped", "unit"]).reset_index(drop=True)
        x = np.arange(len(test))
        _intervals(
            ax,
            x,
            test["lower"].to_numpy(),
            test["upper"].to_numpy(),
            test["open_ended"].to_numpy(bool),
            top,
            t,
        )
        ax.plot(
            x, test["capped"], color=t["truth"], linewidth=1.6, label=f"True RUL (capped at {cap})"
        )
        ax.scatter(
            x,
            test["point"],
            s=9,
            color=t["series"][0],
            linewidths=0,
            zorder=3,
            label="Predicted RUL",
        )
        m = run["metrics"]["test"]["models"]["lstm"]["per_repeat"][str(run["served_repeat"])]
        ax.set_title(
            f"{subset}  ·  RMSE {m['rmse']:.1f}  ·  {m['coverage']:.0%} inside the nominal 90%",
            pad=6,
        )
        ax.set_xlim(-2, len(test) + 1)
        ax.set_ylim(0, top)
        above = int((test["rul_true"] > cap).sum())
        ax.set_xlabel(f"{len(test)} test engines, sorted by capped true RUL ({above} above {cap})")
        ax.set_ylabel("Cycles")
        ax.grid(axis="x", visible=False)
    for ax in list(axes.flat)[len(subsets) :]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    handles += [
        plt.Line2D([], [], color=t["series"][0], alpha=0.35, linewidth=4),
        plt.Line2D([], [], color=t["series"][0], alpha=0.15, linewidth=4),
    ]
    labels += ["Nominal 90% interval", "Open-ended: at least the lower end"]
    fig.legend(handles, labels, loc="upper right", ncols=2, bbox_to_anchor=(0.99, 1.0), fontsize=8)
    _title(
        fig,
        t,
        "Official test sets: predicted vs true remaining useful life",
        "One prediction per engine at its last recorded cycle, from the served networks (repeat "
        f"{next(iter(runs.values()))['served_repeat']}).",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return _save(fig, out_dir, "test-sets", theme)


def _first_alert(group: pd.DataFrame, threshold: float):
    fired = group[(group["probability"] >= threshold) & (group["rul"] >= 1)]
    return fired.iloc[0] if len(fired) else None


def fig_lives(run: dict, t: dict, out_dir: Path, theme: str, n: int = 4) -> Path:
    """Whole lives of a spread of training engines, each predicted out of fold."""
    oof = _served(run["oof"], run)
    cfg = run["metrics"]["config"]
    threshold, horizon, cap = cfg["alert_threshold"], cfg["alert_cycles"], cfg["max_rul"]
    lengths = oof.groupby("unit")["cycle"].max().sort_values()
    picks = lengths.index[np.linspace(0, len(lengths) - 1, n).round().astype(int)]
    fig, axes = plt.subplots(1, n, figsize=(11, 3.6), sharey=True)
    for ax, unit in zip(axes, picks, strict=True):
        g = oof[oof["unit"] == unit]
        ax.axhspan(0, horizon, color=t["zone"], alpha=0.09, linewidth=0)
        upper = np.where(g["open_ended"], 130, g["upper"])
        ax.fill_between(
            g["cycle"],
            g["lower"],
            upper,
            color=t["series"][0],
            alpha=0.16,
            lw=0,
            label="Nominal 90% interval (open-ended at the cap)",
        )
        ax.plot(
            g["cycle"],
            np.minimum(g["rul"], cap),
            color=t["truth"],
            lw=1.6,
            label="True RUL (capped)",
        )
        ax.plot(g["cycle"], g["point"], color=t["series"][0], lw=1.6, label="Predicted RUL")
        alert = _first_alert(g, threshold)
        if alert is not None:
            ax.axvline(alert["cycle"], color=t["serious"], linewidth=1)
            ax.annotate(
                f"alert: {int(alert['rul'])} cycles\nbefore failure",
                (alert["cycle"], 4),
                xytext=(-4, 0),
                textcoords="offset points",
                ha="right",
                va="bottom",
                fontsize=8,
                color=t["text2"],
                bbox={"boxstyle": "round,pad=0.25", "fc": t["surface"], "ec": "none", "alpha": 0.9},
            )
        ax.set_title(f"Engine {unit}  ·  {int(g['cycle'].max())} cycles", pad=6)
        ax.set_xlim(1, g["cycle"].max())
        ax.set_ylim(0, 130)
        ax.set_xlabel("Cycle")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Remaining useful life (cycles)")
    axes[0].text(
        2,
        horizon + 1.5,
        f"alert horizon · {horizon} cycles",
        fontsize=7.5,
        color=t["muted"],
        va="bottom",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    # Below the panels: the title runs most of the way across the top.
    fig.legend(handles, labels, loc="lower center", ncols=3, bbox_to_anchor=(0.5, 0.0), fontsize=8)
    lead = run["metrics"]["out_of_fold"]["lstm"]["per_repeat"][str(run["served_repeat"])]["alert"]
    _title(
        fig,
        t,
        f"{run['metrics']['subset']}: whole lives, each predicted by a network that never saw "
        "the engine",
        f"Out-of-fold predictions for {lead['units']} engines, served repeat: median first alert "
        f"{lead['lead_median']:.0f} cycles before failure, {lead['missed']} missed, "
        f"{lead['early']} first alerts more than {horizon} cycles early.",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.85))
    return _save(fig, out_dir, "lives", theme)


def fig_attribution(run: dict, t: dict, out_dir: Path, theme: str) -> Path:
    predictor = Predictor(run["dir"])
    contract = predictor.contract
    replay = pd.read_csv(run["dir"] / "replay.csv")
    lengths = replay.groupby("unit")["cycle"].max()
    unit = int(lengths.index[np.argsort(lengths.to_numpy())[len(lengths) // 2]])
    engine = replay[replay["unit"] == unit].sort_values("cycle").reset_index(drop=True)
    cut = int(np.flatnonzero(engine["rul"].to_numpy() <= 25)[0])
    history = engine.iloc[: cut + 1]
    window = contract.window_from_readings(history)
    attributions = predictor.explain(window)
    prediction = predictor.predict_readings(history, explain=False)
    symbols = [SENSOR_INFO[s][0] for s in contract.sensors]
    cycles = history["cycle"].to_numpy()[-contract.window :]
    if len(cycles) < contract.window:  # padded window: label the padded rows with the first cycle
        cycles = np.r_[np.full(contract.window - len(cycles), cycles[0]), cycles]

    fig = plt.figure(figsize=(11, 4.2))
    grid = fig.add_gridspec(1, 2, width_ratios=[2.1, 1], wspace=0.28)
    ax_heat = fig.add_subplot(grid[0])
    ax_bar = fig.add_subplot(grid[1])
    extent = max(0.25, float(np.abs(attributions).max()))
    cmap = LinearSegmentedColormap.from_list("diverging", t["div"])
    image = ax_heat.imshow(
        attributions.T,
        aspect="auto",
        cmap=cmap,
        norm=TwoSlopeNorm(0, -extent, extent),
        interpolation="nearest",
    )
    ax_heat.set_yticks(range(len(symbols)), symbols)
    ticks = np.arange(0, len(cycles), 5)
    ax_heat.set_xticks(ticks, cycles[ticks])
    ax_heat.set_xlabel("Cycle")
    ax_heat.grid(False)
    for spine in ax_heat.spines.values():
        spine.set_visible(False)
    ax_heat.set_xticks(np.arange(-0.5, len(cycles), 1), minor=True)
    ax_heat.set_yticks(np.arange(-0.5, len(symbols), 1), minor=True)
    ax_heat.grid(which="minor", color=t["surface"], linewidth=1.2)
    ax_heat.tick_params(which="minor", length=0)
    windows = sorted({int(m.window) for m in predictor.model.members})  # type: ignore[arg-type]
    ax_heat.set_title(
        f"Contribution of each reading to the ensemble's prediction (networks read the last "
        f"{', '.join(map(str, windows))} cycles)",
        pad=6,
        fontsize=9,
    )
    bar = fig.colorbar(
        image, ax=ax_heat, orientation="horizontal", fraction=0.05, pad=0.2, aspect=40
    )
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=t["muted"], labelsize=7.5)
    bar.set_label(
        "cycles of RUL  (red pulls the prediction towards failure, blue pushes it away)",
        color=t["text2"],
        fontsize=8,
    )

    totals = attributions.sum(axis=0)
    order = np.argsort(-np.abs(totals))[:10][::-1]
    colours = [t["div"][0] if v < 0 else t["div"][2] for v in totals[order]]
    ax_bar.barh(range(len(order)), totals[order], color=colours, height=0.55)
    ax_bar.set_yticks(range(len(order)), [symbols[i] for i in order])
    ax_bar.axvline(0, color=t["axis"], linewidth=0.8)
    span = float(np.abs(totals[order]).max()) * 1.35
    ax_bar.set_xlim(-span, span)
    for y, v in enumerate(totals[order]):
        ax_bar.text(
            v + (span * 0.03 if v >= 0 else -span * 0.03),
            y,
            f"{v:+.1f}",
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=8,
            color=t["text2"],
        )
    ax_bar.grid(axis="y", visible=False)
    ax_bar.set_xlabel("Net effect on predicted RUL (cycles)")
    ax_bar.set_title("Net effect by sensor (top 10)", pad=6)
    rul = prediction["rul"]
    _title(
        fig,
        t,
        f"Why the model predicts {rul['point']:.0f} cycles for engine {unit} at cycle {cycles[-1]}",
        "Integrated gradients against an average engine in the same operating regime: what the "
        f"model's output depends on, not physical cause. True RUL {int(history['rul'].iloc[-1])}; "
        f"nominal 90% interval {rul['summary']}. Engine {unit} was held out of all training.",
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.8, bottom=0.12)
    return _save(fig, out_dir, "attribution", theme)


def fig_alerts(runs: dict, focus: dict, t: dict, out_dir: Path, theme: str) -> Path:
    fig, (ax_rel, ax_lead) = plt.subplots(
        1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1, 1.25]}
    )
    ax_rel.plot([0, 1], [0, 1], color=t["axis"], linewidth=1)
    repeat = str(focus["served_repeat"])
    horizon = focus["metrics"]["config"]["alert_cycles"]
    for model in ("lstm", "lightgbm"):
        block = focus["metrics"]["out_of_fold"].get(model)
        if not block:
            continue
        r = block["per_repeat"][repeat]
        curve = pd.DataFrame(r["reliability"])
        colour = t["series"][MODEL_SLOT[model]]
        cal = r["calibration"]
        ax_rel.plot(
            curve["mean_probability"],
            curve["observed_rate"],
            color=colour,
            linewidth=1.6,
            marker="o",
            markersize=4.5,
            markeredgecolor=t["surface"],
            markeredgewidth=1,
            label=f"{MODEL_LABEL[model]}: ECE {cal['ece']:.3f} "
            f"({cal['ci_low']:.3f}–{cal['ci_high']:.3f})",
        )
        if model == "lstm":
            for _, b in curve.iterrows():
                point = (b["mean_probability"], b["observed_rate"])
                if b["observed_rate"] < 0.08:
                    # Both curves leave the origin steeply, so that bin's count goes in the
                    # empty space above, with a leader line to its point.
                    ax_rel.annotate(
                        f"{int(b['count']):,}",
                        point,
                        xytext=(0.03, 0.2),
                        textcoords="data",
                        fontsize=6.5,
                        color=t["muted"],
                        arrowprops={
                            "arrowstyle": "-",
                            "color": t["muted"],
                            "linewidth": 0.6,
                            "shrinkA": 1,
                            "shrinkB": 3,
                        },
                    )
                else:
                    ax_rel.annotate(
                        f"{int(b['count']):,}",
                        point,
                        xytext=(5, -9),
                        textcoords="offset points",
                        fontsize=6.5,
                        color=t["muted"],
                    )
    ax_rel.set_xlim(0, 1)
    ax_rel.set_ylim(0, 1)
    ax_rel.set_xlabel(f"Predicted probability of failure within {horizon} cycles")
    ax_rel.set_ylabel("Observed frequency")
    engines = focus["metrics"]["out_of_fold"]["lstm"]["per_repeat"][repeat]["engines"]
    ax_rel.set_title(
        f"{focus['metrics']['subset']}: every cycle of {engines} engines, out of fold", pad=6
    )
    ax_rel.legend(loc="upper left", fontsize=7.5)
    ax_rel.text(
        0.97,
        0.04,
        "numbers: cycles per bin (LSTM)",
        ha="right",
        fontsize=7,
        color=t["muted"],
        transform=ax_rel.transAxes,
    )

    labels = []
    for i, (subset, run) in enumerate(runs.items()):
        alert = run["metrics"]["out_of_fold"]["lstm"]["per_repeat"][str(run["served_repeat"])]
        per_unit = run["metrics"]["out_of_fold"]["lstm"]["per_repeat"][str(run["served_repeat"])][
            "alert_per_unit"
        ]
        leads = np.array([u["lead_time"] for u in per_unit if u["lead_time"] is not None], float)
        jitter = np.random.default_rng(i).uniform(-0.18, 0.18, len(leads))
        ax_lead.scatter(
            leads,
            np.full(len(leads), i) + jitter,
            s=9,
            color=t["series"][0],
            alpha=0.6,
            linewidths=0,
        )
        if len(leads):
            ax_lead.plot([np.median(leads)] * 2, [i - 0.3, i + 0.3], color=t["text"], linewidth=1.6)
        a = alert["alert"]
        labels.append(f"{subset}\n{a['units']} engines, {a['missed']} missed")
    ax_lead.axvline(horizon, color=t["serious"], linewidth=1)
    ax_lead.set_yticks(range(len(runs)), labels)
    ax_lead.set_ylim(len(runs) - 0.5, -0.6)
    ax_lead.set_xlim(left=0)
    ax_lead.set_xlabel("Cycles left when the alert first fired")
    ax_lead.grid(axis="y", visible=False)
    ax_lead.set_title("First alert on every cross-fitted engine", pad=6)
    ax_lead.legend(
        [
            plt.Line2D([], [], color=t["series"][0], marker="o", linestyle="", markersize=4),
            plt.Line2D([], [], color=t["text"], linewidth=1.6),
            plt.Line2D([], [], color=t["serious"], linewidth=1),
        ],
        ["one engine", "median", f"{horizon}-cycle horizon"],
        loc="upper right",
        fontsize=7.5,
    )
    _title(
        fig,
        t,
        "How the alert behaves, out of fold",
        f"Served repeat ({repeat}) of each subset. Left: reliability, with the ECE's 95% interval "
        "from resampling whole engines.\nRight: when the LSTM's alert first fires; dots right of "
        "the line fired before failure entered the horizon.",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.81))
    return _save(fig, out_dir, "alerts", theme)


def fig_results(runs: dict, t: dict, out_dir: Path, theme: str) -> Path:
    """This project's two models under one protocol. Published results use other protocols, so
    they are compared in a table, not on this axis."""
    from predmaint.report import width_stats

    subsets = list(runs)
    fig, (ax_err, ax_cov, ax_width) = plt.subplots(
        1, 3, figsize=(11, 1.6 + 0.6 * len(subsets)), sharey=True
    )
    y = np.arange(len(subsets))
    widths = {s: width_stats(runs[s]["dir"]) for s in subsets}
    marker = {"markersize": 6, "markeredgecolor": t["surface"], "elinewidth": 1.4, "capsize": 0}
    for offset, model in ((-0.14, "lstm"), (0.14, "lightgbm")):
        colour = t["series"][MODEL_SLOT[model]]
        summaries = [
            runs[s]["metrics"]["test"]["models"].get(model, {}).get("summary") for s in subsets
        ]
        if not all(summaries):
            continue
        ax_err.errorbar(
            [s["rmse"]["mean"] for s in summaries],
            y + offset,
            xerr=[s["rmse"]["sd"] or 0 for s in summaries],
            fmt="o",
            color=colour,
            label=MODEL_LABEL[model],
            **marker,
        )
        means = np.array([s["coverage"]["mean"] for s in summaries])
        bounds = np.array([s["coverage_wilson_at_mean"] for s in summaries])
        ax_cov.errorbar(
            100 * means,
            y + offset,
            xerr=[
                100 * np.maximum(means - bounds[:, 0], 0),
                100 * np.maximum(bounds[:, 1] - means, 0),
            ],
            fmt="o",
            color=colour,
            **marker,
        )
        ax_width.errorbar(
            [widths[s][model]["mean"] for s in subsets],
            y + offset,
            xerr=[widths[s][model]["sd"] or 0 for s in subsets],
            fmt="o",
            color=colour,
            **marker,
        )
    ax_cov.axvline(90, color=t["muted"], linewidth=1, linestyle=(0, (3, 3)))
    ax_cov.text(
        90.4, len(subsets) - 0.55, "nominal 90%", fontsize=7.5, color=t["muted"], va="bottom"
    )
    ax_err.set_yticks(y, subsets)
    ax_err.set_ylim(len(subsets) - 0.5, -0.6)
    ax_err.set_xlabel("Test RMSE, cycles\n(capped truth; mean ± sd over repeats)")
    ax_cov.set_xlabel("Test engines inside the interval, %\n(95% Wilson interval)")
    ax_width.set_xlabel("Mean interval width, cycles\n(open-ended counted to the cap)")
    for ax in (ax_err, ax_cov, ax_width):
        ax.grid(axis="y", visible=False)
    ax_err.set_title("Point accuracy", pad=6)
    ax_cov.set_title("Interval coverage", pad=6)
    ax_width.set_title("Interval width (narrower is sharper)", pad=6)
    fig.legend(
        *ax_err.get_legend_handles_labels(), loc="upper right", ncols=2, bbox_to_anchor=(0.99, 0.99)
    )
    _title(
        fig,
        t,
        "The network against a gradient-boosting baseline, under one protocol",
        "Same cross-fitted folds, window choice, calibration and test engines for both models.",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.8))
    return _save(fig, out_dir, "results", theme)


def hero_engine(replay: pd.DataFrame, threshold: float, cap: int) -> int:
    """The replay engine shown in the hero figure, picked by a fixed rule: among engines whose
    interval misses the truth at least once, the one with the median first-alert lead time."""
    rows = []
    for _, g in replay.groupby("unit"):
        truth = np.minimum(g["rul"].to_numpy(), cap)
        inside = (truth >= g["lower"].to_numpy()) & (
            g["open_ended"].to_numpy(bool) | (truth <= g["upper"].to_numpy())
        )
        fired = g[(g["probability"] >= threshold) & (g["rul"] >= 1)]
        lead = int(fired["rul"].to_numpy()[0]) if len(fired) else -1
        rows.append((int(g["unit"].to_numpy()[0]), int((~inside).sum()), lead))
    table = pd.DataFrame(rows, columns=["unit", "misses", "lead"])
    pool = table[table["misses"] > 0] if (table["misses"] > 0).any() else table
    pool = pool.sort_values(["lead", "unit"]).reset_index(drop=True)
    return int(pool["unit"].to_numpy()[len(pool) // 2])


def fig_hero(runs: dict, focus: dict, t: dict, out_dir: Path, theme: str) -> Path:
    cfg = focus["metrics"]["config"]
    cap, horizon, threshold = cfg["max_rul"], cfg["alert_cycles"], cfg["alert_threshold"]
    replay = _served(pd.read_csv(focus["dir"] / "predictions" / "replay-holdout.csv.gz"), focus)
    unit = hero_engine(replay, threshold, cap)
    g = replay[replay["unit"] == unit].sort_values("cycle")
    cycle = g["cycle"].to_numpy()
    truth = np.minimum(g["rul"].to_numpy(), cap)
    open_ended = g["open_ended"].to_numpy(bool)
    inside = (truth >= g["lower"].to_numpy()) & (open_ended | (truth <= g["upper"].to_numpy()))

    fig = plt.figure(figsize=(11, 4.1))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.9, 1], hspace=0.75, wspace=0.22)
    ax = fig.add_subplot(grid[:, 0])
    ax_err = fig.add_subplot(grid[0, 1])
    ax_cov = fig.add_subplot(grid[1, 1], sharey=ax_err)

    ax.axhspan(0, horizon, color=t["zone"], alpha=0.09, linewidth=0)
    top = cap + 10
    upper = np.where(open_ended, top, g["upper"].to_numpy())
    ax.fill_between(
        cycle,
        g["lower"],
        upper,
        color=t["series"][0],
        alpha=0.16,
        lw=0,
        label="Nominal 90% interval",
    )
    ax.plot(cycle, truth, color=t["truth"], lw=1.6, label=f"True RUL (capped at {cap})")
    ax.plot(cycle, g["point"], color=t["series"][0], lw=1.8, label="Predicted RUL")
    if (~inside).any():
        ax.scatter(
            cycle[~inside],
            truth[~inside],
            s=10,
            color=t["serious"],
            zorder=4,
            label=f"Truth outside the interval ({int((~inside).sum())} cycles)",
        )
    fired = g[(g["probability"] >= threshold) & (g["rul"] >= 1)]
    if len(fired):
        first_cycle = int(fired["cycle"].to_numpy()[0])
        first_left = int(fired["rul"].to_numpy()[0])
        ax.axvline(first_cycle, color=t["serious"], lw=1)
        ax.annotate(
            f"first alert: {first_left} cycles\nbefore failure",
            (first_cycle, 4),
            xytext=(-5, 0),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=8,
            color=t["text2"],
            bbox={"boxstyle": "round,pad=0.25", "fc": t["surface"], "ec": "none", "alpha": 0.9},
        )
    ax.text(
        cycle[0] + 2,
        top - 3,
        "open-ended: at least the lower end",
        fontsize=7.5,
        color=t["muted"],
        va="top",
    )
    ax.set_xlim(1, cycle.max())
    ax.set_ylim(0, top)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Remaining useful life (cycles)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.25), fontsize=7.5)
    ax.set_title(f"{focus['metrics']['subset']} engine {unit}, held out of all training", pad=6)

    subsets = list(runs)
    y = np.arange(len(subsets))
    for offset, model in ((-0.15, "lstm"), (0.15, "lightgbm")):
        summaries = [
            runs[s]["metrics"]["test"]["models"].get(model, {}).get("summary") for s in subsets
        ]
        if not all(summaries):
            continue
        colour = t["series"][MODEL_SLOT[model]]
        ax_err.errorbar(
            [s["rmse"]["mean"] for s in summaries],
            y + offset,
            xerr=[s["rmse"]["sd"] or 0 for s in summaries],
            fmt="o",
            color=colour,
            markersize=5,
            markeredgecolor=t["surface"],
            elinewidth=1.2,
            capsize=0,
            label=MODEL_LABEL[model],
        )
        ax_cov.plot(
            [100 * s["coverage"]["mean"] for s in summaries],
            y + offset,
            "o",
            color=colour,
            markersize=5,
            markeredgecolor=t["surface"],
        )
    ax_cov.axvline(90, color=t["muted"], lw=1, linestyle=(0, (3, 3)))
    ax_err.set_yticks(y, subsets)
    ax_err.set_ylim(len(subsets) - 0.5, -0.6)
    ax_err.set_title("Test RMSE (cycles)", pad=4, fontsize=9)
    ax_cov.set_title("Test coverage, nominal 90% (%)", pad=4, fontsize=9)
    for a in (ax_err, ax_cov):
        a.grid(axis="y", visible=False)
        a.tick_params(labelsize=8)
    ax_err.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncols=2, fontsize=7)
    _title(
        fig,
        t,
        "How many cycles are left, how sure is the model, and when should it alert?",
        "Left: one engine the model never saw, cycle by cycle. Right: the network and the "
        "gradient-boosting baseline on all four test sets, mean over repeats.",
    )
    fig.subplots_adjust(left=0.06, right=0.98, top=0.8, bottom=0.12)
    return _save(fig, out_dir, "hero", theme)


def fig_attribution_checks(focus: dict, t: dict, out_dir: Path, theme: str) -> Path | None:
    path = focus["dir"] / "attribution-checks.json"
    if not path.is_file():
        return None
    report = json.loads(path.read_text())
    curves, s = report["curves"], report["summary"]
    x = 100 * np.asarray(curves["fractions"])
    fig, (ax_del, ax_ins, ax_rho) = plt.subplots(
        1, 3, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1, 1, 1.25]}
    )
    styles = (
        ("ig", "integrated gradients", t["series"][0], "-", 1.8),
        ("input_deviation", "|input - baseline| alone", t["series"][1], "-", 1.3),
        ("randomised_ig", "integrated gradients, randomised weights", t["series"][2], "-", 1.3),
        ("random", "random order (mean of 20)", t["muted"], (0, (4, 3)), 1.3),
    )
    for ax, key, title in (
        (ax_del, "deletion", "Deletion: highest-ranked readings removed first"),
        (ax_ins, "insertion", "Insertion: highest-ranked readings added first"),
    ):
        for name, label, colour, style, width in styles:
            if f"{key}_{name}" in curves:
                ax.plot(
                    x, curves[f"{key}_{name}"], color=colour, lw=width, linestyle=style, label=label
                )
        ax.set_xlabel("Readings replaced (%)")
        ax.set_title(title, pad=6, fontsize=9)
        ax.set_ylim(-0.2, 1.2)
        ax.axhline(0, color=t["axis"], lw=0.8)
        ax.axhline(1, color=t["axis"], lw=0.8)
    ax_del.set_ylabel("Share of the prediction's\ndeparture from the baseline (median)")
    fig.legend(
        *ax_del.get_legend_handles_labels(),
        loc="lower left",
        ncols=4,
        bbox_to_anchor=(0.06, 0.0),
        fontsize=7.5,
    )
    labels = [
        ("spearman_vs_full_randomisation", "vs randomised weights"),
        ("spearman_vs_head_randomisation", "vs randomised output heads"),
        ("spearman_vs_own_baseline", "vs the engine's first cycles as baseline"),
        ("spearman_vs_input_deviation", "vs |input - baseline| alone"),
    ]
    values = [s[k]["median"] for k, _ in labels]
    lows = [s[k]["median"] - s[k]["p10"] for k, _ in labels]
    highs = [s[k]["p90"] - s[k]["median"] for k, _ in labels]
    yy = np.arange(len(labels))
    ax_rho.barh(
        yy,
        values,
        color=t["series"][0],
        height=0.5,
        xerr=[lows, highs],
        error_kw={"ecolor": t["text2"], "elinewidth": 1},
    )
    ax_rho.set_yticks(yy, [label for _, label in labels], fontsize=8)
    ax_rho.set_xlim(-0.2, 1)
    ax_rho.axvline(0, color=t["axis"], lw=0.8)
    ax_rho.set_ylim(len(labels) - 0.5, -0.5)
    ax_rho.set_xlabel("Rank correlation of attribution maps\n(median, 10th to 90th percentile)")
    ax_rho.set_title("Similarity to the served model's maps", pad=6, fontsize=9)
    ax_rho.grid(axis="y", visible=False)
    _title(
        fig,
        t,
        "Sanity checks on the attributions",
        f"{report['windows']} windows from {report['engines']} held-out engines at "
        f"{', '.join(map(str, report['stages_cycles_left']))} cycles before failure. "
        f"Completeness gap: median {s['completeness_gap_cycles']['median']:.2f} cycles.",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.84))
    return _save(fig, out_dir, "attribution-checks", theme)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m predmaint.figures",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--artifacts", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "docs" / "figures")
    parser.add_argument("--subsets", nargs="+", default=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument(
        "--focus", default="FD001", help="subset used for the single-engine figures"
    )
    args = parser.parse_args(argv)
    artifacts = args.artifacts or Settings.from_env().artifacts_dir
    runs = {
        s: _load(artifacts, s)
        for s in args.subsets
        if (artifacts / s / "metrics.json").is_file()
        and json.loads((artifacts / s / "metrics.json").read_text()).get("served")
    }
    if not runs:
        parser.error(f"No evaluated subsets found in {artifacts}")
    args.out.mkdir(parents=True, exist_ok=True)
    focus = runs.get(args.focus) or next(iter(runs.values()))
    written = []
    for theme, t in THEMES.items():
        _style(t)
        written += [
            fig_hero(runs, focus, t, args.out, theme),
            fig_test_sets(runs, t, args.out, theme),
            fig_lives(focus, t, args.out, theme),
            fig_alerts(runs, focus, t, args.out, theme),
            fig_results(runs, t, args.out, theme),
        ]
        if (focus["dir"] / "model.pt").is_file():
            written.append(fig_attribution(focus, t, args.out, theme))
        if checks := fig_attribution_checks(focus, t, args.out, theme):
            written.append(checks)
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
