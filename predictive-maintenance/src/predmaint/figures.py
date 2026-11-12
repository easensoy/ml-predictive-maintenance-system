"""Render the figures from trained artifacts, in a light and a dark variant.

    python -m predmaint.figures hero --draft        # preview one figure: light, 110 dpi, .cache/
    python -m predmaint.figures                     # release: every figure, both themes, 240 dpi
    python -m predmaint.figures --changed           # release, skipping figures that are current
    python -m predmaint.figures --out /tmp/figures  # rebuild every figure elsewhere (CI)
    python -m predmaint.figures --list              # the figure names

Every number in every figure is read or computed from ``artifacts/<subset>/``: nothing is typed
in by hand. Release figures are 8 inches wide at 240 dpi, so that at the 850-1000 pixels of a
GitHub README no text is smaller than about 11 pixels. Drafts go to ``.cache/figure-drafts/``,
never to ``docs/figures``.

One visual system runs through every figure:

- **Identity.** The LSTM is blue with round markers and solid lines; LightGBM is green with
  diamond markers and dashed lines, so neither depends on colour alone (the pair passes a
  colour-vision-deficiency check against both backgrounds). Grey is the truth: solid for the
  capped target, dashed for the actual remaining life. Orange is the alert and its 30-cycle
  horizon. Red marks a failure: the truth outside the interval, or an attribution that ranks
  worse than chance. The red/blue diverging pair with a grey midpoint is for attributions.
- **Hierarchy.** A short title names what is shown; the finding and how to read the figure are
  in the Markdown caption beside it. Panels carry small titles, series are labelled directly
  where that removes a legend, and statistics are annotated compactly.

Each rendering run loads every artifact once and shares it between the figures and both
themes. The attribution map behind ``attribution`` is computed once and cached in ``.cache/``
under a key that covers the model, its inputs and the source code, so redrawing the layout does
not load PyTorch. ``--changed`` skips a figure only when its key (inputs, source code, theme,
resolution and library versions) matches the last render and the image on disk is the one that
render wrote.

The dashboard screenshots are not drawn here: ``python -m predmaint.screenshots`` captures them.
"""

import argparse
import functools
import hashlib
import importlib.metadata
import json
import logging
import sys
import textwrap
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import matplotlib

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)  # quiet font fallback
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from predmaint.calibration import covered  # noqa: E402
from predmaint.config import PROJECT_ROOT, SUBSETS, Settings  # noqa: E402
from predmaint.data import SENSOR_INFO  # noqa: E402
from predmaint.gallery import RUL_LABELS  # noqa: E402
from predmaint.inference import Predictor  # noqa: E402
from predmaint.report import (  # noqa: E402
    closed_widths,
    coverage_by_rul,
    first_alert,
    hero_state,
    pooled_coverage,
    single_models,
    width_stats,
)

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "text2": "#52514e",
        "muted": "#898781",
        "grid": "#ecebe6",
        "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#11865d"],
        "truth": "#52514e",
        "div": ["#e34948", "#f0efec", "#2a78d6"],
        "fail": "#d03b3b",
        "zone": "#ec835a",
        "alert": "#e0601f",
        "band": 0.18,
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#f0efec",
        "text2": "#c3c2b7",
        "muted": "#8f8d87",
        "grid": "#2a2a28",
        "axis": "#4a4a46",
        "series": ["#3987e5", "#d95926", "#199e70"],
        "truth": "#c3c2b7",
        "div": ["#e66767", "#383835", "#3987e5"],
        "fail": "#ef6b6b",
        "zone": "#ec835a",
        "alert": "#f08a5d",
        "band": 0.28,
    },
}
WIDTH = 8.0  # inches
RELEASE_DPI = 240
DRAFT_DPI = 110
NOTE_CHARS = 150  # characters per note line at 7.5 pt across the 8-inch width
CACHE = PROJECT_ROOT / ".cache"
DRAFTS = CACHE / "figure-drafts"
MODELS = ("lstm", "lightgbm")
MODEL_LABEL = {"lstm": "LSTM", "lightgbm": "LightGBM"}
MODEL_SLOT = {"lstm": 0, "lightgbm": 2}
MODEL_MARKER = {"lstm": "o", "lightgbm": "D"}
MODEL_SIZE = {"lstm": 30, "lightgbm": 22}  # equal visual weight for a circle and a diamond
MODEL_DASH: dict[str, Any] = {"lstm": "-", "lightgbm": (0, (4, 2))}
OFFSET = {"lstm": -0.17, "lightgbm": 0.17}  # LSTM row above LightGBM row in every subset


def _style(t: dict) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": t["surface"],
            "axes.facecolor": t["surface"],
            "savefig.facecolor": t["surface"],
            "font.family": ["Segoe UI", "Arial", "DejaVu Sans"],
            "font.size": 8,
            "text.color": t["text"],
            "axes.labelcolor": t["text2"],
            "axes.labelsize": 8,
            "axes.edgecolor": t["axis"],
            "axes.linewidth": 0.7,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": t["grid"],
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "xtick.color": t["axis"],
            "ytick.color": t["axis"],
            "xtick.labelcolor": t["text2"],
            "ytick.labelcolor": t["text2"],
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 8.5,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlecolor": t["text"],
            "axes.titlepad": 5,
            "legend.frameon": False,
            "legend.fontsize": 7.5,
            "legend.labelcolor": t["text2"],
            "legend.handlelength": 1.6,
            "legend.columnspacing": 1.4,
            "lines.solid_capstyle": "round",
        }
    )


def _header(fig, t: dict, title: str, note: str = "") -> float:
    """A short title naming what is shown, and at most one line of context, in the top margin;
    returns the figure fraction just below them. Findings and reading notes belong in the
    caption next to the figure, not here."""
    h = fig.get_figheight()
    fig.text(0.012, 1 - 0.07 / h, title, ha="left", va="top", fontsize=10, weight="semibold")
    if not note:
        return 1 - 0.34 / h
    note = textwrap.fill(" ".join(note.split()), NOTE_CHARS)
    fig.text(
        0.012,
        1 - 0.29 / h,
        note,
        ha="left",
        va="top",
        fontsize=7.5,
        color=t["text2"],
        linespacing=1.3,
    )
    return 1 - (0.36 + 0.14 * (note.count("\n") + 1)) / h


def _span_text(values: list[float], spec: str, unit: str = "") -> str:
    low, high = format(min(values), spec), format(max(values), spec)
    return f"{low}{unit}" if low == high else f"{low}–{high}{unit}"


def _identity_handles(t: dict) -> list:
    return [_model_handle(t, m) for m in MODELS]


def _model_handle(t: dict, model: str, line: bool = False) -> Line2D:
    colour = t["series"][MODEL_SLOT[model]]
    return Line2D(
        [],
        [],
        color=colour,
        lw=1.6 if line else 0,
        linestyle=MODEL_DASH[model] if line else "",
        marker=MODEL_MARKER[model],
        markersize=5.2 if model == "lstm" else 4.4,
        markerfacecolor=colour,
        markeredgecolor=colour,
    )


def _legend(fig, handles: list, labels: list[str], top: float, ncols: int) -> float:
    """A legend row (or rows) under the header; returns the figure fraction below it."""
    rows = int(np.ceil(len(handles) / ncols))
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.004, top),
        ncols=ncols,
        handletextpad=0.5,
        borderaxespad=0,
        borderpad=0,
    )
    return top - (0.18 * rows + 0.1) / fig.get_figheight()


def _save(fig, out_dir: Path, name: str, theme: str) -> Path:
    path = out_dir / f"{name}-{theme}.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------------------------
# Loading: each artifact file is parsed once per rendering run, on first use


class Run:
    """One subset's artifacts. ``run["test"]`` and the like parse the file on first use and keep
    it, so the figures and both themes share one copy."""

    FILES: ClassVar[dict[str, str]] = {
        "test": "predictions/test.csv",
        "members": "predictions/test-members.csv",
        "oof": "predictions/oof.csv.gz",
        "replay": "predictions/replay-holdout.csv.gz",
    }

    def __init__(self, model_dir: Path):
        self.dir = model_dir
        self.subset = model_dir.name
        self._cache: dict[str, Any] = {}

    def __getitem__(self, key: str) -> Any:
        if key == "dir":
            return self.dir
        if key not in self._cache:
            if key == "metrics":
                self._cache[key] = json.loads((self.dir / "metrics.json").read_text())
            elif key == "served_repeat":
                self._cache[key] = self["metrics"]["served"]["repeat"]
            elif key in self.FILES:
                self._cache[key] = pd.read_csv(self.dir / self.FILES[key])
            else:
                raise KeyError(key)
        return self._cache[key]


class Context:
    """What the figures of one rendering run read: the runs, and every derived table, computed
    once and shared by all figures and both themes."""

    def __init__(self, artifacts: Path, subsets: list[str], focus: str, cache: Path = CACHE):
        self.artifacts, self.cache = artifacts, cache
        self.runs = {
            s: Run(artifacts / s)
            for s in subsets
            if (artifacts / s / "metrics.json").is_file()
            and json.loads((artifacts / s / "metrics.json").read_text()).get("served")
        }
        self.focus = self.runs.get(focus) or next(iter(self.runs.values()), None)
        self._memo: dict[Any, Any] = {}

    def memo(self, key: Any, build: Callable[[], Any]) -> Any:
        if key not in self._memo:
            self._memo[key] = build()
        return self._memo[key]

    @property
    def subsets(self) -> list[str]:
        return list(self.runs)

def _rows(ax, subsets: list[str]) -> None:
    """One row per subset, FD001 on top, with the two models' sub-rows inside it."""
    ax.set_yticks(range(len(subsets)), subsets)
    ax.set_ylim(len(subsets) - 0.5, -0.5)
    ax.grid(axis="y", visible=False)
    for i in range(1, len(subsets)):
        ax.axhline(i - 0.5, color=plt.rcParams["grid.color"], linewidth=0.7, zorder=0)


def _served(frame: pd.DataFrame, run: Run, model: str = "lstm") -> pd.DataFrame:
    return frame[(frame["model"] == model) & (frame["repeat"] == run["served_repeat"])]


# --------------------------------------------------------------------------------------------
# Hero


def fig_hero(focus: dict, t: dict, out_dir: Path, theme: str) -> Path:
    cfg = focus["metrics"]["config"]
    cap, horizon, threshold = cfg["max_rul"], cfg["alert_cycles"], cfg["alert_threshold"]
    state = hero_state(focus["dir"])
    g, unit = state["life"], state["unit"]
    cycle = g["cycle"].to_numpy()
    actual = g["rul"].to_numpy()
    truth = np.minimum(actual, cap)
    lower, upper = g["lower"].to_numpy(), g["upper"].to_numpy()
    open_ended = g["open_ended"].to_numpy(bool)
    prob = g["probability"].to_numpy()
    inside = covered(lower, upper, open_ended, truth)
    below = ~inside & (truth < lower)
    alert = first_alert(g, threshold)
    n_miss = int((~inside).sum())
    top = cap + 38

    if alert is not None:
        title = (
            f"Held-out engine {unit}: alert {int(alert['rul'])} cycles before failure; "
            f"interval missed {n_miss} of {len(cycle)} cycles"
        )
    else:
        title = f"Held-out engine {unit}: no alert; interval missed {n_miss} of {len(cycle)} cycles"
    fig = plt.figure(figsize=(WIDTH, 4.45))
    y = _header(
        fig,
        t,
        title,
        f"{focus['metrics']['subset']} engine never seen in training or calibration, predicted "
        "cycle by cycle by the served LSTM. Picked by a fixed rule, not by eye.",
    )
    handles = [
        Line2D([], [], color=t["series"][0], lw=2),
        Patch(color=t["series"][0], alpha=t["band"], lw=0),
        Line2D([], [], color=t["truth"], lw=1.6),
        Line2D([], [], color=t["truth"], lw=1.2, linestyle=(0, (4, 2.5))),
    ]
    labels = [
        "Predicted RUL",
        "Nominal 90% interval",
        f"True RUL capped at {cap} (the target)",
        "Actual remaining life",
    ]
    y = _legend(fig, handles, labels, y, ncols=len(handles))
    grid = fig.add_gridspec(
        2, 1, height_ratios=[3.4, 1], hspace=0.14, left=0.082, right=0.985, top=y, bottom=0.105
    )
    ax = fig.add_subplot(grid[0])
    axp = fig.add_subplot(grid[1], sharex=ax)

    ax.axhspan(0, horizon, color=t["zone"], alpha=0.09, lw=0, zorder=0)
    ax.fill_between(
        cycle, lower, np.minimum(upper, cap), color=t["series"][0], alpha=t["band"], lw=0
    )
    # Open-ended: the interval has no upper end. A fading band above the cap says "and more".
    for start, stop, alpha in ((0, 11, 0.65 * t["band"]), (11, 22, 0.3 * t["band"])):
        ax.fill_between(
            cycle,
            cap + start,
            cap + stop,
            where=open_ended,
            color=t["series"][0],
            alpha=alpha,
            lw=0,
            interpolate=False,
            step="mid",
        )
    ax.axhline(cap, color=t["muted"], lw=0.8, linestyle=(0, (1, 2.5)), zorder=1)
    ax.plot(cycle, truth, color=t["truth"], lw=1.6, zorder=3)
    above_cap = actual >= cap
    ax.plot(
        cycle[above_cap],
        actual[above_cap],
        color=t["truth"],
        lw=1.2,
        linestyle=(0, (4, 2.5)),
        zorder=3,
    )
    ax.plot(cycle, g["point"], color=t["series"][0], lw=1.9, zorder=4)
    ax.scatter(
        cycle[~inside],
        truth[~inside],
        marker="x",
        s=22,
        linewidths=1.2,
        color=t["text"],
        zorder=5,
    )
    ax.text(
        cycle.max(),
        cap + 2,
        f"{cap}-cycle cap",
        ha="right",
        va="bottom",
        fontsize=7.5,
        color=t["muted"],
    )
    open_cycles = cycle[open_ended]
    if len(open_cycles):
        ax.text(
            open_cycles[0] + 2,
            cap + 13,
            "open-ended: “at least the lower end”",
            fontsize=7.5,
            color=t["text2"],
            va="bottom",
        )
    ax.text(3, horizon - 2, f"last {horizon} cycles", fontsize=7.5, color=t["muted"], va="top")
    lines = []
    if n_miss:
        span = cycle[~inside]
        if below[~inside].all():
            where = "below the interval"
        elif not below[~inside].any():
            where = "above the interval"
        else:
            where = "outside the interval"
        lines.append(
            f"\u00d7 truth {where} on {n_miss} of {len(cycle)} cycles ({span.min()}–{span.max()})"
        )
    else:
        lines.append(f"Truth inside the interval on all {len(cycle)} cycles")
    if alert is not None:
        a_cycle, lead = int(alert["cycle"]), int(alert["rul"])
        after = prob[cycle >= a_cycle] >= threshold
        flaps = int(np.sum(after[:-1] & ~after[1:]))
        lines.append(
            "After the first alert it stays on to failure"
            if flaps == 0
            else f"After the first alert it switches off {flaps} times"
        )
        for a in (ax, axp):
            a.axvline(a_cycle, color=t["alert"], lw=1.3, zorder=6)
        ax.text(
            a_cycle - 2,
            top - 3,
            f"first alert\n{lead} cycles left",
            ha="right",
            va="top",
            fontsize=7.5,
            color=t["alert"],
            linespacing=1.25,
        )
    else:
        lines.append("The alert never fires")
    ax.text(
        4,
        horizon + 5,
        "\n".join(lines),
        fontsize=8,
        color=t["text"],
        va="bottom",
        ha="left",
        linespacing=1.45,
        bbox={"boxstyle": "round,pad=0.45", "fc": t["surface"], "ec": t["grid"], "lw": 0.8},
        zorder=7,
    )
    ax.set_xlim(1, cycle.max())
    ax.set_ylim(0, top)
    ax.set_yticks([0, 25, 50, 75, 100, 125])
    ax.set_ylabel("Remaining useful life\n(cycles)")
    ax.grid(axis="x", visible=False)
    ax.tick_params(labelbottom=False)

    within = actual <= horizon
    axp.fill_between(cycle, 0, 1, where=within, color=t["zone"], alpha=0.12, lw=0, step="mid")
    axp.plot(cycle, prob, color=t["series"][0], lw=1.5)
    axp.fill_between(
        cycle,
        threshold,
        prob,
        where=prob >= threshold,
        color=t["alert"],
        alpha=0.45,
        lw=0,
        interpolate=True,
    )
    axp.axhline(threshold, color=t["muted"], lw=0.8, linestyle=(0, (3, 3)))
    axp.text(
        3,
        threshold + 0.06,
        f"alert threshold {threshold:g}",
        fontsize=7.5,
        color=t["muted"],
        va="bottom",
    )
    axp.set_ylim(0, 1.05)
    axp.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
    axp.set_ylabel(f"P(failure\nwithin {horizon})")
    axp.set_xlabel("Cycle")
    axp.grid(axis="x", visible=False)
    return _save(fig, out_dir, "hero", theme)


# --------------------------------------------------------------------------------------------
# Accuracy


def _rmse_title(subsets: list[str], means: dict, diffs: dict) -> str:
    lgbm_lower = [s for s in subsets if means[s]["lightgbm"] < means[s]["lstm"]]
    clear = [s for s in subsets if diffs[s]["ci_low"] > 0 or diffs[s]["ci_high"] < 0]
    if 2 * len(lgbm_lower) >= len(subsets):
        head = f"LightGBM's test RMSE is lower on {len(lgbm_lower)} of {len(subsets)} subsets"
    else:
        head = (
            f"The LSTM's test RMSE is lower on {len(subsets) - len(lgbm_lower)} of "
            f"{len(subsets)} subsets"
        )
    if not clear:
        return f"{head}; no 95% interval of the difference excludes 0"
    if len(clear) == 1:
        return f"{head}; only {clear[0]}'s 95% interval excludes 0"
    return f"{head}; the 95% intervals exclude 0 on {', '.join(clear)}"


def fig_comparison(runs: dict, t: dict, out_dir: Path, theme: str) -> Path:
    subsets = list(runs)
    stats = {s: single_models(runs[s]["dir"]) for s in subsets}
    diffs = {s: runs[s]["metrics"]["test"]["lstm_minus_lightgbm"]["rmse"] for s in subsets}
    means = {s: {m: float(np.mean(stats[s][m]["averages"])) for m in MODELS} for s in subsets}
    fig = plt.figure(figsize=(WIDTH, 1.3 + 0.6 * len(subsets)))
    y = _header(fig, t, _rmse_title(subsets, means, diffs))
    neutral = t["text2"]
    handles = [
        *_identity_handles(t),
        Line2D([], [], marker="o", linestyle="", markersize=6, color=neutral),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=5.5,
            markerfacecolor="none",
            markeredgecolor=neutral,
        ),
    ]
    labels = ["LSTM", "LightGBM", "Five-fold ensemble, one per repeat", "Single fold model"]
    y = _legend(fig, handles, labels, y, ncols=len(handles))
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[2.2, 1],
        wspace=0.06,
        left=0.075,
        right=0.985,
        top=y - 0.04,
        bottom=0.58 / fig.get_figheight(),
    )
    ax = fig.add_subplot(grid[0])
    axd = fig.add_subplot(grid[1], sharey=ax)
    rng = np.random.default_rng(0)
    values = [v for s in subsets for m in MODELS for v in stats[s][m]["singles"]]
    low, high = min(values) - 0.35, max(values) + 1.15
    for i, s in enumerate(subsets):
        best = min(means[s], key=lambda m: means[s][m])
        for m in MODELS:
            row = i + OFFSET[m]
            colour = t["series"][MODEL_SLOT[m]]
            singles = np.asarray(stats[s][m]["singles"])
            ax.scatter(
                singles,
                row + rng.uniform(-0.07, 0.07, len(singles)),
                s=17,
                facecolors="none",
                edgecolors=colour,
                linewidths=0.9,
                alpha=0.9,
                zorder=2,
            )
            ax.scatter(
                stats[s][m]["averages"],
                np.full(len(stats[s][m]["averages"]), row),
                s=38,
                color=colour,
                edgecolors=t["surface"],
                linewidths=0.8,
                zorder=3,
            )
            ax.text(
                high - 0.05,
                row,
                f"{MODEL_LABEL[m]} {means[s][m]:.2f}",
                ha="right",
                va="center",
                fontsize=8,
                color=t["text"] if m == best else t["text2"],
                weight="semibold" if m == best else "normal",
            )
    ax.set_xlim(low, high)
    _rows(ax, subsets)
    ax.set_xlabel("Test RMSE (cycles; truth capped at 125). Right: mean of the three ensembles")
    ax.set_title(
        "Lower is better; the lower mean is in bold",
        fontsize=8.5,
        weight="normal",
        color=t["text2"],
    )

    for i, s in enumerate(subsets):
        d = diffs[s]
        significant = d["ci_low"] > 0 or d["ci_high"] < 0
        axd.errorbar(
            d["difference"],
            i,
            xerr=[[d["difference"] - d["ci_low"]], [d["ci_high"] - d["difference"]]],
            fmt="o",
            markersize=6,
            color=t["text2"],
            ecolor=t["text2"],
            elinewidth=1.3,
            capsize=0,
            markerfacecolor=t["text"] if significant else t["surface"],
            markeredgecolor=t["text"],
        )
    axd.axvline(0, color=t["muted"], lw=0.9)
    span = max(abs(diffs[s][k]) for s in subsets for k in ("ci_low", "ci_high")) * 1.15
    axd.set_xlim(-span, span)
    axd.text(
        0.02,
        1.01,
        "← LSTM lower",
        transform=axd.transAxes,
        fontsize=7.5,
        color=t["muted"],
        va="bottom",
    )
    axd.text(
        0.98,
        1.01,
        "LightGBM lower →",
        transform=axd.transAxes,
        fontsize=7.5,
        color=t["muted"],
        va="bottom",
        ha="right",
    )
    axd.tick_params(labelleft=False)
    axd.grid(axis="y", visible=False)
    for i in range(1, len(subsets)):
        axd.axhline(i - 0.5, color=plt.rcParams["grid.color"], linewidth=0.7, zorder=0)
    axd.set_xlabel("LSTM minus LightGBM, 95% paired\nbootstrap (filled: excludes 0)")
    return _save(fig, out_dir, "comparison", theme)


# --------------------------------------------------------------------------------------------
# Uncertainty


def _coverage_title(runs: dict, by_rul: pd.DataFrame, nominal: float) -> str:
    coverage = [
        100 * runs[s]["metrics"]["test"]["models"][m]["summary"]["coverage"]["mean"]
        for s in runs
        for m in MODELS
    ]
    worst = by_rul.iloc[int(np.argmin(by_rul["coverage"].to_numpy()))]
    head = f"Test coverage is {_span_text(coverage, '.0f', '%')} at nominal {nominal:.0f}%"
    if 100 * worst["coverage"] >= nominal:
        return f"{head} on every subset and at every remaining life"
    return (
        f"{head}, but {100 * worst['coverage']:.0f}% for engines {worst['bin']} cycles from failure"
    )


def fig_uncertainty(runs: dict, t: dict, out_dir: Path, theme: str) -> Path:
    subsets = list(runs)
    first = next(iter(runs.values()))["metrics"]
    nominal = 100 * (1 - first["config"]["alpha"])
    cap = first["config"]["max_rul"]
    by_rul = coverage_by_rul({s: runs[s]["dir"] for s in subsets})
    widths = {s: width_stats(runs[s]["dir"]) for s in subsets}
    repeats = len(first["config"]["repeats"])

    fig = plt.figure(figsize=(WIDTH, 5.5))
    y = _header(fig, t, _coverage_title(runs, by_rul, nominal))
    neutral = t["text2"]
    handles = [
        *_identity_handles(t),
        Line2D([], [], marker="o", linestyle="-", color=neutral, markersize=5.5, lw=1.3),
        Line2D(
            [],
            [],
            marker="D",
            linestyle="",
            markerfacecolor="none",
            markeredgecolor=neutral,
            markersize=5,
        ),
    ]
    labels = [
        "LSTM",
        "LightGBM",
        "Test sets, with 95% Wilson interval",
        "Out of fold, whole lives",
    ]
    y = _legend(fig, handles, labels, y, ncols=len(handles))
    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.12, 1],
        width_ratios=[1.3, 1, 0.34],
        hspace=0.62,
        wspace=0.08,
        left=0.075,
        right=0.985,
        top=y - 0.035,
        bottom=0.125,
    )
    ax_cov = fig.add_subplot(grid[0, 0])
    ax_w = fig.add_subplot(grid[0, 1], sharey=ax_cov)
    ax_open = fig.add_subplot(grid[0, 2], sharey=ax_cov)
    ax_rul = fig.add_subplot(grid[1, :])

    for i, s in enumerate(subsets):
        m = runs[s]["metrics"]
        for model in MODELS:
            row = i + OFFSET[model]
            colour = t["series"][MODEL_SLOT[model]]
            summary = m["test"]["models"][model]["summary"]
            mean = 100 * summary["coverage"]["mean"]
            lo, hi = (100 * v for v in summary["coverage_wilson_at_mean"])
            ax_cov.errorbar(
                mean,
                row,
                xerr=[[max(mean - lo, 0)], [max(hi - mean, 0)]],
                fmt="o",
                markersize=5.5,
                color=colour,
                markeredgecolor=t["surface"],
                elinewidth=1.3,
                capsize=0,
                zorder=3,
            )
            oof = 100 * m["out_of_fold"][model]["summary"]["coverage_per_engine_weighted"]["mean"]
            ax_cov.plot(
                oof,
                row,
                marker="D",
                markersize=5,
                markerfacecolor=t["surface"],
                markeredgecolor=colour,
                markeredgewidth=1.3,
                linestyle="",
                zorder=4,
            )
            w = widths[s][model]
            ax_w.errorbar(
                w["mean"],
                row,
                xerr=w["sd"] or 0,
                fmt="o",
                markersize=5.5,
                color=colour,
                markeredgecolor=t["surface"],
                elinewidth=1.3,
                capsize=0,
            )
            ax_open.text(
                0.5,
                row,
                f"{100 * summary['open_ended_share']['mean']:.0f}%",
                ha="center",
                va="center",
                fontsize=8,
                color=t["text"],
                transform=ax_open.get_yaxis_transform(),
            )
    ax_cov.axvline(nominal, color=t["muted"], lw=1, linestyle=(0, (3, 3)), zorder=1)
    _rows(ax_cov, subsets)
    ax_cov.set_xlim(77, 100.5)
    ax_cov.set_xlabel("Inside the interval (%): test engines, or engine-weighted out of fold")
    ax_cov.set_title(f"Empirical coverage per subset (dashed: nominal {nominal:.0f}%)")
    for a in (ax_w, ax_open):
        a.tick_params(labelleft=False)
        a.grid(axis="y", visible=False)
        for i in range(1, len(subsets)):
            a.axhline(i - 0.5, color=plt.rcParams["grid.color"], linewidth=0.7, zorder=0)
    ax_w.set_xlabel(f"Cycles, test engines; open-ended\nintervals truncated at {cap}")
    ax_w.set_title("Mean width")
    ax_open.set_xticks([])
    ax_open.grid(False)
    for spine in ax_open.spines.values():
        spine.set_visible(False)
    ax_open.set_title("Open-ended", fontsize=8.5)

    x = np.arange(len(RUL_LABELS))
    counts = by_rul.drop_duplicates("bin").set_index("bin")["engines"]
    for model, shift in (("lstm", -0.14), ("lightgbm", 0.14)):
        rows = by_rul[by_rul["model"] == model].set_index("bin").reindex(RUL_LABELS)
        colour = t["series"][MODEL_SLOT[model]]
        cov = 100 * rows["coverage"].to_numpy()
        ax_rul.errorbar(
            x + shift,
            cov,
            yerr=[cov - 100 * rows["low"].to_numpy(), 100 * rows["high"].to_numpy() - cov],
            fmt="o",
            markersize=5.5,
            color=colour,
            markeredgecolor=t["surface"],
            elinewidth=1.3,
            capsize=0,
            zorder=3,
        )
        worst = int(np.nanargmin(cov))
        ax_rul.annotate(
            f"{cov[worst]:.0f}%",
            (x[worst] + shift, cov[worst]),
            xytext=(7, -1),
            textcoords="offset points",
            fontsize=7.5,
            color=t["text"],
            va="center",
        )
    ax_rul.axhline(nominal, color=t["muted"], lw=1, linestyle=(0, (3, 3)), zorder=1)
    ax_rul.text(
        len(x) - 0.45,
        nominal + 0.6,
        f"nominal {nominal:.0f}%",
        ha="right",
        va="bottom",
        fontsize=7.5,
        color=t["muted"],
    )
    ax_rul.set_xticks(x, [f"{label}\n{int(counts[label])} engines" for label in RUL_LABELS])
    ax_rul.set_xlim(-0.5, len(x) - 0.5)
    ax_rul.set_ylim(66, 101)
    ax_rul.set_ylabel("Test engines inside (%)")
    ax_rul.set_xlabel("True RUL at the last observed cycle (cycles), all four test sets pooled")
    ax_rul.grid(axis="x", visible=False)
    ax_rul.set_title(f"Coverage by true remaining life (test sets, mean of {repeats} repeats)")
    return _save(fig, out_dir, "uncertainty", theme)


# --------------------------------------------------------------------------------------------
# Alerts


def _alert_title(subsets: list[str], flaps: dict) -> str:
    more = [s for s in subsets if flaps[s]["lightgbm"]["mean"] > flaps[s]["lstm"]["mean"]]
    if len(more) == len(subsets):
        return (
            "LightGBM's alert switches off before failure more often than the LSTM's, "
            "on every subset"
        )
    if not more:
        return "The LSTM's alert switches off before failure at least as often as LightGBM's"
    return (
        f"LightGBM's alert switches off before failure more often on {len(more)} of "
        f"{len(subsets)} subsets"
    )


def fig_alerts(runs: dict, t: dict, out_dir: Path, theme: str) -> Path:
    subsets = list(runs)
    first = next(iter(runs.values()))
    horizon = first["metrics"]["config"]["alert_cycles"]
    repeat = str(first["served_repeat"])
    repeats = len(first["metrics"]["config"]["repeats"])
    clip = 120
    flaps = {
        s: {
            m: runs[s]["metrics"]["out_of_fold"][m]["summary"]["alert.flaps_per_engine"]
            for m in MODELS
        }
        for s in subsets
    }
    medians = [
        runs[s]["metrics"]["out_of_fold"][m]["per_repeat"][repeat]["alert"]["lead_median"]
        for s in subsets
        for m in MODELS
    ]
    medians = [v for v in medians if v is not None]

    fig = plt.figure(figsize=(WIDTH, 1.25 + 0.66 * len(subsets)))
    y = _header(fig, t, _alert_title(subsets, flaps))
    handles = [
        *_identity_handles(t),
        Line2D([], [], color=t["text"], marker="|", linestyle="", markersize=10, mew=1.6),
        Line2D([], [], color=t["alert"], lw=1.3),
    ]
    labels = ["LSTM", "LightGBM", "Median", f"{horizon}-cycle horizon"]
    y = _legend(fig, handles, labels, y, ncols=len(handles))
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.95, 1.0, 0.62],
        wspace=0.07,
        left=0.075,
        right=0.985,
        top=y - 0.05,
        bottom=0.62 / fig.get_figheight(),
    )
    ax_lead = fig.add_subplot(grid[0])
    ax_flap = fig.add_subplot(grid[1], sharey=ax_lead)
    ax_tab = fig.add_subplot(grid[2], sharey=ax_lead)

    rng = np.random.default_rng(1)
    flap_max = max(f["mean"] + (f["sd"] or 0) for per in flaps.values() for f in per.values())
    for i, s in enumerate(subsets):
        m = runs[s]["metrics"]
        for model in MODELS:
            row = i + OFFSET[model]
            colour = t["series"][MODEL_SLOT[model]]
            alert = m["out_of_fold"][model]["per_repeat"][repeat]
            leads = np.array(
                [u["lead_time"] for u in alert["alert_per_unit"] if u["lead_time"] is not None],
                float,
            )
            shown = np.minimum(leads, clip)
            ax_lead.scatter(
                shown,
                row + rng.uniform(-0.09, 0.09, len(leads)),
                s=7,
                color=colour,
                alpha=0.55,
                linewidths=0,
                zorder=2,
            )
            beyond = leads > clip
            if beyond.any():
                ax_lead.scatter(
                    np.full(beyond.sum(), clip),
                    np.full(beyond.sum(), row),
                    marker=">",
                    s=22,
                    color=colour,
                    zorder=3,
                )
                ax_lead.text(
                    clip - 3,
                    row - 0.13,
                    ", ".join(f"{v:.0f}" for v in sorted(leads[beyond])),
                    ha="right",
                    va="bottom",
                    fontsize=7,
                    color=t["text2"],
                )
            if len(leads):
                ax_lead.plot(
                    [np.median(leads)] * 2,
                    [row - 0.13, row + 0.13],
                    color=t["text"],
                    lw=1.6,
                    zorder=4,
                )
            f = flaps[s][model]
            ax_flap.errorbar(
                f["mean"],
                row,
                xerr=f["sd"] or 0,
                fmt="o",
                markersize=5.5,
                color=colour,
                markeredgecolor=t["surface"],
                elinewidth=1.3,
                capsize=0,
                zorder=3,
            )
            ax_flap.text(
                f["mean"] + (f["sd"] or 0) + 0.025 * flap_max,
                row,
                f"{f['mean']:.2f}",
                va="center",
                fontsize=7.5,
                color=t["text2"],
            )
            a = alert["alert"]
            for col, value in enumerate((a["units"], a["missed"], a["early_twice_horizon"])):
                ax_tab.text(
                    col,
                    row,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=t["text2"] if col == 0 else t["text"],
                )
    _rows(ax_lead, subsets)
    ax_lead.axvline(horizon, color=t["alert"], lw=1.2, zorder=1)
    ax_lead.set_xlim(0, clip + 4)
    ax_lead.set_title(
        f"First alert: median {_span_text(medians, '.0f')} cycles before failure", fontsize=9
    )
    ax_lead.set_xlabel(f"Cycles left at the first alert, one dot per engine (> {clip}: arrow)")
    for a in (ax_flap, ax_tab):
        a.tick_params(labelleft=False)
        a.grid(axis="y", visible=False)
        for i in range(1, len(subsets)):
            a.axhline(i - 0.5, color=plt.rcParams["grid.color"], linewidth=0.7, zorder=0)
    ax_flap.set_xlim(0, flap_max * 1.35)
    ax_flap.set_title("Switch-offs before failure", fontsize=9)
    ax_flap.set_xlabel(f"Per engine (mean ± sd, {repeats} repeats)")
    ax_tab.set_xlim(-0.6, 2.6)
    ax_tab.set_xticks([0, 1, 2], ["n", "missed", f"> {2 * horizon}"])
    ax_tab.xaxis.set_ticks_position("top")
    ax_tab.tick_params(labeltop=True, labelbottom=False)
    ax_tab.grid(False)
    for spine in ax_tab.spines.values():
        spine.set_visible(False)
    ax_tab.set_xlabel(
        f"> {2 * horizon}: first alert with\nmore than {2 * horizon} cycles left", fontsize=7.5
    )
    return _save(fig, out_dir, "alerts", theme)


def fig_alert_calibration(runs: dict, t: dict, out_dir: Path, theme: str) -> Path:
    subsets = list(runs)
    first = next(iter(runs.values()))
    horizon = first["metrics"]["config"]["alert_cycles"]
    repeat = str(first["served_repeat"])
    aps = [
        runs[s]["metrics"]["test"]["models"][m]["summary"]["alert_ap"]["mean"]
        for s in subsets
        for m in MODELS
    ]
    rates = [
        100
        * runs[s]["metrics"]["test"]["models"]["lstm"]["per_repeat"]["0"]["alert_positives"]
        / runs[s]["metrics"]["data"]["test_engines"]
        for s in subsets
    ]

    fig = plt.figure(figsize=(WIDTH, 4.6))
    y = _header(
        fig,
        t,
        f"Alert ranking: average precision {_span_text(aps, '.3f')} against base rates of "
        f"{_span_text(rates, '.0f', '%')}",
        f"Left: test sets, P(failure within {horizon} cycles) at each engine's last observed "
        "cycle. "
        "Right: reliability out of fold over whole lives (served repeat), with cycles per bin; "
        "cycles of one engine are dependent.",
    )
    handles = [
        *_identity_handles(t),
        Line2D([], [], color=t["muted"], marker="|", linestyle="", markersize=11, mew=1.8),
    ]
    y = _legend(fig, handles, ["LSTM", "LightGBM", "Base rate"], y, ncols=3)
    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=[0.55, 3.2],
        wspace=0.24,
        left=0.075,
        right=0.985,
        top=y - 0.05,
        bottom=0.11,
    )
    ax_ap = fig.add_subplot(outer[0])
    lower = outer[1].subgridspec(2, len(subsets), height_ratios=[3, 1.05], hspace=0.12, wspace=0.16)
    for i, s in enumerate(subsets):
        m = runs[s]["metrics"]
        engines = m["data"]["test_engines"]
        positives = m["test"]["models"]["lstm"]["per_repeat"]["0"]["alert_positives"]
        ax_ap.plot(positives / engines, i, marker="|", color=t["muted"], markersize=13, mew=1.8)
        for model in MODELS:
            ap = m["test"]["models"][model]["summary"]["alert_ap"]["mean"]
            ax_ap.plot(
                ap,
                i + OFFSET[model],
                "o",
                color=t["series"][MODEL_SLOT[model]],
                markersize=5.5,
                markeredgecolor=t["surface"],
            )
    _rows(ax_ap, subsets)
    ax_ap.set_xlim(0, 1.04)
    ax_ap.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax_ap.set_title("Average precision", fontsize=9)
    ax_ap.set_xlabel("Test sets")

    for j, s in enumerate(subsets):
        ax_r = fig.add_subplot(lower[0, j])
        ax_c = fig.add_subplot(lower[1, j], sharex=ax_r)
        ax_r.plot([0, 1], [0, 1], color=t["axis"], lw=0.9)
        eces = []
        for model, shift in (("lstm", -0.022), ("lightgbm", 0.022)):
            block = runs[s]["metrics"]["out_of_fold"][model]["per_repeat"][repeat]
            rel = pd.DataFrame(block["reliability"])
            colour = t["series"][MODEL_SLOT[model]]
            ax_r.plot(
                rel["mean_probability"],
                rel["observed_rate"],
                color=colour,
                lw=1.3,
                marker="o",
                markersize=3.5,
                markeredgecolor=t["surface"],
                markeredgewidth=0.6,
            )
            centres = (rel["bin_low"] + rel["bin_high"]) / 2
            ax_c.bar(centres + shift, rel["count"], width=0.04, color=colour, lw=0)
            eces.append(f"{MODEL_LABEL[model]} {block['calibration']['ece']:.3f}")
        ax_r.text(
            0.03,
            0.97,
            "ECE\n" + "\n".join(eces),
            transform=ax_r.transAxes,
            fontsize=7.5,
            va="top",
            color=t["text2"],
            linespacing=1.3,
        )
        ax_r.set_xlim(0, 1)
        ax_r.set_ylim(0, 1)
        ax_r.set_xticks([0, 0.5, 1])
        ax_r.set_yticks([0, 0.5, 1], ["0", "0.5", "1"] if j == 0 else ["", "", ""])
        ax_r.tick_params(labelbottom=False)
        ax_r.set_title(s, fontsize=9)
        ax_c.set_yscale("log")
        ax_c.set_ylim(1, max(10, ax_c.get_ylim()[1]))
        ax_c.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
        ax_c.grid(axis="x", visible=False)
        ax_c.tick_params(axis="y", labelsize=7)
        if j == 0:
            ax_r.set_ylabel("Observed frequency")
            ax_c.set_ylabel("Cycles", fontsize=7.5)
        else:
            ax_c.tick_params(labelleft=False)
        ax_c.set_xlabel("Predicted probability", fontsize=7.5)
    return _save(fig, out_dir, "alert-calibration", theme)


# --------------------------------------------------------------------------------------------
# Attribution


def fig_attribution(run: dict, t: dict, out_dir: Path, theme: str) -> Path:
    predictor = Predictor(run["dir"])
    contract = predictor.contract
    state = hero_state(run["dir"])
    unit, cycle = state["unit"], state["cycle"]
    replay = pd.read_csv(run["dir"] / "replay.csv")
    engine = replay[replay["unit"] == unit].sort_values("cycle")
    history = engine[engine["cycle"] <= cycle]
    window = contract.window_from_readings(history)
    attributions = predictor.explain(window)
    symbols = [SENSOR_INFO[s][0] for s in contract.sensors]
    cycles = history["cycle"].to_numpy()[-contract.window :]
    if len(cycles) < contract.window:  # padded window: the padded rows repeat the first cycle
        cycles = np.r_[np.full(contract.window - len(cycles), cycles[0]), cycles]
    member_windows = sorted(int(m.window) for m in predictor.model.members)  # type: ignore[arg-type]
    shortest = member_windows[0]
    long_members = sum(w > shortest for w in member_windows)

    totals = attributions.sum(axis=0)
    ranked = np.argsort(-np.abs(totals))
    lead = [symbols[i] for i in ranked[:3]]
    signs = np.sign(totals[ranked[:3]])
    verb = (
        "lower the predicted RUL most"
        if (signs < 0).all()
        else "raise the predicted RUL most"
        if (signs > 0).all()
        else "move the predicted RUL most"
    )
    fig = plt.figure(figsize=(WIDTH, 4.05))
    y = _header(
        fig,
        t,
        f"Engine {unit} at its first alert: the {lead[0]}, {lead[1]} and {lead[2]} readings {verb}",
        "Integrated gradients of the served LSTM against a regime-mean baseline, in cycles of "
        "predicted RUL: what the model uses, not a cause.",
    )
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[2.25, 1],
        wspace=0.22,
        left=0.075,
        right=0.985,
        top=y - 0.07,
        bottom=0.12,
    )
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
    ax_heat.set_yticks(range(len(symbols)), symbols, fontsize=7.5)
    ticks = np.arange(len(cycles) - 1, -1, -10)[::-1]
    ax_heat.set_xticks(ticks, cycles[ticks])
    ax_heat.set_xlabel(f"Cycle (the model's {contract.window}-cycle input window)")
    ax_heat.grid(False)
    for spine in ax_heat.spines.values():
        spine.set_visible(False)
    if long_members and shortest < contract.window:
        edge = contract.window - shortest - 0.5
        ax_heat.axvline(edge, color=t["text"], lw=1, linestyle=(0, (2, 2)))
        ax_heat.text(
            edge + 0.6,
            -0.9,
            f"read by all {len(member_windows)} networks →",
            fontsize=7.5,
            color=t["text2"],
            va="bottom",
        )
    bar = fig.colorbar(
        image, ax=ax_heat, orientation="horizontal", fraction=0.045, pad=0.2, aspect=45
    )
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=t["text2"], labelsize=7.5)
    bar.set_label(
        "cycles of predicted RUL: red lowers the prediction, blue raises it",
        color=t["text2"],
        fontsize=8,
    )

    order = ranked[:10][::-1]
    colours = [t["div"][0] if v < 0 else t["div"][2] for v in totals[order]]
    ax_bar.barh(range(len(order)), totals[order], color=colours, height=0.6)
    ax_bar.set_yticks(range(len(order)), [symbols[i] for i in order], fontsize=7.5)
    ax_bar.axvline(0, color=t["axis"], lw=0.8)
    span = float(np.abs(totals[order]).max()) * 1.45
    ax_bar.set_xlim(-span, span)
    for row, v in enumerate(totals[order]):
        ax_bar.text(
            v + (span * 0.03 if v >= 0 else -span * 0.03),
            row,
            f"{v:+.1f}",
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=7.5,
            color=t["text2"],
        )
    ax_bar.grid(axis="y", visible=False)
    ax_bar.set_xlabel("Net effect per sensor (cycles)")
    ax_bar.set_title("Ten largest sensor totals", fontsize=9)
    return _save(fig, out_dir, "attribution", theme)


# --------------------------------------------------------------------------------------------
# Figures for docs/evaluation.md


def fig_test_sets(runs: dict, t: dict, out_dir: Path, theme: str) -> Path:
    subsets = list(runs)
    cols = 2 if len(subsets) > 1 else 1
    rows = int(np.ceil(len(subsets) / cols))
    fig = plt.figure(figsize=(WIDTH, 2.75 * rows + 1.15))
    first = next(iter(runs.values()))
    y = _header(
        fig,
        t,
        "Every official test engine: predicted against true remaining useful life",
        "One prediction per engine at its last recorded cycle, from the served networks (LSTM, "
        f"repeat {first['served_repeat']}). Engines sorted by capped true RUL.",
    )
    handles = [
        Line2D([], [], color=t["truth"], lw=1.6),
        Line2D([], [], color=t["series"][0], marker="o", linestyle="", markersize=4),
        Line2D([], [], color=t["series"][0], alpha=0.4, lw=4),
        Line2D([], [], color=t["series"][0], alpha=0.15, lw=4),
    ]
    cap = first["metrics"]["config"]["max_rul"]
    labels = [
        f"True RUL, capped at {cap}",
        "Predicted RUL",
        "Nominal 90% interval",
        "Open-ended interval (at least the lower end)",
    ]
    y = _legend(fig, handles, labels, y, ncols=4)
    grid = fig.add_gridspec(
        rows, cols, hspace=0.48, wspace=0.12, left=0.07, right=0.985, top=y, bottom=0.08
    )
    top = cap + 15.0
    for k, subset in enumerate(subsets):
        ax = fig.add_subplot(grid[k // cols, k % cols])
        run = runs[subset]
        test = _served(run["test"], run).copy()
        test["capped"] = np.minimum(test["rul_true"], cap)
        test = test.sort_values(["capped", "unit"]).reset_index(drop=True)
        x = np.arange(len(test))
        closed = ~test["open_ended"].to_numpy(bool)
        lo, hi = test["lower"].to_numpy(), test["upper"].to_numpy()
        ax.vlines(x[closed], lo[closed], hi[closed], color=t["series"][0], alpha=0.35, lw=1.3)
        ax.vlines(x[~closed], lo[~closed], top, color=t["series"][0], alpha=0.14, lw=1.3)
        ax.plot(x, test["capped"], color=t["truth"], lw=1.5)
        ax.scatter(x, test["point"], s=6, color=t["series"][0], linewidths=0, zorder=3)
        m = run["metrics"]["test"]["models"]["lstm"]["per_repeat"][str(run["served_repeat"])]
        ax.set_title(
            f"{subset} · RMSE {m['rmse']:.1f} · {m['coverage']:.0%} inside the interval",
            fontsize=9,
        )
        ax.set_xlim(-2, len(test) + 1)
        ax.set_ylim(0, top)
        above = int((test["rul_true"] > cap).sum())
        ax.set_xlabel(f"{len(test)} test engines ({above} with true RUL above {cap})")
        if k % cols == 0:
            ax.set_ylabel("Cycles")
        ax.grid(axis="x", visible=False)
    return _save(fig, out_dir, "test-sets", theme)


def fig_lives(run: dict, t: dict, out_dir: Path, theme: str, n: int = 4) -> Path:
    """Whole lives of a spread of training engines, each predicted out of fold."""
    oof = _served(run["oof"], run)
    cfg = run["metrics"]["config"]
    threshold, horizon, cap = cfg["alert_threshold"], cfg["alert_cycles"], cfg["max_rul"]
    lengths = oof.groupby("unit")["cycle"].max().sort_values()
    picks = lengths.index[np.linspace(0, len(lengths) - 1, n).round().astype(int)]
    fig = plt.figure(figsize=(WIDTH, 5.4))
    lead = run["metrics"]["out_of_fold"]["lstm"]["per_repeat"][str(run["served_repeat"])]["alert"]
    y = _header(
        fig,
        t,
        f"{run['metrics']['subset']}: whole lives, each predicted by a network that never saw the "
        "engine",
        f"The shortest, longest and two evenly spaced lives of {lead['units']} cross-fitted "
        f"engines (served repeat). Over all of them: median first alert {lead['lead_median']:.0f} "
        f"cycles before failure,\n{lead['missed']} missed, {lead['early']} with more than "
        f"{horizon} cycles left.",
    )
    handles = [
        Line2D([], [], color=t["series"][0], lw=1.8),
        Patch(color=t["series"][0], alpha=t["band"], lw=0),
        Line2D([], [], color=t["truth"], lw=1.6),
        Line2D([], [], color=t["alert"], lw=1.3),
    ]
    labels = ["Predicted RUL", "Nominal 90% interval", f"True RUL, capped at {cap}", "First alert"]
    y = _legend(fig, handles, labels, y, ncols=4)
    grid = fig.add_gridspec(
        2, 2, hspace=0.5, wspace=0.1, left=0.07, right=0.985, top=y, bottom=0.09
    )
    for k, unit in enumerate(picks):
        ax = fig.add_subplot(grid[k // 2, k % 2])
        g = oof[oof["unit"] == unit]
        ax.axhspan(0, horizon, color=t["zone"], alpha=0.09, linewidth=0)
        upper = np.where(g["open_ended"], cap + 10, g["upper"])
        ax.fill_between(g["cycle"], g["lower"], upper, color=t["series"][0], alpha=t["band"], lw=0)
        ax.plot(g["cycle"], np.minimum(g["rul"], cap), color=t["truth"], lw=1.5)
        ax.plot(g["cycle"], g["point"], color=t["series"][0], lw=1.6)
        alert = first_alert(g, threshold)
        if alert is not None:
            ax.axvline(alert["cycle"], color=t["alert"], linewidth=1.2)
            ax.text(
                alert["cycle"] - 3,
                4,
                f"alert {int(alert['rul'])} cycles\nbefore failure",
                ha="right",
                va="bottom",
                fontsize=7.5,
                color=t["text2"],
                bbox={"boxstyle": "round,pad=0.2", "fc": t["surface"], "ec": "none", "alpha": 0.85},
            )
        ax.set_title(f"Engine {unit} · {int(g['cycle'].max())} cycles", fontsize=9)
        ax.set_xlim(1, g["cycle"].max())
        ax.set_ylim(0, cap + 10)
        if k % 2 == 0:
            ax.set_ylabel("RUL (cycles)")
        if k >= 2:
            ax.set_xlabel("Cycle")
        ax.grid(axis="x", visible=False)
    return _save(fig, out_dir, "lives", theme)


def fig_attribution_checks(focus: dict, t: dict, out_dir: Path, theme: str) -> Path | None:
    path = focus["dir"] / "attribution-checks.json"
    if not path.is_file():
        return None
    report = json.loads(path.read_text())
    curves, s = report["curves"], report["summary"]
    x = 100 * np.asarray(curves["fractions"])
    fig = plt.figure(figsize=(WIDTH, 5.6))
    y = _header(
        fig,
        t,
        "Sanity checks on the attributions",
        f"{report['windows']} windows from {report['engines']} held-out engines at "
        f"{', '.join(map(str, report['stages_cycles_left']))} cycles before failure, served "
        f"five-network average. Completeness gap: median "
        f"{s['completeness_gap_cycles']['median']:.2f} cycles.",
    )
    styles: tuple[tuple[str, str, str, Any, float], ...] = (
        ("ig", "Integrated gradients", t["series"][0], "-", 1.9),
        ("input_deviation", "|input − baseline| alone", t["series"][1], "-", 1.3),
        ("randomised_ig", "Integrated gradients, randomised weights", t["series"][2], "-", 1.3),
        ("random", "Random order (mean of 20)", t["muted"], (0, (4, 3)), 1.3),
    )
    handles = [Line2D([], [], color=c, lw=w, linestyle=st) for _, _, c, st, w in styles]
    y = _legend(fig, handles, [label for _, label, *_ in styles], y, ncols=2)
    split = y - (y - 0.1) * 0.56
    grid = fig.add_gridspec(1, 2, wspace=0.12, left=0.1, right=0.985, top=y, bottom=split)
    ax_del = fig.add_subplot(grid[0, 0])
    ax_ins = fig.add_subplot(grid[0, 1], sharey=ax_del)
    ax_rho = fig.add_subplot(
        fig.add_gridspec(1, 1, left=0.34, right=0.985, top=split - 0.12, bottom=0.1)[0]
    )
    for ax, key, title in (
        (ax_del, "deletion", "Deletion: highest-ranked readings removed first"),
        (ax_ins, "insertion", "Insertion: highest-ranked readings added first"),
    ):
        for name, _, colour, style, width in styles:
            if f"{key}_{name}" in curves:
                ax.plot(x, curves[f"{key}_{name}"], color=colour, lw=width, linestyle=style)
        ax.set_xlabel("Readings replaced (%)")
        ax.set_title(title, fontsize=9)
        ax.set_ylim(-0.15, 1.15)
        ax.axhline(0, color=t["axis"], lw=0.8)
        ax.axhline(1, color=t["axis"], lw=0.8)
    ax_ins.tick_params(labelleft=False)
    ax_del.set_ylabel("Share of the prediction's\ndeparture from the baseline")
    labels = [
        ("spearman_vs_full_randomisation", "vs randomised weights"),
        ("spearman_vs_head_randomisation", "vs randomised output heads"),
        ("spearman_vs_own_baseline", "vs the engine's first cycles as baseline"),
        ("spearman_vs_input_deviation", "vs |input − baseline| alone"),
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
    ax_rho.set_yticks(yy, [label for _, label in labels])
    ax_rho.set_xlim(-0.2, 1)
    ax_rho.axvline(0, color=t["axis"], lw=0.8)
    ax_rho.set_ylim(len(labels) - 0.5, -0.5)
    ax_rho.set_xlabel(
        "Rank correlation with the served model's maps (median, 10th–90th percentile over windows)"
    )
    ax_rho.set_title("How much the maps change", fontsize=9)
    ax_rho.grid(axis="y", visible=False)
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
            fig_hero(focus, t, args.out, theme),
            fig_comparison(runs, t, args.out, theme),
            fig_uncertainty(runs, t, args.out, theme),
            fig_alerts(runs, t, args.out, theme),
            fig_alert_calibration(runs, t, args.out, theme),
            fig_test_sets(runs, t, args.out, theme),
            fig_lives(focus, t, args.out, theme),
        ]
        if (focus["dir"] / "model.pt").is_file() and (focus["dir"] / "replay.csv").is_file():
            written.append(fig_attribution(focus, t, args.out, theme))
        if checks := fig_attribution_checks(focus, t, args.out, theme):
            written.append(checks)
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
