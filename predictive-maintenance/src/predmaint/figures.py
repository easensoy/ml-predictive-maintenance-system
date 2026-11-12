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
def _rows(ax, subsets: list[str]) -> None:
    """One row per subset, FD001 on top, with the two models' sub-rows inside it."""
    ax.set_yticks(range(len(subsets)), subsets)
    ax.set_ylim(len(subsets) - 0.5, -0.5)
    ax.grid(axis="y", visible=False)
    for i in range(1, len(subsets)):
        ax.axhline(i - 0.5, color=plt.rcParams["grid.color"], linewidth=0.6, zorder=0)


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

    def by_rul(self) -> pd.DataFrame:
        return self.memo(
            "by_rul", lambda: coverage_by_rul({s: r.dir for s, r in self.runs.items()})
        )

    def widths(self, subset: str) -> dict:
        return self.memo(("widths", subset), lambda: width_stats(self.runs[subset].dir))

    def closed(self, subset: str) -> dict:
        return self.memo(("closed", subset), lambda: closed_widths(self.runs[subset].dir))

    def singles(self, subset: str) -> dict:
        return self.memo(("singles", subset), lambda: single_models(self.runs[subset].dir))

    def hero(self) -> dict:
        focus = self.focus
        assert focus is not None
        return self.memo("hero", lambda: hero_state(focus.dir))

    def attribution(self) -> dict | None:
        """The served model's attribution map for the hero engine at its first alert, from the
        cache when model, inputs and code are unchanged; None without the model or replay rows."""
        return self.memo("attribution", lambda: _attribution_map(self))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_digest() -> str:
    """One identifier for all of predmaint's source code: any change invalidates every cache."""
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode() + b"\0" + path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


ATTRIBUTION_INPUTS = (
    "model.pt",
    "contract.json",
    "metrics.json",
    "replay.csv",
    "predictions/replay-holdout.csv.gz",
)


def _attribution_map(ctx: Context) -> dict | None:
    run = ctx.focus
    if run is None or not all((run.dir / name).is_file() for name in ATTRIBUTION_INPUTS):
        return None
    key = hashlib.sha256(
        json.dumps(
            {
                "source": source_digest(),
                "inputs": {name: _sha256(run.dir / name) for name in ATTRIBUTION_INPUTS},
                "torch": importlib.metadata.version("torch"),
                "numpy": np.__version__,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:24]
    path = ctx.cache / "figures" / f"attribution-{run.subset}-{key}.npz"
    if path.is_file():
        stored = np.load(path, allow_pickle=False)
        return {
            "unit": int(stored["unit"]),
            "cycle": int(stored["cycle"]),
            "attributions": stored["attributions"],
            "cycles": stored["cycles"],
            "symbols": [str(s) for s in stored["symbols"]],
            "member_windows": [int(w) for w in stored["member_windows"]],
            "cached": True,
        }
    from predmaint.inference import Predictor  # PyTorch is imported only on a cache miss

    predictor = Predictor(run.dir)
    contract = predictor.contract
    state = ctx.hero()
    unit, cycle = state["unit"], state["cycle"]
    replay = pd.read_csv(run.dir / "replay.csv")
    engine = replay[replay["unit"] == unit].sort_values("cycle")
    history = engine[engine["cycle"] <= cycle]
    attributions = predictor.explain(contract.window_from_readings(history))
    cycles = history["cycle"].to_numpy()[-contract.window :]
    if len(cycles) < contract.window:  # padded window: the padded rows repeat the first cycle
        cycles = np.r_[np.full(contract.window - len(cycles), cycles[0]), cycles]
    result = {
        "unit": int(unit),
        "cycle": int(cycle),
        "attributions": attributions,
        "cycles": cycles,
        "symbols": [SENSOR_INFO[s][0] for s in contract.sensors],
        "member_windows": sorted(int(m.window) for m in predictor.model.members),  # type: ignore[arg-type]
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {k: np.asarray(v) for k, v in result.items()}
    np.savez(path, **arrays)
    return {**result, "cached": False}


def _served(frame: pd.DataFrame, run: Run, model: str = "lstm") -> pd.DataFrame:
    return frame[(frame["model"] == model) & (frame["repeat"] == run["served_repeat"])]


# --------------------------------------------------------------------------------------------
# Hero


def fig_hero(ctx: Context, t: dict) -> Figure:
    focus = ctx.focus
    assert focus is not None
    cfg = focus["metrics"]["config"]
    cap, horizon, threshold = cfg["max_rul"], cfg["alert_cycles"], cfg["alert_threshold"]
    state = ctx.hero()
    g, unit = state["life"], state["unit"]
    cycle = g["cycle"].to_numpy()
    actual = g["rul"].to_numpy()
    truth = np.minimum(actual, cap)
    point = g["point"].to_numpy()
    lower, upper = g["lower"].to_numpy(), g["upper"].to_numpy()
    open_ended = g["open_ended"].to_numpy(bool)
    prob = g["probability"].to_numpy()
    inside = covered(lower, upper, open_ended, truth)
    alert = first_alert(g, threshold)
    top = cap + 36
    blue = t["series"][0]

    fig = plt.figure(figsize=(WIDTH, 4.5))
    y = _header(fig, t, f"Held-out engine {unit} ({focus.subset}), predicted at every cycle")
    handles = [
        Line2D([], [], color=blue, lw=1.8),
        Patch(color=blue, alpha=t["band"], lw=0),
        Line2D([], [], color=t["truth"], lw=1.5),
        Line2D([], [], color=t["truth"], lw=1.1, linestyle=(0, (4, 2.5))),
    ]
    labels = [
        "predicted RUL",
        "nominal 90% interval",
        f"target: true RUL capped at {cap}",
        "actual RUL",
    ]
    y = _legend(fig, handles, labels, y, ncols=4)
    grid = fig.add_gridspec(
        2, 1, height_ratios=[3.3, 1], hspace=0.1, left=0.066, right=0.988, top=y, bottom=0.1
    )
    ax = fig.add_subplot(grid[0])
    axp = fig.add_subplot(grid[1], sharex=ax)

    ax.axhspan(0, horizon, color=t["zone"], alpha=0.07, lw=0, zorder=0)
    ax.fill_between(cycle, lower, np.minimum(upper, cap), color=blue, alpha=t["band"], lw=0)
    # Open-ended: no upper end. The band fades out above the cap instead of stopping at it.
    for start, stop, share in ((0, 8, 0.75), (8, 16, 0.45), (16, 24, 0.2)):
        ax.fill_between(
            cycle,
            cap + start,
            cap + stop,
            where=open_ended,
            color=blue,
            alpha=share * t["band"],
            lw=0,
            step="mid",
        )
    ax.axhline(cap, color=t["muted"], lw=0.7, linestyle=(0, (1, 2.5)), zorder=1)
    ax.plot(cycle, truth, color=t["truth"], lw=1.5, zorder=3)
    above = actual >= cap
    ax.plot(cycle[above], actual[above], color=t["truth"], lw=1.1, linestyle=(0, (4, 2.5)))
    ax.plot(cycle, point, color=blue, lw=1.8, zorder=4)

    small: dict[str, Any] = {"fontsize": 7.5, "zorder": 8}
    if open_ended.any():
        first_open = cycle[open_ended][0]
        ax.text(
            first_open + 2,
            cap + 25,
            "open-ended: at least the lower end",
            color=t["text2"],
            **small,
        )
    if actual.max() > top:
        # Beside the dashed line where it enters the plot: right of where it crosses the label's
        # lower edge, so the text never sits on the line.
        crossing = cycle[np.argmin(np.abs(actual - (top - 11)))]
        ax.text(crossing + 3, top - 2, "actual RUL", color=t["text2"], va="top", **small)
    ax.text(
        cycle[0] + 2,
        horizon - 2,
        f"alert horizon, {horizon} cycles",
        color=t["muted"],
        va="top",
        **small,
    )

    missed = ~inside
    if missed.any():
        ax.scatter(
            cycle[missed],
            truth[missed],
            marker="x",
            s=20,
            linewidths=1.2,
            color=t["fail"],
            zorder=6,
        )
        side = "above" if (truth[missed] > upper[missed]).all() else "outside"
        last = cycle[missed].max()
        ax.annotate(
            f"truth {side} the interval\non {int(missed.sum())} of {len(cycle)} cycles",
            xy=(last + 1, truth[missed][-1]),
            xytext=(last + 9, truth[missed][-1] + 14),
            color=t["fail"],
            fontsize=7.5,
            va="bottom",
            linespacing=1.2,
            arrowprops={
                "arrowstyle": "-",
                "color": t["fail"],
                "lw": 0.8,
                "shrinkA": 0,
                "shrinkB": 2,
            },
            zorder=8,
        )
    if alert is not None:
        a_cycle, lead = int(alert["cycle"]), int(alert["rul"])
        for a in (ax, axp):
            a.axvline(a_cycle, color=t["alert"], lw=1.2, zorder=6)
        ax.text(
            a_cycle - 2,
            top - 3,
            f"first alert\n{lead} cycles left",
            ha="right",
            va="top",
            color=t["alert"],
            linespacing=1.2,
            **small,
        )
    ax.set_xlim(cycle.min(), cycle.max())
    ax.set_ylim(0, top)
    ax.set_yticks([0, 25, 50, 75, 100, 125])
    ax.set_ylabel("RUL (cycles)")
    ax.grid(axis="x", visible=False)
    ax.tick_params(labelbottom=False)

    within = actual <= horizon
    axp.fill_between(cycle, 0, 1, where=within, color=t["zone"], alpha=0.1, lw=0, step="mid")
    axp.plot(cycle, prob, color=blue, lw=1.4)
    axp.fill_between(
        cycle,
        threshold,
        prob,
        where=prob >= threshold,
        color=t["alert"],
        alpha=0.4,
        lw=0,
        interpolate=True,
    )
    axp.axhline(threshold, color=t["muted"], lw=0.7, linestyle=(0, (3, 3)))
    axp.text(
        cycle[0] + 2,
        threshold + 0.07,
        f"alert threshold {threshold:g}",
        color=t["muted"],
        va="bottom",
        **small,
    )
    axp.set_ylim(0, 1.05)
    axp.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
    axp.set_ylabel(f"P(fail ≤ {horizon})")
    axp.set_xlabel("Cycle")
    axp.grid(axis="x", visible=False)
    return fig


# --------------------------------------------------------------------------------------------
# Accuracy


def _difference_rows(ax, t: dict, subsets: list[str], diffs: dict, label: bool = True) -> None:
    """Paired LSTM-minus-LightGBM differences with their 95% intervals, one row per subset; a
    filled marker where the interval excludes zero."""
    for i, s in enumerate(subsets):
        d = diffs[s]
        clear = d["ci_low"] > 0 or d["ci_high"] < 0
        ax.plot([d["ci_low"], d["ci_high"]], [i, i], color=t["text2"], lw=1.6, zorder=2)
        ax.scatter(
            [d["difference"]],
            [i],
            s=46,
            marker="o",
            color=t["text"] if clear else t["surface"],
            edgecolors=t["text"],
            linewidths=1.2,
            zorder=3,
        )
        if label:
            ax.text(
                1.0,
                i,
                f"{d['difference']:+.2f}  [{d['ci_low']:+.2f}, {d['ci_high']:+.2f}]",
                transform=ax.get_yaxis_transform(),
                ha="left",
                va="center",
                fontsize=7.5,
                color=t["text"] if clear else t["text2"],
                weight="semibold" if clear else "normal",
            )
    ax.axvline(0, color=t["muted"], lw=0.8, zorder=1)
    span = max(abs(diffs[s][k]) for s in subsets for k in ("ci_low", "ci_high")) * 1.12
    ax.set_xlim(-span, span)
    ax.text(0.01, 1.02, "← LSTM lower", transform=ax.transAxes, fontsize=7.5, color=t["muted"])
    ax.text(
        0.99,
        1.02,
        "LightGBM lower →",
        transform=ax.transAxes,
        fontsize=7.5,
        color=t["muted"],
        ha="right",
    )


def fig_comparison(ctx: Context, t: dict) -> Figure:
    """The principal comparison: the paired difference in test RMSE with its bootstrap interval,
    next to each model's RMSE (the three repeat ensembles and their mean)."""
    subsets = ctx.subsets
    diffs = {s: ctx.runs[s]["metrics"]["test"]["lstm_minus_lightgbm"]["rmse"] for s in subsets}
    stats = {s: ctx.singles(s) for s in subsets}
    fig = plt.figure(figsize=(WIDTH, 2.9))
    y = _header(fig, t, "Test RMSE on the official test endpoints, LSTM versus LightGBM")
    handles = [
        *_identity_handles(t),
        Line2D([], [], marker="o", linestyle="", markersize=5.5, color=t["text"]),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=5.5,
            markerfacecolor=t["surface"],
            markeredgecolor=t["text"],
        ),
    ]
    labels = ["LSTM", "LightGBM", "95% interval excludes 0", "includes 0"]
    y = _legend(fig, handles, labels, y, ncols=4)
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.25, 1],
        wspace=0.42,
        left=0.07,
        right=0.985,
        top=y - 0.07,
        bottom=0.19,
    )
    axd = fig.add_subplot(grid[0])
    axm = fig.add_subplot(grid[1], sharey=axd)
    _difference_rows(axd, t, subsets, diffs)
    _rows(axd, subsets)
    axd.set_xlabel("LSTM − LightGBM test RMSE (cycles), paired engine bootstrap")

    values = [v for s in subsets for m in MODELS for v in stats[s][m]["averages"]]
    for i, s in enumerate(subsets):
        means = {m: float(np.mean(stats[s][m]["averages"])) for m in MODELS}
        axm.plot([means["lstm"], means["lightgbm"]], [i, i], color=t["axis"], lw=1.2, zorder=1)
        for m in MODELS:
            colour = t["series"][MODEL_SLOT[m]]
            repeats = stats[s][m]["averages"]
            axm.scatter(
                repeats,
                np.full(len(repeats), i + OFFSET[m] * 0.7),
                s=11,
                marker=MODEL_MARKER[m],
                facecolors="none",
                edgecolors=colour,
                linewidths=0.8,
                alpha=0.85,
                zorder=2,
            )
            axm.scatter(
                [means[m]],
                [i],
                s=MODEL_SIZE[m] * 1.5,
                marker=MODEL_MARKER[m],
                color=colour,
                edgecolors=t["surface"],
                linewidths=0.8,
                zorder=3,
            )
    axm.set_xlim(min(values) - 0.35, max(values) + 0.35)
    axm.tick_params(labelleft=False)
    axm.grid(axis="y", visible=False)
    for i in range(1, len(subsets)):
        axm.axhline(i - 0.5, color=plt.rcParams["grid.color"], linewidth=0.6, zorder=0)
    axm.set_xlabel("Test RMSE (cycles): mean of 3 repeats; small: each repeat")
    return fig


def fig_comparison_folds(ctx: Context, t: dict) -> Figure:
    """Supporting detail for the evaluation notes: every single fold model and every repeat's
    five-fold ensemble, so the gain from averaging and the repeat spread are visible."""
    subsets = ctx.subsets
    stats = {s: ctx.singles(s) for s in subsets}
    means = {s: {m: float(np.mean(stats[s][m]["averages"])) for m in MODELS} for s in subsets}
    fig = plt.figure(figsize=(WIDTH, 1.25 + 0.62 * len(subsets)))
    y = _header(fig, t, "Test RMSE of every single fold model and every five-fold ensemble")
    neutral = t["text2"]
    handles = [
        *_identity_handles(t),
        Line2D([], [], marker="o", linestyle="", markersize=5.5, color=neutral),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=5,
            markerfacecolor="none",
            markeredgecolor=neutral,
        ),
    ]
    labels = ["LSTM", "LightGBM", "five-fold ensemble, one per repeat", "single fold model"]
    y = _legend(fig, handles, labels, y, ncols=4)
    ax = fig.add_subplot(
        fig.add_gridspec(
            1, 1, left=0.075, right=0.985, top=y - 0.03, bottom=0.6 / fig.get_figheight()
        )[0]
    )
    rng = np.random.default_rng(0)
    values = [v for s in subsets for m in MODELS for v in stats[s][m]["singles"]]
    low, high = min(values) - 0.35, max(values) + 1.25
    for i, s in enumerate(subsets):
        best = min(means[s], key=lambda m: means[s][m])
        for m in MODELS:
            row = i + OFFSET[m]
            colour = t["series"][MODEL_SLOT[m]]
            singles = np.asarray(stats[s][m]["singles"])
            ax.scatter(
                singles,
                row + rng.uniform(-0.07, 0.07, len(singles)),
                s=14,
                marker=MODEL_MARKER[m],
                facecolors="none",
                edgecolors=colour,
                linewidths=0.8,
                alpha=0.9,
                zorder=2,
            )
            ax.scatter(
                stats[s][m]["averages"],
                np.full(len(stats[s][m]["averages"]), row),
                s=MODEL_SIZE[m] * 1.2,
                marker=MODEL_MARKER[m],
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
                fontsize=7.5,
                color=t["text"] if m == best else t["text2"],
                weight="semibold" if m == best else "normal",
            )
    ax.set_xlim(low, high)
    _rows(ax, subsets)
    cap = next(iter(ctx.runs.values()))["metrics"]["config"]["max_rul"]
    ax.set_xlabel(f"Test RMSE (cycles; truth capped at {cap})")
    return fig


# --------------------------------------------------------------------------------------------
# Uncertainty


def _open_share_by_rul(ctx: Context) -> pd.DataFrame:
    """Share of open-ended test intervals per model and true-RUL band, all subsets pooled, mean
    over repeats (a descriptive summary defined after the test labels were read)."""
    from predmaint.gallery import RUL_BINS

    def build() -> pd.DataFrame:
        test = pd.concat([run["test"] for run in ctx.runs.values()], ignore_index=True)
        test["bin"] = pd.cut(test["rul_true"], RUL_BINS, labels=RUL_LABELS).astype(str)
        share = test.groupby(["model", "bin", "repeat"])["open_ended"].mean()
        return share.groupby(level=["model", "bin"]).mean().rename("share").reset_index()

    return ctx.memo("open_by_rul", build)


def fig_uncertainty(ctx: Context, t: dict) -> Figure:
    subsets = ctx.subsets
    assert ctx.focus is not None
    first = ctx.focus["metrics"]
    nominal = 100 * (1 - first["config"]["alpha"])
    cap = first["config"]["max_rul"]
    by_rul = ctx.by_rul()
    pooled = ctx.memo("pooled", lambda: pooled_coverage({s: r.dir for s, r in ctx.runs.items()}))
    open_share = _open_share_by_rul(ctx)
    repeats = len(first["config"]["repeats"])

    fig = plt.figure(figsize=(WIDTH, 4.7))
    y = _header(
        fig,
        t,
        f"Coverage of the nominal {nominal:.0f}% intervals on the official test endpoints",
    )
    y = _legend(fig, _identity_handles(t), ["LSTM", "LightGBM"], y, ncols=2)
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.75, 1],
        height_ratios=[2.15, 1],
        wspace=0.3,
        hspace=0.62,
        left=0.072,
        right=0.985,
        top=y - 0.05,
        bottom=0.1,
    )
    ax_rul = fig.add_subplot(grid[0, 0])
    ax_open = fig.add_subplot(grid[1, 0], sharex=ax_rul)
    ax_sub = fig.add_subplot(grid[0, 1])
    ax_w = fig.add_subplot(grid[1, 1])

    # Conditional coverage: by true remaining life, with the aggregate as the first column.
    columns = ["all", *RUL_LABELS]
    x = np.arange(len(columns), dtype=float)
    x[1:] += 0.45  # a gap between the aggregate and the bands
    deficit = [
        label
        for label in RUL_LABELS
        if (by_rul[by_rul["bin"] == label]["coverage"] * 100 < nominal).any()
    ]
    for i, label in enumerate(columns):
        if label in deficit:
            ax_rul.axvspan(x[i] - 0.42, x[i] + 0.42, color=t["fail"], alpha=0.07, lw=0, zorder=0)
    ax_rul.axhline(nominal, color=t["muted"], lw=0.8, linestyle=(0, (3, 3)), zorder=1)
    for m in MODELS:
        colour = t["series"][MODEL_SLOT[m]]
        g = by_rul[by_rul["model"] == m].set_index("bin").reindex(RUL_LABELS)  # empty band: NaN
        rates = [100 * pooled[m]["coverage"], *(100 * g["coverage"].to_numpy(float))]
        lows = [100 * pooled[m]["low"], *(100 * g["low"].to_numpy(float))]
        highs = [100 * pooled[m]["high"], *(100 * g["high"].to_numpy(float))]
        xs = x + (-0.13 if m == "lstm" else 0.13)
        ax_rul.vlines(xs, lows, highs, color=colour, lw=1.3, zorder=2)
        ax_rul.scatter(
            xs,
            rates,
            s=MODEL_SIZE[m] * 1.1,
            marker=MODEL_MARKER[m],
            color=colour,
            edgecolors=t["surface"],
            linewidths=0.7,
            zorder=3,
        )
        worst = int(np.nanargmin(rates[1:])) + 1
        if rates[worst] < nominal:
            ax_rul.text(
                xs[worst] + 0.1,
                rates[worst],
                f"{rates[worst]:.0f}%",
                fontsize=7.5,
                color=t["fail"],
                va="center",
                weight="semibold",
            )
    ax_rul.text(
        x[-1] + 0.46,
        nominal + 0.6,
        f"nominal {nominal:.0f}%",
        ha="right",
        va="bottom",
        fontsize=7.5,
        color=t["muted"],
    )
    ax_rul.set_ylim(68, 101)
    ax_rul.set_ylabel("Inside the interval (%)")
    ax_rul.set_title(
        f"By true RUL at the last observed cycle; all four subsets, mean of {repeats} repeats",
        fontsize=8,
        weight="normal",
        color=t["text2"],
    )
    ax_rul.grid(axis="x", visible=False)
    ax_rul.tick_params(labelbottom=False)

    # Open-ended prevalence in the same bands: why coverage far from failure is nearly certain.
    for m in MODELS:
        colour = t["series"][MODEL_SLOT[m]]
        share_by_bin = open_share[open_share["model"] == m].set_index("bin")["share"]
        test = pd.concat([run["test"] for run in ctx.runs.values()])
        overall = float(test[test["model"] == m].groupby("repeat")["open_ended"].mean().mean())
        shares = [100 * overall] + [100 * float(share_by_bin.get(b, 0.0)) for b in RUL_LABELS]
        ax_open.bar(
            x + (-0.13 if m == "lstm" else 0.13),
            shares,
            width=0.24,
            color=colour,
            alpha=0.85 if m == "lstm" else 0.75,
            hatch=None if m == "lstm" else "////",
            edgecolor=t["surface"],
            lw=0.4,
        )
    ax_open.set_ylim(0, 100)
    ax_open.set_yticks([0, 50, 100])
    ax_open.set_ylabel("Open-ended (%)")
    ax_open.set_xticks(x, ["All engines", *[f"{b}" for b in RUL_LABELS]])
    ax_open.set_xlabel("True RUL (cycles)")
    ax_open.grid(axis="x", visible=False)
    counts = by_rul.drop_duplicates("bin").set_index("bin")["engines"]
    for i, label in enumerate(columns[1:], start=1):
        ax_open.text(
            x[i],
            103,
            f"n={counts.get(label, 0)}",
            ha="center",
            va="bottom",
            fontsize=7,
            color=t["muted"],
        )
    ax_open.text(
        x[0],
        103,
        f"n={pooled['lstm']['engines']}",
        ha="center",
        va="bottom",
        fontsize=7,
        color=t["muted"],
    )

    # Aggregate (marginal) coverage per subset, with 95% Wilson intervals over engines.
    for i, s in enumerate(subsets):
        m_all = ctx.runs[s]["metrics"]["test"]["models"]
        for m in MODELS:
            colour = t["series"][MODEL_SLOT[m]]
            summary = m_all[m]["summary"]
            rate = 100 * summary["coverage"]["mean"]
            low, high = (100 * v for v in summary["coverage_wilson_at_mean"])
            row = i + OFFSET[m]
            ax_sub.plot([low, high], [row, row], color=colour, lw=1.3, zorder=2)
            ax_sub.scatter(
                [rate],
                [row],
                s=MODEL_SIZE[m],
                marker=MODEL_MARKER[m],
                color=colour,
                edgecolors=t["surface"],
                linewidths=0.7,
                zorder=3,
            )
    ax_sub.axvline(nominal, color=t["muted"], lw=0.8, linestyle=(0, (3, 3)))
    _rows(ax_sub, subsets)
    ax_sub.set_xlim(82, 100.5)
    ax_sub.set_title(
        "Per subset, 95% Wilson interval", fontsize=8, weight="normal", color=t["text2"]
    )
    ax_sub.set_xlabel("Inside the interval (%)")

    # Width only where it is finite: engines on which both models' intervals are closed.
    for i, s in enumerate(subsets):
        closed = ctx.closed(s)
        for m in MODELS:
            colour = t["series"][MODEL_SLOT[m]]
            row = i + OFFSET[m]
            ax_w.plot([0, closed[m]], [row, row], color=colour, lw=2.2, solid_capstyle="butt")
            ax_w.scatter(
                [closed[m]],
                [row],
                s=MODEL_SIZE[m] * 0.8,
                marker=MODEL_MARKER[m],
                color=colour,
                zorder=3,
            )
    _rows(ax_w, subsets)
    ax_w.set_yticks(
        range(len(subsets)), [f"{s} · n={ctx.closed(s)['engines']:.0f}" for s in subsets]
    )
    ax_w.set_xlim(0, None)
    ax_w.set_title(
        "Width where both intervals are closed", fontsize=8, weight="normal", color=t["text2"]
    )
    ax_w.set_xlabel(f"Mean width (cycles); open-ended excluded, cap {cap}")
    return fig


# --------------------------------------------------------------------------------------------
# Alerts


def fig_alerts(ctx: Context, t: dict) -> Figure:
    runs = ctx.runs
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
    y = _header(fig, t, "Alert timing and stability over whole lives, out of fold")
    handles = [
        *_identity_handles(t),
        Line2D([], [], color=t["text"], marker="|", linestyle="", markersize=10, mew=1.6),
        Line2D([], [], color=t["alert"], lw=1.3),
    ]
    labels = ["LSTM", "LightGBM", "median", f"{horizon}-cycle horizon"]
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
                s=7 if model == "lstm" else 6,
                marker=MODEL_MARKER[model],
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
                fmt=MODEL_MARKER[model],
                markersize=5.5 if model == "lstm" else 4.6,
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
    return fig


def fig_alert_calibration(ctx: Context, t: dict) -> Figure:
    runs = ctx.runs
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
        f"Alert ranking (test endpoints) and calibration (out of fold), P(fail ≤ {horizon})",
        f"Average precision {_span_text(aps, '.3f')} against base rates of "
        f"{_span_text(rates, '.0f', '%')}; reliability for the first repeat, cycles per bin "
        "below each panel",
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
    return fig


# --------------------------------------------------------------------------------------------
# Attribution


def fig_attribution(ctx: Context, t: dict) -> Figure | None:
    attribution = ctx.attribution()
    if attribution is None:
        return None
    unit = attribution["unit"]
    attributions = attribution["attributions"]
    symbols = attribution["symbols"]
    cycles = attribution["cycles"]
    member_windows = attribution["member_windows"]
    shortest = member_windows[0]
    long_members = sum(w > shortest for w in member_windows)

    totals = attributions.sum(axis=0)
    fig = plt.figure(figsize=(WIDTH, 4.0))
    y = _header(
        fig,
        t,
        f"Integrated gradients for engine {unit} at its first alert, in cycles of predicted RUL",
        "Relative to a reference window at the operating regime's training mean: what the "
        "prediction is sensitive to, not a cause",
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
    window = attributions.shape[0]
    ax_heat.set_xlabel(f"Cycle (the model's {window}-cycle input window)")
    ax_heat.grid(False)
    for spine in ax_heat.spines.values():
        spine.set_visible(False)
    if long_members and shortest < window:
        edge = window - shortest - 0.5
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

    order = np.argsort(-np.abs(totals))[:10][::-1]
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
    ax_bar.set_title("Ten largest sensor totals")
    return fig


# --------------------------------------------------------------------------------------------
# Figures for docs/evaluation.md


def fig_test_sets(ctx: Context, t: dict) -> Figure:
    runs = ctx.runs
    subsets = list(runs)
    cols = 2 if len(subsets) > 1 else 1
    rows = int(np.ceil(len(subsets) / cols))
    fig = plt.figure(figsize=(WIDTH, 2.75 * rows + 1.05))
    first = next(iter(runs.values()))
    y = _header(
        fig,
        t,
        "Every official test engine at its last observed cycle, sorted by capped true RUL",
        f"LSTM five-fold ensemble, repeat {first['served_repeat']}",
    )
    handles = [
        Line2D([], [], color=t["truth"], lw=1.6),
        Line2D([], [], color=t["series"][0], marker="o", linestyle="", markersize=4),
        Line2D([], [], color=t["series"][0], alpha=0.4, lw=4),
        Line2D([], [], color=t["series"][0], alpha=0.15, lw=4),
    ]
    cap = first["metrics"]["config"]["max_rul"]
    labels = [
        f"true RUL, capped at {cap}",
        "predicted RUL",
        "nominal 90% interval",
        "open-ended: at least the lower end",
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
        ax.set_title(f"{subset} · RMSE {m['rmse']:.1f} · {m['coverage']:.0%} inside the interval")
        ax.set_xlim(-2, len(test) + 1)
        ax.set_ylim(0, top)
        above = int((test["rul_true"] > cap).sum())
        ax.set_xlabel(f"{len(test)} test engines ({above} with true RUL above {cap})")
        if k % cols == 0:
            ax.set_ylabel("Cycles")
        ax.grid(axis="x", visible=False)
    return fig


def fig_lives(ctx: Context, t: dict, n: int = 4) -> Figure:
    """Whole lives of a spread of training engines, each predicted out of fold."""
    run = ctx.focus
    assert run is not None
    oof = _served(run["oof"], run)
    cfg = run["metrics"]["config"]
    threshold, horizon, cap = cfg["alert_threshold"], cfg["alert_cycles"], cfg["max_rul"]
    lengths = oof.groupby("unit")["cycle"].max().sort_values()
    picks = lengths.index[np.linspace(0, len(lengths) - 1, n).round().astype(int)]
    fig = plt.figure(figsize=(WIDTH, 5.2))
    lead = run["metrics"]["out_of_fold"]["lstm"]["per_repeat"][str(run["served_repeat"])]["alert"]
    y = _header(
        fig,
        t,
        f"{run.subset} whole lives, each predicted by the single fold model that never saw it",
        f"Shortest, longest and two evenly spaced of {lead['units']} cross-fitted engines; "
        f"over all of them, median first alert {lead['lead_median']:.0f} cycles before failure, "
        f"{lead['missed']} missed, {lead['early']} with more than {horizon} cycles left",
    )
    handles = [
        Line2D([], [], color=t["series"][0], lw=1.8),
        Patch(color=t["series"][0], alpha=t["band"], lw=0),
        Line2D([], [], color=t["truth"], lw=1.6),
        Line2D([], [], color=t["alert"], lw=1.3),
    ]
    labels = ["predicted RUL", "nominal 90% interval", f"true RUL, capped at {cap}", "first alert"]
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
        ax.set_title(f"Engine {unit} · {int(g['cycle'].max())} cycles")
        ax.set_xlim(1, g["cycle"].max())
        ax.set_ylim(0, cap + 10)
        if k % 2 == 0:
            ax.set_ylabel("RUL (cycles)")
        if k >= 2:
            ax.set_xlabel("Cycle")
        ax.grid(axis="x", visible=False)
    return fig


ORDERINGS = (
    ("ig", "Integrated gradients"),
    ("input_deviation", "|input − reference|"),
    ("randomised_ig", "IG, re-initialised net"),
    ("random", "Random order (mean of 20)"),
)


def fig_attribution_checks(ctx: Context, t: dict) -> Figure | None:
    """Every window's deletion and insertion area for each ranking, with the median and the mean
    of the recorded per-window values, and where integrated gradients ranks worse than random
    order or plain input deviation, by stage of life."""
    focus = ctx.focus
    if focus is None or not (focus.dir / "attribution-checks.json").is_file():
        return None
    report = ctx.memo(
        "attribution_checks",
        lambda: json.loads((focus.dir / "attribution-checks.json").read_text()),
    )
    windows = pd.DataFrame(report["per_window"])
    n = len(windows)
    stages = sorted(windows["rul"].unique(), reverse=True)
    colours = {
        "ig": t["series"][0],
        "input_deviation": t["text2"],
        "randomised_ig": t["muted"],
        "random": t["muted"],
    }
    fig = plt.figure(figsize=(WIDTH, 4.9))
    y = _header(
        fig,
        t,
        f"Attribution checks: {n} windows from {report['engines']} held-out "
        f"{report['model']} engines, served ensemble",
    )
    upper = fig.add_gridspec(1, 2, wspace=0.08, left=0.19, right=0.985, top=y - 0.06, bottom=0.545)
    lower = fig.add_gridspec(1, 2, wspace=0.08, left=0.1, right=0.985, top=0.375, bottom=0.1)
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.17, 0.17, n)
    bounds = {"deletion": (-0.8, 1.6), "insertion": (-0.6, 1.6)}
    for col, (kind, better) in enumerate((("deletion", "lower"), ("insertion", "higher"))):
        ax = fig.add_subplot(upper[0, col])
        low, high = bounds[kind]
        for row, (key, _label) in enumerate(ORDERINGS):
            values = windows[f"{kind}_auc_{key}"].to_numpy(float)
            over, under = values > high, values < low
            inside = ~(over | under)
            ax.scatter(
                values[inside],
                row + jitter[inside],
                s=9,
                color=colours[key],
                alpha=0.75,
                lw=0,
                zorder=2,
            )
            for mask, edge, marker, word, extreme in (
                (over, high, ">", "above", values.max()),
                (under, low, "<", "below", values.min()),
            ):
                if not mask.any():
                    continue
                ax.scatter(
                    np.full(int(mask.sum()), edge),
                    row + jitter[mask],
                    s=16,
                    marker=marker,
                    color=t["fail"],
                    lw=0,
                    zorder=3,
                )
                ax.text(
                    edge,
                    row - 0.36,
                    f"{int(mask.sum())} {word} {edge:g} ({extreme:.2f})",
                    ha="right" if marker == ">" else "left",
                    va="bottom",
                    fontsize=7,
                    color=t["fail"],
                )
            median, mean = float(np.median(values)), float(np.mean(values))
            ax.plot([median, median], [row - 0.3, row + 0.3], color=t["text"], lw=1.8, zorder=4)
            ax.scatter(
                [float(np.clip(mean, low, high))],
                [row],
                s=26,
                marker="D",
                facecolors=t["surface"],
                edgecolors=t["text"],
                linewidths=1.1,
                zorder=5,
            )
        ax.set_yticks(range(len(ORDERINGS)), [label for _, label in ORDERINGS] if col == 0 else [])
        ax.set_ylim(len(ORDERINGS) - 0.5, -0.5)
        ax.grid(axis="y", visible=False)
        ax.set_title(
            f"{kind.capitalize()} area per window ({better} is better)",
            fontsize=8,
            weight="normal",
            color=t["text2"],
        )
        ax.set_xlabel("Area under the curve")
        ax.set_xlim(low - 0.05, high + 0.05)
    key_handles = [
        Line2D([], [], color=t["text"], lw=1.8),
        Line2D(
            [],
            [],
            marker="D",
            linestyle="",
            markersize=5,
            markerfacecolor=t["surface"],
            markeredgecolor=t["text"],
        ),
        Line2D([], [], marker=">", linestyle="", markersize=5, color=t["fail"]),
    ]
    fig.legend(
        key_handles,
        [f"median of {n} windows", f"mean of {n} windows", "clipped value"],
        loc="upper right",
        bbox_to_anchor=(0.985, y + 0.005),
        ncols=3,
        handletextpad=0.4,
        borderaxespad=0,
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
