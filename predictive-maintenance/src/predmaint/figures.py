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
    return {
        "metrics": json.loads((model_dir / "metrics.json").read_text()),
        "test": pd.read_csv(model_dir / "predictions" / "test.csv"),
        "holdout": pd.read_csv(model_dir / "predictions" / "holdout.csv"),
        "history": json.loads((model_dir / "history.json").read_text()),
        "dir": model_dir,
    }


def fig_test_sets(runs: dict, t: dict, out_dir: Path, theme: str) -> Path:
    subsets = list(runs)
    cols = 2 if len(subsets) > 1 else 1
    rows = int(np.ceil(len(subsets) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 3.1 * rows + 0.9), squeeze=False)
    for ax, subset in zip(axes.flat, subsets, strict=False):
        run = runs[subset]
        test = run["test"].sort_values(["rul_capped", "point"]).reset_index(drop=True)
        x = np.arange(len(test))
        m = run["metrics"]["test"]
        ax.vlines(x, test["lower"], test["upper"], color=t["series"][0], alpha=0.28, linewidth=1.6)
        ax.plot(
            x, test["rul_capped"], color=t["truth"], linewidth=1.6, label="True RUL (capped at 125)"
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
        coverage = f"{m['interval_coverage']:.0%} inside the 90% interval"
        ax.set_title(f"{subset}  ·  RMSE {m['rmse']:.1f}  ·  {coverage}", pad=6)
        ax.set_xlim(-2, len(test) + 1)
        ax.set_ylim(0, 140)
        ax.set_xlabel(f"{len(test)} test engines, sorted by true RUL")
        ax.set_ylabel("Cycles")
        ax.grid(axis="x", visible=False)
    for ax in list(axes.flat)[len(subsets) :]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=t["series"][0], alpha=0.35, linewidth=4))
    labels.append("90% interval")
    fig.legend(handles, labels, loc="upper right", ncols=3, bbox_to_anchor=(0.99, 0.995))
    _title(
        fig,
        t,
        "Official test sets: predicted vs true remaining useful life",
        "One point per engine at its last recorded cycle. Ground truth from NASA's RUL files.",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return _save(fig, out_dir, "test-sets", theme)


def _first_alert(group: pd.DataFrame, threshold: float):
    fired = group[group["probability"] >= threshold]
    return fired.iloc[0] if len(fired) else None


def fig_holdout_lives(run: dict, t: dict, out_dir: Path, theme: str, n: int = 4) -> Path:
    hold = run["holdout"]
    threshold = run["metrics"]["calibration"]["alert_threshold"]
    horizon = run["metrics"]["config"]["alert_cycles"]
    lengths = hold.groupby("unit")["cycle"].max().sort_values()
    picks = lengths.index[np.linspace(0, len(lengths) - 1, n).round().astype(int)]
    fig, axes = plt.subplots(1, n, figsize=(11, 3.3), sharey=True)
    for ax, unit in zip(axes, picks, strict=True):
        g = hold[hold["unit"] == unit]
        ax.axhspan(0, horizon, color=t["zone"], alpha=0.09, linewidth=0)
        ax.fill_between(
            g["cycle"],
            g["lower"],
            g["upper"],
            color=t["series"][0],
            alpha=0.16,
            linewidth=0,
            label="90% interval",
        )
        ax.plot(
            g["cycle"], g["rul_capped"], color=t["truth"], linewidth=1.6, label="True RUL (capped)"
        )
        ax.plot(g["cycle"], g["point"], color=t["series"][0], linewidth=1.6, label="Predicted RUL")
        alert = _first_alert(g, threshold)
        if alert is not None:
            ax.axvline(alert["cycle"], color=t["serious"], linewidth=1)
            ax.annotate(
                f"alert: {int(alert['rul_true'])} cycles\nbefore failure",
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
    fig.legend(handles, labels, loc="upper right", ncols=3, bbox_to_anchor=(0.99, 0.995))
    subset = run["metrics"]["subset"]
    lt = run["metrics"]["holdout"]["lead_time"]
    _title(
        fig,
        t,
        f"{subset} holdout engines, replayed cycle by cycle",
        "Engines the model never trained or calibrated on. "
        + (
            f"Median alert lead time {lt['median']:.0f} cycles, "
            if lt["median"] is not None
            else ""
        )
        + f"{lt['missed']} of {lt['units']} missed.",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    return _save(fig, out_dir, "holdout-lives", theme)


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
    ax_heat.set_title("Contribution of each reading in the model's 30-cycle window", pad=6)
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
        "Integrated gradients against an average engine in the same operating regime. "
        f"True RUL {int(history['rul'].iloc[-1])}; "
        f"90% interval {rul['lower']:.0f}–{rul['upper']:.0f}.",
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.8, bottom=0.12)
    return _save(fig, out_dir, "attribution", theme)


def fig_calibration(runs: dict, focus: dict, t: dict, out_dir: Path, theme: str) -> Path:
    fig, (ax_rel, ax_lead) = plt.subplots(
        1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1, 1.25]}
    )
    ax_rel.plot([0, 1], [0, 1], color=t["axis"], linewidth=1)
    curve = pd.DataFrame(focus["metrics"]["holdout"]["reliability"])
    ax_rel.plot(
        curve["mean_probability"],
        curve["observed_rate"],
        color=t["series"][0],
        linewidth=1.6,
        marker="o",
        markersize=5,
        markeredgecolor=t["surface"],
        markeredgewidth=1.2,
    )
    ax_rel.set_xlim(0, 1)
    ax_rel.set_ylim(0, 1)
    horizon = focus["metrics"]["config"]["alert_cycles"]
    ax_rel.set_xlabel(f"Predicted probability of failure within {horizon} cycles")
    ax_rel.set_ylabel("Observed frequency")
    ax_rel.set_title(f"Reliability on {focus['metrics']['subset']} holdout cycles", pad=6)
    ax_rel.text(
        0.97,
        0.05,
        "perfect calibration on the diagonal",
        ha="right",
        fontsize=7.5,
        color=t["muted"],
        transform=ax_rel.transAxes,
    )

    labels = []
    for i, (subset, run) in enumerate(runs.items()):
        leads = [u["lead_time"] for u in run["metrics"]["holdout"]["lead_time"]["per_unit"]]
        caught = np.array([v for v in leads if v is not None], dtype=float)
        jitter = np.random.default_rng(i).uniform(-0.16, 0.16, len(caught))
        ax_lead.scatter(
            caught,
            np.full(len(caught), i) + jitter,
            s=16,
            color=t["series"][0],
            alpha=0.75,
            linewidths=0,
        )
        if len(caught):
            ax_lead.plot(
                [np.median(caught)] * 2, [i - 0.3, i + 0.3], color=t["text"], linewidth=1.6
            )
        missed = sum(v is None for v in leads)
        labels.append(f"{subset}\n{len(leads)} engines" + (f", {missed} missed" if missed else ""))
    ax_lead.axvline(horizon, color=t["serious"], linewidth=1)
    ax_lead.set_yticks(range(len(runs)), labels)
    ax_lead.set_ylim(len(runs) - 0.5, -0.6)
    ax_lead.set_xlim(left=0)
    ax_lead.set_xlabel(
        f"Cycles left when the alert first fired (dots: engines, bar: median, "
        f"orange line: {horizon}-cycle horizon)"
    )
    ax_lead.grid(axis="y", visible=False)
    ax_lead.set_title("Alert lead time on holdout engines", pad=6)
    _title(
        fig,
        t,
        "Are the alerts trustworthy?",
        "Left: the calibrated probability matches how often failure really follows. "
        "Right: how early the alert fires on engines that run to failure.",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    return _save(fig, out_dir, "alerts", theme)


# Range of the RMSEs that Table 4 of Chaoub et al. (2021, arXiv:2104.05049) lists for six
# published methods (DA-CNN, DCGAN, MS-DCNN, HDNN, an LSTM and the authors' MLP-LSTM-MLP).
# Literature context only: protocols differ between papers.
PUBLISHED_RMSE_RANGE = {
    "FD001": (10.71, 16.5),
    "FD002": (12.49, 19.49),
    "FD003": (11.48, 15.9),
    "FD004": (13.97, 22.22),
}


def fig_results(runs: dict, t: dict, out_dir: Path, theme: str) -> Path:
    subsets = list(runs)
    fig, ax = plt.subplots(figsize=(10, 1.3 + 0.62 * len(subsets)))
    y = np.arange(len(subsets))
    for i, subset in enumerate(subsets):
        low, high = PUBLISHED_RMSE_RANGE[subset]
        ax.plot(
            [low, high],
            [i, i],
            color=t["grid"],
            linewidth=9,
            solid_capstyle="round",
            zorder=1,
            label="Published methods, range (Chaoub et al. 2021, Table 4)" if i == 0 else None,
        )
    metrics = [runs[s]["metrics"] for s in subsets]
    single = [m["test_single_models"] for m in metrics]
    ax.errorbar(
        [s["rmse_mean"] for s in single],
        y + 0.18,
        xerr=[s["rmse_std"] for s in single],
        fmt="o",
        color=t["series"][1],
        markersize=6,
        markeredgecolor=t["surface"],
        elinewidth=1.4,
        capsize=0,
        zorder=3,
        label="Single LSTM, mean ± sd of 5 seeds",
    )
    lgbm = [(m["baseline_lightgbm"] or {}).get("rmse", np.nan) for m in metrics]
    ax.plot(
        lgbm,
        y - 0.18,
        "D",
        color=t["series"][2],
        markersize=6,
        markeredgecolor=t["surface"],
        zorder=3,
        label="LightGBM on window statistics",
    )
    ensemble = [m["test"]["rmse"] for m in metrics]
    ax.plot(
        ensemble,
        y,
        "o",
        color=t["series"][0],
        markersize=8,
        markeredgecolor=t["surface"],
        markeredgewidth=1.5,
        zorder=4,
        label="LSTM + attention, 5-model ensemble",
    )
    for yi, v in zip(y, ensemble, strict=True):
        ax.annotate(
            f"{v:.2f}",
            (v, yi),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=t["text"],
        )
    ax.set_yticks(y, subsets)
    ax.set_ylim(len(subsets) - 0.5, -0.6)
    ax.set_xlim(9, 23)
    ax.set_xlabel(
        "Test RMSE in cycles (lower is better; this project scores against truth capped at 125)"
    )
    ax.grid(axis="y", visible=False)
    handles, labels = ax.get_legend_handles_labels()
    order = [3, 1, 2, 0]
    fig.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="upper left",
        ncols=2,
        bbox_to_anchor=(0.06, 0.86),
    )
    _title(
        fig,
        t,
        "Test error by subset, with published results for context",
        "FD002 and FD004 mix six flight conditions; FD003 and FD004 have two fault modes.",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.68))
    return _save(fig, out_dir, "results", theme)


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
        s: _load(artifacts, s) for s in args.subsets if (artifacts / s / "metrics.json").is_file()
    }
    if not runs:
        parser.error(f"No trained subsets found in {artifacts}")
    args.out.mkdir(parents=True, exist_ok=True)
    focus = runs.get(args.focus) or next(iter(runs.values()))
    written = []
    for theme, t in THEMES.items():
        _style(t)
        written += [
            fig_test_sets(runs, t, args.out, theme),
            fig_holdout_lives(focus, t, args.out, theme),
            fig_attribution(focus, t, args.out, theme),
            fig_calibration(runs, focus, t, args.out, theme),
            fig_results(runs, t, args.out, theme),
        ]
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
