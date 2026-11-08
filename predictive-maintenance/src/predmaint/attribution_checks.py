"""Check whether the integrated-gradients attributions of the served model mean anything.

    python -m predmaint.attribution_checks      # FD001 -> artifacts/FD001/attribution-checks.json

Run on windows from the replay engines (held out of all training) at four stages of life:

1. **Completeness.** Integrated gradients should add up to f(window) - f(baseline). The gap
   measures how well the 48-step integral used in serving approximates it.
2. **Deletion and insertion.** Replacing the most-attributed readings with the baseline should
   move the prediction towards f(baseline) faster than replacing readings in random order; adding
   them to the baseline first should recover f(window) faster. We compare the area under each
   curve with the same curve for 20 random orders.
3. **Model-parameter randomisation** (Adebayo et al., 2018). Attributions computed on networks
   whose weights were re-initialised should look different from the trained ones. If they do not,
   the map describes the input, not the model. We also compare with the plain input deviation
   |window - baseline|, the trivial map a gradient-times-input method can collapse to.
4. **Baseline sensitivity.** The default baseline (zeros: the average engine in the same
   regime) against each engine's own first cycles, as a "healthy state of this engine" reference.

Results are plain numbers; they describe what the model's output depends on, not physical cause.
"""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from predmaint.config import Settings
from predmaint.inference import Predictor
from predmaint.model import Ensemble, integrated_gradients

STAGES = (100, 60, 30, 10)  # cycles left at the windows we explain
FRACTIONS = np.linspace(0.0, 1.0, 41)
RANDOM_ORDERS = 20
ORDER_NAMES = ("ig", "input_deviation", "randomised_ig", "random")
MIN_EFFECT = 2.0  # cycles; smaller f(x) - f(baseline) makes the normalised curves meaningless


def _point(model: Ensemble, windows: np.ndarray, max_rul: int) -> np.ndarray:
    with torch.no_grad():
        return model.point(torch.from_numpy(windows.astype(np.float32))).numpy() * max_rul


def _ig(model: Ensemble, window: np.ndarray, baseline: np.ndarray, steps: int, max_rul: int):
    out = integrated_gradients(
        model.point, torch.from_numpy(window), torch.from_numpy(baseline), steps=steps
    )
    return out.detach().numpy() * max_rul


def _curves(
    model: Ensemble, window: np.ndarray, baseline: np.ndarray, order: np.ndarray, max_rul: int
) -> tuple[np.ndarray, np.ndarray]:
    """Deletion and insertion curves for one ordering of the window's cells, normalised so the
    window scores 1 and the baseline scores 0."""
    cells = window.size
    deleted, inserted = [], []
    for fraction in FRACTIONS:
        k = round(float(fraction) * cells)
        flat = np.zeros(cells, bool)
        flat[order[:k]] = True
        mask = flat.reshape(window.shape)
        deleted.append(np.where(mask, baseline, window))
        inserted.append(np.where(mask, window, baseline))
    points = _point(model, np.stack(deleted + inserted), max_rul)
    f_x, f_b = points[0], points[len(FRACTIONS)]  # nothing deleted / nothing inserted
    scale = f_x - f_b
    return (points[: len(FRACTIONS)] - f_b) / scale, (points[len(FRACTIONS) :] - f_b) / scale


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation of the attribution magnitudes of two maps."""
    ranks = [pd.Series(np.abs(m).ravel()).rank().to_numpy() for m in (a, b)]
    return float(np.corrcoef(ranks[0], ranks[1])[0, 1])


def randomized(model: Ensemble, seed: int, heads_only: bool = False) -> Ensemble:
    """A copy of the ensemble with re-initialised weights (all layers, or only the output
    heads, the first step of Adebayo et al.'s cascading randomisation)."""
    torch.manual_seed(seed)
    clone = copy.deepcopy(model)
    for member in clone.members:
        layers = [member.rul_head, member.alert_head] if heads_only else list(member.modules())
        for layer in layers:
            reset = getattr(layer, "reset_parameters", None)
            if callable(reset):
                reset()
    return clone.eval()


def cases(predictor: Predictor, replay: pd.DataFrame) -> list[dict]:
    """Windows of every replay engine at fixed numbers of cycles before failure."""
    out = []
    for _, engine in replay.groupby("unit"):
        engine = engine.sort_values("cycle").reset_index(drop=True)
        unit = int(engine["unit"].to_numpy()[0])
        features = predictor.contract.engine_features(engine)
        early = np.asarray(features[: predictor.contract.window])
        own_baseline = np.repeat(early.mean(axis=0, keepdims=True), predictor.contract.window, 0)
        for left in STAGES:
            matches = np.flatnonzero(engine["rul"].to_numpy() == left)
            if not len(matches):  # engine shorter than this stage
                continue
            end = int(matches[0])
            out.append(
                {
                    "unit": unit,
                    "rul": left,
                    "window": predictor.contract.window_from_readings(engine.iloc[: end + 1]),
                    "own_baseline": own_baseline.astype(np.float32),
                }
            )
    return out


def run(model_dir: Path, seed: int = 0) -> dict:
    started = time.time()
    predictor = Predictor(model_dir)
    model, max_rul = predictor.model, predictor.contract.max_rul
    replay = pd.read_csv(model_dir / "replay.csv")
    rng = np.random.default_rng(seed)
    full_random = randomized(model, seed)
    heads_random = randomized(model, seed, heads_only=True)
    rows = []
    curves: dict[str, list[np.ndarray]] = {}
    for case in cases(predictor, replay):
        x = case["window"].astype(np.float32)
        zero = np.zeros_like(x)
        attr = _ig(model, x, zero, 48, max_rul)
        attr_fine = _ig(model, x, zero, 256, max_rul)
        attr_random = _ig(full_random, x, zero, 48, max_rul)
        own = _ig(model, x, case["own_baseline"], 48, max_rul)
        effect = float(_point(model, np.stack([x, zero]), max_rul) @ np.array([1.0, -1.0]))
        top = set(np.argsort(-np.abs(attr.sum(axis=0)))[:3])
        top_own = set(np.argsort(-np.abs(own.sum(axis=0)))[:3])
        row = {
            "unit": case["unit"],
            "rul": case["rul"],
            "effect_cycles": effect,
            "completeness_gap_cycles": abs(float(attr.sum()) - effect),
            "completeness_gap_256_steps": abs(float(attr_fine.sum()) - effect),
            "spearman_vs_full_randomisation": _spearman(attr, attr_random),
            "spearman_vs_head_randomisation": _spearman(
                attr, _ig(heads_random, x, zero, 48, max_rul)
            ),
            "spearman_vs_input_deviation": _spearman(attr, x - zero),
            "spearman_vs_own_baseline": _spearman(attr, own),
            "top3_sensors_kept_with_own_baseline": len(top & top_own),
        }
        if abs(effect) >= MIN_EFFECT:
            # Every ordering is scored on the trained model: only the order of removal differs.
            orders = {
                "ig": np.abs(attr).ravel(),
                "input_deviation": np.abs(x - zero).ravel(),
                "randomised_ig": np.abs(attr_random).ravel(),
            }
            for name, score in orders.items():
                d, i = _curves(model, x, zero, np.argsort(-score, kind="stable"), max_rul)
                curves.setdefault(f"deletion_{name}", []).append(d)
                curves.setdefault(f"insertion_{name}", []).append(i)
                row |= {f"deletion_auc_{name}": float(np.mean(d))}
                row |= {f"insertion_auc_{name}": float(np.mean(i))}
            rd, ri = zip(
                *(
                    _curves(model, x, zero, rng.permutation(x.size), max_rul)
                    for _ in range(RANDOM_ORDERS)
                ),
                strict=True,
            )
            curves.setdefault("deletion_random", []).append(np.mean(rd, axis=0))
            curves.setdefault("insertion_random", []).append(np.mean(ri, axis=0))
            row |= {
                "deletion_auc_random": float(np.mean(rd)),
                "insertion_auc_random": float(np.mean(ri)),
            }
        rows.append(row)

    frame = pd.DataFrame(rows)
    curve_rows = frame.dropna(subset=["deletion_auc_ig"])

    def summary(column: str) -> dict:
        values = frame[column].dropna().to_numpy(float)
        return {
            "median": float(np.median(values)),
            "p10": float(np.percentile(values, 10)),
            "p90": float(np.percentile(values, 90)),
            "n": len(values),
        }

    scored = ("ig", "input_deviation", "randomised_ig")
    auc_columns = [f"{kind}_auc_{o}" for kind in ("deletion", "insertion") for o in scored]
    auc_columns += ["deletion_auc_random", "insertion_auc_random"]
    beats = {
        f"ig_beats_{other}_on_{kind}": int(
            (
                curve_rows[f"{kind}_auc_ig"] < curve_rows[f"{kind}_auc_{other}"]
                if kind == "deletion"
                else curve_rows[f"{kind}_auc_ig"] > curve_rows[f"{kind}_auc_{other}"]
            ).sum()
        )
        for kind in ("deletion", "insertion")
        for other in ("random", "input_deviation", "randomised_ig")
    }
    return {
        "model": model_dir.name,
        "windows": len(frame),
        "engines": int(frame["unit"].nunique()),
        "stages_cycles_left": list(STAGES),
        "integration_steps_served": 48,
        "summary": {
            column: summary(column)
            for column in (
                "effect_cycles",
                "completeness_gap_cycles",
                "completeness_gap_256_steps",
                "spearman_vs_full_randomisation",
                "spearman_vs_head_randomisation",
                "spearman_vs_input_deviation",
                "spearman_vs_own_baseline",
                "top3_sensors_kept_with_own_baseline",
                *auc_columns,
            )
        },
        "curve_windows": len(curve_rows),
        "comparisons": beats,
        # Median over windows: a few windows whose prediction is close to the baseline make the
        # normalised curves explode, and a mean would follow them.
        "curves": {"fractions": FRACTIONS.tolist()}
        | {name: np.median(values, axis=0).tolist() for name, values in curves.items()},
        "per_window": rows,
        "seconds": round(time.time() - started, 1),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m predmaint.attribution_checks",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--subset", default="FD001")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    model_dir = Settings.from_env().artifacts_dir / args.subset
    report = run(model_dir)
    out = args.out or model_dir / "attribution-checks.json"
    out.write_text(json.dumps(report, indent=1))
    s, c = report["summary"], report["comparisons"]
    n = report["curve_windows"]
    deletion = {k: s[f"deletion_auc_{k}"]["median"] for k in ORDER_NAMES}
    lines = [
        f"{report['windows']} windows from {report['engines']} replay engines",
        f"completeness gap: median {s['completeness_gap_cycles']['median']:.2f} cycles "
        f"(256 steps: {s['completeness_gap_256_steps']['median']:.2f})",
        "deletion AUC, lower is better: " + ", ".join(f"{k} {v:.3f}" for k, v in deletion.items()),
        f"IG beats random in {c['ig_beats_random_on_deletion']}/{n} windows, input deviation "
        f"in {c['ig_beats_input_deviation_on_deletion']}/{n}, randomised-network IG in "
        f"{c['ig_beats_randomised_ig_on_deletion']}/{n}",
        f"Spearman vs randomised weights {s['spearman_vs_full_randomisation']['median']:.2f}, "
        f"vs input deviation {s['spearman_vs_input_deviation']['median']:.2f}, "
        f"vs own-baseline IG {s['spearman_vs_own_baseline']['median']:.2f}",
    ]
    print("\n".join(lines))
    print(f"Written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
