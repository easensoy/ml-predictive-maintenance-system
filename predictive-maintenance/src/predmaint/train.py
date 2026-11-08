"""Cross-fitted training, calibration and out-of-fold evaluation on one C-MAPSS subset.

    python -m predmaint.train --subset FD001            # 3 repeats x 5 folds, LSTM and LightGBM
    python -m predmaint.train --subset FD001 --repeats 0 --epochs 3 --out /tmp/smoke

This command never reads the test labels (RUL_FD00x.txt). It predicts the official test engines
from their sensor readings alone and saves those predictions; ``python -m predmaint.evaluate``
scores them, once, and records that in the experiment ledger.

Protocol, for each repeat (a repeat reruns everything with a new seed):

1. A fixed 10% of the training engines is set aside for the dashboard replay. No model, scaler
   or calibration step ever sees them.
2. The remaining engines are split into 5 folds, stratified by life length.
3. For fold k, the other four folds are split again into inner-train (85%) and inner-validation
   (15%) engines. For each candidate window (30, 40, 50 and 60 cycles) a network is trained on
   inner-train and early-stopped on inner-validation RMSE (every engine weighted equally); the
   window with the lowest inner-validation RMSE is kept as network k. LightGBM gets the same
   folds, splits and window choice, with its own models for the point, the 5% and 95% quantiles
   and the alert.
4. Network k predicts every cycle of every engine in fold k: out-of-fold predictions, made by a
   model whose training, early stopping and window choice never saw that engine.
5. Calibration points look like test points. Each engine is truncated once, at a length drawn
   from the official test engines' trajectory lengths (sensor data only), keeping lengths shorter
   than that engine's life. Its conformity score, from network k, feeds CV+ intervals.
6. The alert temperature is fitted on 10 such truncations per engine, weighted equally per engine.

Sensor scaling and operating-regime clusters use the sensor readings of all non-replay training
engines. They use no labels.
"""

import argparse
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from predmaint import __version__
from predmaint import data as cmapss
from predmaint.calibration import (
    covered,
    cqr_scores,
    cross_conformal_margins,
    cv_plus_interval,
    finalize_interval,
    fit_temperature,
    probability,
)
from predmaint.config import PROJECT_ROOT, Settings
from predmaint.features import FeatureContract
from predmaint.metrics import (
    alert_behaviour,
    ece_with_interval,
    mean_sd,
    per_engine_weights,
    reliability_curve,
    weighted_rmse,
)
from predmaint.model import CHECKPOINT_FORMAT, RULNet, multitask_loss, predict_raw

MODELS = ("lstm", "lightgbm")
LEDGER = PROJECT_ROOT / "docs" / "experiment-ledger.jsonl"
CALIBRATION_FORMAT = 2


@dataclass
class TrainConfig:
    subset: str = "FD001"
    repeats: list[int] = field(default_factory=lambda: [0, 1, 2])
    folds: int = 5
    windows: list[int] = field(default_factory=lambda: [30, 40, 50, 60])
    replay_fraction: float = 0.1
    inner_val_fraction: float = 0.15
    data_seed: int = 42
    max_rul: int = 125
    alert_cycles: int = 30
    hidden: int = 64
    layers: int = 2
    dropout: float = 0.2
    epochs: int = 80
    patience: int = 12
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-5
    alpha: float = 0.1
    alert_threshold: float = 0.5
    temperature_draws: int = 10
    baseline: bool = True
    lgbm_estimators: int = 2000
    lgbm_learning_rate: float = 0.05
    device: str = "auto"


# --------------------------------------------------------------------------------------------
# Provenance


def source_digest() -> str:
    """SHA-256 of the predmaint sources (line endings normalised): names the exact training code."""
    digest = hashlib.sha256()
    package = Path(__file__).resolve().parent
    for path in sorted(package.rglob("*.py")):
        digest.update(path.relative_to(package).as_posix().encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()[:16]


def git_state() -> dict | None:
    """The commit this code came from, and whether its sources differ from that commit."""
    root = Path(__file__).resolve().parents[2]

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()

    try:
        commit = git("rev-parse", "HEAD")
        changed = git("status", "--porcelain", "--", "src", "pyproject.toml", "uv.lock")
    except OSError, subprocess.SubprocessError:
        return None
    return {"commit": commit, "dirty": bool(changed), "changed": changed.splitlines()}


def environment(device: torch.device) -> dict:
    try:
        import lightgbm

        lightgbm_version: str | None = lightgbm.__version__
    except ImportError:
        lightgbm_version = None
    import sklearn

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "lightgbm": lightgbm_version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "training_device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    }


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but this PyTorch build has no CUDA support")
    return torch.device(name)


def ledger_path(out_dir: Path) -> str:
    """How the ledger names a run's output: relative to the project when inside it, otherwise
    just the directory name (local paths stay out of the repository)."""
    try:
        return out_dir.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return f"scratch/{out_dir.name}"


def append_ledger(path: Path, entry: dict) -> None:
    """Append one JSON line to the experiment ledger (one small write, so concurrent runs do not
    interleave)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as ledger:
        ledger.write(json.dumps(entry, separators=(",", ":")) + "\n")


# --------------------------------------------------------------------------------------------
# Data


@dataclass
class Engines:
    """Every row of a set of engines, sorted by unit and cycle, with its model window."""

    unit: np.ndarray
    cycle: np.ndarray
    rul: np.ndarray
    windows: np.ndarray  # (rows, max window, features), float32
    lives: dict[int, int]
    offset: dict[int, int]

    def rows(self, units: np.ndarray) -> np.ndarray:
        return np.isin(self.unit, units)

    def row_of(self, unit: int, cycle: int) -> int:
        return self.offset[unit] + cycle - 1


def engine_table(frame: pd.DataFrame, contract: FeatureContract) -> Engines:
    unit = frame["unit"].to_numpy()
    cycle = frame["cycle"].to_numpy()
    starts = np.flatnonzero(np.r_[True, unit[1:] != unit[:-1]])
    ends = np.r_[starts[1:], len(unit)]
    lives, offset = {}, {}
    for start, end in zip(starts, ends, strict=True):
        u, life = int(unit[start]), int(end - start)
        if not np.array_equal(cycle[start:end], np.arange(1, life + 1)):
            raise cmapss.DataError(f"engine {u}: cycles are not 1..{life}")
        lives[u], offset[u] = life, int(start)
    return Engines(
        unit=unit,
        cycle=cycle,
        rul=frame["rul"].to_numpy(),
        windows=contract.unit_windows(frame),
        lives=lives,
        offset=offset,
    )


# --------------------------------------------------------------------------------------------
# Models


def _point_rmse(model: RULNet, x: torch.Tensor, y: np.ndarray, w: np.ndarray, cfg) -> float:
    model.eval()
    with torch.no_grad():
        points = np.concatenate(
            [
                model(x[i : i + 8192])["quantiles"][:, 1].cpu().numpy()
                for i in range(0, len(x), 8192)
            ]
        )
    return weighted_rmse(points * cfg.max_rul, y * cfg.max_rul, w)


def train_network(
    seed: int,
    window: int,
    train: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    val: tuple[torch.Tensor, np.ndarray, np.ndarray],
    cfg: TrainConfig,
    n_features: int,
) -> tuple[RULNet, list[dict], float]:
    """Train one network on device tensors; return it on the CPU with its best val RMSE."""
    device = train[0].device
    torch.manual_seed(seed)  # weight init and dropout; shuffling uses its own seeded generator
    shuffle = torch.Generator().manual_seed(seed)
    model = RULNet(n_features, cfg.hidden, cfg.layers, cfg.dropout, window=window).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4)
    x, y, a = train
    best_rmse, best_state, best_epoch = float("inf"), None, 0
    history = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total = 0.0
        order = torch.randperm(len(x), generator=shuffle).to(device)
        for start in range(0, len(x), cfg.batch_size):
            batch = order[start : start + cfg.batch_size]
            optimizer.zero_grad()
            loss = multitask_loss(model(x[batch]), y[batch], a[batch])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(batch)
        val_rmse = _point_rmse(model, *val, cfg)
        scheduler.step(val_rmse)
        history.append(
            {"epoch": epoch, "train_loss": total / len(x), "val_rmse": round(val_rmse, 4)}
        )
        if val_rmse < best_rmse - 1e-4:
            best_rmse, best_epoch = val_rmse, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= cfg.patience:
            break
    model = model.cpu()
    assert best_state is not None, "the first epoch always sets best_state"
    model.load_state_dict(best_state)
    model.eval()
    return model, history, best_rmse


def window_summary(windows: np.ndarray) -> np.ndarray:
    """Hand-crafted features for the gradient-boosting baseline: last, mean, std, slope."""
    t = np.arange(windows.shape[1], dtype=np.float64)
    t -= t.mean()
    slope = (windows * t[None, :, None]).sum(axis=1) / (t**2).sum()
    return np.concatenate([windows[:, -1], windows.mean(1), windows.std(1), slope], axis=1)


class LightGBMFold:
    """LightGBM's counterpart of one network: point, 5% and 95% quantile, and alert models
    on window statistics of the chosen window length."""

    def __init__(self, window: int, models: dict[str, Any]):
        self.window = window
        self.models = models

    def predict(self, windows: np.ndarray, max_rul: int) -> dict[str, np.ndarray]:
        x = window_summary(windows[:, -self.window :])
        low = self.models["lower"].predict(x) / max_rul
        high = self.models["upper"].predict(x) / max_rul
        return {
            "quantiles": np.stack(
                [
                    np.minimum(low, high),
                    self.models["point"].predict(x) / max_rul,
                    np.maximum(low, high),
                ],
                axis=1,
            ),
            "alert_logit": self.models["alert"].predict(x, raw_score=True),
        }


def train_lightgbm(
    seed: int,
    windows: list[int],
    train: tuple[np.ndarray, np.ndarray, np.ndarray],
    val: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    cfg: TrainConfig,
) -> tuple[LightGBMFold, list[dict]]:
    import lightgbm as lgb

    x_tr, y_tr, a_tr = train
    x_va, y_va, a_va, w_va = val
    params: dict[str, Any] = {
        "n_estimators": cfg.lgbm_estimators,
        "learning_rate": cfg.lgbm_learning_rate,
        "num_leaves": 31,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "random_state": seed,
        "n_jobs": 4,
        "verbose": -1,
    }
    stop = [lgb.early_stopping(100, verbose=False)]

    def fit(model, f_tr, target_tr, f_va, target_va):
        model.fit(
            f_tr,
            target_tr,
            eval_X=(f_va,),
            eval_y=(target_va,),
            eval_sample_weight=[w_va],
            callbacks=stop,
        )
        return model

    scores, best = [], None
    for window in windows:
        f_tr, f_va = window_summary(x_tr[:, -window:]), window_summary(x_va[:, -window:])
        point = fit(lgb.LGBMRegressor(**params), f_tr, y_tr * cfg.max_rul, f_va, y_va * cfg.max_rul)
        score = weighted_rmse(point.predict(f_va), y_va * cfg.max_rul, w_va)
        scores.append({"window": window, "val_rmse": round(score, 4)})
        if best is None or score < best[0]:
            best = (score, window, point, f_tr, f_va)
    assert best is not None
    _, window, point, f_tr, f_va = best
    models = {"point": point}
    for name, q in (("lower", 0.05), ("upper", 0.95)):
        models[name] = fit(
            lgb.LGBMRegressor(objective="quantile", alpha=q, **params),
            f_tr,
            y_tr * cfg.max_rul,
            f_va,
            y_va * cfg.max_rul,
        )
    models["alert"] = fit(lgb.LGBMClassifier(**params), f_tr, a_tr, f_va, a_va)
    return LightGBMFold(window, models), scores


# --------------------------------------------------------------------------------------------
# One repeat


def _targets(rul: np.ndarray, cfg: TrainConfig) -> tuple[np.ndarray, np.ndarray]:
    capped = (np.minimum(rul, cfg.max_rul) / cfg.max_rul).astype(np.float32)
    return capped, (rul <= cfg.alert_cycles).astype(np.float32)


def crossfit(
    repeat: int,
    pool: Engines,
    test_windows: np.ndarray,
    replay: Engines,
    cfg: TrainConfig,
    device: torch.device,
) -> dict:
    """Train every fold of one repeat; return raw out-of-fold, test and replay outputs."""
    units = np.array(sorted(pool.lives))
    lives = np.array([pool.lives[u] for u in units])
    folds = cmapss.assign_folds(units, lives, cfg.folds, seed=1000 + repeat)
    fold_of_row = np.array([folds[int(u)] for u in pool.unit])
    y_all, a_all = _targets(pool.rul, cfg)
    n_features = pool.windows.shape[2]
    raw: dict[str, Any] = {
        m: {
            "oof_q": np.zeros((len(pool.unit), 3), np.float32),
            "oof_logit": np.zeros(len(pool.unit), np.float32),
            "test_q": [],
            "test_logit": [],
            "replay_q": [],
            "replay_logit": [],
        }
        for m in MODELS
    }
    networks, selection, histories = [], [], {}
    inner_validation: dict[str, list[int]] = {}
    for k in range(cfg.folds):
        started = time.time()
        rest = units[[folds[int(u)] != k for u in units]]
        inner_train, inner_val = cmapss.stratified_subset(
            rest,
            np.array([pool.lives[int(u)] for u in rest]),
            cfg.inner_val_fraction,
            seed=10 * repeat + k,
        )
        inner_validation[str(k)] = [int(u) for u in inner_val]
        tr, va = pool.rows(inner_train), pool.rows(inner_val)
        w_va = per_engine_weights(pool.unit[va])
        x_tr = torch.from_numpy(pool.windows[tr]).to(device)
        train_t = (
            x_tr,
            torch.from_numpy(y_all[tr]).to(device),
            torch.from_numpy(a_all[tr]).to(device),
        )
        val_t = (torch.from_numpy(pool.windows[va]).to(device), y_all[va], w_va)
        best = None
        for window in cfg.windows:
            seed = 100_000 * repeat + 1000 * k + window
            net, history, score = train_network(seed, window, train_t, val_t, cfg, n_features)
            histories[f"{repeat}/{k}/{window}"] = history
            selection.append(
                {
                    "repeat": repeat,
                    "fold": k,
                    "model": "lstm",
                    "window": window,
                    "val_rmse": round(score, 4),
                    "epochs": len(history),
                }
            )
            if best is None or score < best[0]:
                best = (score, window, net)
        assert best is not None
        _, window, net = best
        networks.append(net)
        del x_tr, train_t, val_t
        fold_rows = fold_of_row == k
        fitted: dict[str, Any] = {"lstm": net}
        chosen = {"lstm": window}
        if cfg.baseline:
            lgbm, scores = train_lightgbm(
                100_000 * repeat + 1000 * k,
                cfg.windows,
                (pool.windows[tr], y_all[tr], a_all[tr]),
                (pool.windows[va], y_all[va], a_all[va], w_va),
                cfg,
            )
            selection += [{"repeat": repeat, "fold": k, "model": "lightgbm", **s} for s in scores]
            fitted["lightgbm"] = lgbm
            chosen["lightgbm"] = lgbm.window
        for name, model in fitted.items():
            predict = (
                (lambda x, m=model: predict_raw(m, x))
                if name == "lstm"
                else (lambda x, m=model: m.predict(x, cfg.max_rul))
            )
            out = predict(pool.windows[fold_rows])
            raw[name]["oof_q"][fold_rows] = out["quantiles"]
            raw[name]["oof_logit"][fold_rows] = out["alert_logit"]
            test = predict(test_windows)
            raw[name]["test_q"].append(test["quantiles"])
            raw[name]["test_logit"].append(test["alert_logit"])
            rep = predict(replay.windows)
            raw[name]["replay_q"].append(rep["quantiles"])
            raw[name]["replay_logit"].append(rep["alert_logit"])
        for row in selection:
            if row["repeat"] == repeat and row["fold"] == k:
                row["chosen"] = row["window"] == chosen[str(row["model"])]
        print(
            f"  repeat {repeat} fold {k}: LSTM window {chosen['lstm']}"
            + (f", LightGBM window {chosen['lightgbm']}" if cfg.baseline else "")
            + f" ({time.time() - started:.0f} s)",
            flush=True,
        )
    for name in MODELS:
        for key in ("test_q", "test_logit", "replay_q", "replay_logit"):
            raw[name][key] = np.stack(raw[name][key]) if raw[name][key] else None
    return {
        "folds": folds,
        "inner_validation": inner_validation,
        "fold_of_row": fold_of_row,
        "networks": networks,
        "selection": selection,
        "histories": histories,
        "raw": raw,
        "models": [m for m in MODELS if raw[m]["test_q"] is not None],
    }


def calibrate(
    repeat: int, fit: dict, pool: Engines, test_lengths: np.ndarray, cfg: TrainConfig
) -> dict:
    """Conformity scores at test-like truncation points, and alert temperatures, per model."""
    rng = np.random.default_rng(2000 + repeat)
    points = cmapss.test_like_truncations(pool.lives, test_lengths, 1, rng)
    temp_points = cmapss.test_like_truncations(pool.lives, test_lengths, cfg.temperature_draws, rng)
    rows = np.array([pool.row_of(u, c) for u, c in points])
    temp_rows = np.array([pool.row_of(u, c) for u, c in temp_points])
    member = fit["fold_of_row"][rows]
    target = np.minimum(pool.rul[rows], cfg.max_rul).astype(np.float64)
    alert = (pool.rul[temp_rows] <= cfg.alert_cycles).astype(np.float64)
    weights = np.full(len(temp_rows), 1.0 / cfg.temperature_draws)
    out = {}
    for name in fit["models"]:
        q = fit["raw"][name]["oof_q"].astype(np.float64) * cfg.max_rul
        scores = cqr_scores(q[rows, 0], q[rows, 2], target)
        logits = fit["raw"][name]["oof_logit"].astype(np.float64)
        temp_member = fit["fold_of_row"][temp_rows]
        out[name] = {
            "rows": rows,
            "scores": scores,
            "member": member,
            "temperature": fit_temperature(logits[temp_rows], alert, weights),
            "fold_temperatures": [
                fit_temperature(
                    logits[temp_rows][temp_member != k],
                    alert[temp_member != k],
                    weights[temp_member != k],
                )
                for k in range(cfg.folds)
            ],
            "margins": cross_conformal_margins(scores, member, cfg.folds, cfg.alpha),
        }
    return out


def _interval_frame(
    q_members: np.ndarray, logits: np.ndarray, cal: dict, cfg: TrainConfig
) -> dict[str, np.ndarray]:
    """CV+ intervals and averaged probabilities for points no member trained on."""
    q = q_members.astype(np.float64) * cfg.max_rul  # (members, N, 3)
    lower, upper = cv_plus_interval(q[:, :, 0], q[:, :, 2], cal["scores"], cal["member"], cfg.alpha)
    lower, upper, open_ended = finalize_interval(lower, upper, cfg.max_rul)
    return {
        "point": np.clip(q[:, :, 1].mean(axis=0), 0, cfg.max_rul),
        "lower": lower,
        "upper": upper,
        "open_ended": open_ended,
        "probability": probability(logits, cal["temperature"]),
    }


def _rounded(frame: pd.DataFrame) -> pd.DataFrame:
    """Round once, before anything is computed, so metrics recomputed from the saved files
    match metrics.json exactly."""
    for column in ("point", "lower", "upper"):
        if column in frame:
            frame[column] = frame[column].round(2)
    if "probability" in frame:
        frame["probability"] = frame["probability"].round(4)
    return frame


def outputs(
    repeat: int,
    fit: dict,
    cal: dict,
    pool: Engines,
    replay: Engines,
    test_units: np.ndarray,
    test_lengths: np.ndarray,
    cfg: TrainConfig,
) -> dict[str, pd.DataFrame]:
    """Per-model prediction tables of one repeat: out-of-fold, test (no labels) and replay."""
    tables: dict[str, list[pd.DataFrame]] = {"oof": [], "test": [], "replay": [], "cal": []}
    fold = fit["fold_of_row"]
    for name in fit["models"]:
        c, r = cal[name], fit["raw"][name]
        q = r["oof_q"].astype(np.float64) * cfg.max_rul
        lower, upper, open_ended = finalize_interval(
            q[:, 0] - c["margins"][fold], q[:, 2] + c["margins"][fold], cfg.max_rul
        )
        temps = np.asarray(c["fold_temperatures"])[fold]
        tables["oof"].append(
            pd.DataFrame(
                {
                    "unit": pool.unit,
                    "cycle": pool.cycle,
                    "rul": pool.rul,
                    "repeat": repeat,
                    "model": name,
                    "fold": fold,
                    "point": np.clip(q[:, 1], 0, cfg.max_rul),
                    "lower": lower,
                    "upper": upper,
                    "open_ended": open_ended,
                    "probability": 1.0 / (1.0 + np.exp(-r["oof_logit"].astype(np.float64) / temps)),
                }
            )
        )
        test = _interval_frame(r["test_q"], r["test_logit"], c, cfg)
        tables["test"].append(
            pd.DataFrame(
                {"unit": test_units, "length": test_lengths, "repeat": repeat, "model": name} | test
            )
        )
        rep = _interval_frame(r["replay_q"], r["replay_logit"], c, cfg)
        tables["replay"].append(
            pd.DataFrame(
                {
                    "unit": replay.unit,
                    "cycle": replay.cycle,
                    "rul": replay.rul,
                    "repeat": repeat,
                    "model": name,
                }
                | rep
            )
        )
        rows = c["rows"]
        tables["cal"].append(
            pd.DataFrame(
                {
                    "repeat": repeat,
                    "model": name,
                    "unit": pool.unit[rows],
                    "cycle": pool.cycle[rows],
                    "rul": pool.rul[rows],
                    "member": c["member"],
                    "lower_raw": q[rows, 0],
                    "upper_raw": q[rows, 2],
                    "score": c["scores"],
                }
            )
        )
    return {key: _rounded(pd.concat(frames, ignore_index=True)) for key, frames in tables.items()}


def member_table(repeat: int, fit: dict, test_units: np.ndarray, cfg: TrainConfig) -> pd.DataFrame:
    """Every member's raw outputs on the test engines, so CV+ intervals can be recomputed."""
    frames = []
    for name in fit["models"]:
        q, logit = fit["raw"][name]["test_q"], fit["raw"][name]["test_logit"]
        for k in range(q.shape[0]):
            frames.append(
                pd.DataFrame(
                    {
                        "repeat": repeat,
                        "model": name,
                        "member": k,
                        "unit": test_units,
                        "lower_raw": q[k, :, 0].astype(np.float64) * cfg.max_rul,
                        "point_raw": q[k, :, 1].astype(np.float64) * cfg.max_rul,
                        "upper_raw": q[k, :, 2].astype(np.float64) * cfg.max_rul,
                        "logit": logit[k].astype(np.float64),
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------------------------
# Metrics that need no test labels


def whole_life_metrics(frame: pd.DataFrame, cfg: TrainConfig, seed: int) -> dict:
    """Interval and alert behaviour over every cycle of engines that run to failure."""
    capped = np.minimum(frame["rul"].to_numpy(), cfg.max_rul)
    weights = per_engine_weights(frame["unit"].to_numpy())
    inside = covered(
        frame["lower"].to_numpy(), frame["upper"].to_numpy(), frame["open_ended"].to_numpy(), capped
    )
    label = (frame["rul"].to_numpy() <= cfg.alert_cycles).astype(float)
    prob = frame["probability"].to_numpy()
    behaviour = alert_behaviour(frame, cfg.alert_threshold, cfg.alert_cycles)
    return {
        "engines": int(frame["unit"].nunique()),
        "cycles": len(frame),
        "rmse_per_engine_weighted": weighted_rmse(frame["point"].to_numpy(), capped, weights),
        "coverage_per_engine_weighted": float(np.sum(weights * inside) / np.sum(weights)),
        "open_ended_share": float(frame["open_ended"].mean()),
        "alert": {k: v for k, v in behaviour.items() if k != "per_unit"},
        "alert_per_unit": behaviour["per_unit"],
        "calibration": ece_with_interval(prob, label, frame["unit"].to_numpy(), seed=seed),
        "reliability": reliability_curve(prob, label),
    }


SUMMARY_KEYS = {
    "oof": [
        ("rmse_per_engine_weighted",),
        ("coverage_per_engine_weighted",),
        ("open_ended_share",),
        ("alert", "lead_median"),
        ("alert", "missed"),
        ("alert", "early"),
        ("alert", "early_twice_horizon"),
        ("alert", "false_alerts_per_engine"),
        ("alert", "flaps_per_engine"),
        ("calibration", "ece"),
    ],
}


def _get(d: dict, path: tuple[str, ...]) -> Any:
    for key in path:
        d = d[key]
    return d


def summarise(per_repeat: dict[str, dict], paths: list[tuple[str, ...]]) -> dict:
    """mean ± sd over repeats for each metric path."""
    return {".".join(path): mean_sd([_get(m, path) for m in per_repeat.values()]) for path in paths}


# --------------------------------------------------------------------------------------------
# The command


def run(cfg: TrainConfig, settings: Settings, out_dir: Path, ledger: Path | None = LEDGER) -> dict:
    started = time.time()
    device = resolve_device(cfg.device)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions").mkdir(exist_ok=True)
    git = git_state()

    train_all = cmapss.load_train(settings.data_dir, cfg.subset)
    test_inputs = cmapss.load_test_inputs(settings.data_dir, cfg.subset)
    lives_all = train_all.groupby("unit")["cycle"].max()
    pool_units, replay_units = cmapss.stratified_subset(
        lives_all.index.to_numpy(), lives_all.to_numpy(), cfg.replay_fraction, cfg.data_seed
    )
    pool_frame = train_all[train_all["unit"].isin(pool_units)].reset_index(drop=True)
    replay_frame = train_all[train_all["unit"].isin(replay_units)].reset_index(drop=True)
    contract = FeatureContract.fit(
        pool_frame,
        cfg.subset,
        max(cfg.windows),
        cfg.max_rul,
        cfg.alert_cycles,
        seed=cfg.data_seed,
    )
    pool = engine_table(pool_frame, contract)
    replay = engine_table(replay_frame, contract)
    test_windows, test_units = contract.last_windows(test_inputs)
    test_lengths = test_inputs.groupby("unit")["cycle"].max().loc[test_units].to_numpy()
    print(
        f"{cfg.subset}: {len(pool_units)} engines in {cfg.folds} folds, {len(replay_units)} "
        f"held out for the replay, {len(test_units)} test engines, device {device}",
        flush=True,
    )

    served: dict | None = None
    tables: dict[str, list[pd.DataFrame]] = {"oof": [], "test": [], "replay": [], "cal": []}
    members, selection, histories, calibration_summary = [], [], {}, {}
    folds_by_repeat: dict[str, dict[str, int]] = {}
    inner_by_repeat: dict[str, dict[str, list[int]]] = {}
    for repeat in cfg.repeats:
        fit = crossfit(repeat, pool, test_windows, replay, cfg, device)
        folds_by_repeat[str(repeat)] = {str(u): f for u, f in sorted(fit["folds"].items())}
        inner_by_repeat[str(repeat)] = fit["inner_validation"]
        cal = calibrate(repeat, fit, pool, test_lengths, cfg)
        for key, frame in outputs(
            repeat, fit, cal, pool, replay, test_units, test_lengths, cfg
        ).items():
            tables[key].append(frame)
        members.append(member_table(repeat, fit, test_units, cfg))
        selection += fit["selection"]
        histories.update(fit["histories"])
        calibration_summary[str(repeat)] = {
            name: {
                "calibration_engines": len(c["scores"]),
                "temperature": c["temperature"],
                "fold_temperatures": c["fold_temperatures"],
                "cross_conformal_margins": c["margins"].tolist(),
                "mean_score": float(np.mean(c["scores"])),
            }
            for name, c in cal.items()
        }
        if served is None:
            served = {"repeat": repeat, "fit": fit, "cal": cal["lstm"]}
    assert served is not None
    frames = {key: pd.concat(parts, ignore_index=True) for key, parts in tables.items()}

    # Whole-life metrics, per model and repeat; nothing here touches the test labels.
    oof_metrics: dict[str, Any] = {}
    replay_metrics: dict[str, Any] = {}
    models = sorted(frames["oof"]["model"].unique(), key=MODELS.index)
    for name in models:
        per_repeat = {
            str(r): whole_life_metrics(g, cfg, seed=r)
            for r, g in frames["oof"][frames["oof"]["model"] == name].groupby("repeat")
        }
        oof_metrics[name] = {
            "per_repeat": per_repeat,
            "summary": summarise(per_repeat, SUMMARY_KEYS["oof"]),
        }
        per_repeat = {
            str(r): whole_life_metrics(g, cfg, seed=r)
            for r, g in frames["replay"][frames["replay"]["model"] == name].groupby("repeat")
        }
        replay_metrics[name] = {
            "per_repeat": per_repeat,
            "summary": summarise(per_repeat, SUMMARY_KEYS["oof"]),
        }

    # Artifacts: the served model is the first repeat's LSTM networks.
    networks = served["fit"]["networks"]
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "repeat": served["repeat"],
            "members": [{"config": n.config, "state": n.state_dict()} for n in networks],
        },
        out_dir / "model.pt",
    )
    contract.save(out_dir / "contract.json")
    c = served["cal"]
    calibration = {
        "format": CALIBRATION_FORMAT,
        "method": "CV+ conformalized quantile regression; temperature-scaled alert",
        "repeat": served["repeat"],
        "interval": 1 - cfg.alpha,
        "alpha": cfg.alpha,
        "scores": [round(float(s), 6) for s in c["scores"]],
        "score_member": [int(m) for m in c["member"]],
        "temperature": c["temperature"],
        "alert_threshold": cfg.alert_threshold,
    }
    (out_dir / "calibration.json").write_text(json.dumps(calibration, indent=1))
    (out_dir / "history.json").write_text(json.dumps(histories))
    predictions = out_dir / "predictions"
    frames["test"].to_csv(predictions / "test.csv", index=False)
    pd.concat(members, ignore_index=True).to_csv(
        predictions / "test-members.csv", index=False, float_format="%.6g"
    )
    frames["cal"].to_csv(predictions / "calibration-points.csv", index=False)
    frames["oof"].to_csv(predictions / "oof.csv.gz", index=False, compression="gzip")
    frames["replay"].to_csv(predictions / "replay-holdout.csv.gz", index=False, compression="gzip")
    replay_frame.to_csv(out_dir / "replay.csv", index=False)
    for stale in ("holdout.csv",):
        (predictions / stale).unlink(missing_ok=True)

    metrics = {
        "subset": cfg.subset,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "training_seconds": round(time.time() - started, 1),
        "provenance": {
            "predmaint": __version__,
            "source_sha256": source_digest(),
            "git": git,
            "data": {
                "source": cmapss.SOURCE_URL,
                "archive_sha256": cmapss.OUTER_SHA256,
                "files_sha256": cmapss.file_digests(
                    settings.data_dir, [f"train_{cfg.subset}.txt", f"test_{cfg.subset}.txt"]
                ),
            },
            "environment": environment(device),
            "test_labels_read": False,
        },
        "config": asdict(cfg),
        "data": {
            "train_engines": int(train_all["unit"].nunique()),
            "crossfit_engines": len(pool_units),
            "replay_engines": [int(u) for u in replay_units],
            "test_engines": len(test_units),
            "folds": folds_by_repeat,
            "inner_validation": inner_by_repeat,
        },
        "selection": selection,
        "calibration": calibration_summary,
        "served": {
            "model": "lstm",
            "repeat": served["repeat"],
            "networks": len(networks),
            "windows": [int(n.window) for n in networks],
        },
        "out_of_fold": oof_metrics,
        "replay_holdout": replay_metrics,
        "test": None,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    if ledger is not None:
        append_ledger(
            ledger,
            {
                "event": "train",
                "at": metrics["created_at"],
                "subset": cfg.subset,
                "out": ledger_path(out_dir),
                "git_commit": git["commit"] if git else None,
                "git_dirty": git["dirty"] if git else None,
                "config": asdict(cfg),
                "selection": selection,
                "test_viewed": False,
            },
        )
    return metrics


def main(argv: list[str] | None = None) -> int:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(
        prog="python -m predmaint.train",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--subset", default=defaults.subset, choices=cmapss.SUBSETS)
    parser.add_argument("--repeats", type=int, nargs="+", default=defaults.repeats)
    parser.add_argument("--folds", type=int, default=defaults.folds)
    parser.add_argument("--windows", type=int, nargs="+", default=defaults.windows)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument("--hidden", type=int, default=defaults.hidden)
    parser.add_argument("--no-baseline", action="store_true", help="skip LightGBM")
    parser.add_argument("--out", type=Path, default=None, help="default: artifacts/<subset>")
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    parser.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    parser.add_argument("--device", default=defaults.device, choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="train even if src/, pyproject.toml or uv.lock differ from the git commit",
    )
    args = parser.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    git = git_state()
    if git and git["dirty"] and not args.allow_dirty:
        raise SystemExit(
            "Refusing to train from uncommitted sources (metrics.json would not name the code "
            f"that produced it): {git['changed']}. Commit first, or pass --allow-dirty."
        )

    settings = Settings.from_env()
    cmapss.check(settings.data_dir)
    cfg = TrainConfig(
        subset=args.subset,
        repeats=args.repeats,
        folds=args.folds,
        windows=sorted(args.windows),
        epochs=args.epochs,
        patience=args.patience,
        hidden=args.hidden,
        baseline=not args.no_baseline,
        device=args.device,
    )
    out_dir = args.out or settings.artifacts_dir / args.subset
    metrics = run(cfg, settings, out_dir, ledger=args.ledger)
    for name, block in metrics["out_of_fold"].items():
        s = block["summary"]
        print(
            f"{cfg.subset} {name} out-of-fold: RMSE {s['rmse_per_engine_weighted']['mean']:.2f}, "
            f"coverage {s['coverage_per_engine_weighted']['mean']:.1%}, "
            f"median first alert {s['alert.lead_median']['mean']} cycles before failure"
        )
    print(f"Artifacts written to {out_dir} in {metrics['training_seconds']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
