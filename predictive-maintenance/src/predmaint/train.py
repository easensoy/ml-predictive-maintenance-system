"""Train, calibrate and evaluate the RUL model on one C-MAPSS subset.

    python -m predmaint.train --subset FD001
    python -m predmaint.train --subset FD004 --seeds 0 1 2 --epochs 60

Protocol
--------
* Training engines are split **by engine** into train (70%), validation (15%) and holdout (15%).
  No engine contributes windows to more than one split.
* The feature contract (operating-regime clusters and sensor scaling) is fitted on the train
  split only.
* Each ensemble member early-stops on validation RMSE. Conformal margins and the alert
  temperature are then fitted on the validation engines.
* The official test set is touched once, at the end, using the last window of each engine.
* Holdout engines run to failure, so they measure alert lead time; the dashboard replays them.
"""

import argparse
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from predmaint import __version__
from predmaint import data as cmapss
from predmaint.calibration import calibrated_outputs, conformal_margin, fit_temperature
from predmaint.config import PROJECT_ROOT, Settings
from predmaint.features import FeatureContract
from predmaint.metrics import (
    alert_metrics,
    interval_coverage,
    lead_times,
    mae,
    nasa_score,
    reliability_curve,
    rmse,
)
from predmaint.model import Ensemble, RULNet, multitask_loss, predict_raw


@dataclass
class TrainConfig:
    subset: str = "FD001"
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    split_seed: int = 42
    window: int = 60
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
    ewma_alpha: float | None = None
    baseline: bool = True
    device: str = "auto"


def _targets(frame: pd.DataFrame, max_rul: int, alert_cycles: int) -> tuple[np.ndarray, np.ndarray]:
    rul = frame["rul"].to_numpy()
    capped = np.minimum(rul, max_rul) / max_rul
    return capped.astype(np.float32), (rul <= alert_cycles).astype(np.float32)


def source_digest() -> str:
    """SHA-256 of the predmaint sources (line endings normalised): names the exact training code."""
    digest = hashlib.sha256()
    package = PROJECT_ROOT / "src" / "predmaint"
    for path in sorted(package.rglob("*.py")):
        digest.update(path.relative_to(package).as_posix().encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()[:16]


def _git_commit() -> str | None:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", "src"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except OSError, subprocess.SubprocessError:
        return None
    return f"{commit}{'-dirty' if dirty else ''}"


def _device_name(device: torch.device) -> str:
    return torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but this PyTorch build has no CUDA support")
    return torch.device(name)


def train_member(
    seed: int,
    train_xy: tuple[np.ndarray, np.ndarray, np.ndarray],
    val_xy: tuple[np.ndarray, np.ndarray, np.ndarray],
    cfg: TrainConfig,
    n_features: int,
) -> tuple[RULNet, list[dict]]:
    """Train one ensemble member; it is returned on the CPU, the device that serves it."""
    device = resolve_device(cfg.device)
    torch.manual_seed(seed)  # weight init and dropout; shuffling uses its own seeded generator
    shuffle = torch.Generator().manual_seed(seed)
    model = RULNet(n_features, cfg.hidden, cfg.layers, cfg.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4)
    x, y, a = (torch.from_numpy(arr).to(device) for arr in train_xy)
    x_val, y_val, _ = val_xy
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
        val_point = predict_raw(model, x_val)["quantiles"][:, 1]
        val_rmse = rmse(val_point * cfg.max_rul, y_val * cfg.max_rul)
        scheduler.step(val_rmse)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total / len(train_xy[0]),
                "val_rmse": val_rmse,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        if val_rmse < best_rmse - 1e-4:
            best_rmse, best_epoch = val_rmse, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= cfg.patience:
            break
    model = model.cpu()
    model.load_state_dict(best_state)
    model.eval()
    print(f"  seed {seed}: best val RMSE {best_rmse:.2f} at epoch {best_epoch}/{len(history)}")
    return model, history


def window_summary(windows: np.ndarray) -> np.ndarray:
    """Hand-crafted features for the gradient-boosting baseline: last, mean, std, slope."""
    t = np.arange(windows.shape[1], dtype=np.float64)
    t -= t.mean()
    slope = (windows * t[None, :, None]).sum(axis=1) / (t**2).sum()
    return np.concatenate([windows[:, -1], windows.mean(1), windows.std(1), slope], axis=1)


def lightgbm_baseline(train_xy, val_xy, test_x, cfg: TrainConfig) -> np.ndarray | None:
    try:
        import lightgbm as lgb
    except ImportError:
        print("  lightgbm not installed; skipping baseline")
        return None
    model = lgb.LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=0,
        verbose=-1,
    )
    model.fit(
        window_summary(train_xy[0]),
        train_xy[1] * cfg.max_rul,
        eval_X=(window_summary(val_xy[0]),),
        eval_y=(val_xy[1] * cfg.max_rul,),
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return np.clip(model.predict(window_summary(test_x)), 0, cfg.max_rul)


def regression_report(pred: np.ndarray, true: np.ndarray, max_rul: int) -> dict:
    capped = np.minimum(true, max_rul)
    return {
        "rmse": rmse(pred, capped),
        "mae": mae(pred, capped),
        "nasa_score": nasa_score(pred, capped),
        "rmse_uncapped": rmse(pred, true),
        "nasa_score_uncapped": nasa_score(pred, true),
    }


def run(cfg: TrainConfig, settings: Settings, out_dir: Path) -> dict:
    started = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions").mkdir(exist_ok=True)

    train_all = cmapss.load_train(settings.data_dir, cfg.subset)
    test = cmapss.load_test(settings.data_dir, cfg.subset)
    split = cmapss.split_units(train_all["unit"].to_numpy(), seed=cfg.split_seed)
    parts = {
        name: train_all[train_all["unit"].isin(units)].reset_index(drop=True)
        for name, units in split.items()
    }
    contract = FeatureContract.fit(
        parts["train"],
        cfg.subset,
        cfg.window,
        cfg.max_rul,
        cfg.alert_cycles,
        seed=cfg.split_seed,
        ewma_alpha=cfg.ewma_alpha,
    )

    xy = {}
    for name in ("train", "val", "holdout"):
        windows = contract.unit_windows(parts[name])
        xy[name] = (windows, *_targets(parts[name], cfg.max_rul, cfg.alert_cycles))
    test_x, test_units = contract.last_windows(test)
    test_true = test.groupby("unit", sort=True)["rul"].min().loc[test_units].to_numpy()
    print(
        f"{cfg.subset}: {len(split['train'])}/{len(split['val'])}/{len(split['holdout'])} "
        f"train/val/holdout engines, {len(xy['train'][0])} training windows, "
        f"{len(test_units)} test engines"
    )

    members, histories = [], {}
    for seed in cfg.seeds:
        model, history = train_member(seed, xy["train"], xy["val"], cfg, contract.n_features)
        members.append(model)
        histories[str(seed)] = history
    ensemble = Ensemble(members).eval()

    # Calibration on validation engines.
    val_raw = predict_raw(ensemble, xy["val"][0])
    margin = conformal_margin(
        val_raw["quantiles"][:, 0], val_raw["quantiles"][:, 2], xy["val"][1], cfg.alpha
    )
    temperature = fit_temperature(val_raw["alert_logit"], xy["val"][2])
    calibration = {
        "interval": 1 - cfg.alpha,
        "conformal_margin": margin,
        "temperature": temperature,
        "alert_threshold": cfg.alert_threshold,
    }

    def calibrated(windows: np.ndarray) -> dict[str, np.ndarray]:
        raw = predict_raw(ensemble, windows)
        out = calibrated_outputs(
            raw["quantiles"], raw["alert_logit"], margin, temperature, cfg.max_rul
        )
        out["attention"] = raw["attention"]
        return out

    # Official test set: last window per engine.
    test_out = calibrated(test_x)
    test_capped = np.minimum(test_true, cfg.max_rul)
    test_report = regression_report(test_out["point"], test_true, cfg.max_rul)
    test_report["interval_coverage"] = interval_coverage(
        test_out["lower"], test_out["upper"], test_capped
    )
    test_report["interval_width"] = float(np.mean(test_out["upper"] - test_out["lower"]))
    test_report["alert"] = alert_metrics(
        test_out["probability"], test_true <= cfg.alert_cycles, cfg.alert_threshold
    )
    single = [
        regression_report(
            np.clip(predict_raw(m, test_x)["quantiles"][:, 1] * cfg.max_rul, 0, cfg.max_rul),
            test_true,
            cfg.max_rul,
        )
        for m in members
    ]
    single_summary = {
        f"{metric}_{stat}": float(fn([s[metric] for s in single]))
        for metric in ("rmse", "nasa_score")
        for stat, fn in (("mean", np.mean), ("std", np.std))
    }

    baseline_report = None
    if cfg.baseline:
        baseline_pred = lightgbm_baseline(xy["train"], xy["val"], test_x, cfg)
        if baseline_pred is not None:
            baseline_report = regression_report(baseline_pred, test_true, cfg.max_rul)

    # Holdout engines: every cycle, for lead time and interval coverage along whole lives.
    hold = parts["holdout"]
    hold_out = calibrated(xy["holdout"][0])
    hold_frame = pd.DataFrame(
        {
            "unit": hold["unit"],
            "cycle": hold["cycle"],
            "rul_true": hold["rul"],
            "rul_capped": np.minimum(hold["rul"], cfg.max_rul),
            "point": hold_out["point"],
            "lower": hold_out["lower"],
            "upper": hold_out["upper"],
            "probability": hold_out["probability"],
        }
    )
    hold_report = {
        "rmse_all_cycles": rmse(hold_frame["point"], hold_frame["rul_capped"]),
        "interval_coverage": interval_coverage(
            hold_frame["lower"], hold_frame["upper"], hold_frame["rul_capped"]
        ),
        "alert": alert_metrics(
            hold_frame["probability"].to_numpy(),
            hold_frame["rul_true"].to_numpy() <= cfg.alert_cycles,
            cfg.alert_threshold,
        ),
        "reliability": reliability_curve(
            hold_frame["probability"].to_numpy(),
            (hold_frame["rul_true"].to_numpy() <= cfg.alert_cycles).astype(float),
        ),
        "lead_time": lead_times(hold_frame, cfg.alert_threshold, early_cycles=2 * cfg.alert_cycles),
    }

    # Persist everything the API, the dashboard and the figures need.
    torch.save(
        {"format": 1, "config": members[0].config, "members": [m.state_dict() for m in members]},
        out_dir / "model.pt",
    )
    contract.save(out_dir / "contract.json")
    (out_dir / "calibration.json").write_text(json.dumps(calibration, indent=2))
    (out_dir / "history.json").write_text(json.dumps(histories, indent=2))
    pd.DataFrame(
        {
            "unit": test_units,
            "rul_true": test_true,
            "rul_capped": test_capped,
            "point": test_out["point"],
            "lower": test_out["lower"],
            "upper": test_out["upper"],
            "probability": test_out["probability"],
        }
    ).to_csv(out_dir / "predictions" / "test.csv", index=False, float_format="%.4f")
    hold_frame.to_csv(out_dir / "predictions" / "holdout.csv", index=False, float_format="%.4f")
    hold.to_csv(out_dir / "replay.csv", index=False)

    metrics = {
        "subset": cfg.subset,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "training_seconds": round(time.time() - started, 1),
        "code": {
            "predmaint": __version__,
            "source_sha256": source_digest(),
            "git_commit": _git_commit(),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "platform": platform.platform(),
            "training_device": _device_name(resolve_device(cfg.device)),
        },
        "data": {
            "source": cmapss.SOURCE_URL,
            "archive_sha256": cmapss.OUTER_SHA256,
            "train_engines": int(train_all["unit"].nunique()),
            "test_engines": len(test_units),
            "split_seed": cfg.split_seed,
            "split_engines": {name: [int(u) for u in units] for name, units in split.items()},
        },
        "config": asdict(cfg),
        "calibration": calibration,
        "test": test_report,
        "test_single_models": single_summary,
        "baseline_lightgbm": baseline_report,
        "holdout": hold_report,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main(argv: list[str] | None = None) -> int:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(
        prog="python -m predmaint.train",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--subset", default=defaults.subset, choices=cmapss.SUBSETS)
    parser.add_argument("--seeds", type=int, nargs="+", default=defaults.seeds)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument("--window", type=int, default=defaults.window)
    parser.add_argument("--hidden", type=int, default=defaults.hidden)
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument(
        "--ewma-alpha",
        type=float,
        default=defaults.ewma_alpha,
        help="causal input smoothing weight on the newest cycle (default off)",
    )
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--out", type=Path, default=None, help="default: artifacts/<subset>")
    parser.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    parser.add_argument("--device", default=defaults.device, choices=["auto", "cpu", "cuda"])
    args = parser.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)

    settings = Settings.from_env()
    cmapss.check(settings.data_dir)
    cfg = TrainConfig(
        subset=args.subset,
        seeds=args.seeds,
        epochs=args.epochs,
        patience=args.patience,
        window=args.window,
        hidden=args.hidden,
        dropout=args.dropout,
        lr=args.lr,
        ewma_alpha=args.ewma_alpha,
        baseline=not args.no_baseline,
        device=args.device,
    )
    out_dir = args.out or settings.artifacts_dir / args.subset
    metrics = run(cfg, settings, out_dir)

    t = metrics["test"]
    ap = t["alert"]["average_precision"]
    print(
        f"\n{cfg.subset} test ({metrics['data']['test_engines']} engines): "
        f"RMSE {t['rmse']:.2f}  score {t['nasa_score']:.0f}  "
        f"{t['interval_coverage']:.0%} inside 90% interval (mean width {t['interval_width']:.1f})  "
        f"alert AP {'n/a' if ap is None else f'{ap:.3f}'}"
    )
    if metrics["baseline_lightgbm"]:
        b = metrics["baseline_lightgbm"]
        print(f"LightGBM baseline: RMSE {b['rmse']:.2f}  score {b['nasa_score']:.0f}")
    lt = metrics["holdout"]["lead_time"]
    print(
        f"Holdout alert lead time: median {lt['median']} cycles "
        f"(IQR {lt['p25']}-{lt['p75']}), missed {lt['missed']}/{lt['units']}"
    )
    print(f"Artifacts written to {out_dir} in {metrics['training_seconds']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
