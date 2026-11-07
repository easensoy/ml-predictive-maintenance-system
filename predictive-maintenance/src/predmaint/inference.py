"""Load trained artifacts and turn engine readings into calibrated, explained predictions."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from predmaint.calibration import calibrated_outputs
from predmaint.data import SENSOR_INFO
from predmaint.features import ContractError, FeatureContract
from predmaint.model import Ensemble, RULNet, integrated_gradients, predict_raw

ARTIFACT_FILES = ("model.pt", "contract.json", "calibration.json", "metrics.json")
# Bump when the layout of model.pt changes; loading another format fails loudly.
CHECKPOINT_FORMAT = 1
RETRAIN_HINT = (
    "Retrain with `python -m predmaint.train`, or check out the git commit recorded in the "
    "run's metrics.json"
)


class ArtifactsMissing(FileNotFoundError):
    pass


class ArtifactsIncompatible(RuntimeError):
    """The artifacts exist but cannot be served safely: wrong format, shape or values."""


class NonFinitePrediction(ValueError):
    """The model produced NaN or infinity, which only happens for absurd inputs."""

    def __init__(self, message: str, report: dict):
        super().__init__(message)
        self.report = report


def _load_members(path: Path, n_features: int) -> list[RULNet]:
    try:
        bundle = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ArtifactsIncompatible(f"{path} cannot be read as a checkpoint: {exc}") from exc
    if not isinstance(bundle, dict) or not {"format", "config", "members"} <= bundle.keys():
        raise ArtifactsIncompatible(f"{path} is not a predmaint checkpoint. {RETRAIN_HINT}")
    if bundle["format"] != CHECKPOINT_FORMAT:
        raise ArtifactsIncompatible(
            f"{path} has checkpoint format {bundle['format']}, but this version of predmaint "
            f"reads format {CHECKPOINT_FORMAT}. {RETRAIN_HINT}"
        )
    config = bundle["config"]
    if not isinstance(config, dict) or config.get("n_features") != n_features:
        raise ArtifactsIncompatible(
            f"{path} was trained on {config.get('n_features') if isinstance(config, dict) else '?'}"
            f" features, but contract.json defines {n_features}: the two files come from "
            f"different runs. {RETRAIN_HINT}"
        )
    if not bundle["members"]:
        raise ArtifactsIncompatible(f"{path} holds no networks. {RETRAIN_HINT}")
    members = []
    for i, state in enumerate(bundle["members"]):
        try:
            member = RULNet(**config)
            member.load_state_dict(state)
        except (TypeError, RuntimeError) as exc:
            raise ArtifactsIncompatible(
                f"{path}: network {i} does not match its configuration: {exc}. {RETRAIN_HINT}"
            ) from exc
        members.append(member.eval())
    return members


def _check_calibration(path: Path, calibration: object) -> dict:
    expected = ("interval", "conformal_margin", "temperature", "alert_threshold")
    if not isinstance(calibration, dict) or not all(key in calibration for key in expected):
        raise ArtifactsIncompatible(f"{path} must define {list(expected)}. {RETRAIN_HINT}")
    values = [calibration[key] for key in expected]
    if not all(isinstance(v, int | float) and math.isfinite(v) for v in values):
        raise ArtifactsIncompatible(f"{path}: every value must be a finite number")
    interval, _, temperature, threshold = values
    if not (0 < interval < 1 and temperature > 0 and 0 < threshold < 1):
        raise ArtifactsIncompatible(
            f"{path}: need 0 < interval < 1, temperature > 0 and 0 < alert_threshold < 1"
        )
    return calibration


class Predictor:
    def __init__(self, model_dir: Path):
        missing = [name for name in ARTIFACT_FILES if not (model_dir / name).is_file()]
        if missing:
            raise ArtifactsMissing(
                f"Missing {missing} in {model_dir}. Train first: python -m predmaint.train"
            )
        self.model_dir = model_dir
        try:
            self.contract = FeatureContract.load(model_dir / "contract.json")
        except ContractError as exc:
            raise ArtifactsIncompatible(str(exc)) from exc
        calibration_path = model_dir / "calibration.json"
        self.calibration = _check_calibration(
            calibration_path, json.loads(calibration_path.read_text())
        )
        self.metrics = json.loads((model_dir / "metrics.json").read_text())
        members = _load_members(model_dir / "model.pt", self.contract.n_features)
        self.model = Ensemble(members).eval()

    @property
    def alert_threshold(self) -> float:
        return float(self.calibration["alert_threshold"])

    def predict_windows(self, windows: np.ndarray) -> dict[str, np.ndarray]:
        raw = predict_raw(self.model, windows)
        out = calibrated_outputs(
            raw["quantiles"],
            raw["alert_logit"],
            self.calibration["conformal_margin"],
            self.calibration["temperature"],
            self.contract.max_rul,
        )
        out["attention"] = raw["attention"]
        return out

    def explain(self, window: np.ndarray) -> np.ndarray:
        """Integrated-gradients attribution of the point RUL, in cycles: shape (window, F)."""
        attributions = integrated_gradients(self.model, torch.from_numpy(window.astype(np.float32)))
        return attributions.detach().numpy() * self.contract.max_rul

    def status(self, probability: float, lower: float) -> str:
        """alert: failure within the horizon is more likely than not.
        watch: failure within the horizon is inside the 90% interval.
        healthy: otherwise."""
        if probability >= self.alert_threshold:
            return "alert"
        if lower <= self.contract.alert_cycles:
            return "watch"
        return "healthy"

    def sensor_contributions(self, attributions: np.ndarray, top: int | None = None) -> list[dict]:
        totals = attributions.sum(axis=0)
        order = np.argsort(-np.abs(totals))
        rows = []
        for i in order[:top] if top else order:
            name = self.contract.sensors[i]
            symbol, description, unit = SENSOR_INFO[name]
            rows.append(
                {
                    "sensor": name,
                    "symbol": symbol,
                    "description": description,
                    "unit": unit,
                    "contribution_cycles": round(float(totals[i]), 2),
                }
            )
        return rows

    def predict_readings(self, readings: pd.DataFrame, explain: bool = True) -> dict:
        """Predict for one engine from its readings, oldest cycle first.

        The response always says whether the readings in the model's window lie outside the
        training range; such predictions are extrapolations. Raises ``NonFinitePrediction``
        (with that report attached) when the inputs are so far out that the network overflows.
        """
        ood = self.contract.out_of_distribution(readings.iloc[-self.contract.window :])
        with np.errstate(over="ignore", invalid="ignore"):
            window = self.contract.window_from_readings(readings)
            out = self.predict_windows(window[None])
        if not all(np.isfinite(out[key][0]) for key in ("lower", "point", "upper", "probability")):
            raise NonFinitePrediction(
                "The readings are too far outside the training range for a finite prediction", ood
            )
        result = self.format_prediction({key: value[0] for key, value in out.items()})
        result["cycles_received"] = len(readings)
        result["window_padded"] = len(readings) < self.contract.window
        result["out_of_distribution"] = ood
        if explain:
            with np.errstate(over="ignore", invalid="ignore"):
                attributions = self.explain(window)
            if not np.isfinite(attributions).all():
                result["explanation"] = None
                return result
            result["explanation"] = {
                "method": "integrated gradients vs. an average engine in the same operating regime",
                "baseline_rul_cycles": round(self._baseline_point(), 1),
                "sensors": self.sensor_contributions(attributions),
            }
        return result

    def format_prediction(self, row: dict) -> dict:
        probability = float(row["probability"])
        lower = float(row["lower"])
        return {
            "rul": {
                "point": round(float(row["point"]), 1),
                "lower": round(lower, 1),
                "upper": round(float(row["upper"]), 1),
                "interval": self.calibration["interval"],
                "capped_at": self.contract.max_rul,
            },
            "failure_within_horizon": {
                "horizon_cycles": self.contract.alert_cycles,
                "probability": round(probability, 4),
                "alert": probability >= self.alert_threshold,
            },
            "status": self.status(probability, lower),
        }

    def _baseline_point(self) -> float:
        zeros = np.zeros((1, self.contract.window, self.contract.n_features), np.float32)
        return float(self.predict_windows(zeros)["point"][0])

    def info(self) -> dict:
        m = self.metrics
        return {
            "subset": m["subset"],
            "created_at": m["created_at"],
            "code": m["code"],
            "data": {k: v for k, v in m["data"].items() if k != "split_engines"},
            "ensemble_size": len(self.model.members),
            "window": self.contract.window,
            "max_rul": self.contract.max_rul,
            "alert_cycles": self.contract.alert_cycles,
            "alert_threshold": self.alert_threshold,
            "interval": self.calibration["interval"],
            "sensors": [
                {"sensor": s, "symbol": SENSOR_INFO[s][0], "description": SENSOR_INFO[s][1]}
                for s in self.contract.sensors
            ],
            "test": m["test"],
            "test_single_models": m["test_single_models"],
            "baseline_lightgbm": m["baseline_lightgbm"],
            "holdout": {k: v for k, v in m["holdout"].items() if k != "reliability"}
            | {
                "lead_time": {k: v for k, v in m["holdout"]["lead_time"].items() if k != "per_unit"}
            },
        }
