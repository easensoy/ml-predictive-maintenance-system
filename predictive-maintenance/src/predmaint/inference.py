"""Load trained artifacts and turn engine readings into calibrated, explained predictions."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from predmaint.calibration import calibrated_outputs
from predmaint.data import SENSOR_INFO
from predmaint.features import FeatureContract
from predmaint.model import Ensemble, RULNet, integrated_gradients, predict_raw

ARTIFACT_FILES = ("model.pt", "contract.json", "calibration.json", "metrics.json")


class ArtifactsMissing(FileNotFoundError):
    pass


class Predictor:
    def __init__(self, model_dir: Path):
        missing = [name for name in ARTIFACT_FILES if not (model_dir / name).is_file()]
        if missing:
            raise ArtifactsMissing(
                f"Missing {missing} in {model_dir}. Train first: python -m predmaint.train"
            )
        self.model_dir = model_dir
        self.contract = FeatureContract.load(model_dir / "contract.json")
        self.calibration = json.loads((model_dir / "calibration.json").read_text())
        self.metrics = json.loads((model_dir / "metrics.json").read_text())
        bundle = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
        members = []
        for state in bundle["members"]:
            member = RULNet(**bundle["config"])
            member.load_state_dict(state)
            members.append(member.eval())
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
        """Predict for one engine from its readings, oldest cycle first."""
        window = self.contract.window_from_readings(readings)
        out = self.predict_windows(window[None])
        result = self.format_prediction({key: value[0] for key, value in out.items()})
        result["cycles_received"] = len(readings)
        result["window_padded"] = len(readings) < self.contract.window
        if explain:
            attributions = self.explain(window)
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
