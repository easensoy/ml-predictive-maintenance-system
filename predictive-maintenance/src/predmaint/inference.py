"""Load trained artifacts and turn engine readings into calibrated, explained predictions."""

import hashlib
import json
import math
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from predmaint.calibration import cv_plus_interval, finalize_interval, probability
from predmaint.data import SENSOR_INFO
from predmaint.features import ContractError, FeatureContract
from predmaint.model import CHECKPOINT_FORMAT, Ensemble, RULNet, integrated_gradients, predict_raw

ARTIFACT_FILES = ("model.pt", "contract.json", "calibration.json", "metrics.json")
CALIBRATION_FORMAT = 2
# Attribution maps kept per predictor, keyed by the exact bytes of the window: the replay asks
# again for an engine whose window has not moved (a new watcher, a refresh, a failed engine
# waiting to restart), and integrated gradients dominates the cost of serving one.
EXPLANATION_CACHE_SIZE = 64
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


def _load_members(path: Path, contract: FeatureContract) -> list[RULNet]:
    try:
        bundle = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ArtifactsIncompatible(f"{path} cannot be read as a checkpoint: {exc}") from exc
    if not isinstance(bundle, dict) or not {"format", "members"} <= bundle.keys():
        raise ArtifactsIncompatible(f"{path} is not a predmaint checkpoint. {RETRAIN_HINT}")
    if bundle["format"] != CHECKPOINT_FORMAT:
        raise ArtifactsIncompatible(
            f"{path} has checkpoint format {bundle['format']}, but this version of predmaint "
            f"reads format {CHECKPOINT_FORMAT}. {RETRAIN_HINT}"
        )
    if not bundle["members"]:
        raise ArtifactsIncompatible(f"{path} holds no networks. {RETRAIN_HINT}")
    members = []
    for i, entry in enumerate(bundle["members"]):
        config = entry.get("config") if isinstance(entry, dict) else None
        if not isinstance(config, dict) or config.get("n_features") != contract.n_features:
            raise ArtifactsIncompatible(
                f"{path}: network {i} does not take the {contract.n_features} features that "
                f"contract.json defines, so the two files come from different runs. {RETRAIN_HINT}"
            )
        window = config.get("window")
        if not isinstance(window, int) or not 1 <= window <= contract.window:
            raise ArtifactsIncompatible(
                f"{path}: network {i} needs a {window}-cycle window, but contract.json provides "
                f"{contract.window} cycles. {RETRAIN_HINT}"
            )
        try:
            member = RULNet(**config)
            member.load_state_dict(entry["state"])
        except (TypeError, KeyError, RuntimeError) as exc:
            raise ArtifactsIncompatible(
                f"{path}: network {i} does not match its configuration: {exc}. {RETRAIN_HINT}"
            ) from exc
        members.append(member.eval())
    return members


def _check_calibration(path: Path, calibration: object, n_members: int) -> dict:
    expected = ("format", "interval", "alpha", "scores", "score_member", "temperature")
    if not isinstance(calibration, dict) or not all(key in calibration for key in expected):
        raise ArtifactsIncompatible(f"{path} must define {list(expected)}. {RETRAIN_HINT}")
    if calibration["format"] != CALIBRATION_FORMAT:
        raise ArtifactsIncompatible(
            f"{path} has calibration format {calibration['format']}, but this version of "
            f"predmaint reads format {CALIBRATION_FORMAT}. {RETRAIN_HINT}"
        )
    scalars = [calibration[k] for k in ("interval", "alpha", "temperature", "alert_threshold")]
    if not all(isinstance(v, int | float) and math.isfinite(v) for v in scalars):
        raise ArtifactsIncompatible(f"{path}: every value must be a finite number")
    interval, alpha, temperature, threshold = scalars
    if not (0 < interval < 1 and 0 < alpha < 1 and temperature > 0 and 0 < threshold < 1):
        raise ArtifactsIncompatible(
            f"{path}: need 0 < interval < 1, 0 < alpha < 1, temperature > 0 and "
            "0 < alert_threshold < 1"
        )
    scores, member = calibration["scores"], calibration["score_member"]
    if (
        not isinstance(scores, list)
        or not scores
        or not isinstance(member, list)
        or len(member) != len(scores)
        or not all(isinstance(s, int | float) and math.isfinite(s) for s in scores)
        or not all(isinstance(m, int) and 0 <= m < n_members for m in member)
    ):
        raise ArtifactsIncompatible(
            f"{path}: scores must be finite numbers, each with the index of one of the "
            f"{n_members} networks. {RETRAIN_HINT}"
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
        self.metrics = json.loads((model_dir / "metrics.json").read_text())
        members = _load_members(model_dir / "model.pt", self.contract)
        self.model = Ensemble(members).eval()
        calibration_path = model_dir / "calibration.json"
        self.calibration = _check_calibration(
            calibration_path, json.loads(calibration_path.read_text()), len(members)
        )
        self._scores = np.asarray(self.calibration["scores"], np.float64)
        self._score_member = np.asarray(self.calibration["score_member"], np.int64)
        self._reference_point: float | None = None
        self._explanations: OrderedDict[bytes, np.ndarray] = OrderedDict()
        self._explanations_lock = threading.Lock()

    @property
    def alert_threshold(self) -> float:
        return float(self.calibration["alert_threshold"])

    def predict_windows(self, windows: np.ndarray) -> dict[str, np.ndarray]:
        """Calibrated outputs for (N, window, F) inputs: point, lower, upper, open_ended (the
        upper end reached the cap, so the interval means "at least lower") and probability."""
        raw = predict_raw(self.model, windows)
        q = raw["quantiles"].astype(np.float64) * self.contract.max_rul  # (members, N, 3)
        lower, upper = cv_plus_interval(
            q[:, :, 0], q[:, :, 2], self._scores, self._score_member, self.calibration["alpha"]
        )
        lower, upper, open_ended = finalize_interval(lower, upper, self.contract.max_rul)
        return {
            "point": np.clip(q[:, :, 1].mean(axis=0), 0, self.contract.max_rul),
            "lower": lower,
            "upper": upper,
            "open_ended": open_ended,
            "probability": probability(raw["alert_logit"], self.calibration["temperature"]),
        }

    def explain(self, window: np.ndarray, baseline: np.ndarray | None = None) -> np.ndarray:
        """Integrated-gradients attribution of the ensemble's point RUL, in cycles: shape
        (window, F). Networks with a shorter window attribute nothing to the earlier rows.

        Against the default reference, a window whose float32 bytes equal one explained before
        returns a copy of that earlier map: the networks are fixed once loaded, so recomputing it
        would repeat the same arithmetic. Any other reference is always computed."""
        values = window.astype(np.float32)
        key = None
        if baseline is None:
            key = hashlib.sha256(repr(values.shape).encode() + values.tobytes()).digest()
            with self._explanations_lock:
                cached = self._explanations.get(key)
                if cached is not None:
                    self._explanations.move_to_end(key)
                    return cached.copy()
        attributions = integrated_gradients(
            self.model.point,
            torch.from_numpy(values),
            None if baseline is None else torch.from_numpy(baseline.astype(np.float32)),
        )
        result = attributions.detach().numpy() * self.contract.max_rul
        if key is not None:
            with self._explanations_lock:
                self._explanations[key] = result.copy()
                while len(self._explanations) > EXPLANATION_CACHE_SIZE:
                    self._explanations.popitem(last=False)
        return result

    def clear_explanations(self) -> None:
        """Forget cached attribution maps, so the next call computes afresh (for benchmarks)."""
        with self._explanations_lock:
            self._explanations.clear()

    def status(self, probability: float, lower: float) -> str:
        """alert: failure within the horizon is more likely than not.
        watch: failure within the horizon is inside the interval.
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
                "reading": "what the model's output depends on, not a physical diagnosis",
                "baseline_rul_cycles": round(self._baseline_point(), 1),
                "sensors": self.sensor_contributions(attributions),
            }
        return result

    def format_prediction(self, row: dict) -> dict:
        probability = float(row["probability"])
        lower = float(row["lower"])
        open_ended = bool(row["open_ended"])
        upper = None if open_ended else round(float(row["upper"]), 1)
        cap = self.contract.max_rul
        return {
            "rul": {
                "point": round(float(row["point"]), 1),
                "lower": round(lower, 1),
                "upper": upper,
                "open_ended": open_ended,
                "summary": f"at least {lower:.0f} cycles"
                if open_ended
                else f"{lower:.0f} to {upper:.0f} cycles",
                "interval": self.calibration["interval"],
                "capped_at": cap,
            },
            "failure_within_horizon": {
                "horizon_cycles": self.contract.alert_cycles,
                "probability": round(probability, 4),
                "alert": probability >= self.alert_threshold,
            },
            "status": self.status(probability, lower),
        }

    def _baseline_point(self) -> float:
        """The ensemble's point RUL for the reference window, every scaled reading at its
        regime's training mean. It is fixed for a loaded model, so it is computed once."""
        if self._reference_point is None:
            zeros = np.zeros((1, self.contract.window, self.contract.n_features), np.float32)
            self._reference_point = float(self.predict_windows(zeros)["point"][0])
        return self._reference_point

    def info(self) -> dict:
        """What the dashboard and /api/model-info show: the served model's own numbers."""
        m = self.metrics
        served = m.get("served", {})
        repeat = str(served.get("repeat", 0))
        test = (m.get("test") or {}).get("models", {})
        lstm = test.get("lstm", {}).get("per_repeat", {}).get(repeat)
        lgbm = test.get("lightgbm", {}).get("per_repeat", {}).get(repeat)
        oof = m.get("out_of_fold", {}).get("lstm", {}).get("per_repeat", {}).get(repeat, {})
        alert = oof.get("alert", {})
        provenance = m.get("provenance", {})
        return {
            "subset": m["subset"],
            "created_at": m["created_at"],
            "code": {
                "source_sha256": provenance.get("source_sha256"),
                "git_commit": (provenance.get("git") or {}).get("commit"),
            },
            "data": {
                "test_engines": m["data"]["test_engines"],
                "crossfit_engines": m["data"].get("crossfit_engines"),
                "replay_engines": len(m["data"].get("replay_engines", [])),
            },
            "ensemble_size": len(self.model.members),
            "windows": [int(member.window) for member in self.model.members],  # type: ignore[arg-type]
            "window": self.contract.window,
            "max_rul": self.contract.max_rul,
            "alert_cycles": self.contract.alert_cycles,
            "alert_threshold": self.alert_threshold,
            "interval": self.calibration["interval"],
            "sensors": [
                {"sensor": s, "symbol": SENSOR_INFO[s][0], "description": SENSOR_INFO[s][1]}
                for s in self.contract.sensors
            ],
            "kpis": {
                "test_evaluated": lstm is not None,
                "rmse": lstm and lstm["rmse"],
                "rmse_uncapped": lstm and lstm["rmse_uncapped"],
                "nasa_score": lstm and lstm["nasa_score"],
                "coverage": lstm and lstm["coverage"],
                "open_ended_share": lstm and lstm["open_ended_share"],
                "closed_width": lstm and lstm["closed_width"],
                "alert_ap": lstm and lstm["alert_ap"],
                "alert_positives": lstm and lstm["alert_positives"],
                "lightgbm_rmse": lgbm and lgbm["rmse"],
                "lead_median": alert.get("lead_median"),
                "lead_engines": alert.get("units"),
                "missed": alert.get("missed"),
                "early": alert.get("early"),
            },
        }
