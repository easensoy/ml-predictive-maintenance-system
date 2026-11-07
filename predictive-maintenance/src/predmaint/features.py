"""The feature contract: the single definition of model inputs, shared by training and serving.

Everything that turns raw engine readings into a model window lives here and is saved to
``contract.json`` next to the model, so the API cannot drift from what the model was trained on.
"""

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from predmaint.data import SENSORS, SETTINGS, SUBSETS

# Bump when the meaning or layout of contract.json changes; loading another format fails loudly.
CONTRACT_FORMAT = 2
# A reading is out of distribution when a scaled sensor value lies more than this fraction of
# the training span outside the training range, or its operating settings lie further from
# every regime centre than any training row did (with the same relative slack).
OOD_MARGIN = 0.1

# Sensors 1, 5, 6, 10, 16, 18 and 19 are constant (or nearly so) within each operating regime,
# so they carry no degradation signal. These 14 are the standard informative set.
INFORMATIVE_SENSORS = [
    "sensor_02",
    "sensor_03",
    "sensor_04",
    "sensor_07",
    "sensor_08",
    "sensor_09",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_17",
    "sensor_20",
    "sensor_21",
]
# FD002 and FD004 mix six flight conditions; FD001 and FD003 run at sea level only.
OPERATING_REGIMES = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}


def sliding_windows(rows: np.ndarray, window: int) -> np.ndarray:
    """Return one window per row, ending at that row: shape (T, window, F).

    Rows before the start of the trajectory are filled by repeating the first row, so short
    histories (a new engine, or the shortest test trajectories) still produce a full window.
    """
    if rows.ndim != 2 or len(rows) == 0:
        raise ValueError("rows must be a non-empty (T, F) array")
    padded = np.concatenate([np.repeat(rows[:1], window - 1, axis=0), rows], axis=0)
    view = np.lib.stride_tricks.sliding_window_view(padded, window, axis=0)  # (T, F, window)
    return np.ascontiguousarray(view.transpose(0, 2, 1))


def ewma(rows: np.ndarray, alpha: float) -> np.ndarray:
    """Causal exponential smoothing along time: each output uses only current and past rows."""
    out = np.empty_like(rows)
    out[0] = rows[0]
    for t in range(1, len(rows)):
        out[t] = alpha * rows[t] + (1 - alpha) * out[t - 1]
    return out


class ContractError(ValueError):
    """contract.json is malformed, or was written by an incompatible version of predmaint."""


def _finite_list(value: object, length: int | None = None) -> bool:
    return (
        isinstance(value, list)
        and (length is None or len(value) == length)
        and all(
            isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)
            for v in value
        )
    )


@dataclass
class FeatureContract:
    subset: str
    sensors: list[str]
    window: int
    max_rul: int
    alert_cycles: int
    settings_mean: list[float]
    settings_std: list[float]
    regime_centroids: list[list[float]]
    sensor_mean: list[list[float]]
    sensor_std: list[list[float]]
    feature_min: list[float]
    feature_max: list[float]
    regime_radius: float
    ewma_alpha: float | None = None
    format: int = CONTRACT_FORMAT

    @classmethod
    def fit(
        cls,
        train: pd.DataFrame,
        subset: str,
        window: int = 30,
        max_rul: int = 125,
        alert_cycles: int = 30,
        seed: int = 0,
        ewma_alpha: float | None = None,
    ) -> FeatureContract:
        """Fit regime clusters, per-regime sensor scaling and the training range on training
        engines only."""
        settings = train[SETTINGS].to_numpy(np.float64)
        settings_mean = settings.mean(axis=0)
        settings_std = settings.std(axis=0)
        settings_std[settings_std < 1e-12] = 1.0
        z = (settings - settings_mean) / settings_std

        n_regimes = OPERATING_REGIMES[subset]
        if n_regimes == 1:
            centroids = z.mean(axis=0, keepdims=True)
            labels = np.zeros(len(z), dtype=np.int64)
        else:
            kmeans = KMeans(n_clusters=n_regimes, n_init=10, random_state=seed).fit(z)
            order = np.lexsort(kmeans.cluster_centers_.T[::-1])  # deterministic regime ids
            centroids = kmeans.cluster_centers_[order]
            labels = np.argsort(order)[kmeans.labels_]

        values = train[INFORMATIVE_SENSORS].to_numpy(np.float64)
        means, stds = [], []
        for regime in range(n_regimes):
            in_regime = values[labels == regime]
            if len(in_regime) < 2:
                raise ValueError(f"Regime {regime} has fewer than 2 training rows")
            std = in_regime.std(axis=0)
            std[std < 1e-6] = 1.0
            means.append(in_regime.mean(axis=0).tolist())
            stds.append(std.tolist())

        contract = cls(
            subset=subset,
            sensors=list(INFORMATIVE_SENSORS),
            window=window,
            max_rul=max_rul,
            alert_cycles=alert_cycles,
            settings_mean=settings_mean.tolist(),
            settings_std=settings_std.tolist(),
            regime_centroids=centroids.tolist(),
            sensor_mean=means,
            sensor_std=stds,
            feature_min=[],
            feature_max=[],
            regime_radius=0.0,
            ewma_alpha=ewma_alpha,
        )
        contract.set_training_range(train)
        return contract

    def set_training_range(self, train: pd.DataFrame) -> None:
        """Record the range of scaled sensor values and regime distances seen in training."""
        scaled = self.normalize(train)
        self.feature_min = [float(v) for v in scaled.min(axis=0)]
        self.feature_max = [float(v) for v in scaled.max(axis=0)]
        self.regime_radius = float(self.regime_distance(train[SETTINGS].to_numpy()).max())

    @property
    def n_features(self) -> int:
        return len(self.sensors)

    def _settings_distances(self, settings: np.ndarray) -> np.ndarray:
        z = (np.asarray(settings, np.float64) - self.settings_mean) / self.settings_std
        centroids = np.asarray(self.regime_centroids)
        return np.sqrt(((z[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2))

    def regimes(self, settings: np.ndarray) -> np.ndarray:
        """Assign each row to the nearest operating regime."""
        return self._settings_distances(settings).argmin(axis=1)

    def regime_distance(self, settings: np.ndarray) -> np.ndarray:
        """Distance of each row's scaled operating settings to the nearest regime centre."""
        return self._settings_distances(settings).min(axis=1)

    def out_of_distribution(self, frame: pd.DataFrame) -> dict:
        """Which readings in ``frame`` lie outside what the model saw in training?

        Sensors are compared in scaled units (training standard deviations of their regime).
        Predictions for flagged readings are extrapolations and should not be trusted.
        """
        settings = frame[SETTINGS].to_numpy(np.float64)
        values = frame[self.sensors].to_numpy(np.float64)
        low, high = np.asarray(self.feature_min), np.asarray(self.feature_max)
        slack = OOD_MARGIN * (high - low)
        with np.errstate(over="ignore", invalid="ignore"):  # absurd inputs overflow to inf: fine
            regime = self.regimes(settings)
            scaled = (values - np.asarray(self.sensor_mean)[regime]) / np.asarray(self.sensor_std)[
                regime
            ]
            excess = np.maximum(low - slack - scaled, scaled - high - slack)  # > 0: outside
            distance = self.regime_distance(settings)
        sensors = []
        for i in np.flatnonzero((excess > 0).any(axis=0)):
            name = self.sensors[i]
            sensors.append(
                {
                    "sensor": name,
                    "rows": int((excess[:, i] > 0).sum()),
                    "max_excess_sd": round(float(min(excess[:, i].max(), 1e6)), 2),
                }
            )
        settings_rows = int((~(distance <= self.regime_radius * (1 + OOD_MARGIN))).sum())
        return {
            "flag": bool(sensors) or settings_rows > 0,
            "rows_checked": len(frame),
            "sensors_outside_training_range": sensors,
            "rows_with_unseen_operating_settings": settings_rows,
        }

    def normalize(self, frame: pd.DataFrame) -> np.ndarray:
        """Per-regime z-score of the informative sensors: shape (rows, n_features)."""
        missing = [c for c in (*SETTINGS, *self.sensors) if c not in frame.columns]
        if missing:
            raise KeyError(f"Missing columns: {missing}")
        regime = self.regimes(frame[SETTINGS].to_numpy(np.float64))
        values = frame[self.sensors].to_numpy(np.float64)
        mean = np.asarray(self.sensor_mean)[regime]
        std = np.asarray(self.sensor_std)[regime]
        return ((values - mean) / std).astype(np.float32)

    def engine_features(self, frame: pd.DataFrame) -> np.ndarray:
        """Model features for ONE engine's readings, oldest cycle first: shape (cycles, F)."""
        normalized = self.normalize(frame)
        return ewma(normalized, self.ewma_alpha) if self.ewma_alpha else normalized

    def unit_windows(self, frame: pd.DataFrame) -> np.ndarray:
        """Windows ending at every row, positionally aligned with ``frame``'s rows.

        ``frame`` must be sorted by unit and then cycle; windows never cross engine boundaries.
        """
        units = frame["unit"].to_numpy()
        cycles = frame["cycle"].to_numpy()
        same_unit = units[1:] == units[:-1]
        if np.any(units[1:] < units[:-1]) or np.any(same_unit & (cycles[1:] <= cycles[:-1])):
            raise ValueError("frame must be sorted by unit and strictly increasing cycle")
        boundaries = np.flatnonzero(~same_unit) + 1
        starts = np.concatenate([[0], boundaries])
        ends = np.concatenate([boundaries, [len(units)]])
        return np.concatenate(
            [
                sliding_windows(self.engine_features(frame.iloc[s:e]), self.window)
                for s, e in zip(starts, ends, strict=True)
            ]
        )

    def last_windows(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """One window per unit, ending at its last recorded cycle (the official test protocol)."""
        windows = self.unit_windows(frame)
        units = frame["unit"].to_numpy()
        last = np.flatnonzero(np.r_[units[1:] != units[:-1], True])
        return windows[last], units[last]

    def window_from_readings(self, readings: pd.DataFrame) -> np.ndarray:
        """Model input for the most recent cycle of a single engine: shape (window, F)."""
        return sliding_windows(self.engine_features(readings), self.window)[-1]

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> None:
        self.validate()
        path.write_text(json.dumps(self.to_dict(), indent=2))

    def validate(self) -> None:
        """Check types and shapes, so a bad contract fails at start-up, not mid-request."""
        problems = []
        n_regimes, n_sensors = len(self.regime_centroids), len(self.sensors)
        if self.format != CONTRACT_FORMAT:
            problems.append(f"format is {self.format}, expected {CONTRACT_FORMAT}")
        if self.subset not in SUBSETS:
            problems.append(f"unknown subset {self.subset!r}")
        if not self.sensors or len(set(self.sensors)) != n_sensors:
            problems.append("sensors must be a non-empty list without duplicates")
        elif unknown := [s for s in self.sensors if s not in SENSORS]:
            problems.append(f"unknown sensors {unknown}")
        for name in ("window", "max_rul", "alert_cycles"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                problems.append(f"{name} must be a positive integer")
        if not problems and self.alert_cycles >= self.max_rul:
            problems.append("alert_cycles must be smaller than max_rul")
        for name in ("settings_mean", "settings_std"):
            if not _finite_list(getattr(self, name), len(SETTINGS)):
                problems.append(f"{name} must hold {len(SETTINGS)} finite numbers")
        if n_regimes < 1 or not all(_finite_list(c, len(SETTINGS)) for c in self.regime_centroids):
            problems.append(f"regime_centroids must be a list of {len(SETTINGS)}-number lists")
        for name in ("sensor_mean", "sensor_std"):
            rows = getattr(self, name)
            if len(rows) != n_regimes or not all(_finite_list(r, n_sensors) for r in rows):
                problems.append(f"{name} must be {n_regimes} lists of {n_sensors} finite numbers")
        if _finite_list(self.settings_std) and min(self.settings_std, default=1) <= 0:
            problems.append("settings_std must be positive")
        if all(_finite_list(r) for r in self.sensor_std) and any(
            v <= 0 for r in self.sensor_std for v in r
        ):
            problems.append("sensor_std must be positive")
        for name in ("feature_min", "feature_max"):
            if not _finite_list(getattr(self, name), n_sensors):
                problems.append(f"{name} must hold {n_sensors} finite numbers")
        if not isinstance(self.regime_radius, int | float) or not (
            math.isfinite(self.regime_radius) and self.regime_radius >= 0
        ):
            problems.append("regime_radius must be a finite number >= 0")
        if self.ewma_alpha is not None and not (
            isinstance(self.ewma_alpha, int | float) and 0 < self.ewma_alpha <= 1
        ):
            problems.append("ewma_alpha must be null or in (0, 1]")
        if problems:
            raise ContractError("; ".join(problems))

    @classmethod
    def from_dict(cls, raw: object) -> FeatureContract:
        if not isinstance(raw, dict):
            raise ContractError("expected a JSON object")
        version = raw.get("format", 1)
        if version != CONTRACT_FORMAT:
            raise ContractError(
                f"format {version} is not supported by this version of predmaint, which reads "
                f"format {CONTRACT_FORMAT}. Retrain with `python -m predmaint.train`, or check out "
                "the git commit recorded in the run's metrics.json"
            )
        known = {f.name for f in fields(cls)}
        required = {f.name for f in fields(cls) if f.name not in {"ewma_alpha", "format"}}
        if missing := sorted(required - raw.keys()):
            raise ContractError(f"missing keys {missing}")
        if unknown := sorted(raw.keys() - known):
            raise ContractError(f"unknown keys {unknown}")
        contract = cls(**raw)
        contract.validate()
        return contract

    @classmethod
    def load(cls, path: Path) -> FeatureContract:
        try:
            return cls.from_dict(json.loads(path.read_text()))
        except json.JSONDecodeError as exc:
            raise ContractError(f"{path} is not valid JSON: {exc}") from exc
        except ContractError as exc:
            raise ContractError(f"{path}: {exc}") from exc
