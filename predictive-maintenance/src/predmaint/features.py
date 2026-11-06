"""The feature contract: the single definition of model inputs, shared by training and serving.

Everything that turns raw engine readings into a model window lives here and is saved to
``contract.json`` next to the model, so the API cannot drift from what the model was trained on.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from predmaint.data import SETTINGS

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
    ewma_alpha: float | None = None

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
        """Fit regime clusters and per-regime sensor scaling on training engines only."""
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

        return cls(
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
            ewma_alpha=ewma_alpha,
        )

    @property
    def n_features(self) -> int:
        return len(self.sensors)

    def regimes(self, settings: np.ndarray) -> np.ndarray:
        """Assign each row to the nearest operating regime."""
        z = (np.asarray(settings, np.float64) - self.settings_mean) / self.settings_std
        centroids = np.asarray(self.regime_centroids)
        distances = ((z[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        return distances.argmin(axis=1)

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
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> FeatureContract:
        return cls(**json.loads(path.read_text()))
