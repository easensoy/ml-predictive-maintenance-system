"""Replay held-out C-MAPSS engines as a live fleet.

Each tick advances every engine by one recorded cycle and runs the model on the new window,
exactly as a streaming deployment would. The engines were set aside before training: no network,
scaler or calibration step saw them, and because they run to failure the true RUL is known and
shown.
"""

import threading

import numpy as np
import pandas as pd

from predmaint.data import SENSOR_INFO
from predmaint.inference import Predictor

FAILED_HOLD_TICKS = 3


class FleetReplay:
    def __init__(self, predictor: Predictor, replay: pd.DataFrame, n_engines: int, seed: int = 7):
        self.predictor = predictor
        self.window = predictor.contract.window
        units = sorted(int(u) for u in replay["unit"].unique())[:n_engines]
        self.frames = {
            u: replay[replay["unit"] == u].sort_values("cycle").reset_index(drop=True)
            for u in units
        }
        self.normalized = {u: predictor.contract.engine_features(f) for u, f in self.frames.items()}
        rng = np.random.default_rng(seed)
        # Stagger the engines so the fleet shows a mix of early, mid and late life.
        self.position = {
            u: int(rng.integers(0, int(len(f) * 0.85))) for u, f in self.frames.items()
        }
        self.lap = dict.fromkeys(units, 1)
        self.failed_ticks = dict.fromkeys(units, 0)
        self.history: dict[int, list[dict]] = {u: [] for u in units}
        self.tick = 0
        self.lock = threading.Lock()
        for unit in units:
            self._backfill(unit)

    @property
    def units(self) -> list[int]:
        return list(self.frames)

    def _window_at(self, unit: int, position: int) -> np.ndarray:
        rows = self.normalized[unit][max(0, position - self.window + 1) : position + 1]
        if len(rows) < self.window:
            first = self.normalized[unit][:1]
            rows = np.concatenate([np.repeat(first, self.window - len(rows), axis=0), rows])
        return rows

    def _entry(self, unit: int, position: int, out: dict, i: int) -> dict:
        frame = self.frames[unit]
        row = {key: out[key][i] for key in ("point", "lower", "upper", "open_ended", "probability")}
        formatted = self.predictor.format_prediction(row)
        return {
            "cycle": int(frame["cycle"].to_numpy()[position]),
            "rul_true": int(frame["rul"].to_numpy()[position]),
            "point": formatted["rul"]["point"],
            "lower": formatted["rul"]["lower"],
            "upper": formatted["rul"]["upper"],
            "open_ended": formatted["rul"]["open_ended"],
            "probability": formatted["failure_within_horizon"]["probability"],
            "status": formatted["status"],
        }

    def _backfill(self, unit: int) -> None:
        positions = list(range(self.position[unit] + 1))
        windows = np.stack([self._window_at(unit, p) for p in positions])
        out = self.predictor.predict_windows(windows)
        self.history[unit] = [self._entry(unit, p, out, i) for i, p in enumerate(positions)]

    def seek(self, unit: int, cycle: int) -> None:
        """Put one engine at one of its recorded cycles, with its prediction history up to it, so
        a view of the replay can be reproduced exactly (the README screenshots use this)."""
        with self.lock:
            frame = self.frames[unit]
            found = np.flatnonzero(frame["cycle"].to_numpy() == cycle)
            if not len(found):
                raise ValueError(f"engine {unit} has no recorded cycle {cycle}")
            self.position[unit] = int(found[0])
            self.lap[unit], self.failed_ticks[unit] = 1, 0
            self._backfill(unit)

    def step(self) -> dict:
        with self.lock:
            self.tick += 1
            advancing = []
            for unit, frame in self.frames.items():
                if self.position[unit] >= len(frame) - 1:
                    self.failed_ticks[unit] += 1
                    if self.failed_ticks[unit] <= FAILED_HOLD_TICKS:
                        continue
                    self.position[unit], self.failed_ticks[unit] = -1, 0
                    self.lap[unit] += 1
                    self.history[unit] = []
                self.position[unit] += 1
                advancing.append(unit)
            if advancing:
                windows = np.stack([self._window_at(u, self.position[u]) for u in advancing])
                out = self.predictor.predict_windows(windows)
                for i, unit in enumerate(advancing):
                    self.history[unit].append(self._entry(unit, self.position[unit], out, i))
            return self._snapshot()

    def snapshot(self) -> dict:
        with self.lock:
            return self._snapshot()

    def _snapshot(self) -> dict:
        engines = []
        for unit, frame in self.frames.items():
            latest = dict(self.history[unit][-1])
            failed = self.position[unit] >= len(frame) - 1 and self.failed_ticks[unit] > 0
            latest.update(
                unit=unit,
                total_cycles=int(frame["cycle"].iloc[-1]),
                lap=self.lap[unit],
                status="failed" if failed else latest["status"],
            )
            engines.append(latest)
        return {
            "tick": self.tick,
            "subset": self.predictor.contract.subset,
            "horizon_cycles": self.predictor.contract.alert_cycles,
            "engines": engines,
        }

    def detail(self, unit: int, top_sensors: int = 3) -> dict:
        with self.lock:
            if unit not in self.frames:
                raise KeyError(unit)
            position = self.position[unit]
            history = list(self.history[unit])
            window = self._window_at(unit, position)
            frame = self.frames[unit].iloc[: position + 1]
        attributions = self.predictor.explain(window)
        contributions = self.predictor.sensor_contributions(attributions)
        first_cycle = int(frame["cycle"].iloc[0])
        window_cycles = [
            max(first_cycle, int(frame["cycle"].iloc[-1]) - self.window + 1 + i)
            for i in range(self.window)
        ]
        symbols = [SENSOR_INFO[s][0] for s in self.predictor.contract.sensors]
        traces = {}
        for item in contributions[:top_sensors]:
            traces[item["symbol"]] = {
                "description": item["description"],
                "unit": item["unit"],
                "values": [round(float(v), 4) for v in frame[item["sensor"]]],
            }
        return {
            "unit": unit,
            "lap": self.lap[unit],
            "total_cycles": int(self.frames[unit]["cycle"].iloc[-1]),
            "history": history,
            "attribution": {
                "cycles": window_cycles,
                "sensors": symbols,
                "values": np.round(attributions, 3).tolist(),
            },
            "contributions": contributions,
            "traces": {"cycles": frame["cycle"].astype(int).tolist(), "sensors": traces},
        }
