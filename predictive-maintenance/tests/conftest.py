"""Shared fixtures: a small synthetic dataset in the exact C-MAPSS file format, and a model
trained on it once per test session. Tests never touch the real data/ or artifacts/ folders."""

from pathlib import Path

import numpy as np
import pytest

from predmaint.config import Settings
from predmaint.data import SENSORS
from predmaint.evaluate import evaluate_run
from predmaint.train import TrainConfig, run

FD002_REGIMES = [
    (0.0, 0.0, 100.0),
    (10.0, 0.25, 100.0),
    (20.0, 0.70, 100.0),
    (25.0, 0.62, 60.0),
    (35.0, 0.84, 100.0),
    (42.0, 0.84, 100.0),
]
INFORMATIVE = {2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21}


def _engine_rows(rng, unit: int, n_cycles: int, regimes: list[tuple[float, float, float]]):
    base = 100 + 10 * np.arange(len(SENSORS), dtype=float)
    drift = np.where(
        np.isin(np.arange(1, 22), list(INFORMATIVE)), rng.uniform(2, 6, len(SENSORS)), 0.0
    )
    rows = []
    for cycle in range(1, n_cycles + 1):
        regime = rng.integers(len(regimes))
        settings = np.array(regimes[regime]) + rng.normal(0, 0.001, 3)
        wear = (cycle / n_cycles) ** 2
        values = base + 25 * regime + drift * 10 * wear + rng.normal(0, 0.3, len(SENSORS))
        rows.append([unit, cycle, *settings.round(4), *values.round(4)])
    return rows


def _write(path: Path, rows) -> None:
    path.write_text("".join(" ".join(str(v) for v in row) + " \n" for row in rows))


def write_synthetic_cmapss(data_dir: Path, seed: int = 0, n_units: int = 24) -> None:
    rng = np.random.default_rng(seed)
    for subset, regimes in (("FD001", [(0.0, 0.0, 100.0)]), ("FD002", FD002_REGIMES)):
        train, test, final_rul = [], [], []
        for unit in range(1, n_units + 1):
            train += _engine_rows(rng, unit, int(rng.integers(60, 90)), regimes)
            life = int(rng.integers(60, 90))
            cut = int(rng.integers(12, life - 4))
            test += _engine_rows(rng, unit, life, regimes)[:cut]
            final_rul.append(life - cut)
        _write(data_dir / f"train_{subset}.txt", train)
        _write(data_dir / f"test_{subset}.txt", test)
        (data_dir / f"RUL_{subset}.txt").write_text("".join(f"{r} \n" for r in final_rul))


@pytest.fixture(scope="session")
def cmapss_dir(tmp_path_factory) -> Path:
    data_dir = tmp_path_factory.mktemp("cmapss")
    write_synthetic_cmapss(data_dir)
    return data_dir


TINY = {
    "repeats": [0, 1],
    "folds": 3,
    "windows": [6, 10],
    "epochs": 3,
    "patience": 3,
    "hidden": 16,
    "layers": 1,
    "batch_size": 64,
    "temperature_draws": 3,
    "lgbm_estimators": 40,
}


@pytest.fixture(scope="session")
def artifacts_dir(tmp_path_factory, cmapss_dir) -> Path:
    """Train, then score the test set, exactly as the real pipeline does (never touching the
    repository's experiment ledger)."""
    artifacts = tmp_path_factory.mktemp("artifacts")
    settings = Settings(data_dir=cmapss_dir, artifacts_dir=artifacts)
    run(TrainConfig(subset="FD001", **TINY), settings, artifacts / "FD001", ledger=None)
    evaluate_run(artifacts / "FD001", settings, ledger=None)
    return artifacts


@pytest.fixture
def settings(cmapss_dir, artifacts_dir) -> Settings:
    return Settings(
        data_dir=cmapss_dir, artifacts_dir=artifacts_dir, replay_autostart=False, replay_engines=3
    )
