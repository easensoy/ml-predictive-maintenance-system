"""NASA C-MAPSS turbofan dataset: download, integrity checks, loading and RUL labels.

Source: NASA Prognostics Center of Excellence data repository.
Reference: A. Saxena, K. Goebel, D. Simon and N. Eklund, "Damage Propagation Modeling for
Aircraft Engine Run-to-Failure Simulation", PHM08, Denver CO, 2008.

Usage::

    python -m predmaint.data download            # fetch from NASA and verify checksums
    python -m predmaint.data download --from-zip path/to/downloaded.zip
    python -m predmaint.data check
"""

import argparse
import hashlib
import io
import itertools
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from predmaint.config import Settings

SOURCE_URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
)
# SHA-256 of the archive as published by NASA (12,429,152 bytes, last modified 2022-09-18)
# and of the CMAPSSData.zip it contains.
OUTER_SHA256 = "c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2"
INNER_MEMBER = "6. Turbofan Engine Degradation Simulation Data Set/CMAPSSData.zip"
INNER_SHA256 = "74bef434a34db25c7bf72e668ea4cd52afe5f2cf8e44367c55a82bfd91a5a34f"
MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024

SUBSETS = ("FD001", "FD002", "FD003", "FD004")
EXPECTED_FILES = (
    *(f"{kind}_{subset}.txt" for kind in ("train", "test", "RUL") for subset in SUBSETS),
    "readme.txt",
)

SETTINGS = ["setting_1", "setting_2", "setting_3"]
SENSORS = [f"sensor_{i:02d}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle", *SETTINGS, *SENSORS]

# Sensor symbols and descriptions in the order of Table 2 of Saxena et al. (2008). The dataset
# files number the sensors 1-21 without names; this table order is the conventional mapping.
SENSOR_INFO = {
    "sensor_01": ("T2", "Total temperature at fan inlet", "°R"),
    "sensor_02": ("T24", "Total temperature at LPC outlet", "°R"),
    "sensor_03": ("T30", "Total temperature at HPC outlet", "°R"),
    "sensor_04": ("T50", "Total temperature at LPT outlet", "°R"),
    "sensor_05": ("P2", "Pressure at fan inlet", "psia"),
    "sensor_06": ("P15", "Total pressure in bypass duct", "psia"),
    "sensor_07": ("P30", "Total pressure at HPC outlet", "psia"),
    "sensor_08": ("Nf", "Physical fan speed", "rpm"),
    "sensor_09": ("Nc", "Physical core speed", "rpm"),
    "sensor_10": ("epr", "Engine pressure ratio (P50/P2)", ""),
    "sensor_11": ("Ps30", "Static pressure at HPC outlet", "psia"),
    "sensor_12": ("phi", "Ratio of fuel flow to Ps30", "pps/psi"),
    "sensor_13": ("NRf", "Corrected fan speed", "rpm"),
    "sensor_14": ("NRc", "Corrected core speed", "rpm"),
    "sensor_15": ("BPR", "Bypass ratio", ""),
    "sensor_16": ("farB", "Burner fuel-air ratio", ""),
    "sensor_17": ("htBleed", "Bleed enthalpy", ""),
    "sensor_18": ("Nf_dmd", "Demanded fan speed", "rpm"),
    "sensor_19": ("PCNfR_dmd", "Demanded corrected fan speed", "rpm"),
    "sensor_20": ("W31", "HPT coolant bleed", "lbm/s"),
    "sensor_21": ("W32", "LPT coolant bleed", "lbm/s"),
}


class DataError(RuntimeError):
    """Raised when the dataset is missing, corrupted or malformed."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "predmaint-dataset-downloader"})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read(MAX_DOWNLOAD_BYTES + 1)
    if len(data) > MAX_DOWNLOAD_BYTES:
        raise DataError(f"Download from {url} exceeded {MAX_DOWNLOAD_BYTES} bytes")
    return data


def extract_dataset(archive: bytes, data_dir: Path) -> list[Path]:
    """Verify the NASA archive and write the expected text files into ``data_dir``.

    Only the whitelisted file names are written, so a malicious archive cannot write outside
    ``data_dir`` (no zip-slip) or drop unexpected files.
    """
    if (digest := _sha256(archive)) != OUTER_SHA256:
        raise DataError(f"Archive SHA-256 mismatch: expected {OUTER_SHA256}, got {digest}")
    with zipfile.ZipFile(io.BytesIO(archive)) as outer:
        inner_bytes = outer.read(INNER_MEMBER)
    if (digest := _sha256(inner_bytes)) != INNER_SHA256:
        raise DataError(f"CMAPSSData.zip SHA-256 mismatch: expected {INNER_SHA256}, got {digest}")

    data_dir.mkdir(parents=True, exist_ok=True)
    written = []
    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
        members = set(inner.namelist())
        missing = [name for name in EXPECTED_FILES if name not in members]
        if missing:
            raise DataError(f"Archive is missing expected files: {missing}")
        for name in EXPECTED_FILES:
            target = data_dir / name
            tmp = target.with_suffix(".part")
            tmp.write_bytes(inner.read(name))
            tmp.replace(target)
            written.append(target)
    return written


def download(data_dir: Path, from_zip: Path | None = None, url: str = SOURCE_URL) -> list[Path]:
    """Download (or read a local copy of) the NASA archive, verify it and extract it."""
    archive = from_zip.read_bytes() if from_zip else _fetch(url)
    return extract_dataset(archive, data_dir)


def check(data_dir: Path) -> None:
    missing = [name for name in EXPECTED_FILES if not (data_dir / name).is_file()]
    if missing:
        raise DataError(
            f"C-MAPSS files missing in {data_dir}: {missing}. "
            "Run `python -m predmaint.data download` first."
        )


def _read_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise DataError(f"{path} not found. Run `python -m predmaint.data download` first.")
    df = pd.read_csv(path, sep=r"\s+", header=None)
    if df.shape[1] != len(COLUMNS):
        raise DataError(f"{path.name}: expected {len(COLUMNS)} columns, found {df.shape[1]}")
    df.columns = COLUMNS
    df[["unit", "cycle"]] = df[["unit", "cycle"]].astype("int64")
    if df.isna().to_numpy().any():
        raise DataError(f"{path.name}: contains missing values")
    return df.sort_values(["unit", "cycle"], ignore_index=True)


def load_train(data_dir: Path, subset: str) -> pd.DataFrame:
    """Training trajectories run to failure. Adds ``rul`` = cycles left before failure."""
    df = _read_table(data_dir / f"train_{_check_subset(subset)}.txt")
    last = df.groupby("unit")["cycle"].transform("max")
    df["rul"] = (last - df["cycle"]).astype("int64")
    return df


def load_test_inputs(data_dir: Path, subset: str) -> pd.DataFrame:
    """Test trajectories truncated before failure, without labels.

    Training uses only this: the test labels are read by ``load_test_labels``, which only the
    evaluation step calls.
    """
    return _read_table(data_dir / f"test_{_check_subset(subset)}.txt")


def load_test_labels(data_dir: Path, subset: str) -> pd.Series:
    """True RUL at the last recorded cycle of each test engine, indexed by unit."""
    subset = _check_subset(subset)
    rul_path = data_dir / f"RUL_{subset}.txt"
    if not rul_path.is_file():
        raise DataError(f"{rul_path} not found. Run `python -m predmaint.data download` first.")
    final_rul = np.loadtxt(rul_path, dtype=np.int64, ndmin=1)
    units = np.sort(load_test_inputs(data_dir, subset)["unit"].unique())
    if len(final_rul) != len(units):
        raise DataError(f"RUL_{subset}.txt has {len(final_rul)} rows for {len(units)} test units")
    return pd.Series(final_rul, index=units, name="rul")


def load_test(data_dir: Path, subset: str) -> pd.DataFrame:
    """Test trajectories with ``rul`` at every cycle, from the official RUL file."""
    df = load_test_inputs(data_dir, subset)
    rul_at_last = load_test_labels(data_dir, subset)
    last = df.groupby("unit")["cycle"].transform("max")
    df["rul"] = (last - df["cycle"] + df["unit"].map(rul_at_last)).astype("int64")
    return df


def file_digests(data_dir: Path, names: list[str]) -> dict[str, str]:
    """SHA-256 of raw dataset files, recorded with every trained model and evaluation."""
    return {name: _sha256((data_dir / name).read_bytes()) for name in names}


def _by_life(units: np.ndarray, lives: np.ndarray) -> np.ndarray:
    """Units sorted by life length (ties by unit id), for stratifying splits by life."""
    return np.asarray(units)[np.lexsort((np.asarray(units), np.asarray(lives)))]


def assign_folds(units: np.ndarray, lives: np.ndarray, n_folds: int, seed: int) -> dict[int, int]:
    """Assign engines to ``n_folds`` folds, stratified by life length: engines are sorted by
    life and each consecutive block of ``n_folds`` is dealt out in random order."""
    if len(units) < 2 * n_folds:
        raise DataError(f"{len(units)} engines are too few for {n_folds} folds")
    rng = np.random.default_rng(seed)
    ordered = _by_life(units, lives)
    folds: dict[int, int] = {}
    for start in range(0, len(ordered), n_folds):
        block = ordered[start : start + n_folds]
        for unit, fold in zip(block, rng.permutation(n_folds)[: len(block)], strict=True):
            folds[int(unit)] = int(fold)
    return folds


def stratified_subset(
    units: np.ndarray, lives: np.ndarray, fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split engines into (rest, subset) with ``subset`` about ``fraction`` of them, spread
    across life lengths. Used for the inner validation split and the replay holdout."""
    units, lives = np.asarray(units), np.asarray(lives)
    n_pick = max(1, round(len(units) * fraction))
    if n_pick >= len(units):
        raise DataError(f"Cannot take {n_pick} of {len(units)} engines")
    rng = np.random.default_rng(seed)
    ordered = _by_life(units, lives)
    # One pick per stratum of equal size, at a random position inside it.
    edges = np.linspace(0, len(ordered), n_pick + 1).round().astype(int)
    picked = np.array([ordered[rng.integers(lo, hi)] for lo, hi in itertools.pairwise(edges)])
    rest = np.setdiff1d(units, picked)
    return np.sort(rest), np.sort(picked)


def test_like_truncations(
    lives: dict[int, int], test_lengths: np.ndarray, draws: int, rng: np.random.Generator
) -> list[tuple[int, int]]:
    """(unit, cycle) truncation points that look like the official test set.

    For each training engine, draw a trajectory length from the test engines' observed lengths
    (their sensor readings, never their labels), among those shorter than this engine's life, so
    at least one cycle remains, as in the test set.
    """
    lengths = np.sort(np.asarray(test_lengths))
    points = []
    for unit, life in lives.items():
        eligible = lengths[lengths < life]
        if len(eligible) == 0:
            raise DataError(f"Engine {unit} (life {life}) is shorter than every test trajectory")
        points += [(int(unit), int(c)) for c in rng.choice(eligible, size=draws, replace=True)]
    return points


def _check_subset(subset: str) -> str:
    if subset not in SUBSETS:
        raise DataError(f"Unknown subset {subset!r}; expected one of {SUBSETS}")
    return subset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m predmaint.data", description=__doc__)
    parser.add_argument("command", choices=["download", "check"])
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--from-zip", type=Path, default=None, help="use a local copy of the zip")
    args = parser.parse_args(argv)
    data_dir = args.data_dir or Settings.from_env().data_dir
    try:
        if args.command == "download":
            written = download(data_dir, from_zip=args.from_zip)
            print(f"Verified and extracted {len(written)} files into {data_dir}")
        else:
            check(data_dir)
            print(f"All {len(EXPECTED_FILES)} C-MAPSS files present in {data_dir}")
    except DataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
