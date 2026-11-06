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


def load_test(data_dir: Path, subset: str) -> pd.DataFrame:
    """Test trajectories truncated before failure. Adds ``rul`` from the official RUL file."""
    subset = _check_subset(subset)
    df = _read_table(data_dir / f"test_{subset}.txt")
    rul_path = data_dir / f"RUL_{subset}.txt"
    if not rul_path.is_file():
        raise DataError(f"{rul_path} not found. Run `python -m predmaint.data download` first.")
    final_rul = np.loadtxt(rul_path, dtype=np.int64, ndmin=1)
    units = np.sort(df["unit"].unique())
    if len(final_rul) != len(units):
        raise DataError(f"RUL_{subset}.txt has {len(final_rul)} rows for {len(units)} test units")
    rul_at_last = pd.Series(final_rul, index=units)
    last = df.groupby("unit")["cycle"].transform("max")
    df["rul"] = (last - df["cycle"] + df["unit"].map(rul_at_last)).astype("int64")
    return df


def split_units(
    units: np.ndarray, seed: int = 42, val_fraction: float = 0.15, holdout_fraction: float = 0.15
) -> dict[str, np.ndarray]:
    """Split engine ids (never individual rows) into train / validation / holdout sets."""
    units = np.unique(units)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(units)
    n_val = max(1, round(len(units) * val_fraction))
    n_holdout = max(1, round(len(units) * holdout_fraction))
    if n_val + n_holdout >= len(units):
        raise DataError(f"Not enough units ({len(units)}) for a train/val/holdout split")
    return {
        "val": np.sort(shuffled[:n_val]),
        "holdout": np.sort(shuffled[n_val : n_val + n_holdout]),
        "train": np.sort(shuffled[n_val + n_holdout :]),
    }


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
