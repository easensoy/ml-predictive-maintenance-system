import hashlib
import io
import zipfile

import numpy as np
import pytest

from predmaint import data as cmapss


def test_train_rul_counts_down_to_zero_at_failure(cmapss_dir):
    train = cmapss.load_train(cmapss_dir, "FD001")
    for _, engine in train.groupby("unit"):
        assert engine["rul"].iloc[-1] == 0
        assert engine["rul"].iloc[0] == len(engine) - 1
        assert np.all(np.diff(engine["rul"]) == -1)


def test_test_rul_matches_official_file_at_last_cycle(cmapss_dir):
    test = cmapss.load_test(cmapss_dir, "FD001")
    official = np.loadtxt(cmapss_dir / "RUL_FD001.txt", dtype=int)
    last = test.groupby("unit")["rul"].min().to_numpy()
    np.testing.assert_array_equal(last, official)


def test_split_is_by_engine_disjoint_and_complete():
    units = np.arange(1, 101)
    split = cmapss.split_units(units, seed=42)
    train, val, holdout = (set(split[k]) for k in ("train", "val", "holdout"))
    assert train.isdisjoint(val)
    assert train.isdisjoint(holdout)
    assert val.isdisjoint(holdout)
    assert train | val | holdout == set(units)
    assert (len(split["train"]), len(split["val"]), len(split["holdout"])) == (70, 15, 15)
    np.testing.assert_array_equal(split["val"], cmapss.split_units(units, seed=42)["val"])


def test_rejects_wrong_column_count(tmp_path):
    (tmp_path / "train_FD001.txt").write_text("1 1 0.0 0.0 100.0\n")
    with pytest.raises(cmapss.DataError, match="columns"):
        cmapss.load_train(tmp_path, "FD001")


def test_rejects_unknown_subset(cmapss_dir):
    with pytest.raises(cmapss.DataError, match="Unknown subset"):
        cmapss.load_train(cmapss_dir, "FD009")


def test_archive_with_wrong_checksum_is_rejected(tmp_path):
    with pytest.raises(cmapss.DataError, match="SHA-256 mismatch"):
        cmapss.extract_dataset(b"definitely not the NASA archive", tmp_path)


def _archive(extra_members: dict[str, bytes]) -> tuple[bytes, bytes]:
    inner_buffer = io.BytesIO()
    with zipfile.ZipFile(inner_buffer, "w") as inner:
        for name in cmapss.EXPECTED_FILES:
            inner.writestr(name, f"content of {name}\n")
        for name, payload in extra_members.items():
            inner.writestr(name, payload)
    inner_bytes = inner_buffer.getvalue()
    outer_buffer = io.BytesIO()
    with zipfile.ZipFile(outer_buffer, "w") as outer:
        outer.writestr(cmapss.INNER_MEMBER, inner_bytes)
    return outer_buffer.getvalue(), inner_bytes


def test_extraction_writes_only_whitelisted_files(tmp_path, monkeypatch):
    outer, inner = _archive({"../escape.txt": b"zip slip", "extra.bin": b"unexpected"})
    monkeypatch.setattr(cmapss, "OUTER_SHA256", hashlib.sha256(outer).hexdigest())
    monkeypatch.setattr(cmapss, "INNER_SHA256", hashlib.sha256(inner).hexdigest())
    target = tmp_path / "data"
    written = cmapss.extract_dataset(outer, target)
    assert sorted(p.name for p in written) == sorted(cmapss.EXPECTED_FILES)
    assert sorted(p.name for p in target.iterdir()) == sorted(cmapss.EXPECTED_FILES)
    assert not (tmp_path / "escape.txt").exists()
    cmapss.check(target)
