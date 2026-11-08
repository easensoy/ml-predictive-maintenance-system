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


def test_test_labels_come_only_from_the_rul_file(cmapss_dir):
    inputs = cmapss.load_test_inputs(cmapss_dir, "FD001")
    assert "rul" not in inputs.columns
    labels = cmapss.load_test_labels(cmapss_dir, "FD001")
    np.testing.assert_array_equal(labels.index, np.sort(inputs["unit"].unique()))


LIVES = np.random.default_rng(0).integers(128, 360, 100)
UNITS = np.arange(1, 101)


def test_folds_are_complete_balanced_and_stratified_by_life():
    folds = cmapss.assign_folds(UNITS, LIVES, 5, seed=3)
    assert sorted(folds) == list(UNITS)
    sizes = np.bincount(list(folds.values()))
    assert sizes.tolist() == [20] * 5
    mean_life = [LIVES[[folds[u] == k for u in UNITS]].mean() for k in range(5)]
    assert max(mean_life) - min(mean_life) < 0.1 * LIVES.mean()
    assert folds == cmapss.assign_folds(UNITS, LIVES, 5, seed=3)
    assert folds != cmapss.assign_folds(UNITS, LIVES, 5, seed=4)


def test_stratified_subset_is_disjoint_and_spans_life_lengths():
    rest, picked = cmapss.stratified_subset(UNITS, LIVES, 0.1, seed=42)
    assert len(picked) == 10
    assert set(rest).isdisjoint(picked)
    assert set(rest) | set(picked) == set(UNITS)
    picked_lives = LIVES[np.isin(UNITS, picked)]
    assert (
        picked_lives.min()
        < np.percentile(LIVES, 20)
        < np.percentile(LIVES, 80)
        < picked_lives.max()
    )


def test_truncation_points_use_test_lengths_and_leave_cycles_to_run():
    lives = {1: 150, 2: 300}
    lengths = np.array([31, 120, 200, 290])
    points = cmapss.test_like_truncations(lives, lengths, 50, np.random.default_rng(0))
    assert len(points) == 100
    for unit, cycle in points:
        assert cycle in lengths
        assert cycle < lives[unit]
    assert {c for u, c in points if u == 1} <= {31, 120}


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
