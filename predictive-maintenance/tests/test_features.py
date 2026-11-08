import numpy as np
import pytest

from predmaint import data as cmapss
from predmaint.features import FeatureContract, ewma, sliding_windows


@pytest.fixture
def fd001(cmapss_dir):
    return cmapss.load_train(cmapss_dir, "FD001")


def test_sliding_windows_end_at_each_row_and_pad_with_first_row():
    rows = np.arange(12, dtype=np.float32).reshape(6, 2)
    windows = sliding_windows(rows, 4)
    assert windows.shape == (6, 4, 2)
    np.testing.assert_array_equal(windows[-1], rows[2:6])
    np.testing.assert_array_equal(windows[0], np.repeat(rows[:1], 4, axis=0))
    np.testing.assert_array_equal(windows[2][-3:], rows[:3])


def test_windows_never_cross_engine_boundaries(fd001):
    contract = FeatureContract.fit(fd001, "FD001", window=10)
    windows = contract.unit_windows(fd001)
    assert len(windows) == len(fd001)
    second_engine_start = int(np.flatnonzero(fd001["unit"].to_numpy() == 2)[0])
    first_row = contract.engine_features(fd001[fd001["unit"] == 2])[0]
    np.testing.assert_allclose(windows[second_engine_start], np.repeat(first_row[None], 10, axis=0))


def test_serving_window_equals_training_window(fd001):
    """The bug that broke v1: training and serving must build identical inputs."""
    contract = FeatureContract.fit(fd001, "FD001", window=10, ewma_alpha=0.4)
    engine = fd001[fd001["unit"] == 3].reset_index(drop=True)
    training_windows = contract.unit_windows(engine)
    for cut in (1, 5, len(engine)):
        served = contract.window_from_readings(engine.iloc[:cut])
        np.testing.assert_allclose(served, training_windows[cut - 1], rtol=1e-6)


def test_normalization_is_fitted_on_training_engines_only(fd001):
    train_units = fd001[fd001["unit"] <= 10]
    contract = FeatureContract.fit(train_units, "FD001")
    normalized = contract.normalize(train_units)
    np.testing.assert_allclose(normalized.mean(axis=0), 0, atol=1e-4)
    np.testing.assert_allclose(normalized.std(axis=0), 1, atol=1e-3)
    assert np.abs(contract.normalize(fd001[fd001["unit"] > 10]).mean()) > 0


def test_six_operating_regimes_are_recovered(cmapss_dir):
    fd002 = cmapss.load_train(cmapss_dir, "FD002")
    contract = FeatureContract.fit(fd002, "FD002")
    regimes = contract.regimes(fd002[cmapss.SETTINGS].to_numpy())
    assert len(np.unique(regimes)) == 6
    by_setting = (
        fd002.assign(regime=regimes).groupby(fd002["setting_1"].round())["regime"].nunique()
    )
    assert (by_setting == 1).all(), "each flight condition must map to exactly one regime"


def test_contract_round_trips_through_json(tmp_path, fd001):
    contract = FeatureContract.fit(fd001, "FD001", window=10, ewma_alpha=0.3)
    contract.save(tmp_path / "contract.json")
    loaded = FeatureContract.load(tmp_path / "contract.json")
    assert loaded == contract
    np.testing.assert_array_equal(loaded.unit_windows(fd001), contract.unit_windows(fd001))


def test_unsorted_frames_are_rejected(fd001):
    contract = FeatureContract.fit(fd001, "FD001")
    with pytest.raises(ValueError, match="sorted"):
        contract.unit_windows(fd001.iloc[::-1])


def test_ewma_is_causal():
    rows = np.random.default_rng(0).normal(size=(20, 3))
    changed = rows.copy()
    changed[15:] += 100.0
    np.testing.assert_array_equal(ewma(rows, 0.3)[:15], ewma(changed, 0.3)[:15])
