"""The figure pipeline's incremental render must never keep a stale image."""

import shutil

import pandas as pd

from predmaint.figures import FIGURES, RELEASE_DPI, Context, figure_key, render


def _context(artifacts, cache=None):
    if cache is None:
        return Context(artifacts, ["FD001"], "FD001")
    return Context(artifacts, ["FD001"], "FD001", cache=cache)


def test_a_figure_key_changes_with_its_inputs_theme_and_resolution(tmp_path, artifacts_dir):
    artifacts = tmp_path / "artifacts"
    shutil.copytree(artifacts_dir, artifacts)
    key = figure_key(_context(artifacts), "uncertainty", "light", RELEASE_DPI)
    assert key == figure_key(_context(artifacts), "uncertainty", "light", RELEASE_DPI)
    assert key != figure_key(_context(artifacts), "uncertainty", "dark", RELEASE_DPI)
    assert key != figure_key(_context(artifacts), "uncertainty", "light", 110)
    path = artifacts / "FD001" / "predictions" / "test.csv"
    test = pd.read_csv(path)
    test.loc[0, "point"] += 1
    test.to_csv(path, index=False)
    assert key != figure_key(_context(artifacts), "uncertainty", "light", RELEASE_DPI)
    # A file the figure does not read leaves its key alone.
    hero = figure_key(_context(artifacts), "hero", "light", RELEASE_DPI)
    test.loc[0, "point"] += 1
    test.to_csv(path, index=False)
    assert hero == figure_key(_context(artifacts), "hero", "light", RELEASE_DPI)
