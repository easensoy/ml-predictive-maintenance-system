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


def test_changed_only_skips_current_images_and_redraws_edited_ones(tmp_path, artifacts_dir):
    out, manifest = tmp_path / "figures", tmp_path / "manifest.json"
    image = out / "uncertainty-light.png"

    def draw(changed_only: bool) -> list:
        ctx = _context(artifacts_dir)
        return render(ctx, ["uncertainty"], ["light"], out, 60, changed_only, manifest)

    assert draw(False) == [image]
    assert draw(True) == []
    image.write_bytes(b"edited by hand")
    assert draw(True) == [image]


def test_every_figure_draws_on_a_synthetic_run(tmp_path, artifacts_dir):
    out = tmp_path / "figures"
    ctx = _context(artifacts_dir, cache=tmp_path / "cache")
    written = render(ctx, list(FIGURES), ["light"], out, 40, manifest_path=tmp_path / "m.json")
    assert {p.name for p in written} == {f"{name}-light.png" for name in FIGURES} - (
        set()
        if (artifacts_dir / "FD001" / "attribution-checks.json").is_file()
        else {"attribution-checks-light.png"}
    )
    # The attribution map was computed once and cached outside the repository.
    assert len(list((tmp_path / "cache" / "figures").glob("attribution-*.npz"))) == 1
