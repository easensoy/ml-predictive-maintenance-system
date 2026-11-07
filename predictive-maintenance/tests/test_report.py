import json

import pytest

from predmaint.report import END, START, results_table, update_readme


def _rows(table: str, subset: str) -> list[list[str]]:
    return [
        [cell.strip() for cell in line.strip("|").split("|")]
        for line in table.splitlines()
        if line.startswith(f"| {subset} ")
    ]


def test_results_table_reads_metrics(artifacts_dir):
    table = results_table(artifacts_dir)
    metrics = json.loads((artifacts_dir / "FD001" / "metrics.json").read_text())
    rows = _rows(table, "FD001")
    assert len(rows) == 3  # ensemble, single networks, LightGBM
    assert rows[0][2].strip("*") == f"{metrics['test']['rmse']:.2f}"
    assert rows[2][2].strip("*") == f"{metrics['baseline_lightgbm']['rmse']:.2f}"


def test_results_table_bolds_the_best_model_not_always_the_network(artifacts_dir):
    rows = _rows(results_table(artifacts_dir), "FD001")
    rmse = [float(row[2].strip("*").split(" ± ")[0]) for row in rows]
    bold = [row[2].startswith("**") for row in rows]
    assert bold.count(True) >= 1
    assert all(b == (value == min(rmse)) for b, value in zip(bold, rmse, strict=True))


def test_results_table_shows_nominal_coverage_with_wilson_intervals(artifacts_dir):
    table = results_table(artifacts_dir)
    assert "nominal 90% interval" in table
    ensemble = _rows(table, "FD001")[0]
    assert "(" in ensemble[5]
    assert "–" in ensemble[5]  # e.g. "86% (78–91%)"


def test_update_readme_replaces_only_the_marked_block(tmp_path, artifacts_dir):
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{START}\nold numbers\n{END}\noutro\n", encoding="utf-8")
    update_readme(readme, artifacts_dir)
    text = readme.read_text(encoding="utf-8")
    assert text.startswith("intro\n")
    assert text.endswith("outro\n")
    assert "old numbers" not in text
    assert "| FD001 (" in text


def test_update_readme_requires_markers(tmp_path, artifacts_dir):
    readme = tmp_path / "README.md"
    readme.write_text("no markers here", encoding="utf-8")
    with pytest.raises(SystemExit):
        update_readme(readme, artifacts_dir)
