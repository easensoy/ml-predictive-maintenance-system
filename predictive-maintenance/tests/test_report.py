import json

import pytest

from predmaint.report import END, START, results_table, update_readme


def test_results_table_reads_metrics(artifacts_dir):
    table = results_table(artifacts_dir)
    metrics = json.loads((artifacts_dir / "FD001" / "metrics.json").read_text())
    assert f"**{metrics['test']['rmse']:.2f}**" in table
    assert table.count("| FD001 |") == 1


def test_update_readme_replaces_only_the_marked_block(tmp_path, artifacts_dir):
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{START}\nold numbers\n{END}\noutro\n", encoding="utf-8")
    update_readme(readme, artifacts_dir)
    text = readme.read_text(encoding="utf-8")
    assert text.startswith("intro\n")
    assert text.endswith("outro\n")
    assert "old numbers" not in text
    assert "| FD001 |" in text


def test_update_readme_requires_markers(tmp_path, artifacts_dir):
    readme = tmp_path / "README.md"
    readme.write_text("no markers here", encoding="utf-8")
    with pytest.raises(SystemExit):
        update_readme(readme, artifacts_dir)
