import json

import pytest

from predmaint.report import END, START, results_block, update_readme


def _rows(block: str, subset: str) -> list[list[str]]:
    return [
        [cell.strip() for cell in line.strip("|").split("|")]
        for line in block.splitlines()
        if line.startswith(f"| {subset} (")
    ]


def test_results_block_reads_metrics(artifacts_dir):
    block = results_block(artifacts_dir)
    metrics = json.loads((artifacts_dir / "FD001" / "metrics.json").read_text())
    rows = _rows(block, "FD001")
    assert [r[1] for r in rows] == ["LSTM", "LightGBM"]
    lstm = metrics["test"]["models"]["lstm"]["summary"]["rmse"]
    assert rows[0][2].strip("*").startswith(f"{lstm['mean']:.2f} ± {lstm['sd']:.2f}")
    assert "paired engine-level bootstrap" in block
    assert "out of fold" in block
    assert "replay holdout" in block


def test_results_block_bolds_the_better_model_not_always_the_network(artifacts_dir):
    rows = _rows(results_block(artifacts_dir), "FD001")
    rmse = [float(r[2].strip("*").split(" ± ")[0]) for r in rows]
    bold = [r[2].startswith("**") for r in rows]
    assert bold.count(True) == 1
    assert bold[rmse.index(min(rmse))]


def test_results_block_reports_nominal_coverage_with_an_interval(artifacts_dir):
    row = _rows(results_block(artifacts_dir), "FD001")[0]
    assert "%" in row[5]
    assert "(" in row[5]
    assert "–" in row[5]


def test_update_readme_replaces_only_the_marked_block(tmp_path, artifacts_dir):
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{START}\nold numbers\n{END}\noutro\n", encoding="utf-8")
    update_readme(readme, artifacts_dir)
    text = readme.read_text(encoding="utf-8")
    assert text.startswith("intro\n")
    assert text.endswith("outro\n")
    assert "old numbers" not in text
    assert "| FD001 (" in text


def test_the_opening_keeps_to_its_word_limits(artifacts_dir):
    """The README abstract stays within 80 words and each finding within 35, on the synthetic
    run and on the committed artifacts."""
    from predmaint.config import PROJECT_ROOT
    from predmaint.report import BLOCKS

    runs = [artifacts_dir]
    if (PROJECT_ROOT / "artifacts" / "FD004" / "metrics.json").is_file():
        runs.append(PROJECT_ROOT / "artifacts")
    for artifacts in runs:
        assert len(BLOCKS["headline"](artifacts).split()) <= 80
        findings = BLOCKS["findings"](artifacts).splitlines()
        assert len(findings) == 3
        assert all(len(line.removeprefix("- ").split()) <= 35 for line in findings)


def test_update_readme_requires_markers(tmp_path, artifacts_dir):
    readme = tmp_path / "README.md"
    readme.write_text("no markers here", encoding="utf-8")
    with pytest.raises(SystemExit):
        update_readme(readme, artifacts_dir)
