"""The README's numbers must be reproducible from the committed predictions.

The first test audits the real, committed artifacts and README, so CI fails whenever the three
drift apart. The others check that the audit really catches each kind of drift.
"""

import shutil

import pandas as pd
import pytest

from predmaint.audit import audit
from predmaint.config import PROJECT_ROOT
from predmaint.report import END, START, update_readme

REPO_ARTIFACTS = PROJECT_ROOT / "artifacts"
REPO_DOCS = (PROJECT_ROOT.parent / "README.md", PROJECT_ROOT.parent / "MODEL_CARD.md")


@pytest.mark.skipif(not REPO_DOCS[0].is_file(), reason="needs the repository checkout")
def test_committed_documents_match_committed_predictions():
    assert audit(REPO_ARTIFACTS, *REPO_DOCS) == []


@pytest.fixture
def audited(tmp_path, artifacts_dir):
    """A copy of the synthetic run plus a README generated from it."""
    artifacts = tmp_path / "artifacts"
    shutil.copytree(artifacts_dir, artifacts)
    readme = tmp_path / "README.md"
    readme.write_text(f"# t\n{START}\n{END}\n", encoding="utf-8")
    update_readme(readme, artifacts)
    return artifacts, readme


def test_audit_passes_on_a_consistent_run(audited):
    assert audit(*audited) == []


def test_audit_catches_edited_predictions(audited):
    artifacts, readme = audited
    path = artifacts / "FD001" / "predictions" / "test.csv"
    test = pd.read_csv(path)
    test.loc[0, "point"] += 40
    test.to_csv(path, index=False)
    problems = audit(artifacts, readme)
    assert any("rmse" in p for p in problems)


def test_audit_catches_a_hand_edited_readme(audited):
    artifacts, readme = audited
    text = readme.read_text(encoding="utf-8")
    readme.write_text(text.replace("| FD001 (", "| FD001  (", 1), encoding="utf-8")
    assert any("README" in p for p in audit(artifacts, readme))


def test_audit_catches_edited_lead_times(audited):
    artifacts, readme = audited
    path = artifacts / "FD001" / "predictions" / "holdout.csv"
    holdout = pd.read_csv(path)
    holdout["probability"] = 0.0
    holdout.to_csv(path, index=False)
    assert any("lead" in p for p in audit(artifacts, readme))
