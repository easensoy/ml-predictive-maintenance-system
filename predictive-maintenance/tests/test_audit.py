"""The published numbers must be reproducible from the committed predictions.

The first test audits the real, committed artifacts and documents, so CI fails whenever they
drift apart. The others check that the audit really catches each kind of drift.
"""

import json
import shutil

import pandas as pd
import pytest

from predmaint.audit import audit
from predmaint.config import PROJECT_ROOT
from predmaint.report import END, START, update_readme

REPO_ARTIFACTS = PROJECT_ROOT / "artifacts"
REPO_DOCS = (PROJECT_ROOT.parent / "README.md", PROJECT_ROOT.parent / "MODEL_CARD.md")


def _repo_is_evaluated() -> bool:
    """True once the repository holds evaluated artifacts from the cross-fitted pipeline."""
    path = REPO_ARTIFACTS / "FD001" / "metrics.json"
    if not (REPO_DOCS[0].is_file() and path.is_file()):
        return False
    test = json.loads(path.read_text()).get("test")
    return isinstance(test, dict) and "models" in test


@pytest.mark.skipif(
    not _repo_is_evaluated(), reason="the repository has no evaluated cross-fitted artifacts"
)
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


def test_audit_catches_edited_test_predictions(audited):
    artifacts, readme = audited
    path = artifacts / "FD001" / "predictions" / "test.csv"
    test = pd.read_csv(path)
    test.loc[0, "point"] += 40
    test.to_csv(path, index=False)
    problems = audit(artifacts, readme)
    assert any("rebuilt" in p for p in problems)
    assert any("rmse" in p for p in problems)


def test_audit_catches_a_score_from_a_network_that_saw_the_engine(audited):
    artifacts, readme = audited
    path = artifacts / "FD001" / "predictions" / "calibration-points.csv"
    points = pd.read_csv(path)
    points.loc[0, "member"] = (points.loc[0, "member"] + 1) % 3
    points.to_csv(path, index=False)
    assert any("trained on that engine" in p for p in audit(artifacts, readme))


def test_audit_catches_edited_out_of_fold_alerts(audited):
    artifacts, readme = audited
    path = artifacts / "FD001" / "predictions" / "oof.csv.gz"
    oof = pd.read_csv(path)
    oof["probability"] = 0.0
    oof.to_csv(path, index=False, compression="gzip")
    assert any("out_of_fold" in p for p in audit(artifacts, readme))


def test_audit_catches_a_hand_edited_readme(audited):
    artifacts, readme = audited
    text = readme.read_text(encoding="utf-8")
    readme.write_text(text.replace("| FD001 (", "| FD001  (", 1), encoding="utf-8")
    assert any("README" in p for p in audit(artifacts, readme))
