"""Stage-1 entry must accept the deployed pin without loosening source checks."""
from pathlib import Path

import pytest

from methods.openpi_rlt.plug_v3_yyshadow import train_stage1 as entry


@pytest.mark.parametrize("revision", [entry.EXPECTED_UPSTREAM_COMMIT, entry.DEPLOYMENT_UPSTREAM_COMMIT])
def test_known_identical_stage1_sources_are_accepted(monkeypatch, revision):
    calls = []

    def fake_git(root, *args):
        calls.append(args)
        return revision if args == ("rev-parse", "HEAD") else ""

    monkeypatch.setattr(entry, "git", fake_git)
    assert entry.verify_upstream(Path("upstream")) == revision
    assert calls[-1] == ("diff", "--name-only", entry.EXPECTED_UPSTREAM_COMMIT, revision, "--", *entry.STAGE1_PATHS)


@pytest.mark.parametrize("revision,dirty,changed,message", [
    ("unknown", "", "", "unexpected upstream commit"),
    (entry.DEPLOYMENT_UPSTREAM_COMMIT, " M src/openpi/models/pi0.py", "", "worktree is dirty"),
    (entry.DEPLOYMENT_UPSTREAM_COMMIT, "", "scripts/train_rlt.py", "sources differ"),
])
def test_unknown_dirty_or_stage1_drift_is_rejected(monkeypatch, revision, dirty, changed, message):
    def fake_git(root, *args):
        if args == ("rev-parse", "HEAD"):
            return revision
        if args == ("status", "--porcelain"):
            return dirty
        return changed

    monkeypatch.setattr(entry, "git", fake_git)
    with pytest.raises(ValueError, match=message):
        entry.verify_upstream(Path("upstream"))
