from __future__ import annotations

from pathlib import Path
import pytest

import yaml


def test_resolved_online_config_routes_every_mutable_asset_to_project(tmp_path) -> None:
    from methods.openpi_rlt.scripts.resolve_online_config import build_runtime_payload

    template = Path(__file__).parents[1] / "configs" / "cobot_in_the_pot_online.yaml"
    project = tmp_path / "project"
    payload = build_runtime_payload(template, project)
    run = project / "runs" / "openpi-rlt" / "online-r1-legacy40-v2.1"

    assert payload["experiment"]["rl"]["action_norm_stats_path"] == str(
        project / "methods/openpi_rlt/configs/stats/legacy40-v2.1-action-delta-chunk10.json"
    )
    assert payload["runtime"]["actor_service"]["snapshot_path"] == str(
        run / "actor_snapshot/actor_snapshot.pkl"
    )
    assert payload["runtime"]["learner_service"]["checkpoint_dir"] == str(run / "checkpoints")
    assert payload["runtime"]["learner_service"]["actor_snapshot_path"] == str(
        run / "actor_snapshot/actor_snapshot.pkl"
    )
    assert payload["runtime"]["replay"]["journal_path"] == str(run / "replay/replay_journal.pkl")
    assert payload["runtime"]["monitoring"]["wandb_dir"] == str(run / "wandb")
    assert yaml.safe_dump(payload)


def test_resolved_online_config_accepts_isolated_versioned_run_root(tmp_path) -> None:
    from methods.openpi_rlt.scripts.resolve_online_config import build_runtime_payload

    template = Path(__file__).parents[1] / "configs" / "cobot_in_the_pot_online.yaml"
    project = tmp_path / "project"
    relative = Path("runs/openpi-rlt/online-r1-legacy40-v2.1-session-v2")

    payload = build_runtime_payload(template, project, run_relative=relative)

    expected = project / relative
    assert payload["runtime"]["replay"]["journal_path"] == str(
        expected / "replay/replay_journal.pkl"
    )


@pytest.mark.parametrize("bad", [Path("/absolute/run"), Path("runs/../escape")])
def test_resolved_online_config_rejects_run_root_escape(tmp_path, bad: Path) -> None:
    from methods.openpi_rlt.scripts.resolve_online_config import build_runtime_payload

    template = Path(__file__).parents[1] / "configs" / "cobot_in_the_pot_online.yaml"

    with pytest.raises(ValueError, match="project root"):
        build_runtime_payload(template, tmp_path / "project", run_relative=bad)
