import json

import pytest


def _manifest(status="frozen"):
    return {
        "schema_version": 1,
        "dataset_id": "plug-pilot",
        "status": status,
        "task_prompt": "Insert the plug into the socket.",
        "episodes": [
            {
                "episode_uuid": "ep-001",
                "status": "complete",
                "frame_count": 40,
                "source_sha256": "a" * 64,
                "approved_segments": [{"start_frame": 5, "end_frame": 34}],
            }
        ],
    }


def test_release_uses_only_frozen_approved_segments(tmp_path) -> None:
    from methods.openpi_rlt.cobot_adapter.segmented_dataset import build_transition_plan

    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(_manifest()), encoding="utf-8")
    plan = build_transition_plan(path, chunk_len=10, stride=2)

    assert plan["status"] == "planned-not-materialized"
    assert plan["unique_episodes"] == 1
    assert plan["transition_count"] == 11
    assert plan["transitions"][0]["start_frame"] == 5
    assert plan["transitions"][-1]["start_frame"] == 25


@pytest.mark.parametrize("status", ["candidate", "collecting", ""])
def test_release_rejects_unfrozen_dataset(tmp_path, status: str) -> None:
    from methods.openpi_rlt.cobot_adapter.segmented_dataset import build_transition_plan

    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(_manifest(status)), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen"):
        build_transition_plan(path, chunk_len=10, stride=2)

