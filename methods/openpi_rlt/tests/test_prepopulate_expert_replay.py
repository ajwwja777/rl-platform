import pytest


def test_prepopulate_payload_is_bounded_and_keeps_episode_count() -> None:
    from methods.openpi_rlt.scripts.prepopulate_expert_replay import (
        build_prepopulate_payload,
    )

    plan = {
        "status": "planned-not-materialized",
        "dataset_id": "plug-pilot",
        "source_manifest_sha256": "a" * 64,
        "unique_episodes": 8,
        "transition_count": 240,
        "chunk_len": 10,
        "stride": 2,
    }
    payload = build_prepopulate_payload(plan, min_transitions=128, max_updates=2000)

    assert payload["status"] == "prepared-not-materialized"
    assert payload["unique_episodes"] == 8
    assert payload["transition_count"] == 240
    assert payload["warmup_update_cap"] == 2000


def test_prepopulate_refuses_too_few_expert_transitions() -> None:
    from methods.openpi_rlt.scripts.prepopulate_expert_replay import (
        build_prepopulate_payload,
    )

    with pytest.raises(ValueError, match="below warmup minimum"):
        build_prepopulate_payload(
            {
                "status": "planned-not-materialized",
                "dataset_id": "small",
                "source_manifest_sha256": "a" * 64,
                "unique_episodes": 1,
                "transition_count": 5,
                "chunk_len": 10,
                "stride": 2,
            },
            min_transitions=128,
            max_updates=2000,
        )

