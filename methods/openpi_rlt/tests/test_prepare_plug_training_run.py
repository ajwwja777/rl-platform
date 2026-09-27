import json

import pytest


def _dataset_manifest(status="frozen"):
    return {
        "schema_version": 1,
        "status": status,
        "dataset_id": "plug-expert-v1",
        "release_sha256": "b" * 64,
        "total_episodes": 80,
        "total_frames": 8000,
        "task_prompt": "Insert the plug into the socket.",
    }


def test_prepare_run_emits_smoke_and_full_commands(tmp_path) -> None:
    from methods.openpi_rlt.scripts.prepare_plug_training_run import build_run_manifest

    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps(_dataset_manifest()), encoding="utf-8")
    payload = build_run_manifest(
        dataset_manifest=dataset,
        dataset_root=tmp_path / "lerobot",
        project_root=tmp_path / "project",
        upstream_root=tmp_path / "upstream",
        base_params=tmp_path / "base",
        run_id="plug-r2-test",
        full_steps=1000,
    )

    assert payload["status"] == "prepared-not-trained"
    assert payload["stage1"]["train_scope"] == "joint_vlm_and_rlt_token"
    assert payload["stage2"]["train_scope"] == "actor_critic_only"
    assert "--num-train-steps 2" in payload["commands"]["smoke"]
    assert "--num-train-steps 1000" in payload["commands"]["full"]


def test_prepare_run_rejects_unfrozen_data(tmp_path) -> None:
    from methods.openpi_rlt.scripts.prepare_plug_training_run import build_run_manifest

    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps(_dataset_manifest("collecting")), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen"):
        build_run_manifest(
            dataset_manifest=dataset,
            dataset_root=tmp_path / "lerobot",
            project_root=tmp_path / "project",
            upstream_root=tmp_path / "upstream",
            base_params=tmp_path / "base",
            run_id="plug-r2-test",
            full_steps=1000,
        )

