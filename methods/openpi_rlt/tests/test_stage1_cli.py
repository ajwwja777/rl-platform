import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRATCH_ROOT = Path("/data/LFT-W02_data/jiaan/scratch/cobot-realworld-rl")
UPSTREAM_ROOT = SCRATCH_ROOT / "openpi-rlt/upstream"
DATASET_ROOT = SCRATCH_ROOT / "fixtures/legacy40-v2.1"
ENTRY = PROJECT_ROOT / "methods/openpi_rlt/scripts/stage1.py"


def test_stage1_validate_cli_reports_fixed_source_and_dataset_without_gpu() -> None:
    """Catches a launcher that cannot prove its source and dataset before GPU use."""
    env = os.environ.copy()
    env["TMPDIR"] = str(SCRATCH_ROOT / "test-tmp")
    result = subprocess.run(
        [
            sys.executable,
            str(ENTRY),
            "validate",
            "--project-root",
            str(SCRATCH_ROOT / "cli-validation-project"),
            "--upstream-root",
            str(UPSTREAM_ROOT),
            "--dataset-root",
            str(DATASET_ROOT),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["upstream_commit"] == "c1e40ac360185778c98cf20da2820e22d2d415e7"
    assert payload["upstream_clean"] is True
    assert payload["dataset_id"] == "legacy40-v2.1"
    assert payload["total_episodes"] == 40
    assert payload["total_frames"] == 29383
    assert payload["fps"] == 30
    assert payload["action_shape"] == [14]
    assert payload["state_shape"] == [14]
    assert payload["camera_keys"] == [
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    ]


@pytest.mark.parametrize(("command", "expected_name"), [("stats", "cobot_rlt_pi05_joint"), ("train", "cobot_rlt_pi05_joint")])
def test_stage1_dry_run_freezes_paths_and_config_without_starting_work(command: str, expected_name: str) -> None:
    """Catches a stats/train entrypoint that starts work before exposing its resolved contract."""
    project = SCRATCH_ROOT / "cli-dry-run-project"
    env = os.environ.copy()
    env["TMPDIR"] = str(SCRATCH_ROOT / "test-tmp")
    result = subprocess.run(
        [
            sys.executable,
            str(ENTRY),
            command,
            "--dry-run",
            "--project-root",
            str(project),
            "--upstream-root",
            str(UPSTREAM_ROOT),
            "--dataset-root",
            str(DATASET_ROOT),
            "--base-params",
            "/models/pi05_base/params",
            "--exp-name",
            "r1-dry-run",
            "--batch-size",
            "8",
            "--num-workers",
            "4",
            "--fsdp-devices",
            "4",
            "--num-train-steps",
            "5000",
            "--rlt-alpha",
            "1.0",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["command"] == command
    assert payload["config_name"] == expected_name
    assert payload["dataset_repo_id"] == "legacy40-v2.1"
    assert payload["base_params"] == "/models/pi05_base/params"
    assert payload["batch_size"] == 8
    assert payload["num_workers"] == 4
    assert payload["fsdp_devices"] == 4
    assert payload["num_train_steps"] == 5000
    assert payload["rlt_alpha"] == 1.0
    assert payload["assets_base_dir"] == str((project / "assets/openpi-rlt").resolve())
    assert payload["checkpoint_base_dir"] == str((project / "checkpoints/openpi-rlt").resolve())


def test_stage1_stats_cli_runs_official_pipeline_on_real_frames(tmp_path: Path) -> None:
    """Catches a stats entrypoint that bypasses the real LeRobot/Cobot transforms."""
    project = tmp_path / "project"
    env = os.environ.copy()
    env["TMPDIR"] = str(SCRATCH_ROOT / "test-tmp")
    result = subprocess.run(
        [
            sys.executable,
            str(ENTRY),
            "stats",
            "--project-root",
            str(project),
            "--upstream-root",
            str(UPSTREAM_ROOT),
            "--dataset-root",
            str(DATASET_ROOT),
            "--base-params",
            "/models/pi05_base/params",
            "--exp-name",
            "stats-test",
            "--batch-size",
            "4",
            "--num-workers",
            "1",
            "--max-frames",
            "8",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    stats = project / "assets/openpi-rlt/cobot_rlt_pi05_joint/legacy40-v2.1/norm_stats.json"
    payload = json.loads(stats.read_text())
    assert len(payload["norm_stats"]["state"]["mean"]) == 14
    assert len(payload["norm_stats"]["actions"]["mean"]) == 14
