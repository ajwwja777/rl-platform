from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRATCH_ROOT = Path("/data/LFT-W02_data/jiaan/scratch/cobot-realworld-rl")
UPSTREAM_ROOT = SCRATCH_ROOT / "openpi-rlt/upstream"
DATASET_ROOT = SCRATCH_ROOT / "fixtures/legacy40-v2.1"
ENTRY = PROJECT_ROOT / "methods/openpi_rlt/scripts/serve_machine_a.py"


def test_machine_a_imports_runtime_before_building_training_config(monkeypatch, tmp_path: Path) -> None:
    """Guards the Cobot pyarrow/JAX import-order crash seen in the legacy π0.5 env."""
    from methods.openpi_rlt.scripts import serve_machine_a

    events: list[str] = []
    runtime = object()
    args = SimpleNamespace(upstream_root=tmp_path / "upstream")
    monkeypatch.setattr(
        serve_machine_a.importlib,
        "import_module",
        lambda name: events.append(f"import:{name}") or runtime,
    )
    monkeypatch.setattr(
        serve_machine_a,
        "_build_config",
        lambda _args: events.append("build-config") or object(),
    )

    loaded_runtime, _config = serve_machine_a._load_runtime_and_config(args)

    assert loaded_runtime is runtime
    assert events == ["import:serve_rlt_policy", "build-config"]


def test_machine_a_dry_run_freezes_cobot_contract_without_loading_model(tmp_path: Path) -> None:
    checkpoint = tmp_path / "5000"
    (checkpoint / "params").mkdir(parents=True)
    env = os.environ.copy()
    env["TMPDIR"] = str(SCRATCH_ROOT / "test-tmp")

    result = subprocess.run(
        [
            sys.executable,
            str(ENTRY),
            "--dry-run",
            "--project-root",
            str(tmp_path / "remote-project"),
            "--upstream-root",
            str(UPSTREAM_ROOT),
            "--dataset-root",
            str(DATASET_ROOT),
            "--base-params",
            "/models/pi05-base/params",
            "--checkpoint-dir",
            str(checkpoint),
            "--port",
            "8100",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["upstream_commit"] == "c1e40ac360185778c98cf20da2820e22d2d415e7"
    assert payload["checkpoint_dir"] == str(checkpoint.resolve())
    assert payload["config_name"] == "cobot_rlt_pi05_joint"
    assert payload["proprio_dim"] == 14
    assert payload["action_dim"] == 14
    assert payload["chunk_len"] == 50
    assert payload["port"] == 8100
    assert payload["shared_prefix_inference"] is True
