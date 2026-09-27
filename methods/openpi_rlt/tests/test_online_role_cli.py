import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENTRY = PROJECT_ROOT / "methods" / "openpi_rlt" / "scripts" / "online_role.py"
_PROJECT_UPSTREAM = PROJECT_ROOT / "code" / "openpi-rlt"
UPSTREAM_ROOT = Path(
    os.environ.get(
        "COBOT_RLT_TEST_UPSTREAM_ROOT",
        str(
            _PROJECT_UPSTREAM
            if _PROJECT_UPSTREAM.is_dir()
            else "/data/LFT-W02_data/jiaan/scratch/cobot-realworld-rl/openpi-rlt/upstream"
        ),
    )
)


def test_online_role_help_reaches_fixed_upstream_cli() -> None:
    """Catches a wrapper that patches too late or cannot enter the upstream role CLI."""
    result = subprocess.run(
        [
            sys.executable,
            str(ENTRY),
            "--upstream-root",
            str(UPSTREAM_ROOT),
            "--help",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout
    assert "--system.role" in result.stdout


def test_ros_shutdown_is_only_expected_after_operator_marker(tmp_path, monkeypatch) -> None:
    from methods.openpi_rlt.scripts.online_role import _is_expected_operator_shutdown

    marker = tmp_path / "operator-shutdown"
    monkeypatch.setenv("COBOT_RLT_OPERATOR_SHUTDOWN_MARKER", str(marker))
    error = RuntimeError("ROS shut down while waiting for object reset")

    assert _is_expected_operator_shutdown(error) is False
    marker.touch()
    assert _is_expected_operator_shutdown(error) is True
    assert _is_expected_operator_shutdown(RuntimeError("unexpected failure")) is False
