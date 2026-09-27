from __future__ import annotations

import subprocess
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "interface_task2_teach_rlt_live.sh"
MODEL_SERVER = Path(__file__).parents[1] / "scripts" / "rlt_model_server.sh"


def test_deploy_script_is_valid_single_entrypoint_with_task5_session_gate() -> None:
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    text = SCRIPT.read_text(encoding="utf-8")
    assert text.startswith("#!/usr/bin/env bash")
    assert 'rlt_model_server.sh" start "${STEP}"' in text
    assert "--system.role replay_manager" in text
    assert "--system.role learner_service" in text
    assert "--system.role actor_service" in text
    assert "--system.role env_driver" in text
    assert 'ONLINE_PY="${PROJECT_ROOT}/envs/rlt-online-py310/bin/python"' in text
    assert "rlt-online-py310-v1" not in text
    assert 'MACHINE_A_OVERLAY="${PROJECT_ROOT}/envs/machine-a-py311-overlay"' in text
    assert "rlt_model_server.sh" in text
    assert "/task2/policy/arm" in text
    assert "/task2/policy/set_paused" in text
    assert "start_task5" not in text
    assert "COBOT_RLT_TASK5_URL" in text
    assert "COBOT_RLT_TASK5_DATA_ROOT" in text
    assert "/home/agilex/cobot_magic/task3/jiaan/realworld_rl/data/task5-rlt-r1" in text
    assert "online-r1-legacy40-v2.1-session-v2" in text
    assert '--run-relative "${RUN_RELATIVE}"' in text
    assert "COBOT_RLT_TASK5_MIN_FREE_BYTES" in text
    assert "COBOT_RLT_SESSION_UI" in text
    assert "8015" in text
    assert "8016" in text
    assert "http://127.0.0.1:${SESSION_UI_PORT}/api/session" in text
    assert 'curl --noproxy "*"' in text
    assert 'COBOT_RLT_MAX_EPISODE_STEPS="0"' in text
    assert 'COBOT_RLT_HOME_AFTER_TERMINAL="1"' in text
    assert 'COBOT_RLT_PROMPT="Open the pot lid, put the object into the pot, then close the lid."' in text
    assert "task2_home_cli.py" in text
    assert "shutdown_session.py" in text
    assert 'setsid "$@" </dev/null >>"${SUPERVISOR_LOG}" 2>&1 &' in text
    assert 'start_console_child "${ONLINE_PY}" "${METHOD_ROOT}/scripts/operator_log_tail.py"' in text
    assert 'start_child "${ONLINE_PY}" "${METHOD_ROOT}/scripts/operator_log_tail.py"' not in text
    assert 'monitor_role "learner" "${LEARNER_PID}"' in text
    assert 'monitor_role "actor" "${ACTOR_PID}"' in text
    assert 'monitor_role "replay" "${REPLAY_PID}"' in text
    assert "Machine B role exited unexpectedly" in text


def test_deploy_script_defaults_to_live_but_requires_operator_enter() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'SHADOW="0"' in text
    assert "read -r" in text
    assert "--shadow" in text
    assert "--auto-reset is not available" in text
    assert "rosservice call /task2/policy/set_paused false" not in text
    assert "开始 Session" in text


def test_model_server_lifecycle_script_is_a_valid_explicit_boundary() -> None:
    result = subprocess.run(
        ["bash", "-n", str(MODEL_SERVER)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    text = MODEL_SERVER.read_text(encoding="utf-8")
    assert 'export NO_PROXY="127.0.0.1,localhost' in text
    assert 'export no_proxy="${NO_PROXY}"' in text
