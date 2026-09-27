from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest


def make_state(**changes):
    from methods.openpi_rlt.cobot_adapter.lifecycle import LifecycleState
    values = dict(
        generation="g-1", phase="loading_machine_a", supervisor_pid=os.getpid(),
        supervisor_start_ticks=None, step=4000, mode="eval",
        updated_at="2026-09-16T10:00:00Z", error_code=None,
        children={}, child_start_ticks={},
    )
    values.update(changes)
    return LifecycleState(**values)


def test_atomic_round_trip_and_no_temp_file(tmp_path: Path):
    from methods.openpi_rlt.cobot_adapter.lifecycle import read_lifecycle, write_lifecycle
    path = tmp_path / "state.json"
    state = make_state()
    write_lifecycle(path, state)
    assert read_lifecycle(path) == state
    assert list(tmp_path.glob(".state.json.*.tmp")) == []


def test_malformed_unknown_field_and_phase_fail_closed(tmp_path: Path):
    from methods.openpi_rlt.cobot_adapter.lifecycle import LifecycleError, read_lifecycle
    path = tmp_path / "state.json"
    for payload in (
        "{",
        json.dumps({"phase": "mystery"}),
        json.dumps({**make_state().__dict__, "extra": 1}),
    ):
        path.write_text(payload, encoding="utf-8")
        with pytest.raises(LifecycleError, match="invalid_lifecycle_state"):
            read_lifecycle(path)


def test_registered_process_validates_start_ticks_and_cmdline():
    from methods.openpi_rlt.cobot_adapter.lifecycle import (
        LifecycleError, process_start_ticks, validate_process,
    )
    ticks = process_start_ticks(os.getpid())
    validate_process(os.getpid(), ticks, "pytest")
    with pytest.raises(LifecycleError, match="process_identity_mismatch"):
        validate_process(os.getpid(), ticks + 1, "pytest")
    with pytest.raises(LifecycleError, match="process_cmdline_mismatch"):
        validate_process(os.getpid(), ticks, "definitely-not-in-command-line")


def test_state_requires_child_ticks_for_each_child(tmp_path: Path):
    from methods.openpi_rlt.cobot_adapter.lifecycle import LifecycleError, write_lifecycle
    with pytest.raises(LifecycleError, match="invalid_lifecycle_state"):
        write_lifecycle(tmp_path / "state.json", make_state(children={"env": 10}))