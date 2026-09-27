"""Atomic, fail-closed lifecycle state for the Cobot RLT backend."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

ALLOWED_PHASES = frozenset((
    "offline", "loading_machine_a", "loading_replay", "loading_learner",
    "loading_actor", "loading_env", "ready_disarmed", "ready", "stopping", "fault",
))


class LifecycleError(RuntimeError):
    pass


@dataclass(frozen=True)
class LifecycleState:
    generation: str
    phase: str
    supervisor_pid: int
    supervisor_start_ticks: Optional[int]
    step: int
    mode: str
    updated_at: str
    error_code: Optional[str]
    children: Dict[str, int]
    child_start_ticks: Dict[str, int]


_FIELDS = frozenset(LifecycleState.__dataclass_fields__)


def _validate(state: LifecycleState) -> None:
    valid = (
        isinstance(state.generation, str) and bool(state.generation)
        and state.phase in ALLOWED_PHASES
        and isinstance(state.supervisor_pid, int) and not isinstance(state.supervisor_pid, bool)
        and state.supervisor_pid > 0
        and (state.supervisor_start_ticks is None or (
            isinstance(state.supervisor_start_ticks, int)
            and not isinstance(state.supervisor_start_ticks, bool)
            and state.supervisor_start_ticks > 0
        ))
        and isinstance(state.step, int) and not isinstance(state.step, bool) and state.step > 0
        and isinstance(state.mode, str) and bool(state.mode)
        and isinstance(state.updated_at, str) and bool(state.updated_at)
        and (state.error_code is None or isinstance(state.error_code, str))
        and isinstance(state.children, dict)
        and isinstance(state.child_start_ticks, dict)
        and set(state.children) == set(state.child_start_ticks)
    )
    if valid:
        valid = all(
            isinstance(name, str) and bool(name)
            and isinstance(pid, int) and not isinstance(pid, bool) and pid > 0
            and isinstance(state.child_start_ticks[name], int)
            and not isinstance(state.child_start_ticks[name], bool)
            and state.child_start_ticks[name] > 0
            for name, pid in state.children.items()
        )
    if not valid:
        raise LifecycleError("invalid_lifecycle_state")


def write_lifecycle(path: Path, state: LifecycleState) -> None:
    _validate(state)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(asdict(state), sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, str(path))
        directory_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def read_lifecycle(path: Path) -> LifecycleState:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or set(payload) != _FIELDS:
            raise TypeError("field mismatch")
        state = LifecycleState(**payload)
        _validate(state)
        return state
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, LifecycleError, ValueError) as error:
        raise LifecycleError("invalid_lifecycle_state") from error


def process_start_ticks(pid: int) -> int:
    try:
        raw = Path("/proc") / str(int(pid)) / "stat"
        text = raw.read_text(encoding="utf-8")
        end = text.rfind(")")
        if end < 0:
            raise ValueError("missing comm terminator")
        return int(text[end + 2 :].split()[19])
    except (OSError, ValueError, IndexError) as error:
        raise LifecycleError("process_unavailable") from error


def validate_process(pid: int, start_ticks: int, cmdline_fragment: str) -> None:
    if process_start_ticks(pid) != int(start_ticks):
        raise LifecycleError("process_identity_mismatch")
    try:
        cmdline = (Path("/proc") / str(int(pid)) / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8")
    except (OSError, UnicodeError) as error:
        raise LifecycleError("process_unavailable") from error
    if not cmdline_fragment or cmdline_fragment not in cmdline:
        raise LifecycleError("process_cmdline_mismatch")