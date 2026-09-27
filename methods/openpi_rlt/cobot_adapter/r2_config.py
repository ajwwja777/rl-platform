"""Validated project-local configuration for the RLT R2 repair path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class WarmupConfig:
    min_transitions: int
    min_updates: int
    max_updates: int
    eval_interval: int
    updates_per_new_episode: int = 32


@dataclass(frozen=True)
class ResetConfig:
    auto_reset: bool
    verified: bool
    gripper_mode: str
    profile: str = ""


@dataclass(frozen=True)
class R2Config:
    mode: str
    active_arm: str
    explore_gripper: bool
    warmup: WarmupConfig
    reset: ResetConfig
    payload: dict[str, Any]


def _positive(name: str, value: Any) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def load_r2_config(path: str | Path) -> R2Config:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("R2 config must contain a mapping")
    mode = str(payload.get("mode", ""))
    if mode not in {"eval", "explore"}:
        raise ValueError("mode must be eval or explore")
    active_arm = str(payload.get("active_arm", ""))
    if active_arm not in {"left", "right", "both"}:
        raise ValueError("active_arm must be left, right, or both")
    explore_gripper = bool(payload.get("explore_gripper", False))
    if explore_gripper:
        raise ValueError("gripper exploration is disabled for the first plug task")

    warmup_raw = payload.get("warmup")
    reset_raw = payload.get("reset")
    if not isinstance(warmup_raw, dict) or not isinstance(reset_raw, dict):
        raise TypeError("warmup and reset mappings are required")
    warmup = WarmupConfig(
        min_transitions=_positive("warmup.min_transitions", warmup_raw.get("min_transitions")),
        min_updates=_positive("warmup.min_updates", warmup_raw.get("min_updates")),
        max_updates=_positive("warmup.max_updates", warmup_raw.get("max_updates")),
        eval_interval=_positive("warmup.eval_interval", warmup_raw.get("eval_interval")),
        updates_per_new_episode=_positive(
            "warmup.updates_per_new_episode", warmup_raw.get("updates_per_new_episode", 32)
        ),
    )
    if warmup.min_updates > warmup.max_updates:
        raise ValueError("warmup min_updates cannot exceed max_updates")
    if warmup.max_updates > 2_000:
        raise ValueError("warmup max_updates exceeds the reviewed first-task bound")
    if warmup.eval_interval > warmup.max_updates:
        raise ValueError("warmup eval_interval cannot exceed max_updates")

    reset = ResetConfig(
        auto_reset=bool(reset_raw.get("auto_reset", False)),
        verified=bool(reset_raw.get("verified", False)),
        gripper_mode=str(reset_raw.get("gripper_mode", "")),
        profile=str(reset_raw.get("profile", "")),
    )
    if reset.gripper_mode != "hold":
        raise ValueError("plug reset gripper_mode must be hold")
    if reset.auto_reset and not reset.verified:
        raise ValueError("auto reset requires a verified reset profile")
    return R2Config(mode, active_arm, explore_gripper, warmup, reset, payload)
