"""Pure conversion helpers from Cobot HIL events to upstream RLT replay semantics."""

from __future__ import annotations

from enum import Enum, IntEnum

import numpy as np


class EpisodeOutcome(str, Enum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILURE = "failure"
    ABORTED = "aborted"
    DONE = "done"
    SAVED = "saved"


class ControlSource(IntEnum):
    """Values fixed by upstream ``rlt_online_rl.replay.TransitionSource``."""

    BASE = 0
    RL = 1
    HUMAN = 2
    MIXED = 3


def _action_chunk(name: str, value: object) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim < 2 or array.shape[-1] != 14:
        raise ValueError(f"{name} must have shape (..., 14), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def prepare_replay_actions(
    policy_reference: object,
    executed_actions: object,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep the VLA reference separate from the action actually executed.

    The online learner selects its BC target through ``source_chunk``. Replacing
    ``ref_chunk`` during HIL would erase the correction the actor must learn.
    """
    reference = _action_chunk("policy_reference", policy_reference)
    executed = _action_chunk("executed_actions", executed_actions)
    if executed.shape != reference.shape:
        raise ValueError(
            f"executed_actions shape {executed.shape} does not match policy_reference {reference.shape}"
        )
    return reference.copy(), executed.copy()


def source_chunk_from_expert_mask(
    expert_mask: object,
    *,
    policy_source: ControlSource | int,
) -> tuple[np.ndarray, np.ndarray]:
    """Map Task2's per-arm takeover mask to upstream's per-step source.

    One-arm takeover becomes ``MIXED`` and two-arm takeover becomes ``HUMAN``.
    The original two-column mask remains a raw Cobot sidecar for audit and any
    future arm-specific objective.
    """
    mask = np.asarray(expert_mask)
    if mask.dtype != np.bool_ or mask.ndim < 1 or mask.shape[-1] != 2:
        raise ValueError(f"expert_mask must be bool with shape (..., 2), got {mask.shape}/{mask.dtype}")

    try:
        source = ControlSource(policy_source)
    except ValueError as error:
        raise ValueError(f"unknown policy_source {policy_source!r}") from error
    if source not in (ControlSource.BASE, ControlSource.RL):
        raise ValueError("policy_source must be BASE or RL")

    takeover_count = np.count_nonzero(mask, axis=-1)
    source_chunk = np.full(mask.shape[:-1], int(source), dtype=np.uint8)
    source_chunk[takeover_count == 1] = int(ControlSource.MIXED)
    source_chunk[takeover_count == 2] = int(ControlSource.HUMAN)
    return source_chunk, takeover_count > 0


def outcome_from_flags(*, success: bool, failure: bool, done: bool) -> EpisodeOutcome:
    if success and failure:
        raise ValueError("success and failure cannot both be true")
    if success:
        return EpisodeOutcome.SUCCESS
    if failure:
        return EpisodeOutcome.FAILURE
    if done:
        return EpisodeOutcome.DONE
    return EpisodeOutcome.PENDING


def terminal_rewards(num_transitions: int, outcome: EpisodeOutcome) -> np.ndarray:
    if num_transitions <= 0:
        raise ValueError("num_transitions must be positive")
    if outcome in (EpisodeOutcome.PENDING, EpisodeOutcome.ABORTED, EpisodeOutcome.SAVED):
        raise ValueError(f"{outcome.value} episode cannot enter replay")

    rewards = np.zeros(num_transitions, dtype=np.float32)
    if outcome is EpisodeOutcome.SUCCESS:
        rewards[-1] = 1.0
    return rewards
