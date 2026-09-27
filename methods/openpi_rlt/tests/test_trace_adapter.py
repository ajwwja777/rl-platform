import importlib.util

import numpy as np
import pytest


def test_trace_adapter_module_is_importable() -> None:
    """Catches a missing project-owned replay trace boundary."""
    assert importlib.util.find_spec("methods.openpi_rlt.cobot_adapter.trace") is not None


def test_hil_replay_preserves_vla_reference_and_records_executed_action() -> None:
    """Catches replacing the VLA reference with HIL actions in online replay."""
    from methods.openpi_rlt.cobot_adapter import trace

    prepare = getattr(trace, "prepare_replay_actions", None)
    assert callable(prepare)
    reference = np.zeros((2, 14), dtype=np.float32)
    executed = np.stack(
        [np.arange(14, dtype=np.float32) + 10, np.arange(14, dtype=np.float32) + 30]
    )

    replay_reference, replay_executed = prepare(reference, executed)

    np.testing.assert_array_equal(replay_reference, reference)
    np.testing.assert_array_equal(replay_executed, executed)
    assert not np.shares_memory(replay_reference, reference)
    assert not np.shares_memory(replay_executed, executed)


def test_replay_actions_reject_shape_mismatch() -> None:
    """Catches reference/executed frame misalignment in replay construction."""
    from methods.openpi_rlt.cobot_adapter.trace import prepare_replay_actions

    with pytest.raises(ValueError, match="does not match"):
        prepare_replay_actions(
            np.zeros((3, 14), dtype=np.float32),
            np.zeros((2, 14), dtype=np.float32),
        )


def test_per_arm_expert_mask_maps_to_upstream_step_sources() -> None:
    """Catches loss of unilateral takeover when upstream expects one source per step."""
    from methods.openpi_rlt.cobot_adapter.trace import (
        ControlSource,
        source_chunk_from_expert_mask,
    )

    expert_mask = np.array(
        [
            [False, False],
            [True, False],
            [False, True],
            [True, True],
        ],
        dtype=bool,
    )

    source_chunk, intervention = source_chunk_from_expert_mask(
        expert_mask,
        policy_source=ControlSource.RL,
    )

    np.testing.assert_array_equal(
        source_chunk,
        np.array(
            [ControlSource.RL, ControlSource.MIXED, ControlSource.MIXED, ControlSource.HUMAN],
            dtype=np.uint8,
        ),
    )
    np.testing.assert_array_equal(intervention, np.array([False, True, True, True]))


def test_source_mapping_rejects_bad_mask_or_non_policy_default() -> None:
    from methods.openpi_rlt.cobot_adapter.trace import (
        ControlSource,
        source_chunk_from_expert_mask,
    )

    with pytest.raises(ValueError, match="expert_mask"):
        source_chunk_from_expert_mask(np.zeros((3, 3), dtype=bool), policy_source=ControlSource.BASE)
    with pytest.raises(ValueError, match="policy_source"):
        source_chunk_from_expert_mask(np.zeros((3, 2), dtype=bool), policy_source=ControlSource.HUMAN)


def test_success_reward_is_sparse_and_terminal_only() -> None:
    """Catches broadcasting a terminal success label across every transition."""
    from methods.openpi_rlt.cobot_adapter import trace

    rewards = trace.terminal_rewards(4, trace.EpisodeOutcome.SUCCESS)

    np.testing.assert_array_equal(rewards, np.array([0, 0, 0, 1], dtype=np.float32))


@pytest.mark.parametrize("outcome_name", ["FAILURE", "DONE"])
def test_non_success_terminal_reward_is_zero(outcome_name: str) -> None:
    """Catches assigning positive reward to failure or unlabeled termination."""
    from methods.openpi_rlt.cobot_adapter import trace

    rewards = trace.terminal_rewards(3, getattr(trace.EpisodeOutcome, outcome_name))

    np.testing.assert_array_equal(rewards, np.zeros(3, dtype=np.float32))


def test_pending_episode_cannot_enter_replay() -> None:
    """Catches training on an episode before its terminal outcome is committed."""
    from methods.openpi_rlt.cobot_adapter import trace

    with pytest.raises(ValueError, match="pending"):
        trace.terminal_rewards(3, trace.EpisodeOutcome.PENDING)


def test_aborted_episode_cannot_enter_replay() -> None:
    """Catches an operator-aborted rollout being trained as a zero-reward failure."""
    from methods.openpi_rlt.cobot_adapter import trace

    with pytest.raises(ValueError, match="aborted"):
        trace.terminal_rewards(3, trace.EpisodeOutcome.ABORTED)


def test_conflicting_manual_outcomes_are_rejected() -> None:
    """Catches an episode being marked both success and failure."""
    from methods.openpi_rlt.cobot_adapter import trace

    with pytest.raises(ValueError, match="success.*failure"):
        trace.outcome_from_flags(success=True, failure=True, done=True)
