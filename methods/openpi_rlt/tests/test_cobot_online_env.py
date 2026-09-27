from __future__ import annotations

from types import SimpleNamespace

import numpy as np


class FakeTask2IO:
    def __init__(self, samples):
        self.samples = list(samples)
        self.published: list[np.ndarray] = []
        self.raw_steps: list[dict] = []
        self.home_count = 0
        self.object_ready_count = 0
        self.arm_wait_count = 0
        self.chunk_ready: list[bool] = []
        self.shadow_mode = False
        self.pending_reasons: list[str] = []
        self.pending_outcome = None
        self.replay_finalized_count = 0

    def wait_armed(self):
        self.arm_wait_count += 1

    def sample(self):
        if not self.samples:
            raise AssertionError("fake Task2 sample stream exhausted")
        return self.samples.pop(0)

    def publish_policy_action(self, action):
        self.published.append(np.asarray(action, dtype=np.float32).copy())
        return True

    def record_raw_step(self, record):
        self.raw_steps.append(record)

    def request_home(self):
        self.home_count += 1

    def wait_object_ready(self, _delay_sec):
        self.object_ready_count += 1

    def set_chunk_ready(self, ready):
        self.chunk_ready.append(bool(ready))

    def wait_episode_ready(self):
        return None

    def mark_terminal_pending(self, reason):
        self.pending_reasons.append(str(reason))

    def wait_terminal_outcome(self):
        if self.pending_outcome is None:
            raise AssertionError("test did not configure a terminal outcome")
        return self.pending_outcome

    def mark_replay_finalized(self):
        self.replay_finalized_count += 1


def _obs(value: float) -> dict:
    return {
        "state": np.full(14, value, dtype=np.float32),
        "images": {
            "base_0_rgb": np.zeros((2, 2, 3), dtype=np.uint8),
            "left_wrist_0_rgb": np.zeros((2, 2, 3), dtype=np.uint8),
            "right_wrist_0_rgb": np.zeros((2, 2, 3), dtype=np.uint8),
        },
        "prompt": "Put the objects into the pot.",
    }


def _sample(value: float, mode: str = "policy", outcome=None, paused: bool = False):
    return SimpleNamespace(
        observation=_obs(value),
        mode=mode,
        outcome=outcome,
        paused=paused,
        timestamp=value,
    )


def _plan(value: float, source: int = 1):
    chunk = np.full((10, 14), value, dtype=np.float32)
    return SimpleNamespace(
        action_chunk=chunk,
        ref_chunk=chunk.copy(),
        source=source,
        actor_param_version=7,
        start_features=SimpleNamespace(
            z_rl=np.zeros(2048, dtype=np.float32),
            proprio=np.zeros(14, dtype=np.float32),
            ref_chunk=chunk.copy(),
        ),
    )


def _indexed_plan(ref_base: float = 100.0):
    action_chunk = np.stack(
        [np.full(14, index * 0.001, dtype=np.float32) for index in range(10)]
    )
    ref_chunk = np.stack(
        [np.full(14, ref_base + index, dtype=np.float32) for index in range(10)]
    )
    return SimpleNamespace(
        action_chunk=action_chunk,
        ref_chunk=ref_chunk,
        source=1,
        actor_param_version=7,
        start_features=SimpleNamespace(
            z_rl=np.zeros(2048, dtype=np.float32),
            proprio=np.zeros(14, dtype=np.float32),
            ref_chunk=ref_chunk.copy(),
        ),
    )


def test_env_records_hil_from_task2_mode_and_fresh_replans_after_release() -> None:
    from methods.openpi_rlt.cobot_adapter.cobot_online_env import CobotOnlineEnv
    from methods.openpi_rlt.cobot_adapter.trace import EpisodeOutcome

    io = FakeTask2IO(
        [
            _sample(0.0),
            _sample(0.01, "manual:left"),
            _sample(0.02, "manual:left"),
            _sample(0.03, "policy"),
            _sample(0.04, "policy", EpisodeOutcome.SUCCESS),
        ]
    )
    env = CobotOnlineEnv(
        io,
        chunk_exec_horizon=10,
        control_frequency_hz=20.0,
        max_episode_steps=100,
        joint_step_limit=0.03,
        gripper_step_limit=0.004,
        sleep=lambda _seconds: None,
    )
    observation = env.reset()
    planned_from: list[float] = []

    def planner(plan_observation, _local_step):
        value = float(plan_observation["state"][0])
        planned_from.append(value)
        result = _plan(0.5)
        result.start_features.z_rl[0] = value
        return result

    _, rewards, done, info = env.execute_chunk(observation, planner)

    assert done is True
    assert rewards[-1] == 1.0
    np.testing.assert_allclose(planned_from, [0.0, 0.03])
    assert [step["source"] for step in info["step_trace"]][:2] == [3, 3]
    assert io.raw_steps[0]["expert_mask"] == [True, False]
    assert all(step["expert_mask"] == [True, False] for step in io.raw_steps[:2])
    assert len(io.published) == 1
    assert io.arm_wait_count == 1
    assert io.chunk_ready[:4] == [False, True, False, True]
    assert float(info["chunk_start_features"].z_rl[0]) == 0.0
    np.testing.assert_allclose(info["policy_anchor_features"][0].z_rl[0], 0.03)


def test_next_episode_homes_then_waits_for_object_reset_without_new_start_click() -> None:
    from methods.openpi_rlt.cobot_adapter.cobot_online_env import CobotOnlineEnv
    from methods.openpi_rlt.cobot_adapter.trace import EpisodeOutcome

    io = FakeTask2IO([_sample(0.0), _sample(0.01, outcome=EpisodeOutcome.FAILURE), _sample(0.0)])
    env = CobotOnlineEnv(
        io,
        chunk_exec_horizon=1,
        control_frequency_hz=20.0,
        max_episode_steps=100,
        joint_step_limit=0.03,
        gripper_step_limit=0.004,
        enable_robot_reset=True,
        auto_next_delay_sec=12.0,
        sleep=lambda _seconds: None,
    )
    first = env.reset()
    env.execute_chunk(first, lambda *_args: _plan(0.0))

    second = env.reset()

    assert second["state"].shape == (14,)
    assert io.home_count == 1
    assert io.object_ready_count == 1
    assert env.current_phase_name() == "warmup"
    assert env.episode_phase_name() == "rollout"


def test_reset_motion_is_not_called_when_robot_reset_is_disabled() -> None:
    from methods.openpi_rlt.cobot_adapter.cobot_online_env import CobotOnlineEnv
    from methods.openpi_rlt.cobot_adapter.trace import EpisodeOutcome

    io = FakeTask2IO([_sample(0.0), _sample(0.01, outcome=EpisodeOutcome.DONE), _sample(0.0)])
    env = CobotOnlineEnv(
        io,
        chunk_exec_horizon=1,
        control_frequency_hz=20.0,
        max_episode_steps=100,
        joint_step_limit=0.03,
        gripper_step_limit=0.004,
        enable_robot_reset=False,
        auto_next_delay_sec=0.0,
        sleep=lambda _seconds: None,
    )
    first = env.reset()
    env.execute_chunk(first, lambda *_args: _plan(0.0))
    env.reset()

    assert io.home_count == 0


def test_pause_service_gap_never_publishes_or_mislabels_policy_as_human() -> None:
    from methods.openpi_rlt.cobot_adapter.cobot_online_env import CobotOnlineEnv
    from methods.openpi_rlt.cobot_adapter.trace import EpisodeOutcome

    io = FakeTask2IO(
        [
            _sample(0.0),
            _sample(0.01, paused=True),
            _sample(0.02, "manual:right", paused=True),
            _sample(0.03, "policy"),
            _sample(0.04, outcome=EpisodeOutcome.DONE),
        ]
    )
    env = CobotOnlineEnv(
        io,
        chunk_exec_horizon=10,
        control_frequency_hz=20.0,
        max_episode_steps=100,
        joint_step_limit=0.03,
        gripper_step_limit=0.004,
        sleep=lambda _seconds: None,
    )

    observation = env.reset()
    _, _, done, info = env.execute_chunk(observation, lambda *_args: _plan(0.2))

    assert done is True
    assert len(io.published) == 1
    assert info["step_trace"][0]["expert_mask"] == [False, True]
    assert all(step["source"] != 2 for step in info["step_trace"])


def test_policy_step_records_reference_at_same_action_index() -> None:
    from methods.openpi_rlt.cobot_adapter.cobot_online_env import CobotOnlineEnv
    from methods.openpi_rlt.cobot_adapter.trace import EpisodeOutcome

    io = FakeTask2IO([_sample(0.0), _sample(0.0), _sample(0.001, outcome=EpisodeOutcome.DONE)])
    env = CobotOnlineEnv(
        io,
        chunk_exec_horizon=1,
        control_frequency_hz=20.0,
        max_episode_steps=100,
        joint_step_limit=0.03,
        gripper_step_limit=0.004,
        sleep=lambda _seconds: None,
    )

    observation = env.reset()
    _, _, _, info = env.execute_chunk(observation, lambda *_args: _indexed_plan())

    np.testing.assert_allclose(info["step_trace"][0]["ref_action"], 100.0)


def test_fresh_replan_resets_reference_index_after_hil_release() -> None:
    from methods.openpi_rlt.cobot_adapter.cobot_online_env import CobotOnlineEnv
    from methods.openpi_rlt.cobot_adapter.trace import EpisodeOutcome

    io = FakeTask2IO(
        [
            _sample(0.0),
            _sample(0.0),
            _sample(0.001),
            _sample(0.002, mode="manual:left"),
            _sample(0.003, mode="policy"),
            _sample(0.004, outcome=EpisodeOutcome.DONE),
        ]
    )
    env = CobotOnlineEnv(
        io,
        chunk_exec_horizon=4,
        control_frequency_hz=20.0,
        max_episode_steps=100,
        joint_step_limit=0.03,
        gripper_step_limit=0.004,
        sleep=lambda _seconds: None,
    )
    plans = iter((_indexed_plan(100.0), _indexed_plan(200.0)))

    observation = env.reset()
    _, _, _, info = env.execute_chunk(observation, lambda *_args: next(plans))

    np.testing.assert_allclose(info["step_trace"][0]["ref_action"], 100.0)
    np.testing.assert_allclose(info["step_trace"][1]["ref_action"], 101.0)
    np.testing.assert_allclose(info["step_trace"][2]["ref_action"], 200.0)


def test_publish_rejected_by_takeover_records_feedback_not_unpublished_target() -> None:
    from methods.openpi_rlt.cobot_adapter.cobot_online_env import CobotOnlineEnv
    from methods.openpi_rlt.cobot_adapter.trace import EpisodeOutcome

    class TakeoverRaceIO(FakeTask2IO):
        def publish_policy_action(self, _action):
            return False

    io = TakeoverRaceIO(
        [
            _sample(0.0),
            _sample(0.0),
            _sample(0.02, mode="manual:left"),
            _sample(0.03, mode="manual:left", outcome=EpisodeOutcome.DONE),
        ]
    )
    env = CobotOnlineEnv(
        io,
        chunk_exec_horizon=3,
        control_frequency_hz=20.0,
        max_episode_steps=100,
        joint_step_limit=0.03,
        gripper_step_limit=0.004,
        sleep=lambda _seconds: None,
    )

    observation = env.reset()
    _, _, _, info = env.execute_chunk(observation, lambda *_args: _plan(0.5))

    first = info["step_trace"][0]
    np.testing.assert_allclose(first["action"], 0.02)
    assert first["source"] == 3
    assert first["expert_mask"] == [True, False]
    assert io.published == []


def test_shadow_advances_and_reports_predicted_chunks_without_replay() -> None:
    from methods.openpi_rlt.cobot_adapter.cobot_online_env import CobotOnlineEnv
    from methods.openpi_rlt.cobot_adapter.trace import EpisodeOutcome

    class ShadowIO(FakeTask2IO):
        shadow_mode = True

        def __init__(self, samples):
            super().__init__(samples)
            self.shadow_mode = True

        def publish_policy_action(self, _action):
            return False

    io = ShadowIO([_sample(0.0), _sample(0.0), _sample(0.01, outcome=EpisodeOutcome.SUCCESS)])
    env = CobotOnlineEnv(
        io,
        chunk_exec_horizon=1,
        control_frequency_hz=20.0,
        max_episode_steps=100,
        joint_step_limit=0.03,
        gripper_step_limit=0.004,
        sleep=lambda _seconds: None,
    )

    observation = env.reset()
    _, _, done, info = env.execute_chunk(observation, lambda *_args: _plan(0.02))

    assert done is True
    assert info["step_trace"][0]["shadow"] is True
    assert info["replay_eligible"] is False
    assert env.replay_commit_allowed() is False


def test_unbounded_episode_does_not_invent_a_terminal_at_previous_600_step_limit() -> None:
    from methods.openpi_rlt.cobot_adapter.cobot_online_env import CobotOnlineEnv

    io = FakeTask2IO(
        [_sample(0.0), _sample(0.0), _sample(0.01), _sample(0.02), _sample(0.03)]
    )
    env = CobotOnlineEnv(
        io,
        chunk_exec_horizon=2,
        control_frequency_hz=20.0,
        max_episode_steps=None,
        joint_step_limit=0.03,
        gripper_step_limit=0.004,
        sleep=lambda _seconds: None,
    )

    observation = env.reset()
    _, rewards, done, info = env.execute_chunk(observation, lambda *_args: _plan(0.01))

    assert io.pending_reasons == []
    assert done is False
    assert rewards == [0.0, 0.0]
    assert info["outcome"] is None
    assert env.replay_commit_allowed() is False


def test_terminal_outcome_is_not_dropped_when_pause_and_abort_arrive_together() -> None:
    from methods.openpi_rlt.cobot_adapter.cobot_online_env import CobotOnlineEnv
    from methods.openpi_rlt.cobot_adapter.trace import EpisodeOutcome

    io = FakeTask2IO(
        [_sample(0.0), _sample(0.01, outcome=EpisodeOutcome.ABORTED, paused=True)]
    )
    env = CobotOnlineEnv(
        io,
        chunk_exec_horizon=10,
        control_frequency_hz=20.0,
        max_episode_steps=None,
        joint_step_limit=0.03,
        gripper_step_limit=0.004,
        sleep=lambda _seconds: None,
    )

    observation = env.reset()
    _, _, done, info = env.execute_chunk(observation, lambda *_args: _plan(0.01))

    assert done is True
    assert info["outcome"] == "aborted"
    assert info["drop_transition"] is True
    assert io.raw_steps == []
    assert io.published == []


def test_paused_waiting_intervals_do_not_enter_training_trace() -> None:
    from methods.openpi_rlt.cobot_adapter.cobot_online_env import CobotOnlineEnv
    from methods.openpi_rlt.cobot_adapter.trace import EpisodeOutcome

    io = FakeTask2IO([
        _sample(0.0),
        _sample(0.01),
        _sample(0.02),
        _sample(1.0, paused=True),
        _sample(2.0, "manual:right", paused=True),
        _sample(2.1, "manual:right", paused=True),
        _sample(3.0, "policy", paused=True),
        _sample(9.0, "policy", outcome=EpisodeOutcome.SUCCESS, paused=True),
    ])
    env = CobotOnlineEnv(
        io,
        chunk_exec_horizon=10,
        control_frequency_hz=20.0,
        max_episode_steps=None,
        joint_step_limit=0.03,
        gripper_step_limit=0.004,
        sleep=lambda _seconds: None,
    )
    observation = env.reset()
    _, rewards, done, info = env.execute_chunk(observation, lambda *_: _plan(0.01))

    assert done is True
    assert len(info["step_trace"]) == 3
    assert [step["timestamp"] for step in info["step_trace"]] == [0.02, 2.0, 2.1]
    assert rewards == [0.0, 0.0, 1.0]
    assert info["step_trace"][-1]["done"] is True
    assert len(io.raw_steps) == 3


def test_terminal_reward_updates_last_real_step_after_pause() -> None:
    from methods.openpi_rlt.cobot_adapter.online_runtime import _mark_previous_step_terminal

    step = SimpleNamespace(done=False, reward=0.0, success=0)
    chunk = SimpleNamespace(done=False, success=0)
    raw_episode = SimpleNamespace(steps=[step], chunks=[chunk])
    _mark_previous_step_terminal(raw_episode, 1)
    assert (step.done, step.reward, step.success) == (True, 1.0, 1)
    assert (chunk.done, chunk.success) == (True, 1)


def test_env_exposes_explicit_post_replay_completion_hook() -> None:
    from methods.openpi_rlt.cobot_adapter.cobot_online_env import CobotOnlineEnv

    io = FakeTask2IO([])
    env = CobotOnlineEnv(
        io,
        chunk_exec_horizon=1,
        control_frequency_hz=20.0,
        max_episode_steps=1,
        joint_step_limit=0.03,
        gripper_step_limit=0.004,
        sleep=lambda _seconds: None,
    )

    assert env.persist_upstream_raw_episode() is False
    env.mark_replay_finalized()
    assert io.replay_finalized_count == 1
