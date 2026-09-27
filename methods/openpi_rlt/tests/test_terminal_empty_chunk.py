from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("success", [0, 1])
def test_terminal_after_pause_marks_last_real_step_without_empty_chunk(success):
    from rlt_online_rl.inference import EnvDriver
    from methods.openpi_rlt.cobot_adapter.online_runtime import install_bimanual_runtime_patch

    install_bimanual_runtime_patch()
    driver = object.__new__(EnvDriver)
    driver._env = SimpleNamespace(cobot_task2_contract=True)
    last_step = SimpleNamespace(done=False, reward=0.0, success=0)
    last_chunk = SimpleNamespace(done=False, success=0)
    raw_episode = SimpleNamespace(steps=[last_step], chunks=[last_chunk])

    observation_idx = driver._append_raw_chunk(
        raw_episode,
        observation_idx=7,
        trace_records=[],
        done=True,
        success=success,
    )

    assert observation_idx == 7
    assert len(raw_episode.steps) == 1
    assert len(raw_episode.chunks) == 1
    assert last_step.done is True
    assert last_step.reward == float(success)
    assert last_step.success == success
    assert last_chunk.done is True
    assert last_chunk.success == success


def test_terminal_without_executed_action_does_not_add_empty_chunk():
    from rlt_online_rl.inference import EnvDriver
    from methods.openpi_rlt.cobot_adapter.online_runtime import install_bimanual_runtime_patch

    install_bimanual_runtime_patch()
    driver = object.__new__(EnvDriver)
    driver._env = SimpleNamespace(cobot_task2_contract=True)
    raw_episode = SimpleNamespace(steps=[], chunks=[])

    assert driver._append_raw_chunk(
        raw_episode,
        observation_idx=0,
        trace_records=[],
        done=True,
        success=0,
    ) == 0
    assert raw_episode.steps == []
    assert raw_episode.chunks == []


def test_terminal_after_pause_keeps_replay_anchor_mapped():
    from rlt_online_rl.inference import EnvDriver
    from methods.openpi_rlt.cobot_adapter.online_runtime import install_bimanual_runtime_patch

    install_bimanual_runtime_patch()
    driver = object.__new__(EnvDriver)
    driver._env = SimpleNamespace(cobot_task2_contract=True)
    driver._rl_config = SimpleNamespace(chunk_len=10)
    steps = [SimpleNamespace(done=False, reward=0.0, success=0) for _ in range(10)]
    chunk = SimpleNamespace(
        step_start=0, step_stop=10, drop_transition=False, done=False, success=0
    )
    episode = SimpleNamespace(steps=steps, chunks=[chunk], policy_start_steps=[])

    driver._append_raw_chunk(
        episode, observation_idx=10, trace_records=[], done=True, success=1
    )
    segments, raw_positions = driver._collect_replay_segments(episode)
    windows, _ = driver._build_chunk_replay_windows(episode, segments, raw_positions)

    assert raw_positions[0] == (0, 0)
    assert len(episode.chunks) == 1
    assert steps[-1].done is True
    assert len(windows) == 1
