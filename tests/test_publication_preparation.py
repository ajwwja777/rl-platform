"""Publication preparation regressions: fake hardware and deterministic time only."""
import json
import numpy as np
import pytest
from test_async_execution import setup

@pytest.mark.parametrize("hz", [20, 30, 40, 50])
def test_normal_preparation_uses_idle_time_not_cumulative_shift(monkeypatch, hz):
    engine, env, io, clock = setup(monkeypatch, hz)
    sample, apply, safe, publish = engine.sample, engine.filter.apply, env._runtime.safe_policy_target, io.publish_policy_action
    def cost_sample():
        clock.value += .0005
        return sample()
    def cost_filter(*args):
        clock.value += .0013
        return apply(*args)
    def cost_safe(*args):
        clock.value += .0001
        return safe(*args)
    def cost_publish(*args):
        value = publish(*args)
        clock.value += .0005
        return value
    engine.sample, engine.filter.apply = cost_sample, cost_filter
    env._runtime.safe_policy_target, io.publish_policy_action = cost_safe, cost_publish
    try:
        for _ in range(12):
            _, rewards, done, _ = env.execute_chunk(io.sample().observation)
            assert not done and len(rewards) == 10
        times = np.array([t for t, _ in io.published])
        assert len(times) == hz * 6
        assert np.diff(times).min() >= 1/hz - 1e-9
        assert len(io.records) == 120
        assert engine.stats['clock_window_shift_ms'] < 50
        for row in io.records:
            np.testing.assert_array_equal(row['action'], row['publications'][-1]['action'])
    finally:
        engine.close()


def test_unsafe_feedback_clamp_is_rejected_before_publication(monkeypatch):
    engine, env, io, clock = setup(monkeypatch, 50)
    env._runtime.safe_policy_target = lambda *args: np.full(7, 1., np.float32)
    try:
        with pytest.raises(RuntimeError, match="physical rate limit"):
            env.execute_chunk(io.sample().observation)
        assert not io.published
        assert io.pauses[-1]
    finally:
        engine.close()


def test_partial_fault_log_keeps_emitted_commands_without_fake_replay(monkeypatch, capsys):
    engine, env, io, clock = setup(monkeypatch, 50)
    original = engine.filter.apply
    calls = []
    def blocked_prepare(*args):
        calls.append(True)
        if len(calls) == 2:
            clock.value += .2
        return original(*args)
    engine.filter.apply = blocked_prepare
    try:
        with pytest.raises(RuntimeError, match="deadline"):
            env.execute_chunk(io.sample().observation)
        assert len(io.published) == 1
        assert not io.records
        lines = [x for x in capsys.readouterr().out.splitlines() if x.startswith('[rlt-execution-fault] ')]
        assert len(lines) == 1
        fault = json.loads(lines[0].split(' ', 1)[1])
        assert fault['replay_eligible'] is False
        assert fault['logical_step'] == 0
        assert len(fault['partial_publications']) == 1
        assert fault['attempt']['published'] is False
        assert fault['attempt']['target_prepare_ms'] >= 200
        assert fault['partial_publications'][0]['actor_param_version'] == 3500
    finally:
        engine.close()


def test_io_exception_keeps_publication_status_unknown(monkeypatch, capsys):
    engine, env, io, clock = setup(monkeypatch, 50)
    def broken_publish(action):
        raise OSError('synthetic transport failure')
    io.publish_policy_action = broken_publish
    try:
        with pytest.raises(OSError, match='synthetic transport failure'):
            env.execute_chunk(io.sample().observation)
        line = next(x for x in capsys.readouterr().out.splitlines() if x.startswith('[rlt-execution-fault] '))
        fault = json.loads(line.split(' ', 1)[1])
        assert fault['attempt']['published'] is None
        assert not fault['partial_publications']
        assert not io.records and io.pauses[-1]
    finally:
        engine.close()


def test_pause_during_preparation_invalidates_instead_of_clock_fault(monkeypatch):
    def state(t):
        return 'policy', .20 <= t < .30, 'failure' if t >= .45 else None
    engine, env, io, clock = setup(monkeypatch, 50, state_at=state)
    original = engine.filter.apply
    stalled = []
    def prepare(*args):
        if io.published and not stalled:
            stalled.append(True)
            clock.value += .2
        return original(*args)
    engine.filter.apply = prepare
    try:
        _, _, done, info = env.execute_chunk(io.sample().observation)
        assert done and info['outcome'] == 'failure'
        assert 'last_error' not in engine.stats
        assert not any(.20 <= t < .30 for t, _ in io.published)
        assert engine.stats['inference_requests'] >= 2
    finally:
        engine.close()
