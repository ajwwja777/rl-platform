import json
from methods.openpi_rlt.cobot_adapter.cobot_ros1 import AtomicEpisodeTraceWriter


def test_paused_outcome_updates_last_action_without_new_step(tmp_path):
    writer = AtomicEpisodeTraceWriter(tmp_path)
    writer.append({'action': [1.1234567], 'timestamp': 123, 'reward': 0., 'done': False})
    writer.finalize('success', identity={'task5_episode_uuid': 'known-uuid'})
    path, = tmp_path.glob('*_success.jsonl')
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]['action'] == [1.1234567]
    assert rows[0]['timestamp'] == 123
    assert rows[0]['done'] and rows[0]['reward'] == 1
    assert rows[0]['task5_episode_uuid'] == 'known-uuid'
    writer.finalize('success')  # Idempotent, no synthetic action/file.
    assert len(list(tmp_path.iterdir())) == 1


def test_new_episode_preserves_but_never_reuses_unfinished_trace(tmp_path):
    writer = AtomicEpisodeTraceWriter(tmp_path)
    writer.append({'episode': 'old', 'done': False})
    pending, = tmp_path.glob('*.pending.jsonl')
    original = pending.read_bytes()
    writer.start_episode()
    writer.append({'episode': 'new', 'done': False})
    writer.finalize('failure')
    assert pending.read_bytes() == original
    final, = tmp_path.glob('*_failure.jsonl')
    assert json.loads(final.read_text())['episode'] == 'new'


def test_uuid_can_arrive_after_terminal_trace_is_published(tmp_path):
    writer = AtomicEpisodeTraceWriter(tmp_path)
    writer.append({'done': True, 'outcome': 'success', 'action': [0.2]})
    writer.finalize('success', identity={'task5_episode_uuid': 'finalized-uuid'})
    path, = tmp_path.glob('*_success.jsonl')
    assert json.loads(path.read_text())['task5_episode_uuid'] == 'finalized-uuid'
    writer.start_episode()
    writer.finalize('failure')
    assert len(list(tmp_path.iterdir())) == 1


def test_finalize_abort_and_empty_episode_create_no_action(tmp_path):
    writer = AtomicEpisodeTraceWriter(tmp_path)
    writer.finalize('success')
    assert list(tmp_path.iterdir()) == []
    writer.append({'done': False})
    writer.finalize('aborted')
    writer.finalize('failure')
    assert list(tmp_path.iterdir()) == []


def test_finalize_callback_occurs_before_session_advances():
    from methods.openpi_rlt.cobot_adapter.cobot_online_env import CobotOnlineEnv
    from methods.openpi_rlt.cobot_adapter.trace import EpisodeOutcome
    calls = []
    class IO:
        def finalize_raw_episode(self, outcome): calls.append(('trace', outcome))
        def mark_replay_finalized(self): calls.append(('session', None))
    env = CobotOnlineEnv.__new__(CobotOnlineEnv)
    env._io = IO()
    env._last_outcome = EpisodeOutcome.SUCCESS
    env.mark_replay_finalized()
    assert calls == [('trace', EpisodeOutcome.SUCCESS), ('session', None)]
