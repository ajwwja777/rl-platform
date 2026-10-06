import json
import pytest
from types import SimpleNamespace
from integrations.cobot_runtime.shared_model_env import CollectionTrace, SharedEpisodeLifecycle
from integrations.cobot_runtime.evaluation_env import NoTraceWriter
from methods.openpi_rlt.cobot_adapter.cobot_ros1 import RosTask2IO
from methods.openpi_rlt.cobot_adapter.cobot_online_env import CobotOnlineEnv
from methods.openpi_rlt.cobot_adapter.trace import EpisodeOutcome
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


@pytest.mark.parametrize("collecting", [True, False])
@pytest.mark.parametrize("outcome", ["success", "failure", "aborted"])
def test_shared_trace_finalizes_through_real_io_and_env(tmp_path, outcome, collecting):
    writer = AtomicEpisodeTraceWriter(tmp_path)
    wrapped = CollectionTrace(writer, SimpleNamespace(collecting=collecting))
    events = []
    io = RosTask2IO.__new__(RosTask2IO)
    io._trace_writer = wrapped
    io._session_application = SimpleNamespace(
        snapshot=lambda: SimpleNamespace(
            session_id="session-actual", episode_id=3, task5_episode_uuid="episode-uuid"),
        mark_replay_finalized=lambda: events.append("session-advanced"))
    env = CobotOnlineEnv.__new__(CobotOnlineEnv)
    env._io = io
    env._last_outcome = EpisodeOutcome(outcome)
    wrapped.start_episode()
    wrapped.append({"done": False, "timestamp": 123, "action": [.1, .2]})
    env.mark_replay_finalized()
    assert events == ["session-advanced"]
    files = list(tmp_path.iterdir())
    if not collecting or outcome == "aborted":
        assert files == []
    else:
        path, = files
        record = json.loads(path.read_text())
        assert path.name.endswith("_" + outcome + ".jsonl")
        assert record["task5_episode_uuid"] == "episode-uuid"
        assert record["session_id"] == "session-actual"
        assert record["session_episode_id"] == 3
        assert record["done"] and record["action"] == [.1, .2]
        assert record["reward"] == float(outcome == "success")
        before = path.read_bytes()
        wrapped.finalize(outcome)
        assert path.read_bytes() == before


def test_shared_trace_uses_latched_purpose_until_finalization(tmp_path):
    mode = ["collection"]

    class Recorder:
        def start_episode(self, identity):
            return identity

        def finish_episode(self, reference, outcome):
            return reference

    lifecycle = SharedEpisodeLifecycle(Recorder(), read_use=lambda: mode[0])
    lifecycle.start_episode(object())
    writer = AtomicEpisodeTraceWriter(tmp_path)
    wrapped = CollectionTrace(writer, lifecycle)
    wrapped.start_episode()
    wrapped.append({"episode": "first", "done": False})
    lifecycle.finish_episode(object(), "success")
    mode[0] = "evaluation"
    wrapped.finalize("success", identity={"task5_episode_uuid": "first-uuid"})
    path, = tmp_path.iterdir()
    before = path.read_bytes()
    lifecycle.start_episode(SimpleNamespace())
    wrapped.start_episode()
    wrapped.append({"episode": "evaluation", "done": False})
    wrapped.finalize("failure")
    wrapped.discard()
    assert list(tmp_path.iterdir()) == [path] and path.read_bytes() == before


def test_collection_next_episode_and_abort_do_not_reuse_previous_trace(tmp_path):
    writer = AtomicEpisodeTraceWriter(tmp_path)
    wrapped = CollectionTrace(writer, SimpleNamespace(collecting=True))
    wrapped.start_episode()
    wrapped.append({"episode": "first", "done": False})
    wrapped.finalize("success")
    first, = tmp_path.iterdir()
    before = first.read_bytes()
    wrapped.start_episode()
    wrapped.append({"episode": "second", "done": False})
    wrapped.finalize("failure")
    second, = tmp_path.glob("*_failure.jsonl")
    assert json.loads(second.read_text())["episode"] == "second"
    assert first.read_bytes() == before
    wrapped.start_episode()
    wrapped.append({"episode": "third", "done": False})
    wrapped.discard()
    wrapped.finalize("aborted")
    assert set(tmp_path.iterdir()) == {first, second}


def test_evaluation_no_trace_writer_supports_real_io_terminal_lifecycle(tmp_path):
    io = RosTask2IO.__new__(RosTask2IO)
    io._trace_writer = NoTraceWriter()
    io._session_application = None
    writer = io._trace_writer
    for outcome in ["success", "failure", "aborted"]:
        writer.start_episode()
        writer.append({"done": True})
        writer.discard()
        io.finalize_raw_episode(EpisodeOutcome(outcome))
    assert list(tmp_path.iterdir()) == []
