from methods.openpi_rlt.cobot_adapter.cobot_ros1 import AtomicEpisodeTraceWriter


def test_abort_removes_pending_trace_and_late_samples_cannot_recreate_it(tmp_path):
    writer = AtomicEpisodeTraceWriter(tmp_path)
    writer.append({'step': 0, 'done': False})
    assert len(list(tmp_path.iterdir())) == 1
    writer.discard()
    writer.append({'step': 1, 'done': False})
    writer.append({'step': 2, 'done': True, 'outcome': 'aborted'})
    assert list(tmp_path.iterdir()) == []
    writer.start_episode()
    writer.append({'step': 0, 'done': True, 'outcome': 'success'})
    assert len(list(tmp_path.glob('*_success.jsonl'))) == 1


def test_abort_before_any_trace_creates_no_file(tmp_path):
    writer = AtomicEpisodeTraceWriter(tmp_path)
    writer.append({'done': True, 'outcome': 'aborted'})
    assert list(tmp_path.iterdir()) == []
