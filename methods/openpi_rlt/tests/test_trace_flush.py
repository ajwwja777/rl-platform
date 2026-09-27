import json
from methods.openpi_rlt.cobot_adapter.cobot_ros1 import AtomicEpisodeTraceWriter

def test_trace_syncs_at_terminal_not_each_control_step(tmp_path, monkeypatch):
    import os
    calls = []
    real = os.fsync
    def fsync(fd):
        calls.append(fd)
        return real(fd)
    monkeypatch.setattr(os, "fsync", fsync)
    writer = AtomicEpisodeTraceWriter(tmp_path)
    for step in range(20):
        writer.append({"step": step, "done": False})
    assert not calls, "Per-step fsync stalls the shared recorder disk"
    assert len(list(tmp_path.glob("*.pending.jsonl"))) == 1
    writer.append({"step": 20, "done": True, "outcome": "failure"})
    assert calls, "Terminal data must be durable before publication"
    paths = list(tmp_path.glob("*_failure.jsonl"))
    assert len(paths) == 1
    assert [json.loads(line)["step"] for line in paths[0].read_text().splitlines()] == list(range(21))
    assert not list(tmp_path.glob("*.pending.jsonl"))
