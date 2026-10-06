import json
from types import SimpleNamespace

import numpy as np
import pytest

from integrations.cobot_runtime.evaluation_trace import evaluation_trace_writer
from integrations.cobot_runtime.shared_model_env import CollectionTrace, SharedEpisodeLifecycle
from methods.openpi_rlt.cobot_adapter.cobot_online_env import CobotOnlineEnv
from methods.openpi_rlt.cobot_adapter.cobot_ros1 import AtomicEpisodeTraceWriter, RosTask2IO
from methods.openpi_rlt.cobot_adapter.trace import EpisodeOutcome


def test_disabled_by_default_does_not_create_output(tmp_path, monkeypatch):
    monkeypatch.delenv("COBOT_RLT_EVALUATION_TRACE_DIR", raising=False)
    assert evaluation_trace_writer() is None
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("root", ["relative", "collection", "collection/nested", "."])
def test_reject_relative_and_collection_overlap(tmp_path, monkeypatch, root):
    monkeypatch.setenv("COBOT_RLT_TRACE_DIR", str(tmp_path / "collection"))
    configured = root if root == "relative" else str(tmp_path / root)
    monkeypatch.setenv("COBOT_RLT_EVALUATION_TRACE_DIR", configured)
    with pytest.raises(ValueError):
        evaluation_trace_writer()
    assert list(tmp_path.iterdir()) == []


def test_each_runtime_gets_unique_namespace(tmp_path, monkeypatch):
    monkeypatch.setenv("COBOT_RLT_EVALUATION_TRACE_DIR", str(tmp_path))
    monkeypatch.delenv("COBOT_RLT_TRACE_DIR", raising=False)
    first, second = evaluation_trace_writer(), evaluation_trace_writer()
    first.append({"action": [.1], "done": False})
    second.append({"action": [.2], "done": False})
    first.finalize("success")
    second.finalize("failure")
    assert len(list(tmp_path.iterdir())) == 2
    assert len(list(tmp_path.rglob("*_success.jsonl"))) == 1
    assert len(list(tmp_path.rglob("*_failure.jsonl"))) == 1


@pytest.mark.parametrize("outcome", ["success", "failure", "aborted"])
def test_numeric_trace_uses_real_terminal_lifecycle_without_collection_write(
        tmp_path, monkeypatch, outcome):
    monkeypatch.setenv("COBOT_RLT_EVALUATION_TRACE_DIR", str(tmp_path / "diagnostic"))
    monkeypatch.setenv("COBOT_RLT_TRACE_DIR", str(tmp_path / "collection"))
    diagnostic = evaluation_trace_writer()
    wrapped = CollectionTrace(AtomicEpisodeTraceWriter(tmp_path / "collection"),
                              SimpleNamespace(collecting=False), evaluation_writer=diagnostic)
    io = RosTask2IO.__new__(RosTask2IO)
    io._trace_writer = wrapped
    events = []
    io._session_application = SimpleNamespace(
        snapshot=lambda: SimpleNamespace(session_id="session", episode_id=7,
                                         task5_episode_uuid="uuid"),
        mark_replay_finalized=lambda: events.append("finalized"))
    env = CobotOnlineEnv.__new__(CobotOnlineEnv)
    env._io, env._last_outcome = io, EpisodeOutcome(outcome)
    wrapped.start_episode()
    record = dict(observation={"state": np.arange(7), "images": {"rgb": np.ones((2, 2, 3))}},
                  next_observation={"state": np.arange(7) + .1, "images": {}},
                  ref_action=np.arange(7) + 1, planned_action=np.arange(7) + 2,
                  action=np.arange(7) + 3, command_publish_started_monotonic=1.1,
                  command_publish_finished_monotonic=1.2, actor_param_version=2500,
                  proposal_semantics="plan_anchor_action; not same-state HIL counterfactual",
                  done=False)
    wrapped.append(record)
    env.mark_replay_finalized()
    assert events == ["finalized"]
    assert list((tmp_path / "collection").iterdir()) == []
    assert "images" in record["observation"]  # Does not mutate live inputs.
    files = list((tmp_path / "diagnostic").rglob("*.jsonl"))
    path, = files
    row = json.loads(path.read_text())
    assert row["ref_action"] == record["ref_action"].tolist()
    assert row["planned_action"] == record["planned_action"].tolist()
    assert row["action"] == record["action"].tolist()
    assert row["next_observation"]["state"] == record["next_observation"]["state"].tolist()
    assert "images" not in row["observation"]
    assert row["trace_purpose"] == "evaluation_diagnostic" and not row["replay_eligible"]
    assert row["task5_episode_uuid"] == "uuid" and row["session_id"] == "session"
    assert row["done"] == (outcome != "aborted")
    assert row["reward"] == float(outcome == "success")
    if outcome == "aborted":
        assert row["truncated"] and not row["replay_eligible"]
    assert row["command_publish_finished_monotonic"] == 1.2
    assert len(row["runtime_provenance"]["code_files_sha256"]) == 5


def test_shared_purpose_switch_does_not_leak_or_finalize_previous_episode(tmp_path, monkeypatch):
    monkeypatch.setenv("COBOT_RLT_EVALUATION_TRACE_DIR", str(tmp_path / "diagnostic"))
    monkeypatch.delenv("COBOT_RLT_TRACE_DIR", raising=False)
    mode = ["evaluation"]
    lifecycle = SharedEpisodeLifecycle(SimpleNamespace(start_episode=lambda x: x), read_use=lambda: mode[0])
    wrapped = CollectionTrace(AtomicEpisodeTraceWriter(tmp_path / "collection"), lifecycle,
                              evaluation_writer=evaluation_trace_writer())
    lifecycle.start_episode(SimpleNamespace())
    wrapped.start_episode()
    wrapped.append({"action": [1.], "done": False})
    mode[0] = "collection"  # UI mode is not the episode's latched purpose.
    wrapped.finalize("success")
    diagnostic, = (tmp_path / "diagnostic").rglob("*_success.jsonl")
    original = diagnostic.read_bytes()
    assert list((tmp_path / "collection").iterdir()) == []
    lifecycle.start_episode(object())
    wrapped.start_episode()
    wrapped.append({"action": [2.], "done": False})
    wrapped.finalize("failure")
    collection, = (tmp_path / "collection").glob("*_failure.jsonl")
    assert json.loads(collection.read_text())["action"] == [2.]
    assert diagnostic.read_bytes() == original
    lifecycle.start_episode(object())
    wrapped.start_episode()
    wrapped.discard()
    wrapped.finalize("aborted")
    assert diagnostic.read_bytes() == original


@pytest.mark.parametrize("shared", [False, True])
def test_opt_in_environment_keeps_replay_gate_closed(tmp_path, monkeypatch, shared):
    from methods.openpi_rlt.plug_v3_yyshadow import right_arm_env
    from integrations.cobot_runtime import evaluation_env, shared_model_env
    from methods.openpi_rlt.cobot_adapter import task5_client
    original_client = task5_client.Task5Client
    monkeypatch.setenv("COBOT_RLT_EVALUATION_TRACE_DIR", str(tmp_path / "diagnostic"))
    monkeypatch.setenv("COBOT_RLT_TRACE_DIR", str(tmp_path / "collection"))
    monkeypatch.setenv("COBOT_DEPLOYMENT_MODEL_ID", "frozen-test")
    recorder = SimpleNamespace(collecting=False, episode_use="evaluation", identity=None)
    application = SimpleNamespace(_task5=recorder, update_takeover=lambda **kw: None,
                                  status=lambda: {"phase": "rollout", "replay_eligible": True})
    io = SimpleNamespace(_session_application=application,
                         _trace_writer=AtomicEpisodeTraceWriter(tmp_path / "collection"))
    env = SimpleNamespace(_io=io, replay_commit_allowed=lambda: True)
    monkeypatch.setattr(right_arm_env, "create_right_arm_online_env", lambda: env)
    created = (shared_model_env.create_shared_env() if shared else evaluation_env.create_evaluation_env())
    assert task5_client.Task5Client is original_client
    assert not created.replay_commit_allowed()
    status = application.status()
    assert status["evaluation_only"] and not status["replay_eligible"]
    io._trace_writer.start_episode()
    io._trace_writer.append({"action": [.1], "done": False})
    io._trace_writer.finalize("success")
    assert len(list((tmp_path / "diagnostic").rglob("*_success.jsonl"))) == 1
    assert list((tmp_path / "collection").iterdir()) == []


def test_collection_abort_default_still_discards(tmp_path):
    writer = AtomicEpisodeTraceWriter(tmp_path)
    writer.append({"action": [1], "done": False})
    writer.finalize("aborted")
    assert not list(tmp_path.iterdir())


def test_diagnostic_abort_in_step_keeps_prior_steps_and_no_fake_action(tmp_path):
    from integrations.cobot_runtime.evaluation_trace import EvaluationTraceWriter
    writer = EvaluationTraceWriter(tmp_path)
    writer.append({"action": [1], "done": False})
    writer.append({"action": [2], "done": True, "outcome": "aborted"})
    writer.finalize("aborted", identity={"session_episode_id": 4})
    path, = tmp_path.glob("*_aborted.jsonl")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[-1]["action"] == [2]
    assert not rows[-1]["done"] and rows[-1]["truncated"]
    assert rows[-1]["session_episode_id"] == 4
    before = path.read_bytes()
    writer.start_episode()
    writer.finalize("aborted")
    assert path.read_bytes() == before


def test_web_next_runtime_optin_settings_are_model_bound(tmp_path, monkeypatch):
    from integrations.cobot_runtime import paths
    monkeypatch.setattr(paths, "RUNTIME_ROOT", tmp_path)
    monkeypatch.delenv("COBOT_RLT_EVALUATION_TRACE_DIR", raising=False)
    monkeypatch.delenv("COBOT_RLT_TRACE_DIR", raising=False)
    settings = tmp_path / "evaluation-diagnostic.json"
    settings.write_text(json.dumps(dict(schema_version=1, enabled=True,
        model_id="plug-v3-warmup-5k", trace_root=str(tmp_path / "numeric"))))
    monkeypatch.setenv("COBOT_DEPLOYMENT_MODEL_ID", "plug-v3-warmup-20k")
    with pytest.raises(ValueError, match="model does not match"):
        evaluation_trace_writer()
    assert not (tmp_path / "numeric").exists()
    monkeypatch.setenv("COBOT_DEPLOYMENT_MODEL_ID", "plug-v3-warmup-5k")
    assert evaluation_trace_writer() is not None
    settings.write_text(json.dumps(dict(enabled=False)))
    assert evaluation_trace_writer() is None
