from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


class FakePublisher:
    def __init__(self, topic):
        self.topic = topic
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class FakeRos:
    def __init__(self):
        self.publishers = {}
        self.subscribers = {}
        self.services = {}

    def init_node(self, *_args, **_kwargs):
        return None

    def Publisher(self, topic, *_args, **_kwargs):
        publisher = FakePublisher(topic)
        self.publishers[topic] = publisher
        return publisher

    def Subscriber(self, topic, _message_type, callback, **_kwargs):
        self.subscribers[topic] = callback
        return callback

    def Service(self, name, _service_type, callback):
        self.services[name] = callback
        return callback

    def ServiceProxy(self, *_args, **_kwargs):
        return lambda *_call_args, **_call_kwargs: SimpleNamespace(success=True, message="ok")


class FakeJointState:
    def __init__(self):
        self.header = SimpleNamespace(stamp=0.0)
        self.name = []
        self.position = []


def _response(success, message):
    return SimpleNamespace(success=success, message=message)


def test_ros1_io_routes_only_to_task2_policy_topics_and_shadow_has_no_publishers(tmp_path: Path) -> None:
    from methods.openpi_rlt.cobot_adapter.cobot_ros1 import RosTask2IO

    live_ros = FakeRos()
    RosTask2IO(
        shadow_mode=False,
        trace_dir=tmp_path / "live",
        ros_api=live_ros,
        bridge=SimpleNamespace(),
        image_type=object,
        joint_state_type=FakeJointState,
        bool_type=object,
        string_type=object,
        set_bool_type=object,
        trigger_type=object,
        response_factory=_response,
    )
    assert set(live_ros.publishers) == {
        "/task2/policy/joint_left",
        "/task2/policy/joint_right",
    }
    assert all(not topic.startswith("/master/") for topic in live_ros.publishers)

    shadow_ros = FakeRos()
    shadow_io = RosTask2IO(
        shadow_mode=True,
        trace_dir=tmp_path / "shadow",
        ros_api=shadow_ros,
        bridge=SimpleNamespace(),
        image_type=object,
        joint_state_type=FakeJointState,
        bool_type=object,
        string_type=object,
        set_bool_type=object,
        trigger_type=object,
        response_factory=_response,
    )
    assert shadow_ros.publishers == {}
    shadow_ros.services["/task2/policy/set_paused"](SimpleNamespace(data=False))
    assert shadow_io.publish_policy_action(np.zeros(14, dtype=np.float32)) is False


def test_ros1_io_exposes_task2_pause_and_rl_episode_services_without_task5(tmp_path: Path) -> None:
    from methods.openpi_rlt.cobot_adapter.cobot_ros1 import RosTask2IO

    ros = FakeRos()
    io = RosTask2IO(
        shadow_mode=True,
        trace_dir=tmp_path,
        ros_api=ros,
        bridge=SimpleNamespace(),
        image_type=object,
        joint_state_type=FakeJointState,
        bool_type=object,
        string_type=object,
        set_bool_type=object,
        trigger_type=object,
        response_factory=_response,
    )
    assert set(ros.services) >= {
        "/task2/policy/set_paused",
        "/task2/policy/arm",
        "/task2/policy/chunk_ready",
        "/cobot_rlt/episode/success",
        "/cobot_rlt/episode/failure",
        "/cobot_rlt/episode/done",
        "/cobot_rlt/object_reset_ready",
    }

    assert ros.services["/task2/policy/set_paused"](SimpleNamespace(data=True)).success
    assert io.paused is True
    assert ros.services["/cobot_rlt/episode/success"](SimpleNamespace()).success
    assert io.consume_outcome().value == "success"
    assert io.consume_outcome() is None


def test_request_home_runs_registered_task2_front_cli_without_a_shell(tmp_path: Path) -> None:
    from methods.openpi_rlt.cobot_adapter.cobot_ros1 import RosTask2IO

    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="front home ok\n", stderr="")

    io = RosTask2IO(
        shadow_mode=False,
        trace_dir=tmp_path,
        ros_api=FakeRos(),
        bridge=SimpleNamespace(),
        image_type=object,
        joint_state_type=FakeJointState,
        bool_type=object,
        string_type=object,
        set_bool_type=object,
        trigger_type=object,
        response_factory=_response,
        home_command_runner=run,
    )

    io.request_home()

    command, kwargs = calls[0]
    assert command == [
        "/home/agilex/miniconda3/envs/aloha/bin/python",
        str(
            Path(__file__).parents[1]
            / "scripts"
            / "home_front_once.py"
        ),
        (
            "/home/agilex/cobot_magic/aloha-devel/Piper-AVP-Teleop/"
            "multi_arm_launch_tools/task2_homing/task2_home_cli.py"
        ),
        "front",
    ]
    assert kwargs["shell"] is False
    assert kwargs["timeout"] == 120


def test_request_home_uses_registered_platform_pose(tmp_path: Path, monkeypatch) -> None:
    from methods.openpi_rlt.cobot_adapter.cobot_ros1 import RosTask2IO

    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="all plug2 home ok\n", stderr="")

    monkeypatch.setenv("COBOT_RLT_HOME_TARGET", "all")
    monkeypatch.setenv("COBOT_RLT_HOME_POSE", "plug2")
    io = RosTask2IO(
        shadow_mode=False,
        trace_dir=tmp_path,
        ros_api=FakeRos(),
        bridge=SimpleNamespace(),
        image_type=object,
        joint_state_type=FakeJointState,
        bool_type=object,
        string_type=object,
        set_bool_type=object,
        trigger_type=object,
        response_factory=_response,
        home_command_runner=run,
    )

    io.request_home()

    command, kwargs = calls[0]
    assert command == [
        "/home/agilex/jiaan/project/cobot-control/scripts/home.sh",
        "all", "--pose", "plug2", "--yes",
    ]
    assert kwargs["shell"] is False
    assert kwargs["timeout"] == 120


def test_home_front_once_does_not_wait_for_non_daemon_ros_threads(tmp_path: Path) -> None:
    target = tmp_path / "home_cli.py"
    target.write_text(
        "import sys, threading, time\n"
        "threading.Thread(target=lambda: time.sleep(30)).start()\n"
        "print('front home ok', flush=True)\n"
        "sys.exit(0)\n",
        encoding="utf-8",
    )
    wrapper = Path(__file__).parents[1] / "scripts" / "home_front_once.py"

    result = __import__("subprocess").run(
        [__import__("sys").executable, str(wrapper), str(target), "front"],
        capture_output=True,
        text=True,
        timeout=3,
        check=False,
    )

    assert result.returncode == 0
    assert "front home ok" in result.stdout


def test_request_home_propagates_front_cli_failure(tmp_path: Path) -> None:
    from methods.openpi_rlt.cobot_adapter.cobot_ros1 import RosTask2IO

    def run(_command, **_kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="service unavailable")

    io = RosTask2IO(
        shadow_mode=False,
        trace_dir=tmp_path,
        ros_api=FakeRos(),
        bridge=SimpleNamespace(),
        image_type=object,
        joint_state_type=FakeJointState,
        bool_type=object,
        string_type=object,
        set_bool_type=object,
        trigger_type=object,
        response_factory=_response,
        home_command_runner=run,
    )

    with pytest.raises(RuntimeError, match="service unavailable"):
        io.request_home()


def test_trace_writer_commits_terminal_episode_and_omits_image_payloads(tmp_path: Path) -> None:
    from methods.openpi_rlt.cobot_adapter.cobot_ros1 import AtomicEpisodeTraceWriter

    writer = AtomicEpisodeTraceWriter(tmp_path)
    writer.append(
        {
            "observation": {"state": np.zeros(14), "images": {"base_0_rgb": np.zeros((8, 8, 3))}},
            "next_observation": {"state": np.ones(14), "images": {"base_0_rgb": np.ones((8, 8, 3))}},
            "action": np.ones(14),
            "ref_action": np.zeros(14),
            "expert_mask": [True, False],
            "source": 3,
            "reward": 1.0,
            "done": True,
            "outcome": "success",
            "timestamp": 123.0,
        }
    )

    assert not list(tmp_path.glob("*.pending.jsonl"))
    committed = list(tmp_path.glob("*_success.jsonl"))
    assert len(committed) == 1
    payload = json.loads(committed[0].read_text().strip())
    assert "images" not in payload["observation"]
    assert payload["expert_mask"] == [True, False]
    assert payload["action"] == [1.0] * 14


def test_task2_release_cannot_unpause_finalizing_web_session(tmp_path: Path) -> None:
    from methods.openpi_rlt.cobot_adapter.cobot_ros1 import RosTask2IO

    class FinalizingSession:
        def snapshot(self):
            return SimpleNamespace(phase=SimpleNamespace(value="finalizing"))

    ros = FakeRos()
    io = RosTask2IO(
        shadow_mode=True,
        trace_dir=tmp_path,
        ros_api=ros,
        bridge=SimpleNamespace(),
        image_type=object,
        joint_state_type=FakeJointState,
        bool_type=object,
        string_type=object,
        set_bool_type=object,
        trigger_type=object,
        response_factory=_response,
    )
    io.attach_session(FinalizingSession(), object())

    response = ros.services["/task2/policy/set_paused"](SimpleNamespace(data=False))

    assert response.success is False
    assert io.paused is True


def test_paused_session_records_hil_and_release_restores_pause(tmp_path: Path) -> None:
    from methods.openpi_rlt.cobot_adapter.cobot_ros1 import RosTask2IO
    from methods.openpi_rlt.cobot_adapter.session import RltSessionController, SessionPhase

    controller = RltSessionController(session_id_factory=lambda: "session-1")
    armed = controller.arm_operator()
    starting = controller.start_session(expected_generation=armed.generation)
    rollout = controller.recording_ready(expected_generation=starting.generation)
    controller.pause_session(expected_generation=rollout.generation)

    ros = FakeRos()
    io = RosTask2IO(
        shadow_mode=True,
        trace_dir=tmp_path,
        ros_api=ros,
        bridge=SimpleNamespace(),
        image_type=object,
        joint_state_type=FakeJointState,
        bool_type=object,
        string_type=object,
        set_bool_type=object,
        trigger_type=object,
        response_factory=_response,
    )
    io.attach_session(controller, object())
    io.set_policy_paused(True)

    ros.subscribers["/task2/teach_handover/mode"](SimpleNamespace(data="manual:right"))
    assert controller.snapshot().phase is SessionPhase.HIL
    assert controller.snapshot().policy_paused is False
    assert io.paused is True  # Task2 manual owns the arm; policy publishing stays gated.

    ros.subscribers["/task2/teach_handover/mode"](SimpleNamespace(data="policy"))
    assert controller.snapshot().phase is SessionPhase.PAUSED
    assert controller.snapshot().policy_paused is True
    assert io.paused is True
