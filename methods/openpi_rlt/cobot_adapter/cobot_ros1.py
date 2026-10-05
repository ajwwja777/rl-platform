"""ROS 1 I/O for Cobot RLT, isolated behind the tested Task2 environment."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from methods.openpi_rlt.cobot_adapter.cobot_online_env import CobotOnlineEnv
from methods.openpi_rlt.cobot_adapter.rollout_phase import CobotRolloutPhaseController
from methods.openpi_rlt.cobot_adapter.schema import map_cameras
from methods.openpi_rlt.cobot_adapter.trace import EpisodeOutcome

ALOHA_PYTHON = "/home/agilex/miniconda3/envs/aloha/bin/python"
CONTROL_PROJECT = Path(os.environ.get(
    "COBOT_CONTROL_PROJECT_ROOT",
    str(Path(__file__).resolve().parents[3].parent / "cobot-control"),
))
TASK2_HOME_CLI = str(CONTROL_PROJECT / "integrations/legacy_control/task2_homing/task2_home_cli.py")
HOME_FRONT_ONCE = str(Path(__file__).resolve().parents[1] / "scripts" / "home_front_once.py")
COBOT_PLATFORM_HOME = str(CONTROL_PROJECT / "scripts/home.sh")
DEFAULT_TASK_PROMPT = "Open the pot lid, put the object into the pot, then close the lid."

try:
    import rospy
    from cv_bridge import CvBridge
    from sensor_msgs.msg import Image, JointState
    from std_msgs.msg import Bool, String
    from std_srvs.srv import SetBool, SetBoolResponse, Trigger, TriggerResponse
except ImportError:  # Pure helpers and injected tests do not require ROS.
    rospy = None
    CvBridge = None
    Image = JointState = Bool = String = None
    SetBool = SetBoolResponse = Trigger = TriggerResponse = None


CAMERA_TOPICS = {
    "cam_high": "/camera_f/color/image_raw",
    "cam_left_wrist": "/camera_l/color/image_raw",
    "cam_right_wrist": "/camera_r/color/image_raw",
}
JOINT_TOPICS = {
    "left": "/puppet/joint_left",
    "right": "/puppet/joint_right",
}
POLICY_TOPICS = {
    "left": "/task2/policy/joint_left",
    "right": "/task2/policy/joint_right",
}
JOINT_NAMES = [f"joint{index}" for index in range(7)]


@dataclass(frozen=True)
class CobotIOSample:
    observation: dict[str, Any]
    mode: str
    outcome: EpisodeOutcome | None
    paused: bool
    timestamp: float


def build_machine_a_observation(
    *,
    left_state: object,
    right_state: object,
    images: dict[str, object],
    prompt: str,
) -> dict[str, Any]:
    left = np.asarray(left_state, dtype=np.float32)
    right = np.asarray(right_state, dtype=np.float32)
    if left.shape != (7,) or right.shape != (7,):
        raise ValueError(f"left/right feedback must each be 7D, got {left.shape}/{right.shape}")
    state = np.concatenate((left, right)).astype(np.float32, copy=False)
    if not np.all(np.isfinite(state)):
        raise ValueError("Cobot joint feedback contains non-finite values")
    return {"images": map_cameras(images), "state": state, "prompt": str(prompt)}


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


class AtomicEpisodeTraceWriter:
    """Append recoverable metadata, then atomically publish only terminal traces."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._pending: Path | None = None
        self._finalized: Path | None = None
        self._lock = threading.RLock()
        self._discarded = False

    def discard(self) -> None:
        with self._lock:
            if self._pending is not None:
                self._pending.unlink(missing_ok=True)
                self._pending = None
            self._discarded = True

    def start_episode(self) -> None:
        with self._lock:
            # Keep unfinished evidence on disk, but never append a new Episode
            # to it. A pending trace is not a failed/successful outcome.
            self._pending = None
            self._finalized = None
            self._discarded = False

    def finalize(self, outcome: str, *, identity: dict[str, Any] | None = None) -> None:
        """Label the last executed step after pause, without adding an action."""
        with self._lock:
            if outcome == "aborted":
                self.discard()
                return
            if outcome not in {"success", "failure"}:
                raise ValueError("Trace terminal outcome must be success, failure or aborted")
            source = self._pending or self._finalized
            if source is None or self._discarded:
                return
            lines = source.read_text(encoding="utf-8").splitlines()
            if not lines:
                return
            last = json.loads(lines[-1])
            last.update(done=True, outcome=outcome, reward=float(outcome == "success"))
            if identity:
                last.update(identity)
            lines[-1] = json.dumps(last, sort_keys=True)
            # Atomic replacement prevents a partial rewrite from destroying
            # the recoverable original. This happens once per Episode.
            temporary = source.with_suffix(".finalizing.tmp")
            with temporary.open("w", encoding="utf-8") as handle:
                handle.write("\n".join(lines) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            target = source.with_name(source.name.replace(".pending.jsonl", f"_{outcome}.jsonl"))
            os.replace(temporary, target)
            if source != target:
                source.unlink()
            self._pending = None
            self._finalized = target

    def append(self, record: dict[str, Any]) -> None:
        with self._lock:
            if str(record.get("outcome")) == "aborted":
                self.discard()
                return
            if not self._discarded:
                self._append(record)

    def _append(self, record: dict[str, Any]) -> None:
        if self._pending is None:
            self._pending = self._root / f"episode_{time.time_ns()}.pending.jsonl"
        payload = dict(record)
        for name in ("observation", "next_observation"):
            observation = dict(payload.get(name, {}))
            observation.pop("images", None)
            payload[name] = observation
        line = (json.dumps(_jsonable(payload), sort_keys=True) + "\n").encode()
        fd = os.open(self._pending, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, line)
            if bool(record.get("done")):
                os.fsync(fd)
        finally:
            os.close(fd)
        if bool(record.get("done")):
            outcome = str(record.get("outcome") or "done")
            target = self._pending.with_name(self._pending.name.replace(".pending.jsonl", f"_{outcome}.jsonl"))
            os.replace(self._pending, target)
            self._pending = None
            self._finalized = target


class RosTask2IO:
    """Read three cameras/14D state and publish only through Task2 policy topics."""

    def __init__(
        self,
        *,
        shadow_mode: bool,
        trace_dir: str | Path,
        prompt: str = DEFAULT_TASK_PROMPT,
        max_sync_skew_sec: float = 0.10,
        ros_api: Any = None,
        bridge: Any = None,
        image_type: Any = None,
        joint_state_type: Any = None,
        bool_type: Any = None,
        string_type: Any = None,
        set_bool_type: Any = None,
        trigger_type: Any = None,
        response_factory: Any = None,
        home_command_runner: Any = subprocess.run,
    ) -> None:
        self.ros = ros_api or rospy
        if self.ros is None:
            raise RuntimeError("ROS 1 Python modules are unavailable")
        self.bridge = bridge or CvBridge()
        self.image_type = image_type or Image
        self.joint_state_type = joint_state_type or JointState
        self.bool_type = bool_type or Bool
        self.string_type = string_type or String
        self.set_bool_type = set_bool_type or SetBool
        self.trigger_type = trigger_type or Trigger
        self._response_factory = response_factory
        self._home_command_runner = home_command_runner
        self._prompt = prompt
        self._max_sync_skew_sec = float(max_sync_skew_sec)
        self._shadow_mode = bool(shadow_mode)
        self._trace_writer = AtomicEpisodeTraceWriter(trace_dir)
        self._trace_replay_episode_id = None
        self._condition = threading.Condition(threading.RLock())
        self._images: dict[str, Any] = {}
        self._joints: dict[str, Any] = {}
        self._mode = "policy"
        self._paused = True
        self._chunk_ready = False
        self._outcome: EpisodeOutcome | None = None
        self._armed = threading.Event()
        self._object_ready = threading.Event()
        self._episode_ready = threading.Event()
        self._last_sample_stamp = float("-inf")
        self._session_application: Any | None = None
        self._session_http_server: Any | None = None
        self._chunk_count = 0
        self._episode_signal_count = 0

        self.ros.init_node("cobot_rlt_task2", anonymous=False)
        for key, topic in CAMERA_TOPICS.items():
            self.ros.Subscriber(topic, self.image_type, self._image_callback(key), queue_size=2)
        for side, topic in JOINT_TOPICS.items():
            self.ros.Subscriber(topic, self.joint_state_type, self._joint_callback(side), queue_size=10)
        self.ros.Subscriber(
            "/task2/teach_handover/mode", self.string_type, self._mode_callback, queue_size=10
        )

        self._publishers: dict[str, Any] = {}
        if not self._shadow_mode:
            self._publishers = {
                side: self.ros.Publisher(topic, self.joint_state_type, queue_size=1)
                for side, topic in POLICY_TOPICS.items()
            }

        self.ros.Service("/task2/policy/set_paused", self.set_bool_type, self._pause_service)
        self.ros.Service("/task2/policy/arm", self.trigger_type, self._arm_service)
        self.ros.Service("/task2/policy/chunk_ready", self.trigger_type, self._chunk_ready_service)
        self.ros.Service("/cobot_rlt/episode/success", self.trigger_type, self._outcome_service(EpisodeOutcome.SUCCESS))
        self.ros.Service("/cobot_rlt/episode/failure", self.trigger_type, self._outcome_service(EpisodeOutcome.FAILURE))
        self.ros.Service("/cobot_rlt/episode/done", self.trigger_type, self._outcome_service(EpisodeOutcome.DONE))
        self.ros.Service("/cobot_rlt/object_reset_ready", self.trigger_type, self._object_ready_service)

    @property
    def paused(self) -> bool:
        with self._condition:
            return self._paused

    @property
    def shadow_mode(self) -> bool:
        return self._shadow_mode

    def attach_session(self, application: Any, http_server: Any) -> None:
        self._session_application = application
        self._session_http_server = http_server

    def is_policy_mode(self) -> bool:
        with self._condition:
            return self._mode == "policy"

    def set_policy_paused(self, paused: bool) -> None:
        with self._condition:
            self._paused = bool(paused)
            if self._paused:
                self._chunk_ready = False
            self._condition.notify_all()

    def signal_policy_armed(self) -> None:
        """Release the EnvDriver arm gate without starting policy execution."""
        with self._condition:
            self._armed.set()
            self._condition.notify_all()

    def submit_outcome(self, outcome: EpisodeOutcome) -> None:
        with self._condition:
            if EpisodeOutcome(outcome) is EpisodeOutcome.ABORTED:
                discard = getattr(self._trace_writer, "discard", None)
                if discard is not None:
                    discard()
            self._outcome = EpisodeOutcome(outcome)
            self._condition.notify_all()

    def signal_episode_ready(self) -> None:
        with self._condition:
            start_trace = getattr(self._trace_writer, "start_episode", None)
            if start_trace is not None:
                start_trace()
            if self._episode_signal_count == 0:
                self._episode_ready.set()
            else:
                self._object_ready.set()
            self._episode_signal_count += 1
            self._condition.notify_all()

    def wait_episode_ready(self) -> None:
        while not self._episode_ready.wait(timeout=0.25):
            if hasattr(self.ros, "is_shutdown") and self.ros.is_shutdown():
                raise RuntimeError("ROS shut down while waiting for RLT session start")
        self._episode_ready.clear()

    def wait_terminal_outcome(self) -> EpisodeOutcome:
        while True:
            with self._condition:
                if self._outcome is not None:
                    outcome = self._outcome
                    self._outcome = None
                    return outcome
                self._condition.wait(timeout=0.25)
            if hasattr(self.ros, "is_shutdown") and self.ros.is_shutdown():
                raise RuntimeError("ROS shut down while waiting for operator outcome")

    def mark_terminal_pending(self, reason: str) -> None:
        self.set_policy_paused(True)
        if self._session_application is not None:
            self._session_application.mark_terminal_pending(str(reason))

    def mark_replay_finalized(self) -> None:
        if self._session_application is not None:
            self._session_application.mark_replay_finalized()

    def finalize_raw_episode(self, outcome: EpisodeOutcome) -> None:
        identity = {}
        if self._session_application is not None:
            snapshot = self._session_application.snapshot()
            identity = {
                "session_id": snapshot.session_id,
                "session_episode_id": snapshot.episode_id,
                "task5_episode_uuid": snapshot.task5_episode_uuid,
            }
        self._trace_writer.finalize(EpisodeOutcome(outcome).value, identity=identity)

    def report_chunk(self, latency_sec: float, actor_version: int) -> None:
        self._chunk_count += 1
        if self._session_application is not None:
            self._session_application.update_metrics(
                chunk_count=self._chunk_count,
                last_inference_latency_sec=float(latency_sec),
                actor_version=int(actor_version),
            )
            self._session_application.ensure_recorder_active()
        prefix = "rlt-shadow" if self._shadow_mode else "rlt-live"
        print(
            f"[{prefix}] chunk={self._chunk_count} latency={float(latency_sec):.3f}s "
            f"actor_version={int(actor_version)}",
            flush=True,
        )

    def _response(self, success: bool, message: str, *, set_bool: bool = False):
        if self._response_factory is not None:
            return self._response_factory(success, message)
        cls = SetBoolResponse if set_bool else TriggerResponse
        return cls(success=success, message=message)

    def _image_callback(self, key: str):
        def callback(message: Any) -> None:
            with self._condition:
                self._images[key] = message
                self._condition.notify_all()

        return callback

    def _joint_callback(self, side: str):
        def callback(message: Any) -> None:
            with self._condition:
                self._joints[side] = message
                self._condition.notify_all()

        return callback

    def _mode_callback(self, message: Any) -> None:
        with self._condition:
            self._mode = str(getattr(message, "data", "fault"))
            self._condition.notify_all()
        if self._session_application is not None:
            value = self._mode.strip().lower()
            if value == "policy":
                snapshot = self._session_application.update_takeover(left=False, right=False)
                if snapshot.phase.value == "rollout":
                    self.set_policy_paused(False)
            elif value.startswith("manual:"):
                self.set_policy_paused(True)
                sides = value.split(":", 1)[1].split("+")
                self._session_application.update_takeover(
                    left="left" in sides,
                    right="right" in sides,
                )

    def _pause_service(self, request: Any):
        requested = bool(getattr(request, "data", True))
        if not requested and self._session_application is not None:
            phase = self._session_application.snapshot().phase.value
            if phase != "rollout":
                self.set_policy_paused(True)
                return self._response(
                    False,
                    f"RLT session phase {phase} keeps policy paused",
                    set_bool=True,
                )
        with self._condition:
            self._paused = requested
            self._chunk_ready = False if self._paused else self._chunk_ready
            self._condition.notify_all()
        return self._response(True, "paused" if self._paused else "fresh resume requested", set_bool=True)

    def _arm_service(self, _request: Any):
        with self._condition:
            if self._mode != "policy":
                return self._response(False, f"cannot arm while Task2 mode is {self._mode}")
            self._armed.set()
        if self._session_application is not None:
            snapshot = self._session_application.snapshot()
            if snapshot.phase.value == "disarmed":
                self._session_application.arm_operator()
        return self._response(True, "Cobot RLT armed; policy remains paused")

    def _chunk_ready_service(self, _request: Any):
        with self._condition:
            ready = self._chunk_ready
        return self._response(ready, "first fresh chunk ready" if ready else "fresh chunk not ready")

    def _outcome_service(self, outcome: EpisodeOutcome):
        def callback(_request: Any):
            if self._session_application is not None:
                return self._response(False, "RLT session mode requires terminal selection in the web UI")
            with self._condition:
                self._outcome = outcome
                self._condition.notify_all()
            return self._response(True, f"episode outcome={outcome.value}")

        return callback

    def _object_ready_service(self, _request: Any):
        self._object_ready.set()
        return self._response(True, "object reset acknowledged")

    def consume_outcome(self) -> EpisodeOutcome | None:
        with self._condition:
            value = self._outcome
            self._outcome = None
            return value

    def wait_armed(self) -> None:
        while not self._armed.wait(timeout=0.25):
            if hasattr(self.ros, "is_shutdown") and self.ros.is_shutdown():
                raise RuntimeError("ROS shut down before Cobot RLT was armed")

    def set_chunk_ready(self, ready: bool) -> None:
        with self._condition:
            self._chunk_ready = bool(ready)

    @staticmethod
    def _stamp(message: Any) -> float:
        stamp = getattr(getattr(message, "header", None), "stamp", None)
        if hasattr(stamp, "to_sec"):
            return float(stamp.to_sec())
        try:
            return float(stamp)
        except (TypeError, ValueError):
            return time.time()

    def sample_control(self, *, right_arm_only=False, max_age_sec=.2) -> CobotIOSample:
        """Latest age-checked feedback; no wait for a new camera frame.

        Opt-in physical publisher only. The original synchronized sample()
        remains unchanged. Images are converted once per received frame set.
        """
        with self._condition:
            if len(self._images) != 3 or len(self._joints) != 2:
                raise RuntimeError("Control feedback requires all registered cameras/joints")
            if hasattr(self.ros, "is_shutdown") and self.ros.is_shutdown():
                raise RuntimeError("ROS shut down during control publication")
            messages = [*self._images.values(), *self._joints.values()]
            stamps = np.asarray([self._stamp(message) for message in messages])
            now = self.ros.Time.now().to_sec() if hasattr(self.ros, "Time") else time.time()
            if not np.isfinite(stamps).all() or max(now-stamps) > max_age_sec or min(now-stamps) < -.1:
                raise RuntimeError("Control observation is stale or has invalid ROS timestamps; publication paused")
            if max(stamps)-min(stamps) > self._max_sync_skew_sec:
                raise RuntimeError("Control camera/joint skew exceeds the registered limit")
            image_key = tuple((id(message), self._stamp(message)) for message in self._images.values())
            left = np.asarray(self._joints["left"].position[:7],np.float32)
            right = np.asarray(self._joints["right"].position[:7],np.float32)
            if image_key != getattr(self, "_control_image_key", None):
                images = {key:self.bridge.imgmsg_to_cv2(message,"passthrough")
                          for key,message in self._images.items()}
                self._control_images = build_machine_a_observation(
                    left_state=left,right_state=right,images=images,prompt=self._prompt)["images"]
                self._control_image_key = image_key
            state = right.copy() if right_arm_only else np.concatenate([left,right])
            if not np.isfinite(state).all() or state.shape != ((7,) if right_arm_only else (14,)):
                raise RuntimeError("Invalid control joint feedback")
            observation = dict(images=self._control_images,state=state,prompt=self._prompt)
            outcome,self._outcome = self._outcome,None
            return CobotIOSample(observation=observation,mode=self._mode,outcome=outcome,
                                 paused=self._paused,timestamp=float(max(stamps)))

    def sample(self) -> CobotIOSample:
        while True:
            with self._condition:
                if len(self._images) == 3 and len(self._joints) == 2:
                    messages = [*self._images.values(), *self._joints.values()]
                    stamps = [self._stamp(message) for message in messages]
                    newest_common = min(stamps)
                    if newest_common > self._last_sample_stamp and max(stamps) - min(stamps) <= self._max_sync_skew_sec:
                        images = {
                            key: self.bridge.imgmsg_to_cv2(message, "passthrough")
                            for key, message in self._images.items()
                        }
                        observation = build_machine_a_observation(
                            left_state=self._joints["left"].position[:7],
                            right_state=self._joints["right"].position[:7],
                            images=images,
                            prompt=self._prompt,
                        )
                        self._last_sample_stamp = newest_common
                        outcome = self._outcome
                        self._outcome = None
                        return CobotIOSample(
                            observation=observation,
                            mode=self._mode,
                            outcome=outcome,
                            paused=self._paused,
                            timestamp=max(stamps),
                        )
                self._condition.wait(timeout=0.05)
            if hasattr(self.ros, "is_shutdown") and self.ros.is_shutdown():
                raise RuntimeError("ROS shut down while waiting for synchronized Cobot observation")

    def _joint_message(self, positions: np.ndarray):
        message = self.joint_state_type()
        if hasattr(self.ros, "Time"):
            message.header.stamp = self.ros.Time.now()
        message.name = list(JOINT_NAMES)
        message.position = np.asarray(positions, dtype=np.float64).tolist()
        return message

    def publish_policy_action(self, action: object) -> bool:
        target = np.asarray(action, dtype=np.float32)
        if target.shape != (14,) or not np.all(np.isfinite(target)):
            raise ValueError("Task2 policy action must be finite 14D")
        with self._condition:
            if self._paused or self._mode != "policy":
                return False
            if self._shadow_mode:
                return False
            self._publishers["left"].publish(self._joint_message(target[:7]))
            self._publishers["right"].publish(self._joint_message(target[7:]))
            return True

    def record_raw_step(self, record: dict[str, Any]) -> None:
        payload = dict(record)
        payload.setdefault("replay_episode_id", self._trace_replay_episode_id)
        payload.setdefault("record_written_monotonic", time.perf_counter())
        payload.setdefault("action_semantics", (
            "measured_joint_feedback" if payload.get("human_controlled")
            else "shadow_target" if payload.get("shadow") else "published_policy_target"
        ))
        # Optional async execution already records its individual physical
        # publications. A missing proposal stays missing, never reconstructed
        # from feedback or a subsequently trained Actor.
        payload.setdefault("proposal_semantics", "not_recorded")
        self._trace_writer.append(payload)

    def request_home(self) -> None:
        pose = os.environ.get("COBOT_RLT_HOME_POSE", "").strip()
        target = os.environ.get("COBOT_RLT_HOME_TARGET", "front").strip()
        if pose:
            if target not in {"front", "rear", "all", "mid"}:
                raise RuntimeError(f"Unsupported RLT home target: {target}")
            command = [COBOT_PLATFORM_HOME, target, "--pose", pose, "--yes"]
        else:
            command = [ALOHA_PYTHON, HOME_FRONT_ONCE, TASK2_HOME_CLI, "front"]
        result = self._home_command_runner(
            command,
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        stdout = str(getattr(result, "stdout", "")).strip()
        stderr = str(getattr(result, "stderr", "")).strip()
        if stdout:
            print(f"[rlt-home] {stdout}", flush=True)
        if int(getattr(result, "returncode", 1)) != 0:
            detail = stderr or stdout or f"exit status {getattr(result, 'returncode', 'unknown')}"
            raise RuntimeError(f"Task2 home failed: {detail}")

    def wait_object_ready(self, delay_sec: float | None) -> None:
        if delay_sec is not None:
            if delay_sec < 0:
                raise ValueError("automatic next-episode delay cannot be negative")
            time.sleep(delay_sec)
            return
        while not self._object_ready.wait(timeout=0.25):
            if hasattr(self.ros, "is_shutdown") and self.ros.is_shutdown():
                raise RuntimeError("ROS shut down while waiting for object reset")
        self._object_ready.clear()


def create_cobot_online_env() -> CobotOnlineEnv:
    """Factory consumed by the fixed upstream ``--env-factory`` boundary."""
    trace_dir = os.environ.get("COBOT_RLT_TRACE_DIR")
    if not trace_dir:
        raise ValueError("COBOT_RLT_TRACE_DIR must be set to a registered Cobot project path")
    delay_raw = os.environ.get("COBOT_RLT_AUTO_NEXT_DELAY_SEC", "")
    delay = None if not delay_raw else float(delay_raw)
    max_steps_raw = int(os.environ.get("COBOT_RLT_MAX_EPISODE_STEPS", "0"))
    if max_steps_raw < 0:
        raise ValueError("COBOT_RLT_MAX_EPISODE_STEPS cannot be negative")
    max_episode_steps = None if max_steps_raw == 0 else max_steps_raw
    io = RosTask2IO(
        shadow_mode=os.environ.get("COBOT_RLT_SHADOW", "1") != "0",
        trace_dir=trace_dir,
        prompt=os.environ.get("COBOT_RLT_PROMPT", DEFAULT_TASK_PROMPT),
    )
    if os.environ.get("COBOT_RLT_SESSION_UI", "0") == "1":
        from methods.openpi_rlt.cobot_adapter.session import RltSessionController
        from methods.openpi_rlt.cobot_adapter.session_http import (
            RltSessionApplication,
            RltSessionHttpServer,
            SessionHooks,
        )
        from methods.openpi_rlt.cobot_adapter.task5_client import (
            Task5Client,
            Task5EpisodeIdentity,
        )

        task5_root = os.environ.get("COBOT_RLT_TASK5_DATA_ROOT")
        if not task5_root:
            raise ValueError("COBOT_RLT_TASK5_DATA_ROOT must be set for session UI")
        checkpoint_id = os.environ.get("COBOT_RLT_CHECKPOINT_ID")
        if not checkpoint_id:
            raise ValueError("COBOT_RLT_CHECKPOINT_ID must be set for session UI")
        identity = Task5EpisodeIdentity(
            task_id=os.environ.get("COBOT_RLT_TASK_ID", "in_the_pot"),
            model_id=os.environ.get("COBOT_RLT_MODEL_ID", "openpi_rlt"),
            checkpoint_id=checkpoint_id,
            dataset_round=os.environ.get("COBOT_RLT_DATASET_ROUND", "online_r1"),
            data_root=task5_root,
            max_timesteps=int(os.environ.get("COBOT_RLT_TASK5_MAX_TIMESTEPS", "3600")),
        )
        task5 = Task5Client(
            os.environ.get("COBOT_RLT_TASK5_URL", "http://127.0.0.1:8015"),
            timeout_sec=float(os.environ.get("COBOT_RLT_TASK5_TIMEOUT_SEC", "30")),
            min_free_bytes=int(os.environ.get("COBOT_RLT_TASK5_MIN_FREE_BYTES", str(30 * 1024**3))),
        )
        application = RltSessionApplication(
            RltSessionController(),
            task5,
            identity_factory=lambda _episode_id: identity,
            hooks=SessionHooks(
                is_policy_mode=io.is_policy_mode,
                set_policy_paused=io.set_policy_paused,
                submit_outcome=io.submit_outcome,
                signal_episode_ready=io.signal_episode_ready,
                request_front_home=io.request_home,
                signal_policy_armed=io.signal_policy_armed,
            ),
            home_after_terminal=os.environ.get("COBOT_RLT_HOME_AFTER_TERMINAL", "0") == "1",
        )
        application.update_metrics(shadow_mode=io.shadow_mode)
        server = RltSessionHttpServer(
            application,
            host=os.environ.get("COBOT_RLT_SESSION_UI_HOST", "127.0.0.1"),
            port=int(os.environ.get("COBOT_RLT_SESSION_UI_PORT", "8016")),
        )
        server.start()
        io.attach_session(application, server)
    phase_controller = None
    if os.environ.get("COBOT_RLT_DISABLE_PHASE_CONTROLLER", "0") != "1":
        import json

        from rlt_online_rl.inference import ActorClient
        from rlt_online_rl.replay import ReplayClient

        replay = ReplayClient(os.environ.get("COBOT_RLT_REPLAY_URL", "http://127.0.0.1:9102"))
        actor = ActorClient(os.environ.get("COBOT_RLT_ACTOR_URL", "http://127.0.0.1:9101"))
        learner_status_path = Path(os.environ["COBOT_RLT_LEARNER_STATUS_PATH"])

        def read_learner_status() -> dict[str, Any]:
            if not learner_status_path.is_file():
                return {}
            payload = json.loads(learner_status_path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}

        phase_controller = CobotRolloutPhaseController(
            replay_stats_getter=replay.stats,
            actor_version_getter=actor.get_actor_param_version,
            learner_status_getter=read_learner_status,
            warmup_min_size=int(os.environ.get("COBOT_RLT_WARMUP_MIN_SIZE", "600")),
            min_online_actor_version=int(os.environ.get("COBOT_RLT_MIN_ONLINE_ACTOR_VERSION", "250")),
            shutdown_requested=getattr(io.ros, "is_shutdown", lambda: False),
        )
    return CobotOnlineEnv(
        io,
        chunk_exec_horizon=int(os.environ.get("COBOT_RLT_CHUNK_EXEC_HORIZON", "10")),
        control_frequency_hz=float(os.environ.get("COBOT_RLT_CONTROL_HZ", "20")),
        max_episode_steps=max_episode_steps,
        joint_step_limit=float(os.environ.get("COBOT_RLT_JOINT_STEP_LIMIT", "0.03")),
        gripper_step_limit=float(os.environ.get("COBOT_RLT_GRIPPER_STEP_LIMIT", "0.004")),
        enable_robot_reset=os.environ.get("COBOT_RLT_ENABLE_ROBOT_RESET", "0") == "1",
        auto_next_delay_sec=delay,
        collection_phase=os.environ.get("COBOT_RLT_COLLECTION_PHASE", "warmup"),
        phase_controller=phase_controller,
    )
