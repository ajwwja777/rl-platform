"""Left-arm EXPO actor with existing Task2 ownership and Web's managed gate.

ROS stays in system Python; model inference stays in its private Python 3.11.
No motion on load. Pauses/HIL invalidate pending inference without waiting on GPU.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
from urllib.request import ProxyHandler, Request, build_opener
import numpy as np
from core import Gate, execution_options, limited_plan, validate_observation


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, allow_nan=False))
    os.replace(temporary, path)


class Runtime:
    def __init__(self, args, ros):
        from sensor_msgs.msg import Image, JointState
        from std_msgs.msg import String
        from std_srvs.srv import SetBool, SetBoolResponse, Trigger, TriggerResponse
        self.args, self.ros = args, ros
        self.rtc_delay = int(getattr(args, "rtc_delay", 2))
        if not 1 <= self.rtc_delay <= 8:raise ValueError("RTC delay must be between1and8")
        self.options = execution_options(json.loads(os.environ.get("COBOT_EXECUTION_OPTIONS", '{"enabled":false}')))
        self.gate = Gate()
        self.lock = threading.RLock()
        self.samples = {}
        self.ready = False
        self.chunk_ready = False
        self.last_resume = float("inf")
        self.path = Path(os.environ["COBOT_MODEL_GATE_STATE"])
        self.server = None
        self.rpc = build_opener(ProxyHandler({}))
        self.joint_type = JointState
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.stop = threading.Event()
        self.published = 0
        self.last_gate_save = 0.
        self.mode_seen = 0.
        self.ros.init_node("cobot_expo_left", anonymous=False, disable_signals=True)
        # Fail before advertising any policy interface if another owner exists.
        import rosgraph
        services = dict(rosgraph.Master(self.ros.get_name()).getSystemState()[2])
        if services.get("/task2/policy/set_paused"):
            raise RuntimeError("Another policy owns Task2 pause; release it first")
        self.publisher = None if args.shadow else ros.Publisher("/task2/policy/joint_left", JointState, queue_size=1)
        for key, topic, typ in [
            ("left_wrist_image", "/camera_l/color/image_raw", Image),
            ("right_wrist_image", "/camera_r/color/image_raw", Image),
            ("state", "/puppet/joint_left", JointState),
        ]:
            ros.Subscriber(topic, typ, self.sample_callback(key), queue_size=1)
        ros.Subscriber("/task2/teach_handover/mode", String, self.mode_callback, queue_size=1)
        ros.Subscriber("/task2/teach_handover/fault", String, self.fault_callback, queue_size=1)
        def pause(request):
            with self.lock:
                previous = self.gate.generation
                ok, message = self.gate.request(bool(request.data), str(getattr(request, "_connection_header", {}).get("callerid", "")))
                if self.gate.generation != previous:
                    self.chunk_ready = False
                    self.last_resume = time.monotonic()
                self.save()
                return SetBoolResponse(success=ok, message=message)
        def arm(_):
            with self.lock:
                ok = self.ready and self.gate.mode == "policy" and not self.gate.hil_active
                return TriggerResponse(success=ok, message="armed; remains paused" if ok else "model/handover not ready")
        def chunk(_):
            return TriggerResponse(success=self.chunk_ready and not self.gate.paused, message="fresh chunk ready" if self.chunk_ready else "not ready")
        ros.Service("/task2/policy/set_paused", SetBool, pause)
        ros.Service("/task2/policy/arm", Trigger, arm)
        ros.Service("/task2/policy/chunk_ready", Trigger, chunk)
        self.save()

    def save(self):
        with self.lock:
            atomic_json(self.path, dict(schema_version=1, pid=os.getpid(), model_id=self.args.model_id,
                checkpoint=str(self.args.bundle), ready=self.ready, paused=self.gate.paused,
                manual_pause=self.gate.manual_pause, hil_active=self.gate.hil_active,
                intervention_count=self.gate.intervention_count, generation=self.gate.generation,
                runtime_fault=self.gate.fault, updated_at=time.time(), logical_hz=10,
                published=self.published, robot_publishers=0 if self.args.shadow else 1,
                execution=self.options, **self.status_extra()))
            self.last_gate_save = time.monotonic()

    def status_extra(self):
        return {}

    def resolve_inference(self, value):
        return value

    def policy_command(self):
        return [str(self.args.project/".venv/bin/python"), str(Path(__file__).with_name("policy_server.py")),
                "--bundle", str(self.args.bundle), "--project", str(self.args.project), "--port", str(self.args.port)]

    def fail(self, reason):
        with self.lock:
            self.gate.fail(reason)
            self.chunk_ready = False
            self.save()
        print("EXPO paused: " + str(reason), flush=True)

    def sample_callback(self, key):
        def callback(message):
            with self.lock:
                self.samples[key] = (message, time.monotonic())
        return callback

    def mode_callback(self, message):
        with self.lock:
            previous = self.gate.generation
            self.gate.update_mode(str(message.data).strip().lower())
            self.mode_seen = time.monotonic()
            if self.gate.generation != previous:
                self.chunk_ready = False
                self.last_resume = time.monotonic()
            self.save()

    def fault_callback(self, message):
        if message.data:
            self.fail("handover: " + str(message.data))

    def observation(self, require_after_resume=False):
        from PIL import Image
        now = time.monotonic()
        with self.lock:
            samples = dict(self.samples)
        result = {}
        for key in ("state", "left_wrist_image", "right_wrist_image"):
            if key not in samples:
                raise ValueError("Missing required input: " + key)
            msg, arrival = samples[key]
            age = now - arrival
            stamp = msg.header.stamp.to_sec()
            if age > (.25 if key == "state" else .5) or (stamp and self.ros.Time.now().to_sec() - stamp > .5):
                raise ValueError("Stale required input: " + key)
            if require_after_resume and arrival <= self.last_resume:
                raise ValueError("Waiting for post-resume input: " + key)
            if key == "state":
                if list(msg.name) != ['joint%d' % i for i in range(7)]:
                    raise ValueError("Unexpected physical joint order")
                result[key] = np.asarray(msg.position, np.float32)
            else:
                if msg.encoding not in ("rgb8", "bgr8"):
                    raise ValueError("Unsupported RGB encoding: " + msg.encoding)
                raw = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
                rgb = raw[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
                if msg.encoding == "bgr8":
                    rgb = rgb[..., ::-1]
                result[key] = np.asarray(Image.fromarray(rgb).resize((320, 180), Image.BICUBIC)).copy()
        return validate_observation(result)

    def infer(self, observation, prefix):
        stream = io.BytesIO()
        np.savez(stream, **observation, prefix=np.asarray(prefix, np.float32).reshape(-1, 7))
        request = Request('http://127.0.0.1:%d/infer' % self.args.port,
                          data=stream.getvalue(), headers={"Content-Type": "application/octet-stream"})
        with self.rpc.open(request, timeout=30) as response:
            payload = json.load(response)
        return np.asarray(payload["actions"], np.float32)

    def validate_server_status(self, info):
        pass

    def start_server(self):
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", self.args.port)) == 0:
                raise RuntimeError("EXPO inference port occupied")
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            env.pop(key, None)
        storage = str(self.args.project.parents[1])
        env.update(CUDA_VISIBLE_DEVICES="0", XLA_PYTHON_CLIENT_PREALLOCATE="false", OMP_NUM_THREADS="8",
            XDG_CACHE_HOME=storage+'/.cache', HF_HOME=storage+'/.cache/huggingface',
            OPENPI_DATA_HOME=storage+'/.cache/openpi', TMPDIR=storage+'/tmp/expo-ft',
            JAX_COMPILATION_CACHE_DIR=storage+'/.cache/jax/expo-web', CUDA_CACHE_PATH=storage+'/.cache/nvidia')
        Path(env['JAX_COMPILATION_CACHE_DIR']).mkdir(parents=True, exist_ok=True)
        command = self.policy_command()
        self.server = subprocess.Popen(command, env=env, stdin=subprocess.PIPE if getattr(self.args,'remote_config',None) else None)
        deadline = time.monotonic() + 600
        while not self.stop.wait(.3):
            if self.server.poll() is not None:
                raise RuntimeError("Native EXPO policy process exited during load")
            if time.monotonic() > deadline:
                raise RuntimeError("EXPO load timeout")
            try:
                with self.rpc.open('http://127.0.0.1:%d/ready' % self.args.port, timeout=1) as response:
                    info = json.load(response)
                if info['ready'] and info['step'] == self.args.step:
                    self.validate_server_status(info)
                    self.ready = True
                    self.save()
                    print("EXPO ready and PAUSED; left 7D, two wrist cameras, logical10Hz", flush=True)
                    return
            except (OSError, ValueError):
                pass

    def make_plan(self, actions, last):
        return limited_plan(actions, last, self.options['smoothing'])

    def prepare_plan(self, plan, index, last):
        return plan

    def interpolate_command(self, last, target, fraction, index):
        return last + (target - last) * fraction

    def run(self):
        self.start_server()
        epoch = -1
        plan = None
        index = 0
        last = None
        next_tick = 0.
        while not self.stop.is_set() and not self.ros.is_shutdown():
            if self.server.poll() is not None:
                self.fail("native policy process exited")
                break
            now = time.monotonic()
            with self.lock:
                paused = self.gate.paused
                generation = self.gate.generation
                if now - self.last_gate_save > .25:
                    self.save()
            if paused or generation != epoch:
                plan, index, last = None, 0, None
                epoch = generation
                if self.future is not None and self.future.done():
                    # GPU results from before HIL/stop are never reused.
                    self.future = None
                if paused:
                    self.stop.wait(.01)
                    continue
            if plan is None:
                if self.future is None:
                    try:
                        obs = self.observation(require_after_resume=True)
                    except ValueError as exc:
                        if time.monotonic() - self.last_resume > 1.:
                            self.fail(exc)
                        self.stop.wait(.02)
                        continue
                    last = obs['state'].copy()
                    self.future = self.pool.submit(self.infer, obs, np.empty((0, 7)))
                    self.future_epoch = epoch
                    self.future_started = time.monotonic()
                if not self.future.done():
                    if time.monotonic() - self.future_started > .5:
                        self.fail('Fresh bootstrap inference exceeded 500ms')
                    self.stop.wait(.005)
                    continue
                try:
                    actions = self.future.result()
                    if self.future_epoch != epoch:
                        self.future = None
                        continue
                    if time.monotonic() - self.future_started > .5:
                        raise ValueError('Bootstrap observation expired during inference')
                    plan = self.make_plan(self.resolve_inference(actions), last)
                    self.future = None
                    index = 0
                    next_tick = time.monotonic()
                    self.chunk_ready = True
                except Exception as exc:
                    self.future = None
                    self.fail(exc)
                    continue
            try:
                plan = self.prepare_plan(plan, index, last)
            except Exception as exc:
                self.fail(exc)
                continue
            if self.options['rtc'] and index == 8-self.rtc_delay and self.future is None:
                try:
                    obs = self.observation()
                    self.future = self.pool.submit(self.infer, obs, plan[8-self.rtc_delay:8].copy())
                    self.future_epoch = epoch
                    self.future_started = time.monotonic()
                except Exception as exc:
                    self.fail(exc)
                    continue
            target = plan[index]
            count = self.options['publish_hz'] // 10
            aborted = False
            for micro in range(1, count + 1):
                deadline = next_tick + (micro - 1) / self.options['publish_hz']
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    self.stop.wait(remaining)
                if self.stop.is_set():
                    aborted = True
                    break
                try:
                    # Fresh feedback only; image conversion stays out of publish path.
                    with self.lock:
                        feedback = self.samples.get('state')
                        if feedback is None or time.monotonic() - feedback[1] > .25:
                            raise ValueError('Left feedback expired before publish')
                        if self.gate.paused or self.gate.generation != epoch:
                            aborted = True
                            break
                        if time.monotonic() - deadline > .05:
                            raise ValueError('EXPO physical publish deadline exceeded 50ms')
                        value = self.interpolate_command(last, target, micro / count, index)
                        if self.publisher is not None:
                            message = self.joint_type()
                            message.header.stamp = self.ros.Time.now()
                            message.name = ['joint%d' % i for i in range(7)]
                            message.position = value.tolist()
                            self.publisher.publish(message)
                        self.published += 1
                except Exception as exc:
                    self.fail(exc)
                    aborted = True
                    break
            if aborted:
                continue
            last = target.copy()
            index += 1
            next_tick += .1
            if index == 8:
                if self.options['rtc']:
                    remaining = next_tick - time.monotonic()
                    if remaining > 0:
                        self.stop.wait(remaining)
                    if self.future is None or not self.future.done():
                        self.fail('RTC inference missed the %d-step/%dms prefix budget' % (self.rtc_delay,100*self.rtc_delay))
                        continue
                    try:
                        actions = self.future.result()
                        self.future = None
                        if self.future_epoch != self.gate.generation:
                            plan = None
                            continue
                        plan = self.make_plan(self.resolve_inference(actions), last)
                        index = 0
                    except Exception as exc:
                        self.future = None
                        self.fail(exc)
                else:
                    plan = None
                    self.last_resume = time.monotonic()
            # Catch-up publishing is forbidden: a late logical clock faults above.

    def close(self):
        self.stop.set()
        self.ready = False
        self.fail('runtime stopped')
        if self.server is not None and self.server.poll() is None:
            self.server.terminate()
            try:
                self.server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.server.kill()
                self.server.wait()
        self.pool.shutdown(wait=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--port', type=int, default=8043)
    parser.add_argument('--shadow', action='store_true')
    args = parser.parse_args()
    import rospy
    runtime = Runtime(args, rospy)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: runtime.stop.set())
    try:
        runtime.run()
    except Exception as exc:
        runtime.fail(exc)
        raise
    finally:
        runtime.close()


if __name__ == '__main__':
    main()
