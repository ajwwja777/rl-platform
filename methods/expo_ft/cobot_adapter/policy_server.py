"""Local-only native EXPO inference; deliberately has no ROS or publishers."""
import argparse
import io
import json
import runpy
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.project))
    api = runpy.run_path(str(args.project / "scripts/cobot_plug/deployment.py"))
    # Bundles are immutable; native loader restores the true optimizer step.
    agent, step = api["load_policy"](args.bundle)
    fixture = args.bundle.parent / "common/dev_observation.npz"
    with np.load(fixture, allow_pickle=False) as sample:
        observation = {k: sample[k].copy() for k in ("state", "left_wrist_image", "right_wrist_image")}
    def infer(observation, prefix):
        nonlocal agent
        delay = len(prefix)
        if delay not in (0, 2):
            raise ValueError("Runtime RTC prefix must be zero or two 10Hz actions")
        padded = None
        if delay:
            transformed = agent.actor.input_transforms({**observation, "actions": prefix.copy()})
            padded = np.pad(np.asarray(transformed["actions"]), ((0, 0), (0, 25)))
        started = time.monotonic()
        actions, updated, info = agent.sample_actions(dict(observation), delay=delay,
                                                     prefix_padded=padded)
        actions = np.asarray(actions)
        if actions.shape[0] < 8 or actions.shape[1:] != (7,) or not np.isfinite(actions).all():
            raise ValueError("Non-finite or malformed native actor output")
        agent = updated
        return {"actions": actions[:8].tolist(), "step": step, "delay": delay,
                "seconds": time.monotonic() - started}
    warm = infer(observation, np.empty((0, 7), np.float32))
    infer(observation, np.asarray(warm["actions"][:2], np.float32))
    print("EXPO native actor ready; boot and RTC-prefix paths compiled; no ROS publishers", flush=True)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def send(self, status, payload):
            body = json.dumps(payload, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def do_GET(self):
            if self.path != "/ready":
                return self.send(404, {"error": "not_found"})
            self.send(200, {"ready": True, "step": step, "pid": __import__("os").getpid()})
        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if self.path != "/infer" or not 0 < length <= 2000000:
                    raise ValueError("invalid inference request")
                with np.load(io.BytesIO(self.rfile.read(length)), allow_pickle=False) as data:
                    observation = {k: data[k].copy() for k in ("state", "left_wrist_image", "right_wrist_image")}
                    prefix = data["prefix"].copy()
                from core import validate_observation
                validate_observation(observation)
                if prefix.ndim != 2 or prefix.shape[1] != 7 or not np.isfinite(prefix).all():
                    raise ValueError("invalid physical prefix")
                self.send(200, infer(observation, prefix))
            except Exception as exc:
                self.send(500, {"error": str(exc)})
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
