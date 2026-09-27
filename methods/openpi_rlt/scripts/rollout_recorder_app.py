"""RLT-owned continuous recorder; reuses Task5 without its teach-only gate."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

CONTRACT = {
    "/api/storage/prepare": "post", "/api/episodes/start": "post",
    "/api/status": "get", "/api/episodes/stop": "post", "/api/episodes": "get",
    "/api/episodes/{episode_uuid}/outcome": "post",
    "/api/episodes/{episode_uuid}/labels": "put",
}

def supports_rollout_contract(schema):
    paths = schema.get("paths", {})
    return all(method in paths.get(path, {}) for path, method in CONTRACT.items())

class RltRosProxy:
    def __init__(self, rospy):
        self._rospy = rospy

    def __getattr__(self, name):
        return getattr(self._rospy, name)

    def init_node(self, _name, **kwargs):
        return self._rospy.init_node("task5_rlt_rollout_recorder", **kwargs)

def create_rollout_app(*, data_root, min_free_bytes=30 * 1024**3):
    from capture_core.api import create_app
    from capture_core.config import RecorderConfig
    from capture_core.labels import LabelStore
    from capture_core.recorder import RolloutRecorder
    from rlt_cached_writer import RltCachedWriter
    from capture_core.ros_cache import LatestMessageCache
    from capture_core.ros_subscriber import RosSubscriberBridge, _load_ros_bindings

    root = Path(data_root)
    if not root.is_absolute():
        raise ValueError("RLT data root must be absolute")
    cache = LatestMessageCache()
    config = RecorderConfig(
        data_root=root, min_free_disk_bytes=min_free_bytes,
        stop_free_disk_bytes=min(min_free_bytes, 20 * 1024**3), api_port=8017)
    # No CaptureGate: preserve both policy execution and human takeover frames.
    recorder = RolloutRecorder(config, cache, writer_factory=RltCachedWriter)
    def bindings():
        original = _load_ros_bindings()
        return replace(original, rospy=RltRosProxy(original.rospy))
    def bridge_factory(shared_cache):
        return RosSubscriberBridge(shared_cache, bindings_loader=bindings)
    app = create_app(recorder=recorder, label_store=LabelStore(root),
                     cache=cache, ros_bridge_factory=bridge_factory)
    @app.get("/rlt/identity")
    def identity():
        return {"service": "rlt-continuous-task5-v1", "data_root": str(root)}
    # Task5 mounts static files at "/"; specific routes must precede that mount.
    identity_route = app.router.routes.pop()
    app.router.routes.insert(0, identity_route)
    return app

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--port", type=int, default=8017)
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(create_rollout_app(data_root=Path(args.data_root)),
                host="127.0.0.1", port=args.port, timeout_graceful_shutdown=5)
