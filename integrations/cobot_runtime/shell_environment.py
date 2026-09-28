"""Print allowlisted shell exports; no web import, model load or ROS side effect."""
import json
import os
import shlex
from .paths import RLT, SETTINGS, DATA, RLT_WARMUP, RLT_MODELS
values = {
    "COBOT_RLT_PROJECT_ROOT": str(RLT),
    "COBOT_RLT_MODEL_ROOT": str(RLT_MODELS),
    "COBOT_RLT_WARMUP_ROOT": str(RLT_WARMUP),
    "COBOT_RLT_TRACE_ROOT": SETTINGS.get("rlt_trace_root", str(RLT / "outputs/rlt/plug_v3_yyshadow/online/traces")),
    "COBOT_RUNTIME_ROOT": str(RLT / "runtime"),
    "COBOT_RLT_TASK5_URL": SETTINGS.get("recorder_url", "http://127.0.0.1:8015/api/rlt-recorder"),
    "TASK5_ROS_SETUP": SETTINGS.get("ros_setup", "/opt/ros/noetic/setup.bash"),
    "COBOT_ASSET_MOUNT": SETTINGS.get("asset_mount", ""),
    "COBOT_ASSET_UUID": SETTINGS.get("asset_uuid", ""),
    "COBOT_DATA_ROOT": str(DATA),
    "COBOT_DATA_ROOT_ALIASES": json.dumps(SETTINGS.get("data_root_aliases", {})),
}
for key, value in values.items():
    print("export " + key + "=" + shlex.quote(os.environ.get(key, value)))
