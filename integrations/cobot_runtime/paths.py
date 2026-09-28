"""RLT machine paths. Web and CLI pass the same explicit environment/config."""
import json
import os
from pathlib import Path
RLT = Path(__file__).resolve().parents[2]
CONFIG = Path(os.environ.get("COBOT_RL_CONFIG", RLT / "configs/local.json"))
SETTINGS = json.loads(CONFIG.read_text()) if CONFIG.is_file() else {}
def path(key, default, env=None):
    return Path(os.environ.get(env or "COBOT_" + key.upper(), SETTINGS.get(key, str(default)))).expanduser()
DATA = path("data_root", "/media/agilex/Getea1/jiaan/data")
LEGACY_DATA = path("legacy_data_root", DATA)
RUNTIME_ROOT = path("runtime_root", RLT / "runtime")
RLT_MODELS = path("rlt_model_root", "/media/agilex/Getea1/jiaan/model/rl-platform/rlt/plug_insertion")
RLT_WARMUP = path("rlt_warmup_root", RLT_MODELS / "warmup_5000")
def migrated_data_path(value):
    value = str(value)
    for old, new in sorted(SETTINGS.get("data_root_aliases", {}).items(), key=lambda pair: len(pair[0]), reverse=True):
        if value == old or value.startswith(old + "/"):
            return new + value[len(old):]
    return value
