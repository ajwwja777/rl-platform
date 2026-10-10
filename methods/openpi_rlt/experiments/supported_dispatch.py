"""Choose a registered training schema before any runtime side effects."""
import importlib
import json
from pathlib import Path

MODULES = {
    "held-gripper-retention-v1": "supported_runtime",
    "held-gripper-hil8-clip-v2": "supported_hil_runtime",
}

def runtime_for(profile_path):
    schema = json.loads(Path(profile_path).read_text())["schema"]
    if schema not in MODULES:
        raise ValueError("Unsupported candidate training contract: " + str(schema))
    return importlib.import_module("methods.openpi_rlt.experiments." + MODULES[schema])
