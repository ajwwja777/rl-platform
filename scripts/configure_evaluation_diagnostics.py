#!/usr/bin/env python3
"""Configure numeric capture for the NEXT existing web-launched frozen runtime.

Never loads models, starts services or sends POST/robot commands. Requires the
model runtime to be released before changing the opt-in settings file.
"""
import argparse
import json
import os
from pathlib import Path
import sys
from urllib.request import ProxyHandler, build_opener
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integrations.cobot_runtime.paths import RUNTIME_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--enable", action="store_true")
    mode.add_argument("--disable", action="store_true")
    parser.add_argument("--trace-root", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--status-url", default="http://127.0.0.1:8015/api/deployment/status")
    args = parser.parse_args()
    if args.enable and (args.trace_root is None or not args.trace_root.is_absolute()):
        parser.error("Enabling requires an absolute independent --trace-root")
    with build_opener(ProxyHandler({})).open(args.status_url, timeout=10) as response:
        status = json.load(response)
    # Actual activity, never stale ready_confirmed from an earlier load.
    if status.get("process_started") is not False or status.get("model_ready") is not False:
        parser.error("Release the model runtime first; active/unknown runtime rejected")
    if status.get("phase") not in {"offline", "released"}:
        parser.error("Deployment phase must be offline/released")
    value = dict(schema_version=1, enabled=args.enable, model_id="plug-v3-warmup-5k",
                 trace_root=str(args.trace_root.resolve()) if args.trace_root else None,
                 purpose="diagnostic_development", images_retained=False,
                 learning_allowed=False, replay_allowed=False)
    if args.enable:
        collection = os.environ.get("COBOT_RLT_TRACE_DIR")
        if collection:
            collection = Path(collection).expanduser().resolve()
            root = args.trace_root.resolve()
            if root == collection or root in collection.parents or collection in root.parents:
                parser.error("Output overlaps collection traces")
    target = RUNTIME_ROOT / "evaluation-diagnostic.json"
    if not args.check_only:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".pending.tmp")
        with tmp.open("w") as handle:
            handle.write(json.dumps(value, indent=2)+"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    print(json.dumps(dict(check_only=args.check_only, settings_path=str(target),
        next_runtime_only=True, model_loaded=False, motion=False, settings=value)))

if __name__ == "__main__":
    main()
