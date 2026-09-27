"""Write managed backend state without importing JAX or ROS."""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from methods.openpi_rlt.cobot_adapter.lifecycle import (
    LifecycleState, read_lifecycle, write_lifecycle, process_start_ticks,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--error", default=None)
    parser.add_argument("--child", action="append", default=[])
    args = parser.parse_args()
    old = read_lifecycle(args.state)
    if old.supervisor_pid != args.pid or old.supervisor_start_ticks != process_start_ticks(args.pid):
        raise RuntimeError("supervisor identity mismatch")
    children = {}
    ticks = {}
    for item in args.child:
        name, pid = item.split("=", 1)
        if not pid:
            continue
        try:
            ticks[name] = process_start_ticks(int(pid))
            children[name] = int(pid)
        except RuntimeError:
            continue
    write_lifecycle(args.state, replace(old, phase=args.phase, step=args.step,
        mode=args.mode, error_code=args.error, children=children, child_start_ticks=ticks,
        updated_at=datetime.now(timezone.utc).isoformat()))
if __name__ == "__main__":
    main()