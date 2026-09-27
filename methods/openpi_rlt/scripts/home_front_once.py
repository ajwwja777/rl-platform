#!/usr/bin/env python3
"""Run the registered Task2 home CLI once, then exit without ROS thread drain."""

from __future__ import annotations

import os
from pathlib import Path
import runpy
import sys


def run_target(target: Path, args: list[str]) -> int:
    target = target.expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError(f"Task2 home CLI is missing: {target}")
    sys.argv = [str(target), *args]
    try:
        runpy.run_path(str(target), run_name="__main__")
    except SystemExit as error:
        if error.code is None:
            return 0
        if isinstance(error.code, int):
            return error.code
        print(error.code, file=sys.stderr)
        return 1
    return 0


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: home_front_once.py TASK2_HOME_CLI [ARGS ...]", file=sys.stderr)
        return 2
    return run_target(Path(sys.argv[1]), sys.argv[2:])


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # rospy can retain non-daemon helper threads after the service response.
    # The motion-owning service has returned, so do not let those threads keep
    # the session finalizer in replay_committing.
    os._exit(exit_code)
