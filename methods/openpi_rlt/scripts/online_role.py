#!/usr/bin/env python3
"""Enter one fixed-upstream online role with Cobot's bimanual patch installed."""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path


def _parse_wrapper_args() -> tuple[Path, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--upstream-root", type=Path, required=True)
    args, upstream_args = parser.parse_known_args()
    return args.upstream_root.expanduser().resolve(), upstream_args


def _is_expected_operator_shutdown(error: RuntimeError) -> bool:
    marker_raw = os.environ.get("COBOT_RLT_OPERATOR_SHUTDOWN_MARKER")
    return bool(
        marker_raw
        and Path(marker_raw).is_file()
        and str(error).startswith("ROS shut down while waiting for")
    )


def main() -> None:
    upstream_root, upstream_args = _parse_wrapper_args()
    runtime_root = upstream_root / "rlt_online_rl"
    runner = runtime_root / "scripts" / "run_online_rl.py"
    runtime_src = runtime_root / "src"
    if not runner.is_file() or not runtime_src.is_dir():
        raise FileNotFoundError(f"invalid fixed upstream root: {upstream_root}")

    project_root = Path(__file__).resolve().parents[3]
    for path in (project_root, runtime_src):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from methods.openpi_rlt.cobot_adapter.online_runtime import (
        install_bimanual_runtime_patch,
    )

    from methods.openpi_rlt.cobot_adapter.replay_precision import install_action_precision_patch
    install_action_precision_patch()
    install_bimanual_runtime_patch()
    if "--config" in upstream_args:
        from integrations.cobot_runtime.replay_audit import install_batch_audit
        install_batch_audit(upstream_args[upstream_args.index("--config") + 1])
    sys.argv = [str(runner), *upstream_args]
    try:
        profile = os.environ.get("COBOT_RLT_EXPERIMENT_PROFILE")
        if profile:
            from methods.openpi_rlt.experiments.runtime import run_registered
            run_registered(upstream_root, upstream_args, profile)
        else:
            runpy.run_path(str(runner), run_name="__main__")
    except RuntimeError as error:
        if not _is_expected_operator_shutdown(error):
            raise
        print("[rlt-session] env driver stopped after operator shutdown.", flush=True)


if __name__ == "__main__":
    main()
