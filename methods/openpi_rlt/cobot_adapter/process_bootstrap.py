"""Install project runtime adapters in every fresh multiprocessing child.

The fixed upstream uses spawn. Parent monkey patches are deliberately not
inherited; workers must install them before constructing services or Replay.
"""
from __future__ import annotations

import functools
import os


def initialize_process(config_path: str | None = None) -> None:
    from .replay_precision import install_action_precision_patch
    install_action_precision_patch()
    from .online_runtime import install_bimanual_runtime_patch
    install_bimanual_runtime_patch()
    if os.environ.get("COBOT_RLT_DIAGNOSTIC_METRICS") == "1":
        from rlt_online_rl import trainer
        from .diagnostic_metrics import install_diagnostic_metrics_patch
        install_diagnostic_metrics_patch(trainer)
    if config_path is not None:
        from integrations.cobot_runtime.replay_audit import install_batch_audit
        install_batch_audit(config_path)


def run_initialized_role(target, config_path, *args, **kwargs):
    """Importable spawn target; no service or model is created by the bootstrap."""
    initialize_process(config_path)
    return target(*args, **kwargs)


def install_spawn_bootstrap(native, config_path: str | None = None) -> None:
    if getattr(native._spawn_process, "_cobot_process_bootstrap", False):
        return
    original = native._spawn_process

    def spawn(name, target, *args, **kwargs):
        worker = functools.partial(run_initialized_role, target, config_path)
        return original(name, worker, *args, **kwargs)

    spawn._cobot_process_bootstrap = True
    native._spawn_process = spawn
