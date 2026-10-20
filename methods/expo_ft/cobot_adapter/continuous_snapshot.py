"""Inference identity for native async defaults, separate from sync snapshots."""
def contract(bundle,project,delay,edit_scale=None):
    import hashlib
    from pathlib import Path
    from actor_snapshot import contract as base
    value=base(bundle,project,delay,edit_scale)
    value.update(recipe='native_async_v1',N=8,async_script_sha256=hashlib.sha256((Path(project)/'train_pi_robo_async.py').read_bytes()).hexdigest())
    return value
