"""Offline ablation: use a separate Actor batch without changing Critic inputs.

No native globals are patched. Target Actor feedback still changes later Critic
updates; identical Critic batches do NOT imply identical Critic trajectories.
"""
from types import FunctionType


def with_separate_actor_batch(run):
    import jax
    original = run.__wrapped__
    native_update = original.__globals__["update_actor"]

    def update_actor(state, batch, *args, **kwargs):
        return native_update(state, batch["actor_batch"], *args, **kwargs)

    private = FunctionType(original.__code__,
        dict(original.__globals__, update_actor=update_actor),
        original.__name__, original.__defaults__, original.__closure__)
    private.__kwdefaults__ = original.__kwdefaults__
    return jax.jit(private, static_argnames=("actor", "critic", "rl_config", "use_action_adapter"))
