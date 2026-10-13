"""Opt-in Actor-only min-Q; Critic/TD and global upstream functions are untouched."""
from dataclasses import dataclass
from types import FunctionType

@dataclass(frozen=True)
class ActorMinimumCritic:
    critic: object

    def q_values(self, *args, **kwargs):
        import jax.numpy as jnp
        q1, q2 = self.critic.q_values(*args, **kwargs)
        return jnp.minimum(q1, q2), q2

def with_actor_min_q(run, enabled=False):
    if not enabled:
        return run
    import jax
    original = run.__wrapped__
    native_update = original.__globals__["update_actor"]
    def update_actor(state, batch, actor, critic, *args, **kwargs):
        return native_update(state, batch, actor, ActorMinimumCritic(critic), *args, **kwargs)
    private = FunctionType(original.__code__, dict(original.__globals__, update_actor=update_actor),
                           original.__name__, original.__defaults__, original.__closure__)
    private.__kwdefaults__ = original.__kwdefaults__
    return jax.jit(private, static_argnames=("actor", "critic", "rl_config", "use_action_adapter"))
