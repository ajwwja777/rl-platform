"""Offline-only critic gradient diagnostic/clip, preserving existing release source hashes.

Copies only the selected private update seam. The native optimizer state format,
RNG consumption and other loss components stay unchanged. No global monkeypatch.
"""
import ast
import inspect
import math
import textwrap
from types import FunctionType

def with_critic_gradient_diagnostics(run, clip=None):
    if clip is not None and (not math.isfinite(clip) or clip <= 0):
        raise ValueError("clip must be finite and positive")
    import jax
    original=run.__wrapped__
    update=original.__globals__["update_critic"]
    tree=ast.parse(textwrap.dedent(inspect.getsource(update)))
    fn=tree.body[0]
    anchors=[i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign)
             and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute)
             and n.value.func.attr=='update']
    if len(anchors)!=1:
        raise RuntimeError("Critic optimizer seam changed; re-audit adapter")
    statements=ast.parse("""
raw_norm = optax.global_norm(grads)
scale = jnp.array(1., dtype=raw_norm.dtype)
if _diagnostic_clip is not None:
    scale = jnp.minimum(1., _diagnostic_clip / jnp.maximum(raw_norm, 1e-12))
    grads = jax.tree_util.tree_map(lambda g: g * scale, grads)
metrics = dict(metrics, critic_grad_norm=raw_norm,
               critic_applied_grad_norm=raw_norm * scale,
               critic_grad_clipped=(scale < 1.).astype(jnp.float32))
""").body
    fn.body[anchors[0]:anchors[0]]=statements
    ast.fix_missing_locations(tree)
    namespace=dict(update.__globals__,**inspect.getclosurevars(update).nonlocals,
                   _diagnostic_clip=clip)
    exec(compile(tree,'<private_critic_gradient_diagnostic>','exec'),namespace)
    private=FunctionType(original.__code__,dict(original.__globals__,update_critic=namespace[update.__name__]),
                         original.__name__,original.__defaults__,original.__closure__)
    private.__kwdefaults__=original.__kwdefaults__
    return jax.jit(private,static_argnames=('actor','critic','rl_config','use_action_adapter'))
