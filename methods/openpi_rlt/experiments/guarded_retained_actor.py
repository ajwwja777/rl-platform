"""Opt-in fixed-teacher retention of native Actor outputs on the active six joints on selected old data.

Native loss, RNG, optimizer and inference interface are preserved. The teacher
is the immutable initial Actor, never a claim that its actions are optimal.
"""
from __future__ import annotations
import ast
import inspect
import textwrap


def make_retained_train_step(mc_weight, teacher_params, retention_weight=0., *, target_policy="native"):
    from .guarded_credit import make_train_step
    run = make_train_step(mc_weight, target_policy=target_policy)
    if retention_weight < 0:
        raise ValueError('Nonnegative retention weight required')
    if retention_weight == 0:
        return run
    import jax
    from rlt_online_rl import trainer

    tree = ast.parse(textwrap.dedent(inspect.getsource(trainer.update_actor)))
    function = tree.body[0]
    loss = next((n for n in function.body if isinstance(n,ast.FunctionDef) and n.name=='loss_fn'), None)
    if loss is None:
        raise RuntimeError('Upstream Actor loss changed; review retention adapter')
    anchors = [i for i,n in enumerate(loss.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='actor_loss' for t in n.targets)]
    if len(anchors)!=1:
        raise RuntimeError('Upstream Actor loss changed; review retention adapter')
    statements=ast.parse("""
retention_teacher = jax.lax.stop_gradient(actor.sample_action(_retention_teacher_params, sample_rng, batch["z_rl"], batch["proprio"], dropped_ref, deterministic=False))
retention_mask = batch["retention_mask"].astype(jnp.float32)[:, None]
retention_error = jnp.mean(jnp.square(action_chunk[..., :6]-retention_teacher[..., :6]), axis=-1)
retention_penalty = jnp.mean(retention_error*retention_mask)/jnp.maximum(jnp.mean(retention_mask), 1e-8)
""").body
    idx=anchors[0]
    original_loss=loss.body[idx]
    original_loss.value=ast.BinOp(left=original_loss.value,op=ast.Add(),right=ast.BinOp(left=ast.Name(id='_retention_weight',ctx=ast.Load()),op=ast.Mult(),right=ast.Name(id='retention_penalty',ctx=ast.Load())))
    loss.body[idx:idx]=statements
    metrics=[n for n in loss.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='metrics'for t in n.targets)]
    if len(metrics)!=1 or not isinstance(metrics[0].value,ast.Dict):
        raise RuntimeError('Upstream metrics changed; review retention adapter')
    metrics[0].value.keys.extend([ast.Constant('retention_penalty'),ast.Constant('weighted_retention')])
    metrics[0].value.values.extend([ast.Name(id='retention_penalty',ctx=ast.Load()),ast.BinOp(left=ast.Name(id='_retention_weight',ctx=ast.Load()),op=ast.Mult(),right=ast.Name(id='retention_penalty',ctx=ast.Load()))])
    ast.fix_missing_locations(tree)
    namespace=dict(trainer.update_actor.__globals__, _retention_teacher_params=teacher_params, _retention_weight=float(retention_weight))
    exec(compile(tree,'<private_native_retention>','exec'),namespace)
    original=run.__wrapped__
    private_globals=dict(original.__globals__,update_actor=namespace['update_actor'])
    train_tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
    train_function=train_tree.body[0]
    train_function.decorator_list=[]
    zeros=[n for n in ast.walk(train_function) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='zero_metrics'for t in n.targets)]
    if len(zeros)!=1 or not isinstance(zeros[0].value,ast.Dict):
        raise RuntimeError('Upstream conditional metrics changed; review retention adapter')
    for name in ['retention_penalty','weighted_retention']:
        zeros[0].value.keys.append(ast.Constant(name))
        zeros[0].value.values.append(ast.parse('jnp.array(0., dtype=jnp.float32)',mode='eval').body)
    ast.fix_missing_locations(train_tree)
    exec(compile(train_tree,'<private_native_retention_step>','exec'),private_globals)
    private=private_globals[original.__name__]
    return jax.jit(private,static_argnames=('actor','critic','rl_config','use_action_adapter'))
