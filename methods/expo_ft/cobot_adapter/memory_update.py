"""Single-device buffer donation around the unchanged native EXPO update.

Native update receives self and agent; targets can initially alias online params.
Deduplicate physical input buffers before donation, reconstruct the exact agent
inside JIT, and preserve all native losses, targets, ordering and RNG operations.
The old agent MUST NOT be used after this call, including on update failure.
"""
import jax
import numpy as np

_COMPILED = {}


def donated_update(agent, batch, utd_ratio=1, actor_batch=None):
    from expo_ft.agents.alg.realtime_expo_ft import RealTimeEXPOFTLearner
    agent = agent.replace(_infer_cache=None)
    leaves, structure = jax.tree_util.tree_flatten(agent)
    unique, indices, seen = [], [], {}
    for leaf in leaves:
        if isinstance(leaf, jax.Array):
            if len(leaf.devices()) != 1:
                raise ValueError("Donated EXPO update requires one GPU")
            key = (leaf.unsafe_buffer_pointer(), leaf.shape, str(leaf.dtype))
        else:
            key = ('scalar',len(unique))
            leaf = jax.numpy.asarray(leaf)
        if key not in seen:
            seen[key] = len(unique)
            unique.append(leaf)
        indices.append(seen[key])
    train_actor = bool(agent.train_base_actor)
    key = (structure, tuple(indices), utd_ratio, train_actor)
    if key not in _COMPILED:
        def update(buffers, critic_batch, success_batch):
            current = jax.tree_util.tree_unflatten(structure, [buffers[i] for i in indices])
            return RealTimeEXPOFTLearner._update_jit.__wrapped__(
                current, current, critic_batch, utd_ratio, success_batch, train_actor=train_actor)
        _COMPILED[key] = jax.jit(update, donate_argnums=(0,))
    updated, info = _COMPILED[key](tuple(unique), batch, actor_batch)
    return updated.cache_infer_params(), info


class HostTargetUpdater:
    """Keep trainable target-Actor EMA on CPU; share invariant frozen weights.

    All loss-bearing networks remain on GPU. Checkpoints materialize the exact
    native agent shape again. Source guard prevents use after upstream starts
    consuming the target Actor in a loss/inference method.
    """
    def __init__(self, agent):
        import ast, inspect, textwrap, dataclasses
        import flax.struct
        import optax
        from expo_ft.agents.alg.realtime_expo_ft import RealTimeEXPOFTLearner
        source=ast.parse(textwrap.dedent(inspect.getsource(RealTimeEXPOFTLearner)))
        for method in source.body[0].body:
            if isinstance(method,(ast.FunctionDef,ast.AsyncFunctionDef)):
                if any(isinstance(n,ast.Attribute) and n.attr=='target_actor_params' for n in ast.walk(method)):
                    if method.name not in ('update_actor',):
                        raise ValueError('Native target Actor is consumed by '+method.name+'; CPU EMA adapter needs review')
        # Reuse the native function verbatim, omitting only its EMA expression.
        method=ast.parse(textwrap.dedent(inspect.getsource(RealTimeEXPOFTLearner.update_actor)))
        replacements=0
        for node in ast.walk(method):
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='target_score_params' for t in node.targets):
                if not (isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='incremental_update'):
                    raise ValueError('Unexpected native EMA implementation')
                node.value=ast.Constant(value=None)
                replacements+=1
        if replacements!=1:raise ValueError('Native EMA assignment changed')
        namespace=dict(RealTimeEXPOFTLearner.update_actor.__globals__)
        exec(compile(ast.fix_missing_locations(method),'<native EXPO actor update without GPU EMA>','exec'),namespace)
        @flax.struct.dataclass
        class HostTargetLearner(RealTimeEXPOFTLearner):
            update_actor=namespace['update_actor']
        self.native_type=RealTimeEXPOFTLearner
        self.agent_type=HostTargetLearner
        self.fields=[f.name for f in dataclasses.fields(agent)]
        self.cpu=jax.devices('cpu')[0]
        self.trainable=agent.actor.train_config.trainable_filter
        self.frozen=agent.actor.train_config.freeze_filter
        frozen_target=agent.target_actor_params.filter(self.frozen)
        frozen_actor=agent.actor.get_params(agent.actor_train_state).filter(self.frozen)
        for old,current in zip(jax.tree.leaves(frozen_target),jax.tree.leaves(frozen_actor)):
            if old.unsafe_buffer_pointer()!=current.unsafe_buffer_pointer() and not bool(jax.numpy.all(old==current)):
                raise ValueError('Frozen target Actor differs from seed; cannot share frozen weights')
        self.target=jax.tree.map(lambda x:np.array(x,copy=True),jax.device_get(agent.target_actor_params.filter(self.trainable)))
        tau=agent.actor_tau
        self.ema=jax.jit(lambda new,old:optax.incremental_update(new,old,tau),
                         donate_argnums=(1,),backend='cpu')

    def prepare(self,agent):
        return self.agent_type(**{name:(None if name in ('target_actor_params','_infer_cache') else getattr(agent,name)) for name in self.fields})

    def update(self,agent,batch,utd_ratio=1,actor_batch=None):
        if getattr(self,'split_batches',False):
            from split_update import update
            agent,info=update(agent,batch,utd_ratio,actor_batch,progress=getattr(self,'progress',None))
        else:
            agent,info=donated_update(agent,batch,utd_ratio,actor_batch)
        if agent.train_base_actor:
            # Bound host memory as well: never copy another full VLA onto CPU.
            current=agent.actor.get_params(agent.actor_train_state).filter(self.trainable)
            for new,old in zip(jax.tree.leaves(current),jax.tree.leaves(self.target)):
                destination=old.reshape(-1)
                flat=new.reshape(-1)
                for start in range(0,flat.size,4*1024*1024):
                    end=min(start+4*1024*1024,flat.size)
                    value=np.asarray(flat[start:end])
                    result=self.ema(jax.device_put(value,self.cpu),jax.device_put(destination[start:end].copy(),self.cpu))
                    destination[start:end]=np.asarray(result)

        return agent,info

    def checkpoint_agent(self,agent):
        import flax.nnx as nnx
        target=nnx.State.merge(agent.actor.get_params(agent.actor_train_state).filter(self.frozen),self.target)
        return self.native_type(**{name:(target if name=='target_actor_params' else None if name=='_infer_cache' else getattr(agent,name)) for name in self.fields})
