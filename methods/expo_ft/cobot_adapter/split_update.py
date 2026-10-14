"""Schedule native updates separately to bound single-GPU working memory.

Full-batch RNG keys and indices are drawn once; each native minibatch is
gathered and augmented on GPU only when needed. Critic
minibatches, actor loss, optimizer steps and ordering are unchanged.
"""
import jax
import numpy as np
from expo_ft.agents.alg.batch_utils import prepare_critic_batch
_COMPILED={}


def donate(agent, batch, phase):
    leaves,structure=jax.tree.flatten(agent)
    unique,indices,seen=[],[],{}
    for value in leaves:
        if not isinstance(value,jax.Array):value=jax.numpy.asarray(value)
        key=(value.unsafe_buffer_pointer(),value.shape,str(value.dtype))
        if key not in seen:seen[key]=len(unique);unique.append(value)
        indices.append(seen[key])
    key=(structure,tuple(indices),phase)
    if key not in _COMPILED:
        def call(buffers,data):
            current=jax.tree.unflatten(structure,[buffers[i] for i in indices])
            if phase=='critic':return current.update_critic(data)
            if phase=='actor':return current.update_actor(data)
            current,info=current.update_edit_actor(data)
            current,temp=current.update_temperature(info['entropy'])
            return current,{**info,**temp}
        _COMPILED[key]=jax.jit(call,donate_argnums=(0,))
    return _COMPILED[key](tuple(unique),batch)


def augmentation(agent):
    import ast,inspect,textwrap
    from expo_ft.utils import augmentation as native
    full=inspect.getclosurevars(agent.data_augmentation_fn).nonlocals['use_full_augmentation']
    original=native.batched_openpi_augmentation if full else native.batched_random_crop_per_image
    tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
    fn=tree.body[0]
    fn.args.args.append(ast.arg(arg='full_keys'))
    fn.args.defaults.append(ast.Constant(None))
    replacements=0
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=='jax.random.split':
            node.value=ast.Name('full_keys',ast.Load());replacements+=1
    if replacements!=1:raise ValueError('Native augmentation RNG changed')
    ns=dict(original.__globals__)
    exec(compile(ast.fix_missing_locations(tree),'<native augmentation with full batch keys>','exec'),ns)
    keyed=ns[original.__name__]
    def apply(keys,images):
        if full:return keyed(None,images,full_keys=keys)
        obs=jax.numpy.concatenate([images['base_0_rgb'],images['left_wrist_0_rgb']],axis=-1)
        obs=keyed(None,obs,padding=12,full_keys=keys)
        return dict(images,base_0_rgb=obs[...,:3],left_wrist_0_rgb=obs[...,3:6])
    return jax.jit(apply)


def update(agent,batch,utd_ratio,actor_batch,progress=None):
    agent=agent.replace(_infer_cache=None)
    count=batch.total if hasattr(batch,'total') else batch['actions'].shape[0]
    assert count%utd_ratio==0
    size=count//utd_ratio
    rng=agent.rng
    keys={}
    for name in ('image','next_image','next_delayed_image'):
        if name=='next_delayed_image' and agent.delay<=0:continue
        rng,key=jax.random.split(rng)
        keys[name]=jax.random.split(key,2*count)
    agent=agent.replace(rng=rng)
    augment=augmentation(agent)
    def prepare(data):
        return prepare_critic_batch(data,agent.actor.model_config.action_dim,
            agent.action_dim,agent.state_dim,agent.action_horizon,agent.replan_steps,agent.critic_camera_keys)
    if hasattr(batch,'total'):batches=iter(batch)
    else:batches=(jax.tree.map(lambda x:x[i*size:(i+1)*size],batch) for i in range(utd_ratio))
    for index,data in enumerate(batches):
        data=dict(data)
        for name,all_keys in keys.items():
            data[name]=augment(all_keys[index*size*2:(index+1)*size*2],data[name])
        minibatch=prepare(data)
        agent,critic_info=donate(agent,minibatch,'critic')
        jax.block_until_ready(agent)
        if progress:progress('critic',int(agent.critic.step))
    actor_info={}
    if agent.train_base_actor:
        if agent.actor_success_only:
            actor_data=dict(actor_batch)
            rng,key=jax.random.split(agent.rng)
            actor_data['image']=agent.data_augmentation_fn(key,actor_data['image'])
            actor_data=prepare(actor_data)
            agent=agent.replace(rng=rng)
        else:actor_data=minibatch
        agent,actor_info=donate(agent,actor_data,'actor')
        jax.block_until_ready(agent)
        if progress:progress('actor',int(agent.actor_train_state.step))
        del actor_data
    if agent.n_edit_samples>0:
        agent,edit_info=donate(agent,minibatch,'edit')
        actor_info={**actor_info,**edit_info}
    return agent.cache_infer_params(),{**actor_info,**critic_info}
