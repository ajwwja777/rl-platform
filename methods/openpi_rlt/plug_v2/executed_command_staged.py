"""Isolated ablation: BC-led 2000, critic-only 6000, joint 2000; no production publish."""
import functools,types,json,sys
from pathlib import Path
from . import executed_command_train as runner
from .executed_command_core import update_critic,train_step
from rlt_online_rl.trainer import soft_update_targets
import jax,jax.numpy as jnp

@functools.partial(jax.jit,static_argnames=('actor','critic','rl_config','use_action_adapter'))
def critic_only(state,batch,*,actor,critic,rl_config,**kwargs):
    state,metrics=update_critic(state,batch,actor,critic,rl_config)
    due=((state.global_step+1)%rl_config.actor_update_period)==0
    def targets(s):return s.replace(target_actor_params=soft_update_targets(s.target_actor_params,s.actor_params,rl_config.target_tau),target_critic_params=soft_update_targets(s.target_critic_params,s.critic_params,rl_config.target_tau))
    state=jax.lax.cond(due,targets,lambda s:s,state);state=state.replace(global_step=state.global_step+1)
    return state,dict(metrics,actor_loss=jnp.array(0.),bc_weight=jnp.asarray(kwargs['bc_weight']),q_weight=jnp.asarray(kwargs['q_weight']),delta_weight=jnp.asarray(kwargs['delta_weight']),did_actor_update=jnp.array(0.),global_step=state.global_step.astype(jnp.float32),actor_version=state.actor_version.astype(jnp.float32),staged_phase=jnp.array(1.))

def staged_step(state,batch,**kwargs):
    step=int(state.global_step)
    if step<2000:
        kwargs['q_weight']=0.;s,m=train_step(state,batch,**kwargs);return s,dict(m,staged_phase=jnp.array(0.))
    if step<8000:return critic_only(state,batch,**kwargs)
    s,m=train_step(state,batch,**kwargs);return s,dict(m,staged_phase=jnp.array(2.))

def main():
    f=runner.Learner.train_once
    bound=types.FunctionType(f.__code__,dict(f.__globals__,train_step=staged_step),f.__name__,f.__defaults__,f.__closure__);bound.__kwdefaults__=f.__kwdefaults__;runner.Learner.train_once=bound
    out=Path(sys.argv[sys.argv.index('--output')+1]);runner.main()
    (out/'staged_protocol.json').write_text(json.dumps({'phase_0':[0,2000,'BC-led actor; Q weight zero; critic also trains'],'phase_1':[2000,8000,'critic-only; actor params/optimizer frozen; target updates every two'],'phase_2':[8000,10000,'original joint actor-critic'],'budget_note':'10000 global/critic updates; 2000 actor updates; staged executed-command contract','production_publish':False},indent=2))
if __name__=='__main__':main()
