import dataclasses
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from methods.openpi_rlt.experiments.review_metrics import review_metadata, episode_groups

def rows():
    result=[]
    for start in [0,5]:
        rewards=np.zeros(10,np.float32)
        if start==5:rewards[-1]=1.
        source=np.zeros(10,np.uint8)
        source[np.arange(start,start+10)>=10]=2
        result.append(dict(collection_phase_id=2,episode_id=1,step_id=start,rewards=rewards,done=start==5,success=1,source_chunk=source))
    return result

def test_overlap_return_uses_control_steps_and_preserves_assisted_credit():
    group, ret, valid, human, outcome, kind=review_metadata(rows(),.99)
    np.testing.assert_allclose(ret,[.99**14,.99**9])
    assert valid.all() and list(kind)==['pre_hil','human_window']
    assert list(outcome)==['assisted_success']*2
    assert ret[0]>0

def test_phase_identity_separates_equal_numeric_ids_and_rejects_duplicate():
    rr=rows(); other=[dict(r,collection_phase_id=1) for r in rr]
    assert len(episode_groups(rr+other))==2
    with pytest.raises(ValueError):episode_groups(rr+rr)

def test_incomplete_return_remains_unknown():
    _,_,valid,_,outcome,_=review_metadata(rows()[:1],.99)
    assert not valid.any() and outcome[0]=='unknown'

def native():
    pytest.importorskip('jax')
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('native_review_fixture',root/'third_party/openpi-rlt/rlt_online_rl/tests/test_trainer.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    import jax
    from rlt_online_rl import trainer
    cfg=mod._config();batch=mod._batch(cfg)
    state,actor,critic=trainer.init_train_state(cfg,rng=jax.random.PRNGKey(42))
    batch.update(mc_valid=np.ones(len(batch['done']),bool),mc_return=np.full(len(batch['done']),.5,np.float32))
    return cfg,batch,state.replace(global_step=state.global_step+1),actor,critic

def test_clip_disabled_native_and_invalid():
    from methods.openpi_rlt.experiments.guarded_credit import make_train_step
    from rlt_online_rl import trainer
    assert make_train_step() is trainer.train_step
    for value in [0,-1,float('inf'),float('nan')]:
        from methods.openpi_rlt.experiments.critic_stability import with_critic_gradient_diagnostics
        with pytest.raises(ValueError):with_critic_gradient_diagnostics(make_train_step(.3),value)

def test_gradient_clipping_changes_optimizer_moments_and_preserves_rng():
    import jax
    from methods.openpi_rlt.experiments.guarded_credit import make_train_step
    cfg,b,s,a,c=native()
    from methods.openpi_rlt.experiments.critic_stability import with_critic_gradient_diagnostics
    x,m=with_critic_gradient_diagnostics(make_train_step(.3,target_policy='clip'))(s,b,actor=a,critic=c,rl_config=cfg)
    y,n=with_critic_gradient_diagnostics(make_train_step(.3,target_policy='clip'),1e-4)(s,b,actor=a,critic=c,rl_config=cfg)
    assert float(n['critic_grad_clipped'])==1
    assert float(n['critic_applied_grad_norm'])<=1.0001e-4
    np.testing.assert_array_equal(x.rng,y.rng)
    assert any(not np.array_equal(u,v) for u,v in zip(jax.tree.leaves(x.critic_opt_state),jax.tree.leaves(y.critic_opt_state)))

@dataclasses.dataclass(frozen=True)
class OppositeCritic:
    def q_values(self,params,z,pro,act):
        import jax.numpy as jnp
        q=jnp.sum(act,axis=(1,2))
        return q,-q-100.

def test_actor_min_q_changes_only_actor_update_not_native_global():
    import jax
    from rlt_online_rl import trainer
    from methods.openpi_rlt.experiments.guarded_credit import make_train_step
    from methods.openpi_rlt.experiments.actor_q_reduction import with_actor_min_q
    cfg,b,s,a,c=native();run=make_train_step(.3,target_policy='clip')
    assert with_actor_min_q(run) is run
    fn=run.__wrapped__.__globals__['update_actor']
    wrapped=with_actor_min_q(run,True).__wrapped__.__globals__['update_actor']
    old=trainer.update_actor
    x,m=fn(s,b,a,OppositeCritic(),cfg,bc_weight=0.,q_weight=1.)
    y,n=wrapped(s,b,a,OppositeCritic(),cfg,bc_weight=0.,q_weight=1.)
    assert float(n['actor_q'])<float(m['actor_q'])
    assert any(not np.array_equal(u,v) for u,v in zip(jax.tree.leaves(x.actor_params),jax.tree.leaves(y.actor_params)))
    for u,v in zip(jax.tree.leaves(x.critic_params),jax.tree.leaves(y.critic_params)):np.testing.assert_array_equal(u,v)
    np.testing.assert_array_equal(x.rng,y.rng)
    assert trainer.update_actor is old


def test_gradient_diagnostics_without_clipping_preserve_entire_state():
    import jax
    from methods.openpi_rlt.experiments.guarded_credit import make_train_step
    from methods.openpi_rlt.experiments.critic_stability import with_critic_gradient_diagnostics
    cfg,b,s,a,c=native();run=make_train_step(.3,target_policy='clip')
    x,_=run(s,b,actor=a,critic=c,rl_config=cfg)
    y,m=with_critic_gradient_diagnostics(run)(s,b,actor=a,critic=c,rl_config=cfg)
    for u,v in zip(jax.tree.leaves(x),jax.tree.leaves(y)):np.testing.assert_array_equal(u,v)
    assert float(m['critic_grad_clipped'])==0


def test_awbc_success_flag_does_not_erase_human_step_supervision():
    import jax
    import jax.numpy as jnp
    from methods.openpi_rlt.experiments.awbc_actor import make_awbc_train_step
    cfg,b,s,a,c=native();b=dict(b,source_chunk=jnp.full_like(b['source_chunk'],2))
    run=make_awbc_train_step(.3,beta=.1,max_weight=5.,failure_anchor=.1)
    update=run.__wrapped__.__globals__['update_actor']
    x,m=update(s,{'actor_batch':dict(b,awbc_success=jnp.zeros(len(b['done'])))},a,c,cfg,bc_weight=5.,q_weight=.1)
    y,n=update(s,{'actor_batch':dict(b,awbc_success=jnp.ones(len(b['done'])))},a,c,cfg,bc_weight=5.,q_weight=.1)
    for u,v in zip(jax.tree.leaves(x),jax.tree.leaves(y)):np.testing.assert_array_equal(u,v)
    assert float(m['human_mask_ratio'])==1 and float(n['human_mask_ratio'])==1
