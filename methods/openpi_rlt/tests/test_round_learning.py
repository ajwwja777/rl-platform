from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
from rlt_online_rl import trainer
from rlt_online_rl.config import RLTOnlineRLConfig
from methods.openpi_rlt.experiments.round_learning import (guarded_target,
    actor_targets, make_actor_step, make_critic_step)


def test_td_guard_respects_terminal_reward_position_and_unclipped_diagnostics():
    class Actor:
        def sample_action(self,*args,**kwargs): return jnp.zeros((2,10,7))
    class Critic:
        def q_values(self,*args): return jnp.ones(2)*10, jnp.ones(2)*12
    state=SimpleNamespace(target_actor_params=None,target_critic_params=None)
    rewards=jnp.zeros((2,10)).at[1,-1].set(1.)
    batch=dict(next_z_rl=jnp.zeros((2,4)),next_proprio=jnp.zeros((2,7)),
               next_ref_chunk=jnp.zeros((2,10,7)),rewards=rewards,done=jnp.array([False,True]))
    target=guarded_target(Actor(),Critic(),state,batch,SimpleNamespace(gamma=.99),jax.random.PRNGKey(0))
    np.testing.assert_allclose(target,[1.,.99**9],rtol=1e-6)
    raw=guarded_target(Actor(),Critic(),state,batch,SimpleNamespace(gamma=.99),jax.random.PRNGKey(0),clip=False)
    assert float(raw[0])>1.


def test_outcome_bc_uses_actual_only_for_human_and_autonomous_success():
    b=dict(source_chunk=jnp.array([[2,2],[1,1],[1,1]]),
           autonomous_success=jnp.array([False,True,False]),
           action_chunk=jnp.ones((3,2,7)),ref_chunk=jnp.zeros((3,2,7)))
    target,weights=actor_targets(b)
    np.testing.assert_array_equal(target[:,0,0],[1.,1.,0.])
    np.testing.assert_allclose(weights[:,0],[1.,1.,.1])
    reference,_=actor_targets(b,successful_executed=False)
    np.testing.assert_array_equal(reference[:,0,0],[1.,0.,0.])


def test_round_phases_keep_actor_frozen_for_critic_and_critic_frozen_for_actor():
    cfg=RLTOnlineRLConfig(z_dim=4,proprio_dim=7,chunk_len=3,action_dim=7,
        actor_hidden_dim=8,critic_hidden_dim=8,actor_num_layers=1,critic_num_layers=1)
    state,actor,critic=trainer.init_train_state(cfg,rng=jax.random.PRNGKey(2))
    batch=dict(z_rl=jnp.ones((2,4)),proprio=jnp.zeros((2,7)),ref_chunk=jnp.zeros((2,3,7)),
        action_chunk=jnp.ones((2,3,7))*.1,rewards=jnp.zeros((2,3)),done=jnp.ones(2,bool),
        next_z_rl=jnp.ones((2,4)),next_proprio=jnp.zeros((2,7)),next_ref_chunk=jnp.zeros((2,3,7)),
        source_chunk=jnp.ones((2,3),int),autonomous_success=jnp.ones(2,bool),mc_return=jnp.zeros(2))
    critic_step=make_critic_step(actor,critic,cfg)
    updated,_=critic_step(state,batch)
    for a,b in zip(jax.tree_util.tree_leaves(state.actor_params),jax.tree_util.tree_leaves(updated.actor_params)):
        np.testing.assert_array_equal(a,b)
    assert int(updated.actor_version)==int(state.actor_version)
    actor_step=make_actor_step(actor,cfg,-jnp.ones(7),jnp.ones(7))
    final,metrics=actor_step(updated,batch)
    for a,b in zip(jax.tree_util.tree_leaves(updated.critic_params),jax.tree_util.tree_leaves(final.critic_params)):
        np.testing.assert_array_equal(a,b)
    assert int(final.actor_version)==int(updated.actor_version)+1
    assert int(final.global_step)==int(updated.global_step)
    assert float(metrics["q_gradient_enabled"])==0.
    assert any(not np.array_equal(a,b) for a,b in zip(jax.tree_util.tree_leaves(updated.actor_params),jax.tree_util.tree_leaves(final.actor_params)))
