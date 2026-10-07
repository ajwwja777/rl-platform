import numpy as np
import pytest
from test_retained_actor import fixture

@pytest.mark.parametrize("different", [False, True])
def test_only_actor_receives_separate_batch_without_global_patch(different):
    import jax
    import jax.numpy as jnp
    from rlt_online_rl import trainer
    from methods.openpi_rlt.experiments.retained_actor import make_retained_train_step
    from methods.openpi_rlt.experiments.separate_actor_batch import with_separate_actor_batch
    mod=fixture();cfg=mod._config();batch=mod._batch(cfg)
    state,actor,critic=trainer.init_train_state(cfg,rng=jax.random.PRNGKey(42))
    state=state.replace(global_step=state.global_step+1)
    n=len(batch['done']);batch.update(retention_mask=np.ones(n,bool),mc_return=np.full(n,.7,np.float32),mc_valid=np.ones(n,bool))
    run=make_retained_train_step(.3,state.actor_params,50.)
    actor_batch=dict(batch)
    if different:
        actor_batch['ref_chunk']=jnp.asarray(batch['ref_chunk'])+.3
        actor_batch['action_chunk']=jnp.asarray(batch['action_chunk'])+.3
    original=trainer.update_actor
    expected,_=run(state,batch,actor=actor,critic=critic,rl_config=cfg)
    actual,_=with_separate_actor_batch(run)(state,dict(batch,actor_batch=actor_batch),actor=actor,critic=critic,rl_config=cfg)
    for a,b in zip(jax.tree.leaves(expected.critic_params),jax.tree.leaves(actual.critic_params)):
        np.testing.assert_array_equal(a,b)
    if different:
        assert any(not np.array_equal(a,b)for a,b in zip(jax.tree.leaves(expected.actor_params),jax.tree.leaves(actual.actor_params)))
    else:
        for a,b in zip(jax.tree.leaves(expected),jax.tree.leaves(actual)):
            np.testing.assert_array_equal(a,b)
    assert trainer.update_actor is original
