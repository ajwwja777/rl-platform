import importlib.util
from pathlib import Path
import numpy as np
import pytest


def fixture():
    pytest.importorskip('jax')
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('retention_native_fixture',root/'third_party/openpi-rlt/rlt_online_rl/tests/test_trainer.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def test_disabled_retention_is_exact_native_function():
    from methods.openpi_rlt.experiments.retained_actor import make_retained_train_step
    from rlt_online_rl import trainer
    assert make_retained_train_step(0.,None,0.) is trainer.train_step
    with pytest.raises(ValueError):make_retained_train_step(0.,None,-1.)


def test_empty_mask_preserves_full_native_update_and_mc():
    import jax
    from rlt_online_rl import trainer
    from methods.openpi_rlt.experiments.credit import make_train_step
    from methods.openpi_rlt.experiments.retained_actor import make_retained_train_step
    mod=fixture();cfg=mod._config();batch=mod._batch(cfg)
    state,actor,critic=trainer.init_train_state(cfg,rng=jax.random.PRNGKey(42))
    state=state.replace(global_step=state.global_step+1)
    batch.update(retention_mask=np.zeros(len(batch['done']),bool),mc_return=np.full(len(batch['done']),.7,np.float32),mc_valid=np.ones(len(batch['done']),bool))
    original=trainer.update_actor
    for mc in [0.,.3]:
        expected,_=make_train_step(mc)(state,batch,actor=actor,critic=critic,rl_config=cfg)
        actual,metrics=make_retained_train_step(mc,state.actor_params,50.)(state,batch,actor=actor,critic=critic,rl_config=cfg)
        for a,b in zip(jax.tree.leaves(expected),jax.tree.leaves(actual)):
            np.testing.assert_allclose(a,b,atol=1e-6,rtol=1e-5)
        assert float(metrics.get('retention_penalty',0))==0
    assert trainer.update_actor is original


def test_selected_teacher_error_changes_actor_not_critic_update():
    import jax
    from rlt_online_rl import trainer
    from methods.openpi_rlt.experiments.retained_actor import make_retained_train_step
    mod=fixture();cfg=mod._config();batch=mod._batch(cfg)
    state,actor,critic=trainer.init_train_state(cfg,rng=jax.random.PRNGKey(42))
    state=state.replace(global_step=state.global_step+1)
    teacher,_,_=trainer.init_train_state(cfg,rng=jax.random.PRNGKey(43))
    batch['retention_mask']=np.ones(len(batch['done']),bool)
    expected,_=trainer.train_step(state,batch,actor=actor,critic=critic,rl_config=cfg)
    actual,metrics=make_retained_train_step(0.,teacher.actor_params,50.)(state,batch,actor=actor,critic=critic,rl_config=cfg)
    assert float(metrics['retention_penalty'])>0
    assert float(metrics['weighted_retention'])==pytest.approx(50*float(metrics['retention_penalty']))
    assert any(not np.allclose(a,b)for a,b in zip(jax.tree.leaves(expected.actor_params),jax.tree.leaves(actual.actor_params)))
    for a,b in zip(jax.tree.leaves(expected.critic_params),jax.tree.leaves(actual.critic_params)):
        np.testing.assert_array_equal(a,b)


def test_held_gripper_teacher_difference_is_not_retention_error():
    import dataclasses
    import jax
    import jax.numpy as jnp
    from rlt_online_rl import trainer
    from methods.openpi_rlt.experiments.retained_actor import make_retained_train_step
    mod=fixture();cfg=dataclasses.replace(mod._config(),action_dim=7);batch=mod._batch(cfg)
    state,actor,critic=trainer.init_train_state(cfg,rng=jax.random.PRNGKey(42))
    state=state.replace(global_step=state.global_step+1)
    teacher=jax.tree.map(lambda x:jnp.array(x),state.actor_params)
    last=teacher['trunk']['layers'][-1]
    last['b']=last['b'].at[jnp.arange(6,cfg.chunk_len*7,7)].add(10.)
    batch['retention_mask']=np.ones(len(batch['done']),bool)
    expected,_=make_retained_train_step(0.,state.actor_params,50.)(state,batch,actor=actor,critic=critic,rl_config=cfg)
    actual,metrics=make_retained_train_step(0.,teacher,50.)(state,batch,actor=actor,critic=critic,rl_config=cfg)
    assert float(metrics['retention_penalty'])==pytest.approx(0.,abs=1e-12)
    for a,b in zip(jax.tree.leaves(expected),jax.tree.leaves(actual)):
        np.testing.assert_allclose(a,b,atol=1e-6,rtol=1e-5)
