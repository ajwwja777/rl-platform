import numpy as np
import pytest
from methods.openpi_rlt.experiments.credit import Profile, episode_credit, sample_indices

def rows():
    return [dict(collection_phase_id=2, episode_id=1, step_id=i,
        rewards=np.array([0., float(i==2)]), done=i==2, success=int(i==2))
        for i in range(3)]

def test_return_uses_control_steps_and_does_not_relabel():
    data=rows(); before=[x["rewards"].copy() for x in data]
    target, valid=episode_credit(data,.9)
    assert valid.all()
    np.testing.assert_allclose(target,[.9**3,.9**2,.9])
    for r,b in zip(data,before):np.testing.assert_array_equal(r["rewards"],b)

def test_unknown_gap_conflict_and_duplicate_are_not_invented():
    for data in [rows()[:2], [rows()[0], dict(rows()[2],step_id=5)],
                 rows()+[rows()[0]], [dict(r,success=0) for r in rows()]]:
        assert not episode_credit(data,.99)[1].any()

def test_failed_episode_return_is_zero():
    data=[dict(r,rewards=np.zeros(2),success=0) for r in rows()]
    target,valid=episode_credit(data,.99)
    assert valid.all() and not target.any()

def test_sampling_keeps_holdout_out_and_terminal_is_explicit():
    meta=[dict(phase="online",episode_id=i//3,done=i%3==2,
               portion=("early","middle","late")[i%3]) for i in range(12)]
    train=np.arange(9)
    for name in ["uniform","terminal","tail_mix","episode_balanced"]:
        ids=sample_indices(np.random.default_rng(1),train,meta,Profile(sampling=name),1000)
        assert set(ids)<=set(train)
        if name=="terminal":assert all(meta[i]["done"] for i in ids)
    ids=sample_indices(np.random.default_rng(3),train,meta,Profile(sampling="tail_mix",tail_fraction=.5),10000)
    assert .63 < np.mean([meta[i]["portion"]=="late" for i in ids]) < .70

def test_invalid_profile_rejected():
    with pytest.raises(ValueError):Profile(mc_weight=2)
    with pytest.raises(ValueError):Profile(sampling="magic")

def test_zero_weight_keeps_exact_function():
    pytest.importorskip("jax")
    from methods.openpi_rlt.experiments.credit import make_train_step
    from rlt_online_rl import trainer
    assert make_train_step(0) is trainer.train_step

def test_private_critic_adapter_preserves_baseline_and_updates_actor():
    jax=pytest.importorskip("jax")
    import importlib.util
    from pathlib import Path
    from rlt_online_rl import trainer
    from methods.openpi_rlt.experiments.credit import make_train_step
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location("credit_fixture",root/"third_party/openpi-rlt/rlt_online_rl/tests/test_trainer.py")
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    cfg=fixture._config();batch=fixture._batch(cfg)
    actor,critic=trainer._make_networks(cfg)
    state,actor,critic=trainer.init_train_state(cfg,rng=jax.random.PRNGKey(42))
    original=trainer.update_critic
    run=make_train_step(.1)
    batch.update(mc_return=np.full(8,.7,np.float32),mc_valid=np.zeros(8,bool))
    kwargs=dict(actor=actor,critic=critic,rl_config=cfg)
    expected,_=trainer.train_step(state,batch,**kwargs)
    actual,_=run(state,batch,**kwargs)
    for a,b in zip(jax.tree.leaves(expected),jax.tree.leaves(actual)):
        np.testing.assert_allclose(a,b,atol=1e-6,rtol=1e-5)
    batch["mc_valid"][:]=True
    changed,metrics=run(state,batch,**kwargs)
    assert float(metrics["mc_effective_weight"])==pytest.approx(.1)
    assert trainer.update_critic is original
    assert int(changed.global_step)==int(state.global_step)+1
