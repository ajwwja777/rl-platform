import json
import pickle
import copy
import numpy as np
import pytest
from methods.openpi_rlt.experiments.supported_sampler import SupportedSampler, unpack_actor_batch
from methods.openpi_rlt.experiments.supported_hil_runtime import SupportedIndex
from methods.openpi_rlt.experiments.supported_dispatch import runtime_for

class Adapter:
    def prepare_training_batch(self, batch):
        return dict(batch, action_chunk=batch["action_chunk"] / 2)

def row(ep, human=False, done=True, phase=2):
    return dict(collection_phase_id=phase, episode_id=ep, step_id=0,
        z_rl=np.zeros(2), proprio=np.zeros(7), ref_chunk=np.zeros((2,7)),
        action_chunk=np.full((2,7), float(ep)), rewards=np.array([0.,0.]),
        next_z_rl=np.zeros(2), next_proprio=np.zeros(7), next_ref_chunk=np.zeros((2,7)),
        source_chunk=np.full(2,2 if human else 1), done=done, success=0,
        source=2 if human else 1, intervention_flag=human)

def sampler(tmp_path, rows, state=None):
    path=tmp_path/"journal.pkl"
    with path.open("wb") as f:
        for r in rows: pickle.dump(r,f)
    if state is None:
        state=dict(base_rng=np.random.default_rng(42).bit_generator.state,
                   actor_rng=np.random.default_rng(1042).bit_generator.state,global_step=7000)
    return SupportedSampler(SupportedIndex(path,.99,[]),Adapter(),state=state)

def test_no_hil_and_normalization_once(tmp_path):
    s=sampler(tmp_path,[row(10000),row(10001)])
    b=unpack_actor_batch(s.sample_batch(128))
    assert s.last_receipt["extra_actor_slots"]==0
    assert s.last_receipt["critic_identities"]==s.last_receipt["actor_identities"]
    np.testing.assert_array_equal(b["actor_batch"]["action_chunk"],b["action_chunk"]/2)

def test_only_recent_real_online_hil_gets_extra_slots(tmp_path):
    s=sampler(tmp_path,[row(-2,True,phase=1),row(9980,True),row(10000),row(10001,True),row(10002,True,False)])
    s.sample_batch(128);r=s.last_receipt
    assert r["recent_hil_pool_windows"]==1
    assert r["extra_actor_slots"]==8
    assert r["available_complete_windows"]==4
    assert all(x[1]!=10002 for x in r["actor_identities"]+r["critic_identities"])
    assert sum(x[1]==10001 for x in r["actor_identities"])>=8
    assert all(a[1]==10001 for a,c in zip(r["actor_identities"],r["critic_identities"]) if a!=c)

def test_both_rngs_resume_exactly(tmp_path):
    rows=[row(10000),row(10001,True)]
    s=sampler(tmp_path,rows);s.sample_batch(128);state=copy.deepcopy(s.state())
    expected=s.sample_batch(128);receipt=s.last_receipt
    s2=sampler(tmp_path,rows,state);actual=s2.sample_batch(128)
    assert s2.last_receipt==receipt
    for k in actual:np.testing.assert_array_equal(actual[k],expected[k])
    assert s2.state()==s.state()

@pytest.mark.parametrize("schema,module",[("held-gripper-retention-v1","supported_runtime"),("held-gripper-hil8-clip-v2","supported_hil_runtime")])
def test_dispatch_preserves_old_runtime(tmp_path,schema,module):
    p=tmp_path/"profile.json";p.write_text(json.dumps(dict(schema=schema)))
    assert runtime_for(p).__name__.endswith("."+module)

def test_unknown_schema_rejected_before_model_import(tmp_path):
    p=tmp_path/"profile.json";p.write_text('{"schema":"invalid"}')
    with pytest.raises(ValueError,match="Unsupported"):runtime_for(p)

def test_missing_actor_contract_rejected():
    with pytest.raises(ValueError,match="Missing Actor"):unpack_actor_batch({"rewards":np.zeros(2)})


def test_bounded_target_and_native_critic_remains_untouched():
    import importlib.util
    import jax
    from rlt_online_rl import trainer
    from methods.openpi_rlt.experiments.guarded_credit import make_train_step
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location("guard_fixture",root/"third_party/openpi-rlt/rlt_online_rl/tests/test_trainer.py")
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    cfg=fixture._config();batch=fixture._batch(cfg)
    state,actor,critic=trainer.init_train_state(cfg,rng=jax.random.PRNGKey(42))
    original=trainer.update_critic
    # Synthetic out-of-range targets exercise both limits; these are NOT accepted journal rewards.
    batch.update(mc_return=np.array([-1.,2.]*4,np.float32),mc_valid=np.ones(8,bool))
    _,metrics=make_train_step(1.,target_policy="clip")(state,batch,actor=actor,critic=critic,rl_config=cfg)
    assert float(metrics["target_q_mean"])==pytest.approx(.5)
    assert trainer.update_critic is original
    assert make_train_step(0) is trainer.train_step


def test_reward_outside_sparse_terminal_contract_is_not_admitted(tmp_path):
    r=row(10000);r['rewards'][-1]=2.
    s=sampler(tmp_path,[r])
    with pytest.raises(ValueError,match="complete eligible"):s.sample_batch(128)
