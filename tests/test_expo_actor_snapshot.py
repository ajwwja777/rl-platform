import dataclasses
from pathlib import Path
import sys
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import flax.struct as struct
import flax.nnx as nnx
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"methods/expo_ft/cobot_adapter"))
from actor_snapshot import export,restore

@dataclasses.dataclass
class State:
    params: object
    step: int=0
    ema_params: object=None
    opt_state: object=()

@struct.dataclass
class Net:
    params: object

@struct.dataclass
class Agent:
    actor: object=struct.field(pytree_node=False)
    actor_train_state: object
    batch_encoder: object
    edit_actor: object
    target_critic: object
    rng: object
    _infer_cache: object=None
    def cache_infer_params(self):return self

def make_agent(offset=0):
    params=nnx.State({'train':nnx.VariableState(nnx.Param,jnp.arange(12,dtype=jnp.float32).reshape(3,4)+offset),
                      'frozen':nnx.VariableState(nnx.Param,jnp.ones(4))})
    actor=SimpleNamespace(get_params=lambda state:state.params,train_config=SimpleNamespace(
        trainable_filter=nnx.PathContains('train'),freeze_filter=nnx.PathContains('frozen')))
    return Agent(actor,State(params,offset),Net({'v':jnp.arange(7,dtype=jnp.float32)+offset}),
        Net({'v':jnp.arange(5,dtype=jnp.float32)+offset}),Net({'v':jnp.arange(3,dtype=jnp.float32)+offset}),jax.random.PRNGKey(42))

def test_complete_inference_state_roundtrip_and_seed(tmp_path):
    identity={'recipe':'test'}
    original=make_agent();manifest=export(tmp_path,original,0,identity)
    assert all(r['seed'] for r in manifest['records'] if r['section']=='actor')
    restored,_=restore(tmp_path/'0',make_agent(),identity)
    updated=make_agent(2);export(tmp_path,updated,2,identity)
    restored,_=restore(tmp_path/'2',restored,identity)
    for expected,actual in zip(jax.tree.leaves(updated.actor_train_state.params),jax.tree.leaves(restored.actor_train_state.params)):
        np.testing.assert_array_equal(expected,actual)
    np.testing.assert_array_equal(restored.target_critic.params['v'],updated.target_critic.params['v'])
    assert restored.actor_train_state.step==2
    assert restored.actor_train_state.opt_state==()
    with pytest.raises(ValueError,match='differs'):export(tmp_path,make_agent(3),2,identity)

def test_corrupt_snapshot_and_wrong_contract_rejected(tmp_path):
    manifest=export(tmp_path,make_agent(),0,{'recipe':'test'})
    with pytest.raises(ValueError,match='contract'):restore(tmp_path/'0',make_agent(),{'recipe':'other'})
    item=next(r for r in manifest['records'] if not r['seed'])
    (tmp_path/'0'/item['file']).write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='checksum'):restore(tmp_path/'0',make_agent(),{'recipe':'test'})
