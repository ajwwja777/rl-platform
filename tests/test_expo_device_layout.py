"""Run with XLA_FLAGS=--xla_force_host_platform_device_count=2 JAX_PLATFORMS=cpu."""
import sys
from pathlib import Path
from types import SimpleNamespace
import dataclasses
import numpy as np
import pytest
import jax
import jax.numpy as jnp
import flax.struct as struct
import flax.nnx as nnx
sys.path.insert(0,str(Path(__file__).parents[1]/'methods/expo_ft/cobot_adapter'))
from device_layout import buffer_identity
from stream_checkpoint import save,restore
from actor_snapshot import export

@struct.dataclass
class State:
 params:object
 step:object
 opt_state:object
@struct.dataclass
class Net:
 params:object
@struct.dataclass
class Agent:
 actor:object=struct.field(pytree_node=False)
 actor_train_state:object
 rng:object
 batch_encoder:object
 edit_actor:object
 target_critic:object
 _infer_cache:object=None

def layout():
 if len(jax.devices())<2:pytest.skip('Needs two virtual CPU devices')
 mesh=jax.sharding.Mesh(np.asarray(jax.devices()[:2]),('data',))
 return jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec()),jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec('data'))

def make_agent(rep,split):
 def put(x,s):return jax.device_put(np.asarray(x,dtype=np.float32),s)
 params=nnx.State({'train':nnx.VariableState(nnx.Param,put(np.arange(16).reshape(4,4),split)),
                   'frozen':nnx.VariableState(nnx.Param,put(np.ones(4),rep))})
 actor=SimpleNamespace(get_params=lambda state:state.params,train_config=SimpleNamespace(trainable_filter=nnx.PathContains('train'),freeze_filter=nnx.PathContains('frozen')))
 net=Net({'v':put(np.arange(8),split)})
 return Agent(actor,State(params,jax.device_put(np.int32(3),rep),{'m':put(np.arange(16).reshape(4,4),split)}),jax.device_put(jax.random.PRNGKey(3),rep),net,net,net)

def test_buffer_identity_deduplicates_only_complete_aliases():
 rep,split=layout();x=jax.device_put(np.arange(8,dtype=np.float32),rep)
 assert buffer_identity(x)==buffer_identity(x)
 y=jax.device_put(x,split)
 assert buffer_identity(x)!=buffer_identity(y)
 assert len(buffer_identity(x)[2])==2
 z=jax.device_put(np.arange(8,dtype=np.float32),rep)
 assert buffer_identity(x)!=buffer_identity(z)
 # A donated compiled call with one argument for two aliases stays valid.
 f=jax.jit(lambda unique:unique[0]+unique[0],donate_argnums=(0,))
 np.testing.assert_array_equal(f((x,)),np.arange(8)*2)

def test_checkpoint_preserves_rng_optimizer_and_reshards_between_topologies(tmp_path):
 rep,split=layout();a=make_agent(rep,split)
 u=SimpleNamespace(frozen=a.actor.train_config.freeze_filter,target={'ema':np.arange(16,dtype=np.float32).reshape(4,4)})
 save(tmp_path/'checkpoints',a,u,1)
 single=jax.sharding.SingleDeviceSharding(jax.devices()[0])
 template=make_agent(single,single)
 restored=restore(tmp_path/'checkpoints',template,u,1)
 for expected,actual in zip(jax.tree.leaves(a),jax.tree.leaves(restored)):
  np.testing.assert_array_equal(expected,actual)
  assert actual.devices()=={jax.devices()[0]}
 save(tmp_path/'single',restored,u,1)
 back=restore(tmp_path/'single',make_agent(rep,split),u,1)
 for expected,actual in zip(jax.tree.leaves(a),jax.tree.leaves(back)):
  np.testing.assert_array_equal(expected,actual)
  assert actual.sharding==expected.sharding
 left=export(tmp_path/'dual-actors',a,3,{'test':1})
 right=export(tmp_path/'single-actors',restored,3,{'test':1})
 assert [x['raw_sha256'] for x in left['records']]==[x['raw_sha256'] for x in right['records']]


def test_donated_native_adapter_handles_replicated_aliases(monkeypatch):
 import types
 from functools import partial
 from memory_update import donated_update
 @struct.dataclass
 class SmallAgent:
  weights:object
  target:object
  data_sharding:object=struct.field(pytree_node=False)
  actor:object=struct.field(pytree_node=False)
  train_base_actor:bool=struct.field(pytree_node=False,default=True)
  _infer_cache:object=None
  def cache_infer_params(self):return self
 @partial(jax.jit,static_argnames=('utd_ratio','train_actor'))
 def native(self,agent,batch,utd_ratio,actor_batch,train_actor=True):
  updated=agent.weights-jnp.mean(batch,axis=0)
  return agent.replace(weights=updated,target=agent.target*.9+updated*.1),{'loss':jnp.sum(updated)}
 module=types.ModuleType('expo_ft.agents.alg.realtime_expo_ft')
 module.RealTimeEXPOFTLearner=SimpleNamespace(_update_jit=native)
 for name in ('expo_ft','expo_ft.agents','expo_ft.agents.alg'):
  monkeypatch.setitem(sys.modules,name,types.ModuleType(name))
 monkeypatch.setitem(sys.modules,'expo_ft.agents.alg.realtime_expo_ft',module)
 rep,split=layout();value=jax.device_put(np.arange(4,dtype=np.float32),rep)
 agent=SmallAgent(value,value,split,SimpleNamespace(replicated_sharding=rep))
 batch=jax.device_put(np.ones((8,4),np.float32),split)
 result,metrics=donated_update(agent,batch,20,None,cache_inference=False)
 np.testing.assert_allclose(result.weights,np.arange(4)-1)
 np.testing.assert_allclose(result.target,np.arange(4)-.1,rtol=1e-6,atol=1e-6)
 assert len(result.weights.devices())==2
