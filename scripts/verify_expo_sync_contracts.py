"""CPU native contracts for bounded sampling, augmentation and gradient mapping."""
import argparse,sys,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--adapter',type=Path,required=True);a=p.parse_args()
sys.path.insert(0,str(a.adapter))
from native_online import source_path,configuration,replay_buffer
source_path(a.project)
import jax,jax.numpy as jnp
from stream_sample import Batch
from split_update import augmentation
from batch_memory import mapped
from sync_schedule import Schedule
from types import SimpleNamespace
from expo_ft.utils.augmentation import make_data_augmentation_fn
assert jax.default_backend()=='cpu'
checks=[]
config=configuration(a.bundle,2,aligned=True)
buf=replay_buffer(config,'contract test',96)
for i in range(40):
 obs=dict(state=np.full(7,i*.001,np.float32),left_wrist_image=np.full((180,320,3),i+5,np.uint8),right_wrist_image=np.full((180,320,3),i+10,np.uint8))
 buf.insert(dict(observations=obs,actions=obs['state'],rewards=float(i==39),masks=float(i!=39),dones=i==39,is_hil=i%2==0,is_success=True))
buf.rng=jax.random.PRNGKey(18)
full=buf._convert_to_openpi_format(buf.sample_jax(8));expected_rng=np.asarray(buf.rng).copy()
buf.rng=jax.random.PRNGKey(18)
stream=Batch(buf,8,4)
parts=[jax.tree.map(lambda x:np.array(x,copy=True),b) for b in stream]
actual=jax.tree.map(lambda *x:np.concatenate(x),*parts)
for x,y in zip(jax.tree.leaves(full),jax.tree.leaves(actual)):np.testing.assert_array_equal(x,y)
np.testing.assert_array_equal(buf.rng,expected_rng)
checks.append('native Replay same indices, all batch arrays and RNG')
for flag in (False,True):
 fn=make_data_augmentation_fn(flag);bounded=augmentation(SimpleNamespace(data_augmentation_fn=fn))
 images={k:jnp.arange(8*32*32*3,dtype=jnp.float32).reshape(8,32,32,3)/30000 for k in ('base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb')}
 key=jax.random.PRNGKey(71);expected=fn(key,images);keys=jax.random.split(key,16)
 chunks=[bounded(keys[i*2:(i+4)*2],jax.tree.map(lambda x:x[i:i+4],images)) for i in (0,4)]
 actual=jax.tree.map(lambda *x:jnp.concatenate(x),*chunks)
 for x,y in zip(jax.tree.leaves(expected),jax.tree.leaves(actual)):np.testing.assert_allclose(x,y,rtol=2e-6,atol=2e-6)
checks.append('native crop/full augmentation global keys and outputs')
x=jnp.arange(64*5,dtype=jnp.float32).reshape(64,5)/100;mask=jnp.arange(64)%3>0
forward=lambda w,z:jnp.sin(z@w)
def loss(w,bounded):
 pred=mapped(lambda z:forward(w,z),x,4,remat=True) if bounded else forward(w,x)
 return jnp.sum(pred*mask)/jnp.maximum(mask.sum(),1)
w=jnp.arange(5,dtype=jnp.float32)
for x,y in zip(jax.value_and_grad(loss)(w,False),jax.value_and_grad(loss)(w,True)):np.testing.assert_allclose(x,y,rtol=2e-6,atol=2e-6)
checks.append('full-batch masked loss and rematerialized gradient')
s=Schedule()
for i in range(10):s,n=s.finish(100);assert n==0 and s.credit==0
s,n=s.finish(95);assert n==3 and s.credit==5
s,n=s.finish(25);assert n==1 and s.credit==0
s,n=s.finish(601);assert n==20 and s.credit==1
checks.append('10 collection episodes, no catchup, update credit and no cap')
print(json.dumps(dict(passed=True,checks=checks)))
