"""CPU-only real Replay gather check, no optimizer, robot or production mutation."""
import argparse,sys,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser()
for name in ('project','bundle','adapter'):p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();sys.path.insert(0,str(a.adapter))
from native_online import source_path,configuration,replay_buffer
source_path(a.project)
import jax
import expo_ft.agents
from outcome_sampling import sample_critic,pools
assert jax.default_backend()=='cpu'
c=configuration(a.bundle,5,aligned=True)
b=replay_buffer(c,'sampling verification',128)
for i in range(96):
 source=1 if i<48 else 2
 obs=dict(state=np.full(7,source*.1,np.float32),left_wrist_image=np.full((180,320,3),source+5,np.uint8),right_wrist_image=np.full((180,320,3),source+5,np.uint8))
 b.insert(dict(observations=obs,actions=obs['state'],rewards=float(i in (47,71)),masks=float(i not in (47,71,95)),dones=i in (47,71,95),is_hil=i<48,is_success=i<72))
saved=np.array(jax.random.PRNGKey(71));b.rng=saved
available=pools(b,48,[(24,'success'),(24,'failure')])
raw=sample_critic(b,320,available)
assert np.all(np.asarray(raw['is_hil']).reshape(5,64).sum(axis=1)==32)
assert not np.all(np.asarray(raw['is_success']).reshape(5,64)[:,32:])
# Native gather preserves source flags and includes online failure windows.
print('raw keys',list(raw))
full=b._convert_to_openpi_format(raw)
for k,v in full.items():
 if hasattr(v,'shape'):print(k,v.shape)
assert b.prior_sampling_stats['demonstration_samples']==160
assert b.prior_sampling_stats['online_success_samples']==96
assert b.prior_sampling_stats['online_failure_samples']==64
assert np.asarray(raw['is_hil']).sum()==160
assert np.asarray(raw['is_success']).sum()==256
expected=jax.tree.map(lambda x:np.array(x,copy=True),full);rng=np.array(b.rng)
b.rng=saved;again=b._convert_to_openpi_format(sample_critic(b,320,available))
for x,y in zip(jax.tree.leaves(expected),jax.tree.leaves(again)):np.testing.assert_array_equal(x,y)
np.testing.assert_array_equal(b.rng,rng)
print(json.dumps(dict(passed=True,checks=['real native multi-step gather and conversion','fixed per-minibatch source quota','checkpoint RNG deterministic replay'],stats=b.prior_sampling_stats)))
