"""Isolated full native batch update, numeric receipt and portable checkpoint."""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser()
for name in ('project','bundle','seed-dir','output','adapter'):p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--num-devices',type=int,choices=(1,2),required=True)
p.add_argument('--restore',type=Path)
a=p.parse_args()
if 'verification' not in a.output.parts or a.output.exists():raise ValueError('Fresh verification output required')
a.output.mkdir(parents=True)
sys.path.insert(0,str(a.adapter))
from native_online import source_path
source_path(a.project)
import jax
from native_online import load_native,replay_buffer
from memory_update import HostTargetUpdater
from batch_memory import install
from stream_checkpoint import save,restore
from actor_snapshot import export,contract
from success_seed import insert_seed,seed_manifest
start=time.monotonic()
def event(name,**items):
 row=dict(event=name,seconds=time.monotonic()-start,**items)
 print(json.dumps(row),flush=True)
 with (a.output/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
agent,config,prompt=load_native(a.bundle,5,aligned=True,edit_scale=.01,num_devices=a.num_devices)
install()
u=HostTargetUpdater(agent);u.cache_inference=False
agent=u.prepare(agent)
event('loaded',devices=[str(d) for d in jax.devices()])
if a.restore:
 agent=restore(a.restore/'checkpoints',agent,u,1)
 event('restored',critic=int(agent.critic.step),actor=int(agent.actor_train_state.step))
else:
 buffer=replay_buffer(config,prompt,capacity=12000)
 insert_seed(buffer,a.seed_dir,seed_manifest(a.seed_dir))
 batch=buffer._convert_to_openpi_format(buffer.sample_jax(64*20))
 actor_batch=buffer._convert_to_openpi_format(buffer.sample_jax(64,success_only=True))
 def samples(tree):
  result={}
  for i,x in enumerate(jax.tree.leaves(tree)):
   if hasattr(x,'dtype') and np.issubdtype(x.dtype,np.inexact):
    v=np.asarray(x).reshape(-1);idx=np.linspace(0,len(v)-1,min(2048,len(v)),dtype=int)
    result[str(i)]=v[idx].astype(np.float32)
  return result
 before=samples(agent)
 event('update_start')
 t=time.monotonic();agent,metrics=u.update(agent,batch,20,actor_batch);jax.block_until_ready(agent)
 metrics={k:float(np.asarray(v)) for k,v in metrics.items() if np.asarray(v).size==1}
 assert all(np.isfinite(x) for x in metrics.values())
 assert int(agent.critic.step)==20 and int(agent.actor_train_state.step)==1
 event('update_complete',update_seconds=time.monotonic()-t,metrics=metrics)
 after=samples(agent)
 np.savez(a.output/'deltas.npz',**{k:after[k]-before[k] for k in before})
 save(a.output/'checkpoints',agent,u,1)
 event('checkpoint_saved')
manifest=export(a.output/'actors',agent,1,contract(a.bundle,a.project,5,.01))
np.savez(a.output/'rng.npz',rng=np.asarray(agent.rng))
(a.output/'result.json').write_text(json.dumps(dict(passed=True,devices=a.num_devices,critic=int(agent.critic.step),actor=int(agent.actor_train_state.step),actor_records=manifest['records'])))
event('passed')
