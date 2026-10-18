"""Real native batch5 lifecycle; run partial, resume, then restore in fresh processes."""
import argparse,json,sys,time,uuid
from pathlib import Path
from types import SimpleNamespace
import numpy as np
p=argparse.ArgumentParser()
for name in ('project','bundle','seed-dir','output','adapter'):p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--stage',choices=('partial','resume','restore'),required=True)
p.add_argument('--num-devices',type=int,choices=(1,2,4),default=1)
a=p.parse_args()
if 'verification' not in a.output.parts:raise ValueError('Independent verification output required')
if a.stage=='partial' and (a.output/'run').exists():raise ValueError('Fresh run required')
if a.stage!='partial' and not (a.output/'partial.json').exists():raise ValueError('Partial receipt required')
sys.path.insert(0,str(a.adapter))
from native_online import source_path
source_path(a.project)
from online_learner import Learner
from episode_journal import EpisodeJournal
from online_protocol import atomic_json
args=SimpleNamespace(project=a.project,bundle=a.bundle,run_dir=a.output/'run',batch_size=64,capacity=500000,
 warmup_steps=0,steps_per_update=30,max_updates=0,recipe='native_sync',seed_dir=a.seed_dir,rtc_delay=5,
 native_batch_update=True,edit_scale=.01,learner_only=True,num_devices=a.num_devices,episodes_per_update=5)
l=Learner(args)
print('LOADED',json.dumps(l.save_status()),flush=True)
with np.load(a.bundle.parent/'common/dev_observation.npz',allow_pickle=False) as z:
 obs={k:z[k].copy() for k in ('state','left_wrist_image','right_wrist_image')}
j=EpisodeJournal(args.run_dir/'episodes')
def episode(outcome,n):
 identity=str(uuid.uuid4());version=l.meta['updates'];l.begin(identity);j.begin(identity,version)
 for i in range(n+2):j.sample(obs,obs['state'].copy(),100+i*.1,i<n//2)
 j.finish(identity,outcome)
 path=j.root/identity/'finished.json';info=json.loads(path.read_text())
 info['rollout_rng']=np.asarray(l.agent.rng).tolist();atomic_json(path,info)
 l.finish(identity);deadline=time.monotonic()+2400
 while l.phase=='updating':
  if time.monotonic()>deadline:raise TimeoutError('Update timed out')
  time.sleep(.5)
 assert l.phase=='ready',l.error
 committed=json.loads(l.meta_path.read_text());l.finish(identity)
 assert l.meta==committed and identity in committed['accepted']
 print('EPISODE',outcome,json.dumps(l.save_status()),flush=True)
if a.stage=='partial':
 for outcome in ('unknown','aborted'):episode(outcome,6)
 assert l.meta['completed_episodes']==0
 for i in range(10):episode('success' if i%2 else 'failure',16)
 assert l.meta['credit']==0 and l.meta['completed_episodes']==10
 for _ in range(3):episode('failure',6)
 assert l.meta['updates']==0 and l.meta['episodes_since_update']==3 and l.meta['credit']==18
 assert l.meta['checkpoint'] is None and l.actor_manifest['actor_version']==0
elif a.stage=='resume':
 assert l.meta['updates']==0 and l.meta['episodes_since_update']==3 and l.meta['credit']==18
 episode('failure',6)
 assert l.meta['updates']==0 and l.meta['episodes_since_update']==4
 episode('success',6)
 assert l.meta['updates']==1 and l.meta['episodes_since_update']==0 and l.meta['credit']==0
 assert l.meta['critic_updates']==20 and l.meta['actor_updates']==1 and l.actor_manifest['actor_version']==1
else:
 assert l.meta['updates']==1 and l.meta['episodes_since_update']==0 and l.meta['completed_episodes']==15
 assert l.meta['critic_updates']==20 and l.meta['actor_updates']==1 and l.actor_manifest['actor_version']==1
result=dict(passed=True,stage=a.stage,robot_execution=False,synthetic_episodes=True,status=l.save_status())
atomic_json(a.output/(a.stage+'.json'),result);print('PASSED',json.dumps(result),flush=True)
