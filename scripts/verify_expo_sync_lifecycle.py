"""Isolated native-sync warmup, HIL, updates, durability and restart verification."""
import argparse,json,time,uuid,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
p=argparse.ArgumentParser()
for name in ('project','bundle','output','seed-dir','adapter'):p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--edit-scale',type=float);p.add_argument('--learner-only',action='store_true');p.add_argument('--restore-only',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--rtc-delay',type=int,default=2);p.add_argument('--native-batch-update',action='store_true');a=p.parse_args()
if 'verification' not in a.output.parts:raise ValueError('Use a dedicated verification directory')
if a.resume:
 if not (a.output/'lifecycle.json').exists():raise ValueError('Missing first-stage verification')
elif (a.output/'run').exists():raise ValueError('Fresh test requires empty run path')
a.output.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(a.adapter))
from native_online import source_path
source_path(a.project)
from online_learner import Learner
from episode_journal import EpisodeJournal
from online_protocol import atomic_json
args=SimpleNamespace(project=a.project,bundle=a.bundle,run_dir=a.output/'run',batch_size=64,capacity=500000,warmup_steps=0,steps_per_update=30,max_updates=0,recipe='native_sync',seed_dir=a.seed_dir,rtc_delay=a.rtc_delay,native_batch_update=a.native_batch_update,edit_scale=a.edit_scale,learner_only=a.learner_only)
l=Learner(args)
print('LOADED',l.save_status(),flush=True)
if a.edit_scale is not None:
 assert l.agent.edit_scale==l.config.edit_scale==a.edit_scale
 if a.learner_only:assert l.actor_manifest['identity']['edit_scale']==a.edit_scale
start=l.meta['updates']
assert start==(1 if a.resume else 0)
assert l.meta['completed_episodes']==(11 if a.resume else 0)
assert l.meta['actor_updates']==start and l.meta['critic_updates']==20*start
if a.restore_only:
 assert a.resume
 result=dict(passed=True,restored=l.save_status(),checkpoint=l.meta['checkpoint'],actor_identity=l.actor_manifest['identity'])
 (a.output/'restore-only.json').write_text(json.dumps(result,indent=2));print(result,flush=True);raise SystemExit(0)
with np.load(a.bundle.parent/'common/dev_observation.npz',allow_pickle=False) as z:obs={k:z[k].copy() for k in ('state','left_wrist_image','right_wrist_image')}
j=EpisodeJournal(args.run_dir/'episodes')
def episode(outcome,n):
 identity=str(uuid.uuid4());before=l.meta['updates'];l.begin(identity);j.begin(identity,before)
 for i in range(n+2):j.sample(obs,obs['state'].copy(),100+i*.1,i<n//2)
 j.finish(identity,outcome)
 if a.learner_only:
  path=args.run_dir/'episodes'/identity/'finished.json'
  meta=json.loads(path.read_text());meta['rollout_rng']=np.asarray(l.agent.rng).tolist();atomic_json(path,meta)
 l.finish(identity)
 deadline=time.monotonic()+1800
 while l.phase=='updating':
  try:l.begin(str(uuid.uuid4()))
  except ValueError:pass
  else:raise AssertionError('New episode admitted during learner update')
  if time.monotonic()>deadline:raise TimeoutError('Update did not finish')
  time.sleep(.5)
 assert l.phase=='ready',l.error
 assert identity in l.meta['accepted']
 after=l.meta['updates'];l.finish(identity);assert l.meta['updates']==after
 print('EPISODE',outcome,n,l.save_status(),flush=True)
 return before,after
if not a.resume:
 assert episode('unknown',12)==(0,0)
 assert episode('aborted',12)==(0,0)
 assert l.meta['completed_episodes']==0
 for i in range(10):assert episode('success' if i%2==0 else 'failure',16)==(0,0)
 assert l.meta['completed_episodes']==10 and l.meta['credit']==0
assert episode('failure' if a.resume else 'success',30)==(start,start+1)
assert l.meta['actor_updates']==start+1 and l.meta['critic_updates']==20*(start+1)
assert l.meta['checkpoint'] is not None
identity=str(uuid.uuid4());l.begin(identity)
cache=l.infer('/precache',{**obs,'episode_uuid':np.asarray(identity),'prefix':np.empty((0,7),np.float32)})
selected=l.infer('/select',{**obs,'episode_uuid':np.asarray(identity),'precached':np.asarray(cache['precached']),'delay':0,'actor_version':start+1})
assert np.isfinite(selected['actions']).all()
l.cancel_start(identity)
result=dict(passed=True,edit_scale=float(l.agent.edit_scale),robot_execution=False,synthetic_online_episodes=True,seed_episodes=25,version=l.meta['updates'],actor_updates=l.meta['actor_updates'],critic_updates=l.meta['critic_updates'],completed_episodes=l.meta['completed_episodes'],metrics=l.metrics)
(a.output/('resume.json' if a.resume else 'lifecycle.json')).write_text(json.dumps(result));print(result,flush=True)
