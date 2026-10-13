"""No-ROS synthetic lifecycle verification; never point at production run state."""
import argparse,json,time,uuid
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'methods/expo_ft/cobot_adapter'))
from native_online import source_path
p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true');p.add_argument('--outcome',choices=('success','failure'),default='success')
a=p.parse_args()
if a.resume:
    if not (a.output/'lifecycle.json').is_file():raise ValueError('Resume only a completed isolated verification run')
elif (a.output/'run').exists():raise ValueError('Fresh verification requires a new output/run directory')
a.output.mkdir(parents=True,exist_ok=True)
source_path(a.project)
from online_learner import Learner
from episode_journal import EpisodeJournal
args=SimpleNamespace(project=a.project,bundle=a.bundle,run_dir=a.output/'run',batch_size=4,capacity=256,warmup_steps=64,steps_per_update=25,max_updates=2)
l=Learner(args)
start_version=l.meta['updates'];start_accepted=len(l.meta['accepted'])
if a.resume:assert start_version==2 and start_accepted==2,(start_version,start_accepted)
else:assert start_version==0 and start_accepted==0
with np.load(a.bundle.parent/'common/dev_observation.npz',allow_pickle=False) as z:obs={k:z[k].copy() for k in ('state','left_wrist_image','right_wrist_image')}
j=EpisodeJournal(args.run_dir/'episodes')
def episode(outcome,n):
 identity=str(uuid.uuid4());before=l.meta['updates'];l.begin(identity);j.begin(identity,before)
 for i in range(n+1):j.sample(obs,obs['state'].copy(),100+i*.1,i<40)
 j.finish(identity,outcome);l.finish(identity)
 try:l.begin(str(uuid.uuid4()))
 except ValueError:pass
 else:raise AssertionError('start allowed during update')
 while l.phase=='updating':time.sleep(.5)
 assert l.phase=='ready',l.error
 assert identity in l.meta['accepted']
 after=l.meta['updates'];l.finish(identity);assert l.meta['updates']==after
 return before,after
assert episode('unknown',10)==(start_version,start_version)
before,after=episode(a.outcome,80)
assert after==before+2,(before,after)
assert int(l.agent.actor_train_state.step)==(after if a.outcome=='success' else 0)
assert l.meta['checkpoint'] is not None
identity=str(uuid.uuid4());l.begin(identity)
cache=l.infer('/precache',{**obs,'episode_uuid':np.asarray(identity),'prefix':np.empty((0,7),np.float32)})
selected=l.infer('/select',{**obs,'episode_uuid':np.asarray(identity),'precached':np.asarray(cache['precached']),'delay':0,'actor_version':after})
assert np.isfinite(selected['actions']).all()
result=dict(passed=True,start_version=start_version,version=after,accepted=len(l.meta['accepted']),replay_steps=len(l.buffer),actor_updates=int(l.agent.actor_train_state.step),metrics=l.metrics,robot_execution=False)
(a.output/('resume.json' if a.resume else 'lifecycle.json')).write_text(json.dumps(result));print(result,flush=True)
