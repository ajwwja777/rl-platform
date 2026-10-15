"""Load a prepared batch continuation and verify counters/Actor identity; never update."""
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
p=argparse.ArgumentParser()
p.add_argument('--profile',type=Path,required=True);p.add_argument('--adapter',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);p.add_argument('--num-devices',type=int,choices=(1,2),default=2)
a=p.parse_args()
if 'verification' not in a.output.parts:raise ValueError('Verification receipt path required')
c=json.loads(a.profile.read_text());root=Path(c['run_dir'])
meta=(root/'committed.json').read_bytes()
before=json.loads((root/'actors'/str(json.loads(meta)['updates'])/'manifest.json').read_text())
sys.path.insert(0,str(a.adapter))
from native_online import source_path
source_path(Path(c['project']))
from online_learner import Learner
args=SimpleNamespace(project=Path(c['project']),bundle=Path(c['bundle']),run_dir=root,batch_size=64,capacity=500000,
 warmup_steps=0,steps_per_update=30,max_updates=0,recipe='native_sync',seed_dir=Path(c['seed_dir']),rtc_delay=c['rtc_delay'],
 native_batch_update=True,edit_scale=c['edit_scale'],learner_only=True,num_devices=a.num_devices,
 episodes_per_update=c['episodes_per_update'],deployment_id=c['model_id'])
l=Learner(args)
assert l.phase=='ready' and not l.active and not l.error
assert (root/'committed.json').read_bytes()==meta
assert l.actor_manifest==before
assert int(l.agent.critic.step)==l.meta['critic_updates']
assert int(l.agent.actor_train_state.step)==l.meta['actor_updates']
result=dict(passed=True,robot_execution=False,updates_executed=0,status=l.save_status())
a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
