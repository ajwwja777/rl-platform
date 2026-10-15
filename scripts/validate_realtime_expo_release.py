#!/usr/bin/env python3
"""No-ROS CPU integration: real native export, seed budget and online restore."""
import argparse,dataclasses,json,pickle,sys,shutil,time
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('project',type=Path);ap.add_argument('output',type=Path);a=ap.parse_args()
root=Path(__file__).resolve().parents[1];up=a.project/'third_party/openpi-rlt'
sys.path[:0]=[str(root),str(up/'rlt_online_rl/src'),str(up/'packages/openpi-client/src')]
import jax,numpy as np
from rlt_online_rl.config import load_system_config_yaml
from rlt_online_rl.trainer import LearnerService
from rlt_online_rl.inference import ActorService,ActorRequest
from rlt_online_rl.replay import RLTTransition
from methods.openpi_rlt.cobot_adapter.replay_precision import install_action_precision_patch
from methods.openpi_rlt.realtime_expo.release import build,prepare_runtime
install_action_precision_patch(enabled=True)
o=a.output;source=o/'warmup-native-10k';step=json.loads((source/'selection.json').read_text())['selected_step'];bundle=o/'release-local-validation-portable'
build(source,o/'warmup-seed',bundle,bundle,o/'selected-stage1-4999',step,root/'configs/rlt/plug_v3_yyshadow/online_rl.yaml')
prepare_runtime(bundle,'online',bundle/'online/actor_snapshot/actor_snapshot.pkl',bundle/'checked.yaml')
system=load_system_config_yaml(bundle/'checked.yaml')
records=[]
with (o/'warmup-seed/train_expert_journal.pkl').open('rb') as f:
 while True:
  try:records.append(pickle.load(f))
  except EOFError:break
converted=[RLTTransition.from_mapping(r).to_numpy() for r in records]
class Source:
 def __init__(self):self.adds=len(records);self.rng=np.random.default_rng(9)
 def stats(self):return dict(size=self.adds,adds_total=self.adds)
 def sample_batch(self,n):
  indices=self.rng.integers(len(records),size=n)
  return {k:np.stack([converted[i][k] for i in indices]) for k in converted[0]}
s=Source();learner=LearnerService(system.rl,system.learner_service,s,rng=jax.random.PRNGKey(7),metrics_path=str(bundle/'verification-metrics.jsonl'))
assert int(learner.state.global_step)==step
assert learner.train_once() is None,'seed Replay must not earn duplicate update budget'
actor=ActorService(system.rl,system.actor_service)
deadline=time.monotonic()+60
while actor.actor_param_version<0 and time.monotonic()<deadline:time.sleep(.05)
assert actor.actor_param_version==step//2
expected=np.load(source/f'train-deployed-actions-{step}.npy');errors=[]
# Export path is single observation, evaluation used batch128; require FP32 consistency.
for i,r in enumerate(records[:32]):
 result=actor.infer(ActorRequest(z_rl=r['z_rl'],proprio=r['proprio'],ref_chunk=r['ref_chunk'],request_id=str(i),episode_id=r['episode_id'],step_id=r['step_id'],deterministic=True))
 errors.append(float(np.max(np.abs(result.refined_chunk-expected[i]))))
assert max(errors)<2e-5,max(errors)
# Artificial budget test uses only copied offline expert rows; never writes production Replay.
s.adds+=50
for _ in range(50):assert learner.train_once() is not None
assert learner.train_once() is None
assert int(learner.state.global_step)==step+50
assert (bundle/f'online/checkpoints/step_{step+50}.pkl').is_file()
assert (bundle/f'online/actor_snapshot/history/actor_v{(step+50)//2:06d}.pkl').is_file()
deadline=time.monotonic()+30
while actor.actor_param_version<(step+50)//2 and time.monotonic()<deadline:time.sleep(.05)
assert actor.actor_param_version==(step+50)//2
restored=LearnerService(system.rl,system.learner_service,s,rng=jax.random.PRNGKey(0),metrics_path=str(bundle/'resume-metrics.jsonl'))
assert restored.train_once() is None
for x,y in zip(jax.tree.leaves(learner.state),jax.tree.leaves(restored.state),strict=True):np.testing.assert_array_equal(x,y)
actor._stop_event.set();actor._poll_thread.join(timeout=2)
# Exported delivery bundle is fresh and separate from this modified smoke copy.
delivery=o/'release-bundle-portable'
build(source,o/'warmup-seed',delivery,Path('/home/agilex/jiaan/data/rlt/realtime_expo/release_20261015'),Path('/home/agilex/jiaan/data/rlt/realtime_expo/stage1_4999'),step,root/'configs/rlt/plug_v3_yyshadow/online_rl.yaml')
receipt=dict(status='passed',selected_step=step,export_max_abs_error=max(errors),initial_pending_budget=0,synthetic_new_transitions=50,updates=50,saved_step=step+50,actor_version=(step+50)//2,restored_state='all leaves exact',robot_commands=0,delivery_bundle=str(delivery),boundary='CPU-only synthetic update-budget integration; does not prove online success')
(o/'release-native-validation.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
