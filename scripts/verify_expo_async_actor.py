"""No-ROS local Actor hot-install acceptance using isolated returned snapshots."""
from pathlib import Path
from types import SimpleNamespace
import argparse,json,time,uuid,sys,os
import numpy as np
p=argparse.ArgumentParser()
p.add_argument('--project',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True)
p.add_argument('--snapshots',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--initial',type=int,required=True);p.add_argument('--updated',type=int,required=True)
a=p.parse_args();assert a.initial<a.updated
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'methods/expo_ft/cobot_adapter'))
from native_online import source_path
source_path(a.project)
from online_actor import Actor
from online_protocol import atomic_json
args=SimpleNamespace(project=a.project,bundle=a.bundle,snapshot_root=a.snapshots,
    snapshot=a.snapshots/str(a.initial),model_id='expo-async-isolated-cobot-probe',rtc_delay=5,edit_scale=.01)
t=time.monotonic();actor=Actor(args);load=time.monotonic()-t
with np.load(a.bundle.parent/'common/dev_observation.npz',allow_pickle=False) as z:
    obs={k:z[k].copy() for k in ('state','left_wrist_image','right_wrist_image')}
def probe(version):
    identity=str(uuid.uuid4());actor.begin(dict(episode_uuid=identity,actor_version=version,rollout_rng=[0,42]))
    rows=[];prefix=np.zeros((0,7),np.float32)
    for n in range(8):
        t=time.monotonic();cached=actor.infer('/precache',{**obs,'episode_uuid':np.asarray(identity),'prefix':prefix})
        value=actor.infer('/select',{**obs,'episode_uuid':np.asarray(identity),'actor_version':np.asarray(version),
            'precached':np.asarray(cached['precached'],np.float32),'delay':np.asarray(0 if n==0 else 5)})
        actions=np.asarray(value['actions'],np.float32)
        assert actions.shape==(8,7) and np.isfinite(actions).all()
        prefix=actions[-5:];rows.append(time.monotonic()-t)
    return identity,rows
identity,before=probe(a.initial)
try:actor.install(a.snapshots/str(a.updated))
except ValueError as exc:assert 'during an episode' in str(exc)
else:raise AssertionError('Active-episode installation was accepted')
assert actor.status()['actor_version']==a.initial
actor.finish({'episode_uuid':identity})
t=time.monotonic();status=actor.install(a.snapshots/str(a.updated));install=time.monotonic()-t
assert status['ready'] and status['actor_version']==a.updated
identity,after=probe(a.updated);actor.finish({'episode_uuid':identity})
result=dict(passed=True,pid=os.getpid(),initial_actor=a.initial,updated_actor=a.updated,
    load_and_warmup_seconds=load,hot_install_seconds=install,initial_sample_seconds=before,
    updated_sample_seconds=after,process_restart=False,active_install_rejected=True,
    robot_execution=False,ros_publishers=False,production_data=False,status=actor.status())
atomic_json(a.output,result);print(json.dumps(result),flush=True)
