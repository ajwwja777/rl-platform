"""Exercise a real isolated remote Learner over HTTP; never connect to a robot."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid
from urllib.request import Request, build_opener, ProxyHandler
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--url',required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--observation',type=Path,required=True)
p.add_argument('--stage',choices=('partial','resume','restore'),required=True)
p.add_argument('--model-id',required=True)
a=p.parse_args()
if 'verification' not in a.output.parts or not a.model_id.startswith('expo-hpc4-verification-'):
    raise ValueError('Isolated verification identity and output required')
a.output.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'methods/expo_ft/cobot_adapter'))
from episode_journal import EpisodeJournal
from remote_journal import files
from online_protocol import atomic_json
opener=build_opener(ProxyHandler({}))
def rpc(route,data=None,payload=None,headers=None):
    if data is not None:payload=json.dumps(data).encode()
    req=Request(a.url+route,data=payload,headers=headers or {'Content-Type':'application/json'})
    with opener.open(req,timeout=30) as response:return json.load(response)
def status():
    value=rpc('/status')
    assert value['deployment_id']==a.model_id and value['learner_devices']==4
    assert value['training_enabled'] and value['episodes_per_update']==5
    assert value['edit_scale']==.01 and value['rtc_delay']==5
    assert value['phase']!='fault',value.get('error')
    return value
def await_ready():
    deadline=time.monotonic()+3600
    while time.monotonic()<deadline:
        value=status()
        if value['phase']=='ready':return value
        time.sleep(1)
    raise TimeoutError('Isolated Learner did not become ready')
initial=await_ready()
baseline_path=a.output/'baseline.json'
if a.stage=='partial':
    if baseline_path.exists():raise ValueError('Fresh verification required')
    assert initial['batch_episodes_collected']==0 and initial['warmup_remaining']==0
    atomic_json(baseline_path,dict(status=initial,manifest=rpc('/actor/manifest')))
baseline=json.loads(baseline_path.read_text());b=baseline['status']
with np.load(a.observation,allow_pickle=False) as z:
    obs={k:z[k].copy() for k in ('state','left_wrist_image','right_wrist_image')}
journal=EpisodeJournal(a.output/'client-journal')
def episode(outcome):
    start=time.monotonic();identity=str(uuid.uuid4());current=status()
    reply=rpc('/episode/start',dict(episode_uuid=identity))
    journal.begin(identity,current['actor_version'])
    for i in range(16):
        if i==7:
            journal.pause(True)
            journal.sample(obs,obs['state'].copy(),100+i*.1,False)
            journal.pause(False)
        journal.sample(obs,obs['state'].copy(),100+i*.1,i<7)
    journal.finish(identity,outcome)
    path=journal.root/identity/'finished.json';info=json.loads(path.read_text())
    assert info['count']==14,info
    info['rollout_rng']=reply['rollout_rng'];atomic_json(path,info)
    paths=files(journal.root,identity)
    for path in paths:
        body=path.read_bytes();digest=hashlib.sha256(body).hexdigest()
        route='/journal/'+identity+'/'+path.name
        result=rpc(route,payload=body,headers={'Content-Type':'application/octet-stream','X-Content-SHA256':digest})
        assert result.get('sha256')==digest,result
        # Exercise retry/idempotency on the same immutable upload.
        assert rpc(route,payload=body,headers={'Content-Type':'application/octet-stream','X-Content-SHA256':digest})['duplicate']
    rpc('/episode/end',dict(episode_uuid=identity));value=await_ready()
    assert rpc('/episode/receipt',dict(episode_uuid=identity))['accepted']
    again=rpc('/episode/end',dict(episode_uuid=identity))
    for key in ('actor_version','critic_updates','completed_episodes','online_steps'):
        assert again[key]==value[key]
    row=dict(episode=identity,outcome=outcome,seconds=time.monotonic()-start,status=value)
    with (a.output/'episodes.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)
    return value
if a.stage=='partial':
    for i in range(3):
        value=episode('failure' if i%2==0 else 'success')
        assert value['actor_version']==b['actor_version']
        assert value['batch_episodes_collected']==i+1
    assert rpc('/actor/manifest')==baseline['manifest']
elif a.stage=='resume':
    assert initial['batch_episodes_collected']==3 and initial['actor_version']==b['actor_version']
    assert rpc('/actor/manifest')==baseline['manifest']
    value=episode('failure')
    assert value['actor_version']==b['actor_version'] and value['batch_episodes_collected']==4
    value=episode('success')
    assert value['actor_version']==b['actor_version']+2
    assert value['critic_updates']==b['critic_updates']+40
    assert value['actor_updates']==b['actor_updates']+2
    assert value['completed_episodes']==b['completed_episodes']+5
    assert value['online_steps']==b['online_steps']+70
    assert value['batch_episodes_collected']==0
    manifest=rpc('/actor/manifest');atomic_json(a.output/'updated-actor.json',manifest)
    assert manifest['actor_version']==b['actor_version']+2
else:
    assert initial['actor_version']==b['actor_version']+2
    assert initial['critic_updates']==b['critic_updates']+40
    assert initial['actor_updates']==b['actor_updates']+2
    assert initial['completed_episodes']==b['completed_episodes']+5
    assert initial['batch_episodes_collected']==0
    assert rpc('/actor/manifest')==json.loads((a.output/'updated-actor.json').read_text())
atomic_json(a.output/(a.stage+'.json'),dict(passed=True,robot_execution=False,synthetic_episodes=True,status=status()))
print('PASSED',a.stage,flush=True)
