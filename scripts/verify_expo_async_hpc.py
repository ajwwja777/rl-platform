"""Isolated 4-GPU async HTTP/update/restart acceptance; no ROS or production writes."""
from pathlib import Path
import subprocess,json,time,shutil,os,sys,hashlib,uuid
from urllib.request import build_opener,ProxyHandler,Request
import numpy as np
H=Path(sys.argv[1]);cfg=json.loads((H/'qualification-profile.json').read_text());root=Path(cfg['run_dir'])
base=H/'baseline-state';original=(base/'committed.json').read_bytes();initial=json.loads(original)
assert not root.exists();root.mkdir()
for name in ('committed.json','contract.json'):shutil.copy2(base/name,root/name)
for name in ('episodes','checkpoints'):shutil.copytree(base/name,root/name,copy_function=os.link)
op=build_opener(ProxyHandler({}));url='http://127.0.0.1:%d'%cfg['port'];child=None
sys.path.insert(0,str(H/'source/methods/expo_ft/cobot_adapter'))
from online_protocol import atomic_json
from async_state import rollout_rng
from remote_journal import files

def event(phase,**kw):
 value=dict(phase=phase,time=time.time(),**kw);atomic_json(H/'qualification-progress.json',value);print(json.dumps(value),flush=True)
def get():
 with op.open(url+'/status',timeout=5) as r:return json.load(r)
def post(route,data):
 request=Request(url+route,data=json.dumps(data).encode())
 with op.open(request,timeout=10) as r:return json.load(r)
def wait_for(predicate,timeout=1800):
 deadline=time.monotonic()+timeout
 while time.monotonic()<deadline:
  if child.poll() is not None:raise RuntimeError('Learner exited '+str(child.returncode))
  try:
   v=get()
   if v.get('phase')=='fault':raise RuntimeError(v.get('error'))
   if predicate(v):return v
  except (OSError,ValueError):pass
  time.sleep(.5)
 raise TimeoutError('Qualification state timeout')
def start(name):
 global child
 log=(H/(name+'.log')).open('w')
 child=subprocess.Popen([cfg['python'],str(H/'source/methods/expo_ft/cobot_adapter/hpc_learner.py'),'--config',str(H/'qualification-profile.json')],stdout=log,stderr=subprocess.STDOUT)
 atomic_json(H/'qualification-child.json',{'pid':child.pid,'job_id':cfg['job_id']})
 return wait_for(lambda v:v.get('ready') and v.get('phase')=='ready')
def stop():
 global child
 if child is not None and child.poll() is None:
  child.terminate()
  try:child.wait(timeout=30)
  except subprocess.TimeoutExpired:child.kill();child.wait()
 child=None

def sample(outcome='success'):
 identity=str(uuid.uuid4());local=H/'test-upload'/identity;local.mkdir(parents=True)
 source=base/'episodes'/initial['replay'][0]
 for i in range(30):shutil.copy2(source/('%06d.npz'%i),local/('%06d.npz'%i))
 with np.load(source/'000030.npz',allow_pickle=False) as z:
  np.savez(local/'terminal.npz',**{k:z[k] for k in ('state','left_wrist_image','right_wrist_image')})
 meta=dict(episode_uuid=identity,actor_version=initial['updates'],count=0,started_at=time.time())
 atomic_json(local/'started.json',meta)
 meta.update(count=30,outcome=outcome,finished_at=time.time(),training_eligible=True,rollout_rng=rollout_rng('isolated-async',identity))
 atomic_json(local/'finished.json',meta)
 return identity

def upload(identity):
 t=time.monotonic();size=0
 for p in files(H/'test-upload',identity):
  payload=p.read_bytes();size+=len(payload)
  req=Request(url+'/async/journal/'+identity+'/'+p.name,data=payload,headers={'X-Content-SHA256':hashlib.sha256(payload).hexdigest()})
  with op.open(req,timeout=15) as r:assert json.load(r)['sha256']==hashlib.sha256(payload).hexdigest()
 receipt=post('/async/episode/end',{'episode_uuid':identity});assert receipt['received']
 return dict(identity=identity,seconds=time.monotonic()-t,bytes=size,receipt=receipt)

try:
 event('starting');t=time.monotonic();status=start('qualification-first');load=time.monotonic()-t
 assert status['actor_version']==initial['updates'] and status['learner_devices']==4
 uploads=[]
 for i in range(5):uploads.append(upload(sample('success' if i%2 else 'failure')))
 learning=wait_for(lambda v:v.get('pending_update_calls',0)>0)
 concurrent=upload(sample('success'));uploads.append(concurrent)
 after=get();assert after['learner_phase']=='updating' and not concurrent['receipt']['processed']
 event('admitted_while_training',learner=after['learner_step'],upload=concurrent)
 result=wait_for(lambda v:v.get('phase')=='ready' and v.get('completed_episodes')==initial['completed_episodes']+6)
 # Old behavior policy remains admissible even after a new policy is published.
 old_policy=upload(sample('failure'));uploads.append(old_policy)
 result=wait_for(lambda v:v.get('phase')=='ready' and v.get('completed_episodes')==initial['completed_episodes']+7)
 before=json.loads((root/'committed.json').read_text())
 assert before['updates']>initial['updates'] and before['episodes_since_update']==2
 assert before['credit']==initial['credit']+210-(before['updates']-initial['updates'])*30
 assert before['critic_updates']-initial['critic_updates']==20*(before['updates']-initial['updates'])
 assert post('/async/episode/end',{'episode_uuid':uploads[0]['identity']})['processed']
 assert json.loads((root/'committed.json').read_text())==before
 stop();event('restarting',actor=before['updates']);t=time.monotonic();restored=start('qualification-restore');restore_seconds=time.monotonic()-t
 assert json.loads((root/'committed.json').read_text())==before
 assert restored['actor_version']==before['updates'] and restored['batch_episodes_collected']==2
 assert (base/'committed.json').read_bytes()==original
 result=dict(passed=True,job_id=cfg['job_id'],initial_actor=initial['updates'],actor_version=before['updates'],critic_updates=before['critic_updates'],load_seconds=load,restore_seconds=restore_seconds,uploads=uploads,concurrent=concurrent,timings=result.get('timings'),restored=restored,robot_execution=False,production_data=False)
 atomic_json(H/'qualification-result.json',result);event('passed',actor=before['updates'])
 atomic_json(H/'return-ready.json',dict(actor_version=before['updates'],port=cfg['port'],pid=child.pid))
 deadline=time.monotonic()+1800
 while not (H/'return-verified.json').exists():
  if child.poll() is not None:raise RuntimeError('Qualified learner exited before return test')
  if time.monotonic()>deadline:raise TimeoutError('Returned Actor acceptance timeout')
  time.sleep(2)
finally:stop()
