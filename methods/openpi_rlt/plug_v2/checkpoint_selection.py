# Explicit operator selection of an offline-validated fixed warmup candidate. No robot commands.
import argparse,datetime,fcntl,hashlib,json,os,shutil
from pathlib import Path
from methods.openpi_rlt.plug_v2.learning import ROOT,RUN,load_actor
from methods.openpi_rlt.plug_v2.training_flow import session_phase,atomic

def select(step):
 with (RUN/'learning/operation.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if session_phase() not in ('offline','disarmed','ready','stopped','fault'):
   raise RuntimeError('end Session first; loaded actor is fixed within an episode')
  latest=json.loads((RUN/'learning/checkpoint-comparison-latest.json').read_text())
  folder=Path(latest['directory']).resolve();folder.relative_to((RUN/'learning').resolve())
  report=json.loads((folder/'offline-comparison.json').read_text())
  if report['status']!='passed':raise RuntimeError('offline comparison not complete')
  candidate=folder/'checkpoints'/('step_%d.pt'%step)
  entry=next(x for x in report['candidates'] if x['step']==step)
  if hashlib.sha256(candidate.read_bytes()).hexdigest()!=entry['sha256']:raise RuntimeError('candidate hash mismatch')
  actor,version=load_actor(candidate)
  import torch
  saved=torch.load(candidate,map_location='cpu',weights_only=False)
  if version!=step:raise RuntimeError('actor version mismatch')
  current=RUN/'learning/warmup';stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
  backup=RUN/'learning/selection-history'/stamp;backup.mkdir(parents=True)
  for name in ['actor.pt','ready.json']:
   if (current/name).exists():shutil.copy2(current/name,backup/name)
  ready=json.loads((folder/'ready.json').read_text());ready.update({'actor_version':step,'evaluation':saved['evaluation'],'baseline':saved['baseline'],'selected_source':str(candidate),'selected_utc':stamp,'offline_comparison':str(folder/'offline-comparison.json')})
  temporary=current/('actor.pt.selection.%d.tmp'%os.getpid())
  with candidate.open('rb') as source,temporary.open('xb') as target:
   shutil.copyfileobj(source,target);target.flush();os.fsync(target.fileno())
  if hashlib.sha256(temporary.read_bytes()).hexdigest()!=entry['sha256']:raise RuntimeError('candidate changed during copy; prior actor retained')
  os.replace(temporary,current/'actor.pt');atomic(current/'ready.json',ready)
  atomic(RUN/'learning/operation.json',{'phase':'accepted','command':'warmup',**ready,'automatic_warmup':False})
  print('Fixed warmup selected: actor_version=%d; next run ./scripts/rlt_up.sh --frozen-actor --no-record --restart'%step)
  print('Previous actor retained:',backup)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('step',type=int,choices=[100,500,2000,5000,10000,20000]);args=parser.parse_args();select(args.step)
if __name__=='__main__':main()
