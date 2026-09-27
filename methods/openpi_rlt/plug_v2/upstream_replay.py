"""Offline synchronous RLT reproduction dataset; never a live RTC replay replacement.
Uses pinned upstream window construction. Full executed C=10 windows, terminal
window appended; no padding/fictional rewards/no stitching across missing frames.
"""
import os,sys,json,hashlib,time,argparse
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[3];RUN=ROOT/'runs/plug_v2';BASE=ROOT.parent.parent/'data/rlt/plug_v2'
sys.path.insert(0,str(ROOT/'code/openpi-rlt/rlt_online_rl/src'))
from rlt_online_rl.inference import EnvDriver,ReplaySegment
from rlt_online_rl.replay import TransitionSource
from .storage import releases_for_phase
from .rpc import ModelClient
from .replay import atomic_npz
from .training_flow import session_phase

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def nearest(times,targets):
 at=np.searchsorted(times,targets);a=np.maximum(at-1,0);b=np.minimum(at,len(times)-1)
 return np.where(abs(times[a]-targets)<=abs(times[b]-targets),a,b)
def layout(valid,times,terminal):
 segments=[];current=[]
 for i,ok in enumerate(valid):
  if current and (not ok or times[i]-times[current[-1]]<=.015 or times[i]-times[current[-1]]>.05):
   segments.append(ReplaySegment(current));current=[]
  if ok:current.append(i)
 if current:segments.append(ReplaySegment(current))
 driver=object.__new__(EnvDriver);driver._rl_config=SimpleNamespace(chunk_len=10);driver._env_config=SimpleNamespace(step_trace_stride=10)
 episode=SimpleNamespace(steps=[SimpleNamespace(done=i==terminal) for i in range(len(valid))])
 windows,added=driver._build_dense_replay_windows(episode,segments)
 result=[]
 for w in windows:
  indices=segments[w.segment_id].raw_indices;start=w.start_offset;part=indices[start:start+10];done=part[-1]==terminal
  if not done and start+10==len(indices):continue # No next observation across a dropped segment.
  nxt=part[-1] if done else indices[start+10]
  result.append((part,nxt,done))
 return result,added

def prepare_one(item,rpc,out):
 import pyarrow.parquet as pq,av
 root=Path(item['root']);index=item['index'];target=out/'episodes'/f"{item['uuid']}_{index:06d}.npz"
 if target.exists():
  with np.load(target) as f:return json.loads(str(f['metadata']))
 parquet=root/f'data/chunk-{index//1000:03d}/episode_{index:06d}.parquet'
 table=pq.read_table(parquet);states=np.stack(table['observation.state'].to_pylist()).astype(np.float32);actions=np.stack(table['action'].to_pylist()).astype(np.float32)
 files={str(parquet):sha(parquet)};n=len(states)
 if item['expert']:
  valid=np.isfinite(states).all(1)&np.isfinite(actions).all(1);times=np.arange(n)/30.;video_index=np.arange(n);source=np.full(n,int(TransitionSource.HUMAN),np.uint8)
  terminal=n-1 if item['terminal'] else -1
 else:
  trace=Path(item['trace']);files[str(trace)]=sha(trace)
  with np.load(trace) as f:frames=json.loads(str(f['frames_json']))
  with np.load(root/'training_mask.npz') as f:mask=f['valid'].copy()
  with np.load(root/'source_facts.npz') as f:
   stamps=np.maximum(f['rollout/topic_timestamp/front_left'],f['rollout/topic_timestamp/front_right'])
   coordinator=f['rollout/coordinator_command'].copy();coordinator_t=f['rollout/topic_timestamp/coordinator_right'].copy();coordinator_ok=f['rollout/valid_mask/coordinator_right'].copy()
  if np.any(np.diff(stamps)<0):raise ValueError('nonmonotonic source observation timestamps')
  times=np.array([f['ros_timestamp'] for f in frames]);video_index=nearest(stamps,times);states=np.array([f['state'] for f in frames],np.float32)
  actions=np.zeros_like(states);valid=np.zeros(len(frames),bool);source=np.full(len(frames),int(TransitionSource.RL),np.uint8)
  terminal=max((i for i,f in enumerate(frames) if f['valid_for_training']),default=-1)
  for i,f in enumerate(frames):
   j=video_index[i]
   if not f['valid_for_training'] or not mask[j] or abs(times[i]-stamps[j])>.05:continue
   if f['phase']=='rollout' and f['command'] is not None:actions[i]=f['command'];valid[i]=True
   elif f['phase']=='hil' and 'right' in f['mode'] and coordinator_ok[j] and abs(times[i]-coordinator_t[j])<=.05:
    actions[i]=coordinator[j];valid[i]=True;source[i]=int(TransitionSource.HUMAN)
  valid &= np.isfinite(states).all(1)&np.isfinite(actions).all(1)
  # Never move a success/failure label earlier if terminal observation was lost.
  files[str(root/'source_facts.npz')]=sha(root/'source_facts.npz');files[str(root/'training_mask.npz')]=sha(root/'training_mask.npz')
 windows,added=layout(valid,times,terminal)
 anchors=sorted(set([x[0][0] for x in windows]+[x[1] for x in windows]));needed={int(video_index[i]) for i in anchors};images={i:{} for i in needed}
 for camera,dest in [('cam_high','base_0_rgb'),('cam_left_wrist','left_wrist_0_rgb'),('cam_right_wrist','right_wrist_0_rgb')]:
  video=root/f'videos/chunk-{index//1000:03d}/observation.images.{camera}/episode_{index:06d}.mp4'
  with av.open(str(video)) as container:
   for i,f in enumerate(container.decode(container.streams.video[0])):
    if i in needed:images[i][dest]=f.to_ndarray(format='rgb24')
 features={}
 for i in anchors:
  if len(images[int(video_index[i])])!=3:raise ValueError('missing aligned images')
  r=rpc.infer({'state':states[i],'images':images[int(video_index[i])],'action_prefix':np.zeros((50,14),np.float32),'prefix_length':0})
  features[i]=(np.asarray(r['z_rl'],np.float32),np.asarray(r['ref_chunk'][:10,7:14],np.float32))
 rows=[]
 for indices,nxt,done in windows:
  i=indices[0];reward=np.zeros(10,np.float32)
  if done and item['success']:reward[-1]=1.
  rows.append(dict(z_rl=features[i][0],proprio=states[i,7:14],ref_chunk=features[i][1],action_chunk=actions[indices,7:14],rewards=reward,done=done,next_z_rl=features[nxt][0],next_proprio=states[nxt,7:14],next_ref_chunk=features[nxt][1],source_chunk=source[indices],frame=i,next_frame=nxt,span_sec=float(times[indices[-1]]-times[i]),position=i/max(1,len(times)-1)))
 meta={**item,'format':'pinned_upstream_synchronous_C10','rows':len(rows),'terminal_rows':sum(r['done'] for r in rows),'terminal_window_added':added,'reward_rows':sum(bool(r['rewards'].sum()) for r in rows),'source_sha256':files,'model_metadata':rpc.get_server_metadata(),'not_live_RTC_replay':True}
 arrays={k:np.asarray([r[k] for r in rows]) for k in rows[0]} if rows else {}
 atomic_npz(target,**arrays,metadata=np.array(json.dumps(meta)));print('UPSTREAM_DATA',item['uuid'],len(rows),meta['terminal_rows'],flush=True);return meta

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();out=args.output.resolve();out.relative_to(RUN.resolve());out.mkdir(exist_ok=True)
 if session_phase()!='stopped':raise RuntimeError('offline audit requires ended Session')
 registry=json.loads((RUN/'replay/split_registry.json').read_text());items=[]
 expert=BASE/'demonstrations/lerobot';selection=json.loads((expert/'selection_manifest.json').read_text())['episodes'];last={}
 for e in selection:last[e['source_uuid']]=max(last.get(e['source_uuid'],-1),e['episode_index'])
 for e in selection:items.append(dict(root=str(expert),index=e['episode_index'],uuid=e['source_uuid'],split=e['split'],expert=True,success=True,phase='demonstrations',terminal=e['episode_index']==last[e['source_uuid']]))
 for phase in ('warmup','online'):
  for release in releases_for_phase(phase):
   d=json.loads(release.read_text());m=d['metadata']
   if d.get('status')!='validated' or m['outcome'] not in ('success','failure'):continue
   u=d['source_uuid']
   if u not in registry:raise ValueError('unregistered split; no implicit assignment')
   items.append(dict(root=str(release.parent),index=0,uuid=u,split=registry[u],expert=False,success=m['outcome']=='success',phase=phase,trace=m['trace']))
 rpc=ModelClient();report=[]
 try:
  for item in items:
   if session_phase()!='stopped':raise RuntimeError('Session changed; stop offline feature extraction')
   report.append(prepare_one(item,rpc,out));(out/'progress.json').write_text(json.dumps({'phase':'preparing','pid':os.getpid(),'completed':len(report),'total':len(items),'production_publish':False}))
 finally:rpc.close()
 (out/'manifest.json').write_text(json.dumps(report,indent=2));(out/'progress.json').write_text(json.dumps({'phase':'completed','pid':os.getpid(),'episodes':len(report),'rows':sum(r['rows'] for r in report),'production_publish':False}))
if __name__=='__main__':main()
