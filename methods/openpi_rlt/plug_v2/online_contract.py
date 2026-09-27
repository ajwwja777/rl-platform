# Audited online replay v4: physical takeover links, phase-independent roots.
import json,hashlib,random
from pathlib import Path
import numpy as np,torch
from .learning import Sampler
from .training_contract import validate_snapshot
from .handover_credit import link_handover
from .storage import releases_for_phase
ROOT=Path(__file__).resolve().parents[3];RUN=ROOT/'runs/plug_v2';BASE=ROOT.parent.parent/'data/rlt/plug_v2'
class ContractData:
 def __init__(self,output,handover=True):
  source=Sampler(RUN/'replay/v2');self.model_checkpoint=source.model_checkpoint;self.items=[];self.pools={s:{r:{} for r in ('expert','hil','success_policy','failure_policy')} for s in ('train','val')};self.audit=[]
  roots={}
  for phase in ('warmup','online'):
   for release in releases_for_phase(phase):
    info=json.loads(release.read_text())
    if info.get('status')=='validated':roots[info['source_uuid']]=release.parent
  for path in sorted((RUN/'replay/v2').glob('*.npz')):
   with np.load(path,allow_pickle=False) as f:m=json.loads(str(f['metadata']));d={k:f[k].copy() for k in f.files if k!='metadata'}
   n=len(d['z']);d['eligible']=np.ones(n,bool);d['edge_kind']=np.zeros(n,np.int8);d['handover_dt']=np.zeros(n,np.float64)
   edges=[];rejected=[]
   if m['group']!='expert':
    source_file=RUN/'replay'/(m['source_uuid']+'.source.npz')
    with np.load(source_file,allow_pickle=False) as f:frames=json.loads(str(f['frames_json']))
    with np.load(roots[m['source_uuid']]/'source_facts.npz',allow_pickle=False) as f:stamps=np.maximum(f['rollout/topic_timestamp/front_left'],f['rollout/topic_timestamp/front_right'])
    linked,edges,rejected=link_handover(d,frames,stamps)
    if not handover:
     for e in edges:linked['bootstrap'][e['row']]=False;linked['next_row'][e['row']]=e['row']
     edges=[]
    d['bootstrap']=linked['bootstrap'];d['next_row']=linked['next_row'];d['eligible']=linked['continuous_window']
    # Also validate the actually executed post-prefix action interval.
    for i in np.where(d['stream']==0)[0]:
     first=int(d['frame'][i])+int(round(d['context'][i,-1]*6));count=int(d['future_mask'][i].sum())
     if count>1:
      ts=np.asarray([f['ros_timestamp'] for f in frames[first:first+count]]);dt=np.diff(ts)
      d['eligible'][i]&=bool(len(ts)==count and np.all((dt>0)&(dt<=.05)))
    for e in edges:
     i=e['row'];d['edge_kind'][i]=1;d['handover_dt'][i]=e['control_dt_sec'];d['duration']=d['duration'].astype(np.float64);d['duration'][i]=e['discount_span_ticks']
   for i in np.where(d['bootstrap'])[0]:
    if not(d['eligible'][i] and d['eligible'][d['next_row'][i]]):d['bootstrap'][i]=False;d['next_row'][i]=i
   d['truncated']=~d['done']&~d['bootstrap']
   for k in ('z','context','ref','future_mask'):d['next_'+k]=d[k][d['next_row']]
   validate_snapshot(d)
   d['next_action']=d['action'][d['next_row']].copy()
   d['next_hil']=d['hil'][d['next_row']].copy()
   meta={**m,'replay_version':4,'format':'audited_takeover_TD_and_raw_human_BC','source_v2_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'typed_takeover_edges':edges,'observed_MC_returns_used_for_TD':False}
   from .replay import atomic_npz
   atomic_npz(output/'replay'/path.name,**d,metadata=np.array(json.dumps(meta)))
   d['metadata']=meta;self.items.append(d)
   for i in np.where(d['eligible'])[0]:
    role='expert' if m['group']=='expert' else 'hil' if d['hil'][i] else 'success_policy' if m['group']=='success' else 'failure_policy'
    self.pools[m['split']][role].setdefault(m['source_uuid'],[]).append((d,int(i)))
   self.audit.append({'uuid':m['source_uuid'],'split':m['split'],'excluded_rows':int((~d['eligible']).sum()),'typed_handover_edges':int((d['bootstrap']&(d['edge_kind']==1)).sum()),'source_signature':meta['source_v2_sha256']})
  for split in self.pools:
   if any(not v for v in self.pools[split].values()):raise ValueError('all four source strata required')
  uuids={s:set().union(*[set(v) for v in roles.values()]) for s,roles in self.pools.items()}
  if uuids['train']&uuids['val']:raise ValueError('original UUID leakage')
 def sample(self,size,device,split='train',strategy='stratified'):
  chosen=[]
  if strategy=='stratified':
   counts=[round(.3*size),round(.2*size),round(.25*size)];counts.append(size-sum(counts))
   roles=[role for role,count in zip(self.pools[split],counts) for _ in range(count)];random.shuffle(roles)
   for role in roles:
    pool=self.pools[split][role];uuid=random.choice(list(pool));d,i=random.choice(pool[uuid]);chosen.append((d,i))
  elif strategy=='outcome_balanced':
   groups=random.choices(['expert','success','failure'],weights=[.3,.4,.3],k=size)
   for group in groups:
    episodes=[d for d in self.items if d['metadata']['split']==split and d['metadata']['group']==group and d['eligible'].any()];d=random.choice(episodes);i=int(random.choice(np.where(d['eligible'])[0]));chosen.append((d,i))
  else:raise ValueError('unknown strategy')
  keys=('z','context','ref','action','reward','duration','future_mask','bc_mask','bootstrap','next_z','next_context','next_ref','next_future_mask','hil','next_action','next_hil')
  batch={k:torch.as_tensor(np.asarray([d[k][i] for d,i in chosen]),device=device,dtype=torch.bool if k in ('future_mask','bc_mask','bootstrap','next_future_mask','hil','next_hil') else torch.float32) for k in keys}
  batch['expert']=torch.as_tensor([d['metadata']['group']=='expert' for d,i in chosen],device=device)
  batch['successful']=torch.as_tensor([d['metadata']['group']!='failure' for d,i in chosen],device=device)
  return batch
