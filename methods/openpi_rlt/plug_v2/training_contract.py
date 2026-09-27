"""Isolated Torch contract experiment. No ROS/CAN; never publishes production.
Human raw-command BC, audited physical takeover Bellman links, separate causal
BC augmentation (no reward/TD), and explicit source-stratified sampling.
"""
import argparse,copy,fcntl,hashlib,json,random,time,os,datetime
from pathlib import Path
import numpy as np,torch
from torch import nn
from .learning import Actor,Twin,Sampler,conditioned,JOINTS,EXPECTED_STATS,atomic_torch
from .training_flow import session_phase
from .handover_credit import link_handover
from .learning_diagnostics import bc_metrics
ROOT=Path(__file__).resolve().parents[3];RUN=ROOT/'runs/plug_v2';BASE=ROOT.parent.parent/'data/rlt/plug_v2'

def validate_snapshot(d):
 n=len(d['z']);i=np.arange(n);j=d['next_row'];boot=d['bootstrap'];kind=d['edge_kind']
 if np.any(j<0)|np.any(j>=n):raise ValueError('bad next row')
 if np.any(d['done']&boot):raise ValueError('terminal bootstrap')
 normal=boot&(kind==0);takeover=boot&(kind==1)
 if np.any(boot&~np.isin(kind,[0,1])):raise ValueError('untyped boundary')
 for key in ('stream','segment','generation'):
  if np.any(d[key][j[normal]]!=d[key][i[normal]]):raise ValueError('untyped cross-boundary edge')
 if np.any(d['decision_tick'][j[normal]]!=d['decision_tick'][i[normal]]+d['duration'][i[normal]]):raise ValueError('bad normal clock')
 if np.any((d['stream'][i[takeover]]!=0)|(d['stream'][j[takeover]]!=1)|(d['generation'][j[takeover]]!=d['generation'][i[takeover]]+1)):raise ValueError('invalid handover kind')
 if np.any((d['handover_dt'][takeover]<=0)|(d['handover_dt'][takeover]>.05)):raise ValueError('handover gap')
 if np.any(d['bc_mask']&~d['future_mask']):raise ValueError('padding BC')
 if np.any(d['reward'][~d['done']]):raise ValueError('invented reward')
 if np.any(d['reward']*(np.arange(10)[None,:]>=np.ceil(d['duration'])[:,None])):raise ValueError('reward on unexecuted span')
 for key in ('z','context','ref','action','next_z','next_context','next_ref'):
  if not np.isfinite(d[key]).all():raise ValueError('nonfinite')
 for key in ('z','context','ref','future_mask'):
  np.testing.assert_array_equal(d['next_'+key],d[key][j])
 if np.any(boot&~d['eligible'][j]):raise ValueError('next observation excluded')

class ContractData:
 def __init__(self,output,handover=True):
  source=Sampler(RUN/'replay/v2');self.model_checkpoint=source.model_checkpoint;self.items=[];self.pools={s:{r:{} for r in ('expert','hil','success_policy','failure_policy')} for s in ('train','val')};self.audit=[]
  for path in sorted((RUN/'replay/v2').glob('*.npz')):
   with np.load(path,allow_pickle=False) as f:m=json.loads(str(f['metadata']));d={k:f[k].copy() for k in f.files if k!='metadata'}
   n=len(d['z']);d['eligible']=np.ones(n,bool);d['edge_kind']=np.zeros(n,np.int8);d['handover_dt']=np.zeros(n,np.float64)
   edges=[];rejected=[]
   if m['group']!='expert':
    source_file=RUN/'replay'/(m['source_uuid']+'.source.npz')
    with np.load(source_file,allow_pickle=False) as f:frames=json.loads(str(f['frames_json']))
    with np.load(BASE/'warmup/lerobot'/m['source_uuid']/'source_facts.npz',allow_pickle=False) as f:stamps=np.maximum(f['rollout/topic_timestamp/front_left'],f['rollout/topic_timestamp/front_right'])
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
   meta={**m,'replay_version':3,'format':'audited_takeover_TD_and_raw_human_BC','source_v2_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'typed_takeover_edges':edges,'observed_MC_returns_used_for_TD':False}
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
  keys=('z','context','ref','action','reward','duration','future_mask','bc_mask','bootstrap','next_z','next_context','next_ref','next_future_mask','hil')
  batch={k:torch.as_tensor(np.asarray([d[k][i] for d,i in chosen]),device=device,dtype=torch.bool if k in ('future_mask','bc_mask','bootstrap','next_future_mask','hil') else torch.float32) for k in keys}
  batch['expert']=torch.as_tensor([d['metadata']['group']=='expert' for d,i in chosen],device=device)
  batch['successful']=torch.as_tensor([d['metadata']['group']!='failure' for d,i in chosen],device=device)
  return batch

def smooth_acceleration(p,c,mask):
 prefix=c[:,14:98].reshape(-1,6,14)+c[:,None,:14];d=c[:,-1]>0;anchor=torch.where(d[:,None],prefix[:,-1],c[:,:14])
 previous=torch.where(d[:,None],prefix[:,-1]-prefix[:,-2],torch.zeros_like(anchor))[:,JOINTS]
 velocities=torch.cat([(p[:,0,JOINTS]-anchor[:,JOINTS])[:,None],torch.diff(p[...,JOINTS],dim=1)],dim=1)
 acceleration=torch.cat([(velocities[:,0]-previous)[:,None],torch.diff(velocities,dim=1)],dim=1)
 valid=torch.cat([mask[:,:1],mask[:,1:]&mask[:,:-1]],dim=1)[...,None]
 return (acceleration.square()*valid).sum()/torch.clamp(valid.sum()*6,min=1)/(.05**2)
def masked_mse(p,target,mask):return ((p[...,JOINTS]-target[...,JOINTS]).square()*mask[...,None]).sum()/torch.clamp(mask.sum()*6,min=1)/(.05**2)
def update(actor,critic,ta,tq,b,ao,qo,step,augmentation):
 with torch.no_grad():
  next_action=conditioned(ta(b['next_z'],b['next_context'],b['next_ref']),b['next_context']);value=torch.minimum(*tq(b['next_z'],b['next_context'],next_action,b['next_future_mask']));target=(b['reward']*(.99**torch.arange(10,device=b['z'].device))).sum(-1)+b['bootstrap']*(.99**b['duration'])*value
 q1,q2=critic(b['z'],b['context'],b['action'],b['future_mask']);qloss=((q1-target).square()+(q2-target).square()).mean();qo.zero_grad();qloss.backward();nn.utils.clip_grad_norm_(critic.parameters(),1.);qo.step();metrics={'critic_loss':float(qloss.detach())}
 if step%2==0:
  for p in critic.parameters():p.requires_grad_(False)
  raw=actor(b['z'],b['context'],b['ref'],dropout=.5);pred=conditioned(raw,b['context']);human=b['hil'][:,None]&b['bc_mask'];policy=(~b['hil'])[:,None]&b['bc_mask'];anchor=(~b['hil'])[:,None]&~b['bc_mask']&b['future_mask']
  human_bc=masked_mse(raw,b['action'],human);policy_bc=masked_mse(pred,b['action'],policy);reference_bc=masked_mse(pred,conditioned(b['ref'],b['context']),anchor);smooth=smooth_acceleration(pred,b['context'],b['future_mask']);q=critic.q1(b['z'],b['context'],pred,b['future_mask']).mean()
  extra=augmentation.sample(64,b['z'].device);extra_raw=actor(extra['z'],extra['c'],extra['ref'],dropout=.5);extra_bc=masked_mse(extra_raw,extra['target'],extra['mask'])
  loss=10*(human_bc+policy_bc+extra_bc)+reference_bc+smooth-.1*q
  ao.zero_grad();loss.backward();nn.utils.clip_grad_norm_(actor.parameters(),1.);ao.step()
  for p in critic.parameters():p.requires_grad_(True)
  with torch.no_grad():metrics.update(bc_metrics(raw.detach(),b['context'],b['ref'],b['action'],b['hil'],b['future_mask'],b['expert']))
  metrics.update(actor_loss=float(loss.detach()),human_raw_BC=float(human_bc.detach()),causal_d6_raw_BC=float(extra_bc.detach()),reference_anchor=float(reference_bc.detach()),command_acceleration=float(smooth.detach()),sample_human_intervention_ratio=float((b['hil']&~b['expert']).float().mean()),sample_expert_ratio=float(b['expert'].float().mean()))
 with torch.no_grad():
  for dst,src in ((ta,actor),(tq,critic)):
   for x,y in zip(dst.parameters(),src.parameters()):x.lerp_(y,.005)
 for k,v in metrics.items():
  if isinstance(v,(float,int)) and not np.isfinite(v):raise ValueError('nonfinite metric '+k)
 return metrics
class Augmentation:
 def __init__(self):
  groups={g:[] for g in ('expert','success')};self.train_uuids=set();self.val_uuids=set()
  folder=RUN/'learning/rtc-contract-validation-20260918-v3/rtc-bc'
  for p in sorted(folder.glob('*.npz')):
   with np.load(p,allow_pickle=False) as f:
    m=json.loads(str(f['metadata']));at=f['prefix_max_state_delta']<=.04+1e-6
    (self.train_uuids if m['split']=='train' else self.val_uuids).add(m['source_uuid'])
    if m['split']=='train' and at.any():groups[m['group']].append({k:f[k][at].copy() for k in ('z','c','ref','target','mask')})
  if self.train_uuids&self.val_uuids:raise ValueError('BC augmentation leakage')
  self.groups={g:{k:np.concatenate([d[k] for d in values]) for k in values[0]} for g,values in groups.items()}
 def sample(self,size,device):
  rows=[]
  for g in ('expert','success'):
   d=self.groups[g];at=np.random.randint(len(d['z']),size=size//2);rows.append({k:v[at] for k,v in d.items()})
  return {k:torch.as_tensor(np.concatenate([x[k] for x in rows]),device=device,dtype=torch.bool if k=='mask' else torch.float32) for k in rows[0]}

def train(args):
 out=args.output_dir.resolve();out.relative_to((RUN/'learning').resolve());out.mkdir(exist_ok=False);torch.set_num_threads(4);torch.set_float32_matmul_precision('high');random.seed(42);np.random.seed(42);torch.manual_seed(42)
 with (RUN/'learning/operation.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if session_phase() not in ('offline','disarmed','ready','paused','terminal_pending','waiting_scene','stopped','fault'):raise RuntimeError('pause/end Session first')
  prod=RUN/'learning/warmup/actor.pt';before=hashlib.sha256(prod.read_bytes()).hexdigest();(out/'operation.json').write_text(json.dumps({'phase':'preparing','pid':os.getpid(),'production_publish':False,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}))
  try:
   data=ContractData(out);(out/'data-audit.json').write_text(json.dumps(data.audit,indent=2));augmentation=Augmentation();device='cuda:0';actor=Actor().to(device);critic=Twin().to(device);ta=copy.deepcopy(actor);tq=copy.deepcopy(critic);ao=torch.optim.Adam(actor.parameters(),lr=1e-4,fused=True);qo=torch.optim.Adam(critic.parameters(),lr=1e-4,fused=True)
   # Audit sampler composition with the currently deployed actor before training.
   saved=torch.load(prod,map_location=device,weights_only=False);deployed=Actor().to(device);deployed.load_state_dict(saved['actor']);deployed.eval();sampling={}
   for strategy in ('outcome_balanced','stratified'):
    collected=[]
    for _ in range(100):
     b=data.sample(128,device,strategy=strategy)
     with torch.no_grad():m=bc_metrics(deployed(b['z'],b['context'],b['ref']),b['context'],b['ref'],b['action'],b['hil'],b['future_mask'],b['expert'])
     m.update(sample_human_intervention_ratio=float((b['hil']&~b['expert']).float().mean()),sample_expert_ratio=float(b['expert'].float().mean()));collected.append(m)
    sampling[strategy]={k:float(np.mean([m[k] for m in collected if m[k] is not None])) for k in collected[0] if k!='metric_units'}
   (out/'sampling-audit.json').write_text(json.dumps(sampling,indent=2));del deployed,saved
   random.seed(42);np.random.seed(42);torch.manual_seed(42)
   (out/'operation.json').write_text(json.dumps({'phase':'training','pid':os.getpid(),'max_steps':args.steps,'sampling':'expert30/HIL20/assisted-success-policy25/failure-policy25','production_publish':False}));started=time.perf_counter()
   for step in range(1,args.steps+1):
    metrics=update(actor,critic,ta,tq,data.sample(128,device),ao,qo,step,augmentation)
    if step%100==0:
     with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps({'step':step,**metrics})+'\n')
     print('CONTRACT_UPDATE',step,json.dumps(metrics),flush=True)
    if step in (100,500,2000,5000,args.steps):
     snapshot={'cohort':'plug_v2','status':'diagnostic_only','global_step':step,'rtc_context_dim':99,'stats_sha256':EXPECTED_STATS,'stage1_checkpoint':data.model_checkpoint,'replay_version':3,'actor':actor.state_dict(),'critic':critic.state_dict(),'target_actor':ta.state_dict(),'target_critic':tq.state_dict(),'actor_optimizer':ao.state_dict(),'critic_optimizer':qo.state_dict(),'not_production_published':True}
     atomic_torch(out/'checkpoints'/f'step_{step}.pt',snapshot)
   after=hashlib.sha256(prod.read_bytes()).hexdigest();assert before==after
   (out/'operation.json').write_text(json.dumps({'phase':'completed','critic_updates':args.steps,'actor_updates':args.steps//2,'seconds':time.perf_counter()-started,'production_sha256':after,'production_actor_unchanged':True,'production_publish':False,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2))
  except Exception as e:
   (out/'operation.json').write_text(json.dumps({'phase':'failed','error':str(e),'production_publish':False},indent=2));raise

def main():
 p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=5000);p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args()
 if not 100<=args.steps<=20000:raise ValueError('bounded experimental steps required')
 train(args)
if __name__=='__main__':main()
