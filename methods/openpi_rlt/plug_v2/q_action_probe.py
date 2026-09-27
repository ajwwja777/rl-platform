"""Read-only CPU Q action-direction and conditional visual-token probes.
Counterfactual actions/BC contexts are diagnostics, never factual transitions.
No ROS/CAN, training, production publication, or fabricated reward.
"""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np,torch
from .learning import Actor,Twin,JOINTS,conditioned,EXPECTED_STATS
from .learning_diagnostics import auc
from .training_contract import validate_snapshot
ROOT=Path(__file__).resolve().parents[3];RUN=ROOT/'runs/plug_v2'

def row_mse(pred,target,mask):
 return (((pred[...,JOINTS]-target[...,JOINTS])**2)*mask[...,None]).sum((1,2))/(mask.sum(1)*6).clamp(min=1)

def action_pair(ref,target,context):
 delta=(target[...,JOINTS]-ref[...,JOINTS]).clamp(-.05,.05)
 good=ref.clone();bad=ref.clone();good[...,JOINTS]+=delta;bad[...,JOINTS]-=delta
 return conditioned(good,context),conditioned(bad,context)

def direction_summary(qgood,qbad,egood,ebad):
 informative=(ebad-egood)>1e-8
 return {'rows':len(qgood),'informative_direction_rows':int(informative.sum()),
   'Q_prefers_toward_teacher_fraction':None if not informative.any() else float((qgood[informative]>qbad[informative]).float().mean()),
   'mean_Q_toward_minus_opposite':None if not informative.any() else float((qgood-qbad)[informative].mean()),
   'mean_MSE_opposite_minus_toward':None if not informative.any() else float((ebad-egood)[informative].mean())}

def collect(folder,causal=False):
 items=[]
 for p in sorted(Path(folder).glob('*.npz')):
  with np.load(p,allow_pickle=False) as f:
   meta=json.loads(str(f['metadata']))
   if meta['split']!='val':continue
   if causal:
    at=f['prefix_max_state_delta']<=.04+1e-6
    d={k:f[k][at].copy() for k in ('z','c','ref','target','mask')}
   else:
    d={k:f[k].copy() for k in f.files if k!='metadata'};validate_snapshot(d);at=d['eligible']
    d={k:v[at] for k,v in d.items()}
   if len(d['z']):items.append((meta,d))
 return items

def merge(items,group=None,policy=False,causal=False):
 values=[];ids=[];ticks=[]
 for m,d in items:
  if group and m['group']!=group:continue
  at=np.ones(len(d['z']),bool) if causal else d['stream']==(0 if policy else 1)
  if not at.any():continue
  keys=('z','c','ref','target','mask') if causal else ('z','context','ref','action','future_mask')
  v={k:d[k][at] for k in keys}
  if not causal:v={'z':v['z'],'c':v['context'],'ref':v['ref'],'target':v['action'],'mask':v['future_mask']}
  values.append(v);ids.extend([m['source_uuid']]*int(at.sum()))
  if not causal:ticks.extend(zip(d['generation'][at].tolist(),d['decision_tick'][at].tolist()))
 if not values:raise ValueError('empty scope')
 return {k:torch.as_tensor(np.concatenate([v[k] for v in values]),dtype=torch.bool if k=='mask' else torch.float32) for k in values[0]},ids,ticks

def qmin(critic,b,action,z=None):return torch.minimum(*critic(b['z'] if z is None else z,b['c'],action,b['mask']))

def visual_probe(actor,critic,items):
 b,ids,ticks=merge(items,policy=True);labels={m['source_uuid']:m['group'] for m,d in items}
 uuids=sorted(set(ids));indices=[]
 for uid in uuids:
  at=sorted([i for i,x in enumerate(ids) if x==uid and b['c'][i,-1]>.5],key=lambda i:ticks[i])[:3]
  if len(at)!=3:raise ValueError('matched three-window visual probe unavailable')
  indices.append(at)
 ix=torch.tensor(indices);take=ix.flatten();bb={k:v[take] for k,v in b.items()}
 pred=conditioned(actor(bb['z'],bb['c'],bb['ref']),bb['c']);base=qmin(critic,bb,pred)
 positive=np.array([labels[u]=='success' for u in uuids]);negative=np.array([labels[u]=='failure' for u in uuids])
 def score(q):
  ep=q.reshape(len(uuids),3).mean(1).numpy();return auc(ep[positive],ep[negative])
 shifts=[]
 for shift in range(1,len(uuids)):
  donor=torch.roll(ix,shift,0).flatten();z=b['z'][donor]
  fixed=qmin(critic,bb,pred,z)
  newpred=conditioned(actor(z,bb['c'],bb['ref']),bb['c']);both=qmin(critic,bb,newpred,z)
  shifts.append({'episode_shift':shift,'critic_only_AUC':score(fixed),'actor_and_critic_AUC':score(both),'mean_abs_critic_Q_change':float((fixed-base).abs().mean())})
 return {'episodes':len(uuids),'HIL_success':int(positive.sum()),'failure':int(negative.sum()),'original_AUC':score(base),'critic_only_shuffled_AUC_mean':float(np.mean([x['critic_only_AUC'] for x in shifts])),'actor_and_critic_shuffled_AUC_mean':float(np.mean([x['actor_and_critic_AUC'] for x in shifts])),'shifts':shifts,'interpretation':'Conditional RL-token permutation with reference/context fixed. Reference still contains VLA visual information; neither a complete vision ablation nor in-distribution causal proof.'}

def evaluate(output):
 output=Path(output).resolve();output.relative_to((RUN/'learning').resolve());output.mkdir(exist_ok=False)
 experiment=RUN/'learning/human-contract-r1-20260918';assert json.loads((experiment/'operation.json').read_text())['phase']=='completed'
 factual=collect(experiment/'replay');causal=collect(RUN/'learning/rtc-contract-validation-20260918-v3/rtc-bc',True)
 beta=json.loads((RUN/'learning/rtc-contract-validation-20260918-v3/bc-validation.json').read_text())['brake']['train_selected_beta']
 paths={'baseline20000':RUN/'learning/warmup/actor.pt',**{f'contract{n}':experiment/f'checkpoints/step_{n}.pt' for n in (2000,5000)}}
 production_before=hashlib.sha256(paths['baseline20000'].read_bytes()).hexdigest();torch.set_num_threads(4);report={'status':'completed','models':{},'brake_beta_train_selected':beta,'limitations':['No heldout autonomous success class.','Counterfactual Q order is not a physical outcome or an observed action value.','Causal BC contexts are not factual RL transitions; reachable-direction checks do not make them factual.','Same tiny validation set reused for checkpoint diagnosis; no independent confirmation.']}
 for name,path in paths.items():
  saved=torch.load(path,map_location='cpu',weights_only=False)
  if saved.get('stats_sha256')!=EXPECTED_STATS or saved.get('rtc_context_dim')!=99 or saved.get('cohort')!='plug_v2':raise ValueError('wrong model contract')
  if saved.get('stage1_checkpoint')!=json.loads((ROOT/'deployments/plug_v2/manifest.json').read_text())['checkpoint']:raise ValueError('wrong frozen checkpoint')
  a=Actor();a.load_state_dict(saved['actor']);a.eval();q=Twin();q.load_state_dict(saved['critic']);q.eval();result={'model_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'action_direction':{},'policy_action_order':{}}
  with torch.no_grad():
   for group in ('expert','success'):
    for kind,items in (('factual_teacher_d0',factual),('supported_causal_d6_BC',causal)):
     b,ids,_=merge(items,group=group,causal=kind.startswith('supported'))
     good,bad=action_pair(b['ref'],b['target'],b['c']);eg=row_mse(good,b['target'],b['mask']);eb=row_mse(bad,b['target'],b['mask'])
     scope=direction_summary(qmin(q,b,good),qmin(q,b,bad),eg,eb);per=[]
     for uid in sorted(set(ids)):
      at=torch.tensor([x==uid for x in ids]);v=direction_summary(qmin(q,b,good)[at],qmin(q,b,bad)[at],eg[at],eb[at]);v['uuid']=uid;per.append(v)
     valid=[x['Q_prefers_toward_teacher_fraction'] for x in per if x['Q_prefers_toward_teacher_fraction'] is not None]
     scope.update(original_uuids=len(per),episode_macro_direction_accuracy=None if not valid else float(np.mean(valid)),episodes=per)
     result['action_direction'][f'{kind}/{group}']=scope
   for group in ('success','failure'):
    b,ids,_=merge(factual,group=group,policy=True);ref=conditioned(b['ref'],b['c']);pred=conditioned(a(b['z'],b['c'],b['ref']),b['c'])
    anchor=b['c'][:,:14]+b['c'][:,84:98];hold=conditioned(anchor[:,None,:].expand_as(b['ref']),b['c']);raw=b['ref'].clone();raw[...,JOINTS]+=(beta*(b['c'][:,None,JOINTS]-raw[...,JOINTS])).clamp(-.05,.05);brake=conditioned(raw,b['c'])
    qq={k:qmin(q,b,v) for k,v in {'reference':ref,'actor':pred,'hold_prefix_anchor':hold,'brake':brake}.items()};refq=qq['reference']
    result['policy_action_order'][group]={'rows':len(ref),'original_uuids':len(set(ids)),'comparisons':{k:{'mean_Q':float(v.mean()),'Q_over_reference_fraction':float((v>refq+1e-7).float().mean()),'mean_Q_minus_reference':float((v-refq).mean())} for k,v in qq.items()}}
   result['conditional_visual_token_probe']=visual_probe(a,q,factual)
  report['models'][name]=result
  print(name,json.dumps({'direction':{k:{kk:vv for kk,vv in v.items() if kk!='episodes'} for k,v in result['action_direction'].items()},'policy':result['policy_action_order'],'visual':{k:v for k,v in result['conditional_visual_token_probe'].items() if k not in ('shifts','interpretation')}}),flush=True)
 report['production_actor_unchanged']=production_before==hashlib.sha256(paths['baseline20000'].read_bytes()).hexdigest();report['published']=False
 (output/'report.json').write_text(json.dumps(report,indent=2)+'\n');return report

def main():
 p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True,type=Path);a=p.parse_args();evaluate(a.output_dir)
if __name__=='__main__':main()
