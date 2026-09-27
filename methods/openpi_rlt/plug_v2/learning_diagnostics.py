"""Read-only episode Q and human-BC diagnostics for the Torch plug_v2 contract.
Upstream JAX replay_journal/snapshot artifacts are intentionally not converted.
"""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np,torch
from .learning import Actor,Twin,conditioned,JOINTS,load_actor
from .replay_contract import validate_arrays
ROOT=Path(__file__).resolve().parents[3];RUN=ROOT/'runs/plug_v2'
def auc(positive,negative):
 if len(positive)==0 or len(negative)==0:return None
 x=np.asarray(positive)[:,None];y=np.asarray(negative)[None,:]
 return float(np.mean((x>y)+.5*(x==y)))
def bc_metrics(raw,c,reference,action,human,mask,expert=None):
 """rad^2 metrics, valid positions/right six joints; padding excluded."""
 predicted=conditioned(raw,c);ref=conditioned(reference,c);teacher=conditioned(action,c)
 masks={'human':human[:,None]&mask,'ref':~human[:,None]&mask}
 if expert is not None:masks.update(expert=expert[:,None]&mask,hil=(human&~expert)[:,None]&mask)
 def mse(x,y,selected):
  n=selected.sum()*6
  return None if not n else float(((x[...,JOINTS]-y[...,JOINTS]).square()*selected[...,None]).sum()/n)
 out={'human_mask_ratio':float(masks['human'].sum()/torch.clamp(mask.sum(),min=1)),
      'bc_ref_penalty':mse(predicted,ref,masks['ref']),
      'bc_human_penalty':mse(predicted,action,masks['human']),
      'bc_human_raw_penalty':mse(raw,action,masks['human']),
      'bc_human_filtered_target_penalty':mse(predicted,teacher,masks['human']),
      'metric_units':'rad^2; different normalization/dimensions from upstream JAX'}
 if expert is not None:
  out.update(hil_mask_ratio=float(masks['hil'].sum()/torch.clamp(mask.sum(),min=1)),
             expert_mask_ratio=float(masks['expert'].sum()/torch.clamp(mask.sum(),min=1)),
             bc_hil_penalty=mse(predicted,action,masks['hil']),
             bc_expert_penalty=mse(predicted,action,masks['expert']))
 return out

def evaluate(replay,model,output,split='val',allow_experimental=False):
 output=Path(output).resolve();output.relative_to((RUN/'learning').resolve());output.mkdir(parents=True,exist_ok=False)
 saved=torch.load(model,map_location='cpu',weights_only=False)
 if allow_experimental and saved.get('status')=='diagnostic_only':
  from .learning import EXPECTED_STATS
  if saved.get('cohort')!='plug_v2' or saved.get('rtc_context_dim')!=99 or saved.get('stats_sha256')!=EXPECTED_STATS:raise ValueError('invalid experimental model')
  if saved.get('stage1_checkpoint')!=json.loads((ROOT/'deployments/plug_v2/manifest.json').read_text())['checkpoint']:raise ValueError('wrong frozen checkpoint')
  actor=Actor();actor.load_state_dict(saved['actor']);actor.eval();version=int(saved['global_step'])
 else:actor,version=load_actor(model)
 if saved.get('required_inference_wrapper')=='rtc-corrective-r1':
  from .corrective_policy import CorrectiveActor
  actor=CorrectiveActor(saved['inference_config']);actor.load_state_dict(saved['actor']);actor.eval()
 elif saved.get('model_kind') not in (None,'legacy'):
  raise ValueError('unsupported inference policy: do not silently evaluate as legacy Actor')
 critic=Twin();critic.load_state_dict(saved['critic']);critic.eval();torch.set_num_threads(4)
 episodes=[];rows=[];teacher_groups={g:[] for g in ('expert','success')};uuid_seen={}
 for path in sorted(Path(replay).glob('*.npz')):
  with np.load(path,allow_pickle=False) as f:
   m=json.loads(str(f['metadata']));d={k:f[k].copy() for k in f.files if k!='metadata'}
  if m.get('replay_version')==3:
   from .training_contract import validate_snapshot
   validate_snapshot(d)
   d={k:v[d['eligible']] for k,v in d.items()}
  else:validate_arrays(d)
  if m['split']!=split:continue
  if m['source_uuid'] in uuid_seen and uuid_seen[m['source_uuid']]!=m['split']:raise ValueError('UUID split leakage')
  uuid_seen[m['source_uuid']]=m['split'];b={k:torch.as_tensor(d[k],dtype=torch.bool if k in ('future_mask','hil','bc_mask') else torch.float32) for k in ('z','context','ref','action','future_mask','hil','bc_mask')}
  with torch.no_grad():
   raw=actor(b['z'],b['context'],b['ref']);pred=conditioned(raw,b['context']);ref=conditioned(b['ref'],b['context']);qd=torch.minimum(*critic(b['z'],b['context'],b['action'],b['future_mask']));qr=torch.minimum(*critic(b['z'],b['context'],ref,b['future_mask']));qa=torch.minimum(*critic(b['z'],b['context'],pred,b['future_mask']))
  for i in range(len(d['z'])):
   rows.append({'source_uuid':m['source_uuid'],'group':m['group'],'split':m['split'],'stream':int(d['stream'][i]),'generation':int(d['generation'][i]),'segment':int(d['segment'][i]),'decision_tick':int(d['decision_tick'][i]),'source_frame':int(d['frame'][i]),'delay':int(round(float(d['context'][i,-1])*6)),'q_data':float(qd[i]),'q_ref':float(qr[i]),'q_actor_mean':float(qa[i]),'done':bool(d['done'][i]),'reward':float(d['reward'][i].sum()),'bootstrap':bool(d['bootstrap'][i]),'human':bool(d['hil'][i])})
  if m['group'] in teacher_groups and b['hil'].any():teacher_groups[m['group']].append({k:v[b['hil']] for k,v in b.items()}|{'raw':raw[b['hil']]})
 # Aggregate by original UUID (expert fragments never treated as new episodes).
 for uuid in sorted(uuid_seen):
  data=[r for r in rows if r['source_uuid']==uuid];g=data[0]['group'];policy=sorted([r for r in data if r['stream']==0],key=lambda r:(r['generation'],r['decision_tick']))
  scopes={'full_recording':data,'policy_only':policy,'policy_first_three_windows':policy[:3]}
  ep={'source_uuid':uuid,'group':g,'outcome':'success' if g!='failure' else 'failure','success_semantics':'expert_demonstration' if g=='expert' else 'HIL_assisted_episode' if g=='success' else 'autonomous_failure_episode','scopes':{}}
  for scope,subset in scopes.items():
   if subset:ep['scopes'][scope]={'rows':len(subset),**{key:float(np.mean([r[key] for r in subset])) for key in ('q_data','q_ref','q_actor_mean')}}
  episodes.append(ep)
 separation={}
 for scope in ('full_recording','policy_only','policy_first_three_windows'):
  separation[scope]={}
  for key in ('q_data','q_ref','q_actor_mean'):
   success=[e['scopes'][scope][key] for e in episodes if e['group']=='success' and scope in e['scopes']];failure=[e['scopes'][scope][key] for e in episodes if e['group']=='failure' and scope in e['scopes']]
   separation[scope][key]={'HIL_success_episodes':len(success),'failure_episodes':len(failure),'AUROC_HIL_success_vs_failure':auc(success,failure),'success_mean':None if not success else float(np.mean(success)),'failure_mean':None if not failure else float(np.mean(failure))}
 metrics={}
 for g,values in teacher_groups.items():
  if not values:continue
  b={k:torch.cat([d[k] for d in values]) for k in values[0]};metrics[g]=bc_metrics(b['raw'],b['context'],b['ref'],b['action'],b['hil'],b['future_mask'])
 with (output/'episode_q_rows.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 report={'status':'completed','actor_mode':'mean','actor_version':version,'model_path':str(Path(model).resolve()),'model_sha256':hashlib.sha256(Path(model).read_bytes()).hexdigest(),'split':split,'episodes':episodes,'separation':separation,'human_BC':metrics,'autonomous_success_episodes':0,'limitations':['HIL success versus autonomous failure is not an autonomous success/failure test.','Full-recording Q can separate by human/cold versus policy/RTC support; policy-only results must be checked.','Only 11 heldout HIL successes / 4 failures; descriptive AUC, not independent task accuracy.','bc_human_filtered_target_penalty uses derived labels, never factual reward targets.']}
 (output/'report.json').write_text(json.dumps(report,indent=2)+'\n');return report

def main():
 p=argparse.ArgumentParser();p.add_argument('--replay-path',type=Path,default=RUN/'replay/v2');p.add_argument('--model-path',type=Path);p.add_argument('--model-dir',type=Path,default=RUN/'learning/warmup');p.add_argument('--actor-mode',choices=['mean'],default='mean');p.add_argument('--allow-experimental',action='store_true');p.add_argument('--split',choices=['train','val'],default='val');p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
 model=a.model_path or a.model_dir/'actor.pt';report=evaluate(a.replay_path,model,a.output_dir,a.split,a.allow_experimental);print(json.dumps({'version':report['actor_version'],'separation':report['separation'],'human_BC':report['human_BC']},indent=2))
if __name__=='__main__':main()
