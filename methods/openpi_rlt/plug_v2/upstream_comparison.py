"""Same-input offline comparison only. No live rollout claim."""
import sys,json,pickle
from pathlib import Path
import numpy as np,torch,pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[3];RUN=ROOT/'runs/plug_v2';sys.path.insert(0,str(ROOT/'code/openpi-rlt/rlt_online_rl/src'))
import jax,jax.numpy as jnp
from rlt_online_rl.config import RLTOnlineRLConfig
from rlt_online_rl.networks import ChunkActor
from rlt_online_rl.action_representation import ActionRepresentationAdapter
from .learning import Actor
from .conditioning_v2 import conditioned

def main():
 torch.set_num_threads(2)
 old_saved=torch.load(RUN/'learning/warmup/actor.pt',map_location='cpu',weights_only=False);old=Actor();old.load_state_dict(old_saved['actor']);old.eval()
 bundles=[]
 for folder,step in [('upstream-baseline-20260919-r3',2000),('upstream-baseline-20260919-r3',5000),('upstream-baseline-20260919-r3',20000),('upstream-delta300-20260919',2000),('upstream-delta300-20260919',5000)]:
  root=RUN/'learning'/folder;cfg=RLTOnlineRLConfig(**json.loads((root/'config.json').read_text())['rl_config']);adapter=ActionRepresentationAdapter.from_config(cfg);saved=pickle.load((root/f'checkpoints/step_{step}.pkl').open('rb'));a=ChunkActor(cfg.z_dim,cfg.proprio_dim,cfg.chunk_len,cfg.action_dim,cfg.actor_hidden_dim,cfg.actor_num_layers,cfg.fixed_std)
  def make(a,params):return jax.jit(lambda z,p,r:a.actor_mean(params,z,p,r))
  bundles.append((folder+'/'+str(step),adapter,make(a,saved['state']['actor_params'])))
 output=[]
 for path in sorted((RUN/'diagnostics/upstream-sync-20260919/episodes').glob('*.npz')):
  with np.load(path) as f:m=json.loads(str(f['metadata']));d={k:f[k].copy() for k in f.files if k!='metadata'}
  if m['split']!='val' or not m['rows']:continue
  if m['expert']:
   index=m['index'];table=pq.read_table(Path(m['root'])/f'data/chunk-{index//1000:03d}/episode_{index:06d}.parquet');full=np.stack(table['observation.state'].to_pylist())[d['frame']].astype(np.float32)
  else:
   with np.load(m['trace']) as f:frames=json.loads(str(f['frames_json']))
   full=np.asarray([frames[i]['state'] for i in d['frame']],np.float32)
  np.testing.assert_allclose(full[:,7:14],d['proprio'],atol=1e-7)
  c=np.concatenate((full,np.zeros((len(full),85),np.float32)),axis=1);ref=np.repeat(full[:,None,:],10,axis=1);ref[:,:,7:14]=d['ref_chunk'];target=np.repeat(full[:,None,:],10,axis=1);target[:,:,7:14]=d['action_chunk'];human=np.isin(d['source_chunk'],[2,3])
  with torch.no_grad():predictions={'reference':ref,'old_warmup_'+str(old_saved['global_step']):old(torch.tensor(d['z_rl']),torch.tensor(c),torch.tensor(ref)).numpy()}
  for name,adapter,fn in bundles:
   pieces=[]
   for start in range(0,len(c),128):
    at=np.minimum(np.arange(start,start+128),len(c)-1);n=min(128,len(c)-start);normalized=adapter.normalize_ref_chunk(d['ref_chunk'][at],d['proprio'][at]);v=fn(jnp.asarray(d['z_rl'][at]),jnp.asarray(d['proprio'][at]),jnp.asarray(normalized));pieces.append(adapter.denormalize_to_abs_chunk(np.asarray(v)[:n],d['proprio'][at][:n]))
   v=ref.copy();v[:,:,7:14]=np.concatenate(pieces);predictions[name]=v
  for name,pred in predictions.items():
   raw_error=((pred[:,:,7:13]-target[:,:,7:13])**2).mean(-1)
   with torch.no_grad():cmd=conditioned(torch.tensor(pred),torch.tensor(c)).numpy()
   error=((cmd[:,:,7:13]-target[:,:,7:13])**2).mean(-1);delta=np.diff(cmd[:,:,7:13],axis=1);rawdelta=np.diff(pred[:,:,7:13],axis=1);turn=((delta[:,1:]*delta[:,:-1]<0)&(abs(delta[:,1:])>1e-4)&(abs(delta[:,:-1])>1e-4)).sum((1,2))
   output.append({'model':name,'uuid':m['uuid'],'expert':m['expert'],'success':m['success'],'windows':len(c),'human_frames':int(human.sum()),'raw_human_bc_rad2':float(raw_error[human].mean()) if human.any() else None,'conditioned_human_bc_rad2':float(error[human].mean()) if human.any() else None,'raw_step_p95_rad':float(np.quantile(abs(rawdelta),.95)),'conditioned_step_p95_rad':float(np.quantile(abs(delta),.95)),'conditioned_reversals_per_window':float(turn.mean()),'conditioned_delta_rms':float(np.sqrt(np.mean(delta**2)))})
 summary={}
 for name in sorted({r['model'] for r in output}):
  rows=[r for r in output if r['model']==name];h=[r for r in rows if r['raw_human_bc_rad2'] is not None];summary[name]={k:float(np.mean([r[k] for r in (h if 'human_bc' in k else rows)])) for k in ('raw_human_bc_rad2','conditioned_human_bc_rad2','raw_step_p95_rad','conditioned_reversals_per_window','conditioned_delta_rms')}
 result={'scope':'same heldout observations, d0 context, fixed shared conditioner; not closed loop or live RTC validation','summary':summary,'episodes':output}
 (RUN/'diagnostics/upstream-audit-20260919/model-comparison.json').write_text(json.dumps(result,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
