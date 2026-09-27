"""Frozen candidate comparison and value diagnostics; never robot I/O."""
import json,pickle,sys,argparse
from pathlib import Path
import numpy as np,torch
from .rtc_upstream_core import original_context
from .chunk_projection import project_correction
from .conditioning_v2 import conditioned
from .upstream_baseline import load,ROOT,RUN
from .learning import Actor
import jax,jax.numpy as jnp
from rlt_online_rl.config import RLTOnlineRLConfig
from rlt_online_rl.networks import TwinCritic
from rlt_online_rl.action_representation import ActionRepresentationAdapter

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--step',type=int,default=5000);p.add_argument('--output',type=Path);args=p.parse_args();torch.set_num_threads(2);data=args.data;run=args.run;episodes=load(data)
    oldstate=torch.load(RUN/'learning/warmup/actor.pt',map_location='cpu',weights_only=False);old=Actor();old.load_state_dict(oldstate['actor']);old.eval()
    cfg=RLTOnlineRLConfig(**json.loads((run/'config.json').read_text())['rl_config']);adapter=ActionRepresentationAdapter.from_config(cfg);state=pickle.load((run/f'checkpoints/step_{args.step}.pkl').open('rb'))['state'];critic=TwinCritic(cfg.z_dim,cfg.proprio_dim,cfg.chunk_len,cfg.action_dim,cfg.critic_hidden_dim,cfg.critic_num_layers)
    qfn=jax.jit(lambda z,p,a:critic.q_values(state['critic_params'],z,p,a)[0]);comparisons=[];curves=[]
    for m,d in episodes:
        q=[]
        for start in range(0,len(d['z_rl']),128):
            at=np.minimum(np.arange(start,start+128),len(d['z_rl'])-1);n=min(128,len(d['z_rl'])-start);b=adapter.prepare_training_batch({k:v[at] for k,v in d.items()})
            q.append(np.asarray(qfn(*[jnp.asarray(b[k]) for k in ('z_rl','proprio','action_chunk')]))[:n])
        q=np.concatenate(q);policy=(d['source_chunk'][:,0]!=2);idx=np.where(policy)[0][:3]
        curves.append(dict(uuid=m['uuid'],split=m['split'],success=m['success'],expert=m['expert'],has_hil=bool((d['source_chunk'][:,0]==2).any() and not m['expert']),early_policy_q=float(q[idx].mean()) if len(idx) else None,frame=d['frame'].tolist(),delay=d['delay'].tolist(),q=q.tolist(),done=d['done'].tolist(),human=(d['source_chunk'][:,0]==2).tolist()))
        if m['split']!='val':continue
        c=original_context(d['proprio']);full=np.tile(c[:,None,:14],(1,10,1));ref=full.copy();ref[:,:,7:14]=d['ref_chunk'];target=full.copy();target[:,:,7:14]=d['action_chunk']
        with np.load(run/f'eval_{args.step}_{m["uuid"]}_{m["index"]}.npz') as f:pred=f['pred'].copy()
        with torch.no_grad():oldpred=old(torch.tensor(d['z_rl']),torch.tensor(c),torch.tensor(ref)).numpy()[:,:,7:14];targetcmd=conditioned(torch.tensor(target),torch.tensor(c)).numpy()[:,:,7:13]
        models={'reference':d['ref_chunk'],'old_warmup_raw':oldpred,'old_warmup_linear':project_correction(oldpred,d['ref_chunk'],1),'new_rtc_linear':project_correction(pred,d['ref_chunk'],1)}
        for name,value in models.items():
            proposed=full.copy();proposed[:,:,7:14]=value
            with torch.no_grad():cmd=conditioned(torch.tensor(proposed),torch.tensor(c)).numpy()[:,:,7:13]
            for delay in (0,6):
                h=(d['source_chunk'][:,0]==2)&(d['delay']==delay)
                if not h.any():continue
                comparisons.append(dict(model=name,uuid=m['uuid'],delay=delay,raw_mse=float(np.mean((value[h,:,:6]-d['action_chunk'][h,:,:6])**2)),matched_mse=float(np.mean((cmd[h]-targetcmd[h])**2)),actual_human_mse=float(np.mean((cmd[h]-d['action_chunk'][h,:,:6])**2)),accel_rms=float(np.sqrt(np.mean(np.diff(cmd[h],n=2,axis=1)**2)))))
    summary={}
    for name in ('reference','old_warmup_raw','old_warmup_linear','new_rtc_linear'):
        summary[name]={}
        for delay in (0,6):
            selected=[r for r in comparisons if r['model']==name and r['delay']==delay];summary[name][str(delay)]={k:float(np.mean([r[k] for r in selected])) for k in ('raw_mse','matched_mse','actual_human_mse','accel_rms')}
    strata={}
    for split in ('train','val'):
        for outcome in (True,False):
            selected=[r for r in curves if r['split']==split and r['success']==outcome and r['early_policy_q'] is not None]
            strata[f'{split}_{outcome}']={'episodes':len(selected),'autonomous_successes':sum(r['success'] and not r['has_hil'] for r in selected),'early_q_mean':float(np.mean([r['early_policy_q'] for r in selected])) if selected else None}
    report={'scope':'reused UUID same-input regression, not closed-loop/independent success validation','summary':summary,'q_strata':strata,'rows':comparisons,'q_curves':curves};(args.output or data/'candidate-comparison.json').write_text(json.dumps(report,indent=2));print(json.dumps({'summary':summary,'q_strata':strata},indent=2))
if __name__=='__main__':main()
