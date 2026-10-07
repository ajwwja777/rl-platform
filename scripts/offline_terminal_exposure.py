"""Matched CPU-only replay of actual batches with a failed-terminal quota.

This diagnostic writes only --output. It never opens runtime HTTP or publishes.
All comparisons are TRAIN, not independent capability evaluation.
"""
from __future__ import annotations
import argparse,dataclasses,json,pickle,hashlib,sys,time
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'third_party/openpi-rlt/rlt_online_rl/src')]

def quota_indices(base,pool,quota,rng):
    result=np.asarray(base,dtype=np.int64).copy()
    if quota and len(pool):
        positions=rng.choice(len(result),quota,replace=False)
        result[positions]=rng.choice(pool,quota,replace=True)
    return result

def validate_output_path(output, assets):
    output = Path(output).resolve()
    assets = Path(assets).resolve()
    if output == assets or assets in output.parents:
        raise ValueError("Offline output must be outside the immutable asset directory")
    return output

def read_journal(path):
    rows=[]
    with Path(path).open('rb')as f:
        while True:
            try:rows.append(pickle.load(f))
            except EOFError:break
    return rows

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--assets',type=Path,required=True);parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output = validate_output_path(args.output, args.assets)
    args.output.mkdir(parents=True,exist_ok=True)
    import jax,jax.numpy as jnp
    from rlt_online_rl import trainer
    from rlt_online_rl.config import RLTOnlineRLConfig
    from rlt_online_rl.action_representation import ActionRepresentationAdapter
    from methods.openpi_rlt.experiments.held_gripper import HeldGripperCritic
    from methods.openpi_rlt.experiments.retained_actor import make_retained_train_step
    from methods.openpi_rlt.experiments.target_attribution import reconstruct_observed_returns
    assert all(d.platform=='cpu'for d in jax.devices()),'CPU only'
    initial=pickle.loads((args.assets/'checkpoints/latest.pkl').read_bytes());profile=json.loads((args.assets/'profile.json').read_text());teacher=pickle.loads((args.assets/'teacher.pkl').read_bytes())
    cfgd=dict(initial['rl_config']);cfgd['action_norm_stats_path']=str(args.assets/'action_norm_stats.json');cfg=RLTOnlineRLConfig(**cfgd)
    assert cfg.actor_lr==1e-5 and cfg.critic_lr==1e-4 and cfg.actor_update_period==2 and profile['mc_weight']==.3
    sp=initial['state'];assert sp['global_step']==7000 and sp['actor_version']==3500
    dummy,actor,native=trainer.init_train_state(cfg,rng=jax.random.PRNGKey(0))
    init=trainer.RLTTrainState(**{k:trainer._tree_to_jax(sp[k])for k in ['actor_params','target_actor_params','critic_params','target_critic_params','actor_opt_state','critic_opt_state','rng','global_step','actor_version']},actor_tx=dummy.actor_tx,critic_tx=dummy.critic_tx)
    adapter=ActionRepresentationAdapter.from_config(cfg);critic=HeldGripperCritic(native,float(adapter.stats.q01[6]),float(adapter.stats.q99[6]))
    train=make_retained_train_step(.3,trainer._tree_to_jax(teacher['actor_params']),50.)
    rows=read_journal(args.assets/'replay/replay_journal.pkl')+read_journal(args.input/'new204-delta.pkl');assert len(rows)==2700
    keys=['z_rl','proprio','ref_chunk','action_chunk','rewards','done','next_z_rl','next_proprio','next_ref_chunk','source_chunk','collection_phase_id','episode_id','step_id','success','source','intervention_flag']
    raw={k:np.stack([r[k]for r in rows])for k in keys};mc,return_reports=reconstruct_observed_returns(raw,.99);assert np.isfinite(mc).all()
    data=adapter.prepare_training_batch(raw);data['mc_return']=mc.astype('float32');data['mc_valid']=np.ones(len(rows),bool)
    retained={tuple(x)for x in profile['retention_episodes']}
    ids=[(int(r['collection_phase_id']),int(r['episode_id']),int(r['step_id']))for r in rows];assert len(set(ids))==len(ids)
    data['retention_mask']=np.asarray([k[:2]in retained for k in ids]);lookup={k:i for i,k in enumerate(ids)}
    manifest=json.loads((args.input/'snapshot-and-batches.json').read_text());batches=manifest['batches'];assert [x['step']for x in batches]==list(range(7001,7205))
    states={name:init for name in ['recorded_baseline','terminal_quota1','terminal_quota4']};quotas=dict(zip(states,[0,1,4]));rngs={name:np.random.default_rng(42)for name in states}
    seen_episodes=set(k[:2]for k in ids if k[1]<10000);sampled_terminal={name:0 for name in states};evaluations={name:[]for name in states}
    log=(args.output/'metrics.jsonl').open('w');start=time.time()
    pro=raw['proprio'];z=raw['z_rl'];ref=data['ref_chunk'];human=np.isin(raw['source_chunk'],[2,3]);bc=np.where(human[...,None],data['action_chunk'],ref);expert=raw['episode_id']<0;recent=(raw['collection_phase_id']==2)&(raw['episode_id']>=10000)
    epgroups=defaultdict(list)
    for i,k in enumerate(ids):epgroups[k[:2]].append(i)
    success=np.zeros(len(rows),bool);assist=np.zeros(len(rows),bool)
    for idx in epgroups.values():success[idx]=raw['success'][idx].any();assist[idx]=human[idx].any()
    masks={'expert_TRAIN':expert,'old_autonomous_success_TRAIN':~expert&~recent&success&~assist,'old_assisted_success_TRAIN':~expert&~recent&success&assist,'old_failure_TRAIN':~expert&~recent&~success,'old_HIL_windows_excluding_experts_TRAIN':~expert&~recent&human.any(axis=1),'new_success_TRAIN':recent&success,'new_failure_TRAIN':recent&~success}
    amean=jax.jit(actor.actor_mean);qvalue=jax.jit(critic.q_values)
    def forward(fun,params,aa=None,next_state=False):
        zz=raw['next_z_rl']if next_state else z;pp=raw['next_proprio']if next_state else pro
        a=(data['next_ref_chunk']if next_state else ref)if aa is None else aa
        vals=[]
        for i in range(0,len(rows),128):
            sl=slice(i,i+128);value=fun(params,jnp.asarray(zz[sl]),jnp.asarray(pp[sl]),jnp.asarray(a[sl]))
            vals.append(np.stack([np.asarray(t)for t in value],axis=1)if isinstance(value,tuple)else np.asarray(value))
        return np.concatenate(vals)
    old_pred=forward(amean,init.actor_params)
    terminal_index=lookup[(2,10004,137)]
    def evaluate(state):
        pred=forward(amean,state.actor_params);q=forward(qvalue,state.critic_params,data['action_chunk']);qp=forward(qvalue,state.critic_params,pred);qo=forward(qvalue,state.critic_params,old_pred)
        nxt=forward(amean,state.target_actor_params,next_state=True);nq=forward(qvalue,state.target_critic_params,nxt,next_state=True)
        td=(raw['rewards']*(.99**np.arange(10))).sum(axis=1)+(~raw['done'])*(.99**10)*nq.min(axis=1);target=.7*td+.3*mc
        fit=((pred[...,:6]-bc[...,:6])**2).mean(axis=(1,2));change=((pred[...,:6]-old_pred[...,:6])**2).mean(axis=(1,2))
        values={'BC6_mse':fit,'Q1_vs_behavior_return_mse':(q[:,0]-mc)**2,'mixed_target_residual_mse':((q-target[:,None])**2).mean(axis=1),'Q1_new_actor_minus_original_same_critic':qp[:,0]-qo[:,0],'actor6_change_mse':change}
        result={'step':int(state.global_step),'actor_version':int(state.actor_version),'terminal_failure_q':q[terminal_index].tolist(),'terminal_target':float(target[terminal_index]),'groups':{}}
        for g,mask in masks.items():
            result['groups'][g]={k:float(np.mean([v[[i for i in ix if mask[i]]].mean()for ix in epgroups.values()if mask[ix].any()]))for k,v in values.items()}
        return result
    initial_eval=evaluate(init);(args.output/'initial-evaluation.json').write_text(json.dumps(initial_eval,indent=2))
    def save(name,state):
        path=args.output/name;path.mkdir(exist_ok=True)
        payload={'rl_config':dataclasses.asdict(cfg),'state':{k:trainer._tree_to_numpy(getattr(state,k))for k in sp},'candidate_profile_sha256':initial.get('candidate_profile_sha256'),'offline_sampling_variant':name,'production_release':False}
        f=path/f'step_{int(state.global_step)}.pkl'
        with f.open('wb')as h:pickle.dump(payload,h,pickle.HIGHEST_PROTOCOL)
    for n,record in enumerate(batches,1):
        base=np.asarray([lookup[(1 if ph=='warmup'else 2,int(ep),int(step))]for ph,ep,step in record['identities']],dtype=np.int64);assert len(base)==128
        seen_episodes.update(ids[i][:2]for i in base)
        pool=[i for i,k in enumerate(ids)if k[0]==2 and k[1]>=10000 and k[:2]in seen_episodes and raw['done'][i]and not raw['success'][i]]
        for name,state in list(states.items()):
            selected=quota_indices(base,pool,quotas[name],rngs[name]);sampled_terminal[name]+=int(np.count_nonzero(selected==terminal_index))
            batch={k:jnp.asarray(v[selected])for k,v in data.items()}
            updated,metrics=train(state,batch,actor=actor,critic=critic,rl_config=cfg,bc_weight=5.,q_weight=.1,delta_weight=10.,use_action_adapter=True,action_q01=jnp.asarray(adapter.stats.q01),action_q99=jnp.asarray(adapter.stats.q99))
            vals={k:float(v)for k,v in jax.device_get(metrics).items()};assert all(np.isfinite(v)for v in vals.values()),(name,n,vals)
            assert int(updated.global_step)==record['step'];states[name]=updated
            log.write(json.dumps({'variant':name,'update_index':n,'quota':quotas[name],'terminal_draws_cumulative':sampled_terminal[name],**vals})+'\n')
            if n in [1,175,180,190,200,204]:evaluations[name].append(evaluate(updated))
            if n in [200,204]:save(name,updated)
        if n%25==0 or n==204:
            log.flush();progress={'updates':n,'elapsed_sec':time.time()-start,'terminal_draws':sampled_terminal,'states':{k:{'step':int(v.global_step),'actor':int(v.actor_version)}for k,v in states.items()}}
            (args.output/'progress.json').write_text(json.dumps(progress,indent=2));print(json.dumps(progress),flush=True)
    log.close();result={'elapsed_sec':time.time()-start,'terminal_draws':sampled_terminal,'evaluations':evaluations,'initial':initial_eval,'boundary':'Matched complete-state restore and actual batch identity replay; modified sampled slots only. CPU-only TRAIN diagnostic, no production publication.'};(args.output/'comparison.json').write_text(json.dumps(result,indent=2));print('COMPLETE',flush=True)
if __name__=='__main__':main()
