"""Matched CPU-only study of recent-window versus recent-Episode sampling.

This diagnostic writes only --output. It never opens runtime HTTP or publishes.
Comparisons use TRAIN and repeatedly used DEV, not independent TEST.
"""
from __future__ import annotations
import argparse,dataclasses,json,pickle,hashlib,sys,time
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'third_party/openpi-rlt/rlt_online_rl/src')]

def paired_recent_indices(recent, warmup, human, available, episode_ids, rng):
    """Same pool quotas and non-recent draws; only recent selection differs."""
    available = np.asarray(available, dtype=np.int64)
    if not len(available):
        raise ValueError("No available Replay")
    recent = np.asarray(recent, dtype=np.int64)
    u = rng.random(51)
    pool = recent if len(recent) else available
    flat = pool[(u * len(pool)).astype(np.int64)]
    if len(recent):
        episodes = np.unique(episode_ids[recent])
        positions = u * len(episodes)
        bins = positions.astype(np.int64)
        chosen, v = episodes[bins], positions - bins
        balanced = np.asarray([recent[episode_ids[recent] == ep][int(t * np.count_nonzero(episode_ids[recent] == ep))] for ep, t in zip(chosen, v)])
    else:
        balanced = flat.copy()
    other = np.concatenate([rng.choice(p if len(p) else available, n, replace=True) for p, n in [(warmup, 38), (human, 26), (available, 13)]])
    perm = rng.permutation(128)
    result = {"window_recent": np.concatenate([flat, other])[perm], "episode_recent": np.concatenate([balanced, other])[perm]}
    assert np.array_equal(result["window_recent"][perm >= 51], result["episode_recent"][perm >= 51])
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
    parser=argparse.ArgumentParser();parser.add_argument('--assets',type=Path,required=True);parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--dev',type=Path,required=True);args=parser.parse_args()
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
    rows=read_journal(args.assets/'replay/replay_journal.pkl')+read_journal(args.input/'new204-delta.pkl')+read_journal(args.input/'new77-delta.pkl');assert len(rows)==2777
    keys=['z_rl','proprio','ref_chunk','action_chunk','rewards','done','next_z_rl','next_proprio','next_ref_chunk','source_chunk','collection_phase_id','episode_id','step_id','success','source','intervention_flag']
    raw={k:np.stack([r[k]for r in rows])for k in keys};mc,return_reports=reconstruct_observed_returns(raw,.99);assert np.isfinite(mc).all()
    data=adapter.prepare_training_batch(raw);data['mc_return']=mc.astype('float32');data['mc_valid']=np.ones(len(rows),bool)
    retained={tuple(x)for x in profile['retention_episodes']}
    ids=[(int(r['collection_phase_id']),int(r['episode_id']),int(r['step_id']))for r in rows];assert len(set(ids))==len(ids)
    data['retention_mask']=np.asarray([k[:2]in retained for k in ids]);lookup={k:i for i,k in enumerate(ids)}
    states={f"{mode}_seed{seed}":init for seed in [41,42,43] for mode in ["window_recent","episode_recent"]}
    rngs={seed:np.random.default_rng(seed) for seed in [41,42,43]}
    sampled_terminal={name:0 for name in states};sampled_hil={name:0 for name in states};evaluations={name:[] for name in states};sample_indices={name:[] for name in states}
    arrivals=[];consumed=0
    for ep in dict.fromkeys(int(r['episode_id']) for r in rows[2496:]):
        count=sum(int(r['episode_id'])==ep for r in rows[2496:])
        arrivals.append(dict(first_update=consumed+1,episode_id=ep,windows=count,available=2496+consumed+count));consumed+=count
    assert consumed==281 and [a['first_update'] for a in arrivals]==[1,60,114,176,205,225,243]
    (args.output/'arrival_schedule.json').write_text(json.dumps(arrivals,indent=2))
    log=(args.output/'metrics.jsonl').open('w');start=time.time()
    pro=raw['proprio'];z=raw['z_rl'];ref=data['ref_chunk'];human=np.isin(raw['source_chunk'],[2,3]);bc=np.where(human[...,None],data['action_chunk'],ref);expert=raw['episode_id']<0;recent=(raw['collection_phase_id']==2)&(raw['episode_id']>=10000)
    epgroups=defaultdict(list)
    for i,k in enumerate(ids):epgroups[k[:2]].append(i)
    success=np.zeros(len(rows),bool);assist=np.zeros(len(rows),bool)
    for idx in epgroups.values():success[idx]=raw['success'][idx].any();assist[idx]=human[idx].any()
    masks={'expert_TRAIN':expert,'old_autonomous_success_TRAIN':~expert&~recent&success&~assist,'old_assisted_success_TRAIN':~expert&~recent&success&assist,'old_failure_TRAIN':~expert&~recent&~success,'old_HIL_windows_excluding_experts_TRAIN':~expert&~recent&human.any(axis=1),'new_success_TRAIN':recent&success&~assist,'new_failure_TRAIN':recent&~success,'new_assisted_success_TRAIN':recent&success&assist,'new_HIL_TRAIN':recent&human.any(axis=1)}
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
    terminals=np.flatnonzero(recent&raw['done']&~success)
    def evaluate(state):
        pred=forward(amean,state.actor_params);q=forward(qvalue,state.critic_params,data['action_chunk']);qp=forward(qvalue,state.critic_params,pred);qo=forward(qvalue,state.critic_params,old_pred)
        nxt=forward(amean,state.target_actor_params,next_state=True);nq=forward(qvalue,state.target_critic_params,nxt,next_state=True)
        td=(raw['rewards']*(.99**np.arange(10))).sum(axis=1)+(~raw['done'])*(.99**10)*nq.min(axis=1);target=.7*td+.3*mc
        fit=((pred[...,:6]-bc[...,:6])**2).mean(axis=(1,2));change=((pred[...,:6]-old_pred[...,:6])**2).mean(axis=(1,2))
        values={'BC6_mse':fit,'Q1_vs_behavior_return_mse':(q[:,0]-mc)**2,'mixed_target_residual_mse':((q-target[:,None])**2).mean(axis=1),'Q1_new_actor_minus_original_same_critic':qp[:,0]-qo[:,0],'actor6_change_mse':change}
        result={'step':int(state.global_step),'actor_version':int(state.actor_version),'terminal_failure_q':q[terminal_index].tolist(),'terminal_target':float(target[terminal_index]),'all_failed_terminal_q':{str(int(raw['episode_id'][i])):q[i].tolist() for i in terminals},'groups':{}}
        for g,mask in masks.items():
            result['groups'][g]={k:float(np.mean([v[[i for i in ix if mask[i]]].mean()for ix in epgroups.values()if mask[ix].any()]))for k,v in values.items()}
        result['repeated_DEV']=evaluate_dev(state)
        return result
    dev_rows=read_journal(args.dev);assert len(dev_rows)==370
    dev_raw={k:np.stack([r[k] for r in dev_rows]) for k in keys};dev_data=adapter.prepare_training_batch(dev_raw)
    dev_returns,_=reconstruct_observed_returns(dev_raw,.99);assert np.isfinite(dev_returns).all()
    dev_eps=defaultdict(list)
    for i,row in enumerate(dev_rows):dev_eps[(int(row['collection_phase_id']),int(row['episode_id']))].append(i)
    assert len(dev_eps)==20 and not (set(dev_eps)&set(epgroups))
    dev_success=np.zeros(len(dev_rows),bool);dev_assist=np.zeros(len(dev_rows),bool)
    dev_human=np.isin(dev_raw['source_chunk'],[2,3]);dev_bc=np.where(dev_human[...,None],dev_data['action_chunk'],dev_data['ref_chunk'])
    for ix in dev_eps.values():dev_success[ix]=dev_raw['success'][ix].any();dev_assist[ix]=dev_human[ix].any()
    def evaluate_dev(state):
        ps=[];qs=[]
        for k in range(0,len(dev_rows),128):
            sl=slice(k,k+128);z=jnp.asarray(dev_raw['z_rl'][sl],jnp.float32);pro=jnp.asarray(dev_raw['proprio'][sl],jnp.float32)
            ps.append(np.asarray(amean(state.actor_params,z,pro,jnp.asarray(dev_data['ref_chunk'][sl]))))
            qq=qvalue(state.critic_params,z,pro,jnp.asarray(dev_data['action_chunk'][sl]));qs.append(np.stack([np.asarray(x) for x in qq],axis=1))
        pred=np.concatenate(ps);q=np.concatenate(qs);err=pred[...,:6]-dev_bc[...,:6]
        scale=(np.asarray(adapter.stats.q99[:6])-np.asarray(adapter.stats.q01[:6])+1e-6)/2
        physical=err*scale;groups={};per_episode=[]
        for ep,ix in dev_eps.items():
            group='failure' if not dev_success[ix[0]] else ('assisted' if dev_assist[ix[0]] else 'autonomous')
            per_episode.append(dict(phase=ep[0],episode=ep[1],group=group,bc6_mse=float((err[ix]**2).mean()),bc6_rmse_mrad=float(np.sqrt((physical[ix]**2).mean())*1000),bc6_abs_p95_mrad=float(np.quantile(np.abs(physical[ix]),.95)*1000),q1_behavior_mse=float(((q[ix,0]-dev_returns[ix])**2).mean())))
        for group in ['autonomous','assisted','failure']:
            rr=[e for e in per_episode if e['group']==group];groups[group]={key:float(np.mean([x[key] for x in rr])) for key in ['bc6_mse','bc6_rmse_mrad','bc6_abs_p95_mrad','q1_behavior_mse']}
        return dict(groups=groups,episodes=per_episode)
    initial_eval=evaluate(init);(args.output/'initial-evaluation.json').write_text(json.dumps(initial_eval,indent=2))
    def save(name,state):
        path=args.output/name;path.mkdir(exist_ok=True)
        payload={'rl_config':dataclasses.asdict(cfg),'state':{k:trainer._tree_to_numpy(getattr(state,k))for k in sp},'candidate_profile_sha256':initial.get('candidate_profile_sha256'),'offline_sampling_variant':name,'production_release':False}
        f=path/f'step_{int(state.global_step)}.pkl'
        with f.open('wb')as h:pickle.dump(payload,h,pickle.HIGHEST_PROTOCOL)
    available=2496
    for n in range(1,282):
        for arrival in arrivals:
            if arrival['first_update']==n:available=arrival['available']
        all_pool=np.arange(available);phase=raw['collection_phase_id'][:available];ep=raw['episode_id'][:available]
        recent_pool=all_pool[(phase==2)&(ep>=int(ep.max())-19)];warmup_pool=all_pool[phase==1]
        human_pool=all_pool[raw['intervention_flag'][:available]|np.isin(raw['source'][:available],[2,3])|human[:available].any(axis=1)]
        for seed in [41,42,43]:
            paired=paired_recent_indices(recent_pool,warmup_pool,human_pool,all_pool,raw['episode_id'],rngs[seed])
            for mode,selected in paired.items():
                name=f"{mode}_seed{seed}";state=states[name];assert selected.max()<available
                sample_indices[name].append(selected.astype(np.int32));sampled_terminal[name]+=int(np.isin(selected,terminals).sum());sampled_hil[name]+=int(((raw['episode_id'][selected]==10009)&human[selected].any(axis=1)).sum())
                batch={k:jnp.asarray(v[selected]) for k,v in data.items()}
                updated,metrics=train(state,batch,actor=actor,critic=critic,rl_config=cfg,bc_weight=5.,q_weight=.1,delta_weight=10.,use_action_adapter=True,action_q01=jnp.asarray(adapter.stats.q01),action_q99=jnp.asarray(adapter.stats.q99))
                vals={k:float(v) for k,v in jax.device_get(metrics).items()};assert all(np.isfinite(v) for v in vals.values())
                assert int(updated.global_step)==7000+n;states[name]=updated
                log.write(json.dumps({'variant':name,'update_index':n,'terminal_draws_cumulative':sampled_terminal[name],'new_HIL_draws_cumulative':sampled_hil[name],**vals})+'\n')
                if n in [204,224,242,281]:evaluations[name].append(evaluate(updated))
                if n==281:save(name,updated)
        if n%40==0 or n==281:
            log.flush();progress=dict(updates=n,elapsed_sec=time.time()-start,terminal_draws=sampled_terminal,new_HIL_draws=sampled_hil)
            (args.output/'progress.json').write_text(json.dumps(progress,indent=2));print(json.dumps(progress),flush=True)
    np.savez_compressed(args.output/'sample_indices.npz',**{name:np.stack(ix) for name,ix in sample_indices.items()})
    log.close();result={'elapsed_sec':time.time()-start,'terminal_draws':sampled_terminal,'evaluations':evaluations,'initial':initial_eval,'new_HIL_draws':sampled_hil,'boundary':'Matched controlled resampling, same initial numerical state and actual arrival windows. Only recent selection changes; identical other-pool draws, CPU-only TRAIN and repeated DEV. No TEST or production publication.'};(args.output/'comparison.json').write_text(json.dumps(result,indent=2));print('COMPLETE',flush=True)
if __name__=='__main__':main()
