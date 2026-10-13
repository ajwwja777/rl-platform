#!/usr/bin/env python3
"""CPU-only teacher50 vs teacher5 continuation from packaged HIL8 7281. No publishing."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['JAX_PLATFORMS']='cpu'
os.environ.setdefault('OMP_NUM_THREADS','4')
from pathlib import Path
import argparse,sys,json,pickle,hashlib,time,dataclasses,collections
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'third_party/openpi-rlt/rlt_online_rl/src'),str(ROOT/'third_party/openpi-rlt/packages/openpi-client/src')]
import numpy as np
from methods.openpi_rlt.experiments.journal_import import remap_collection
from methods.openpi_rlt.experiments.supported_sampler import SupportedSampler,FIELDS,unpack_actor_batch
from methods.openpi_rlt.experiments.supported_hil_runtime import SupportedIndex

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    result=[]
    with Path(p).open('rb') as f:
        while True:
            try:result.append(pickle.load(f))
            except EOFError:break
    return result

def key(r):return tuple(int(r[k]) for k in ['collection_phase_id','episode_id','step_id'])
def epkey(r):return key(r)[:2]
def save(p,obj):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));tmp.replace(p)
def arrays(rows):return {k:np.stack([r[k] for r in rows]) for k in FIELDS}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',choices=['teacher','critic'],default='teacher');ap.add_argument('--assets',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--preflight-only',action='store_true');ap.add_argument('--prepared',type=Path);a=ap.parse_args()
    arms=({'teacher50':dict(teacher=50.,clip=None,minq=False),'teacher5':dict(teacher=5.,clip=None,minq=False)} if a.study=='teacher' else {'teacher5':dict(teacher=5.,clip=None,minq=False),'clip10':dict(teacher=5.,clip=10.,minq=False),'minq':dict(teacher=5.,clip=None,minq=True)})
    p=a.assets;out=a.output;out.mkdir(exist_ok=False,parents=True)
    inputs=p/'outputs/matched-learning-20261012';manifest=json.loads((inputs/'comparison-manifest.json').read_text())
    base=p/'models/rlt/plug_v3_yyshadow';package=base/'history/candidates/supported_hil8_clip_20261010'
    paths={'new':inputs/'exploration-29ep-1068.pkl','old':package/'replay/replay_journal.pkl','initial':package/'checkpoints/latest.pkl','teacher':package/'teacher.pkl','actor_snapshot':package/'actor_snapshot/actor_snapshot.pkl','norm':package/'action_norm_stats.json','dev':p/'outputs/rlt/plug_v3_yyshadow/history/warmup_20260924_v1/holdout/replay/replay_journal.pkl','initial_journal':p/'outputs/rlt/plug_v3_yyshadow/history/warmup_20260925_trials/experts120_5000/replay/replay_journal.pkl','manifest':inputs/'comparison-manifest.json'}
    protected={str(f):sha(f) for f in paths.values()}
    assert protected[str(paths['new'])]=='919ff1adb31b6fcae0f3b3380ca80dc2e881b7f2501566d7042237674d0dcad8'
    old,new,dev=read(paths['old']),read(paths['new']),read(paths['dev'])
    nt,nd=remap_collection(old,new,manifest['new_data'])
    initial_rows=read(paths['initial_journal'])
    def canonical(r):
        phase,ep=epkey(r)
        if phase!=2 and ep>=100000:ep=-(ep-100000+1)
        return phase,ep
    assert not {canonical(r) for r in initial_rows}&{canonical(r) for r in dev},'initial training overlaps external DEV'
    assert not {epkey(r) for r in old+nt}&{epkey(r) for r in nd+dev},'continuation training/DEV leak'
    records=old+nt+nd+dev;keys=[key(r) for r in records];assert len(set(keys))==len(keys)
    lookup={k:i for i,k in enumerate(keys)}
    split=np.array(['old_train']*len(old)+['new_train']*len(nt)+['new_dev']*len(nd)+['old_dev']*len(dev))
    data=arrays(records);ntrain=len(old)+len(nt)
    profile=json.loads((package/'profile.json').read_text())
    # Preserve source order, with each entire Episode admitted before its budget.
    episodes=list(dict.fromkeys(int(r['episode_id']) for r in nt));arrival=[]
    for ep in episodes:arrival.append([i for i in range(len(old),ntrain) if int(data['episode_id'][i])==ep])
    save(out/'input-check.json',dict(source_sha256=protected,initial_external_dev_overlap=False,training_dev_overlap=False,counts=dict(collections.Counter(split.tolist())),episodes_new_train=len(episodes),namespace_map=manifest['new_data'],reference='All evaluations are reused DEV/TRAIN; no independent TEST',arrival_order=episodes))
    from rlt_online_rl.config import RLTOnlineRLConfig
    from rlt_online_rl.action_representation import ActionRepresentationAdapter
    from rlt_online_rl.replay import ReplayBuffer
    payload=pickle.loads(paths['initial'].read_bytes());assert int(payload['state']['global_step'])==7281 and int(payload['state']['actor_version'])==3640
    teacher_payload=pickle.loads(paths['teacher'].read_bytes())
    assert sha(paths['teacher'])==profile['teacher']['sha256']
    cfgdict=dict(payload['rl_config'],action_norm_stats_path=str(paths['norm']))
    adapter=ActionRepresentationAdapter.from_config(RLTOnlineRLConfig(**cfgdict))
    # Derive real base/HIL8 batches before training, and preserve their exact IDs.
    schedules={};sampling=[]
    human=np.isin(data['source_chunk'],[2,3]).any(1);newmask=np.isin(split,['new_train','new_dev'])
    # Independent old/new identity diagnostics, no claim of a production incident.
    def pools(rows):
        ar=arrays(rows);ep=ar['episode_id'];recent=(ar['collection_phase_id']==2)&(ep>=int(ep.max())-19)
        h=np.isin(ar['source_chunk'],[2,3]).any(1)
        return recent, recent&(ep>=10000)&h
    wrong_recent,wrong_hil=pools(old+[r for r in new if any(m['original_native_episode']==int(r['episode_id']) and m['split']=='TRAIN' for m in manifest['new_data'])])
    right_recent,right_hil=pools(old+nt)
    save(out/'namespace-regression.json',dict(before_new_recent=int(wrong_recent[len(old):].sum()),before_new_extra_hil=int(wrong_hil[len(old):].sum()),after_new_recent=int(right_recent[len(old):].sum()),after_new_extra_hil=int(right_hil[len(old):].sum()),production_modified=False))
    if a.prepared is None:
        for seed in manifest['training_budget']['paired_seeds']:
            journal=out/('admission_seed%d.pkl'%seed)
            with journal.open('xb') as f:
                for r in old:pickle.dump(r,f,pickle.HIGHEST_PROTOCOL)
            ix=SupportedIndex(journal,.99,profile['retention_episodes'])
            ss=dict(base_rng=np.random.default_rng(seed).bit_generator.state,actor_rng=np.random.default_rng(seed+100000).bit_generator.state,global_step=7281)
            hs=SupportedSampler(ix,adapter,state=ss)
            native=ReplayBuffer(ntrain,seed=seed,sample_strategy='uniform')
            schedule={name:[[],[]] for name in arms};admitted=len(old)
            for ids in arrival:
                with journal.open('ab') as f:
                    for i in ids:pickle.dump(records[i],f,pickle.HIGHEST_PROTOCOL)
                admitted+=len(ids);native._size=admitted
                for _ in ids:
                    ni=native._sample_uniform_indices(128)
                    # Native draws are unused; both arms consume identical SupportedSampler receipts.
                    hs.sample_batch(128);rc=hs.last_receipt
                    for arm in arms:
                        schedule[arm][0].append([lookup[tuple(k)] for k in rc['critic_identities']])
                        schedule[arm][1].append([lookup[tuple(k)] for k in rc['actor_identities']])
            for arm,(ci,ai) in schedule.items():
                ci=np.asarray(ci,np.int32);ai=np.asarray(ai,np.int32);assert ci.shape==(846,128)
                assert (ci<ntrain).all() and (ai<ntrain).all()
                schedules[(seed,arm)]=(ci,ai)
                np.savez_compressed(out/('%s_seed%d_batches.npz'%(arm,seed)),critic=ci,actor=ai,identities=np.array(keys,dtype=np.int64),splits=split)
                item=dict(seed=seed,arm=arm,critic_updates=len(ci),actor_updates=len(ai[0::2]),available_new_episodes=len(episodes))
                for role,draws in [('critic',ci),('actor',ai[0::2])]:
                    item[role]=dict(new_ratio=float(newmask[draws].mean()),human_ratio=float(human[draws].mean()),new_human_ratio=float((newmask&human)[draws].mean()),unique_windows=int(len(np.unique(draws))),dev_draws=int(np.isin(split[draws],['new_dev','old_dev']).sum()))
                sampling.append(item)
    else:
        import shutil
        previous=json.loads((a.prepared/'input-check.json').read_text())
        assert previous['source_sha256']==protected,'Prepared inputs changed'
        sampling=json.loads((a.prepared/'sampling.json').read_text())
        if a.study=='critic':
            sampling=[dict(v,arm=arm) for v in sampling if v['arm']=='teacher5' for arm in arms]
        for seed in manifest['training_budget']['paired_seeds']:
            for arm in arms:
                name='%s_seed%d_batches.npz'%(arm,seed);source_name='%s_seed%d_batches.npz'%('teacher5' if a.study=='critic' else arm,seed);v=np.load(a.prepared/source_name)
                np.testing.assert_array_equal(v['identities'],np.array(keys,dtype=np.int64))
                np.testing.assert_array_equal(v['splits'],split)
                schedules[(seed,arm)]=(v['critic'],v['actor'])
                shutil.copyfile(a.prepared/source_name,out/name)
    for seed in [41,42,43]:
        for arm in arms:
            for x,y in zip(schedules[(seed,arm)],schedules[(seed,'teacher5')]):np.testing.assert_array_equal(x,y)
    save(out/'sampling.json',sampling)
    # Credit validity is checked for all admitted TRAIN rows before any gradient.
    alljournal=out/'training.pkl'
    with alljournal.open('xb') as f:
        for r in old+nt:pickle.dump(r,f,pickle.HIGHEST_PROTOCOL)
    index=SupportedIndex(alljournal,.99,profile['retention_episodes']);index.refresh()
    assert all(v[1] for v in index.lookup.values()),'Invalid/incomplete TRAIN credit'
    if a.preflight_only:return
    import jax,jax.numpy as jnp,optax
    from rlt_online_rl import trainer,networks
    from methods.openpi_rlt.experiments.held_gripper import HeldGripperCritic
    from methods.openpi_rlt.experiments.guarded_retained_actor import make_retained_train_step
    from methods.openpi_rlt.experiments.separate_actor_batch import with_separate_actor_batch
    from methods.openpi_rlt.experiments.target_attribution import reconstruct_observed_returns
    assert all(d.platform=='cpu' for d in jax.devices())
    observed,integrity=reconstruct_observed_returns(data,.99)
    prepared=adapter.prepare_training_batch(data)
    bb={k:jnp.asarray(v) for k,v in prepared.items()}
    mc=np.where(np.isfinite(observed),observed,0).astype(np.float32)
    retained={tuple(k) for k in profile['retention_episodes']}
    bb.update(mc_return=jnp.asarray(mc),mc_valid=jnp.asarray(np.isfinite(observed)),retention_mask=jnp.asarray([epkey(r) in retained for r in records]))
    groups={}
    for i,r in enumerate(records):groups.setdefault((split[i],*epkey(r)),[]).append(i)
    acfg=RLTOnlineRLConfig(**cfgdict);actor,plaincritic=trainer._make_networks(acfg)
    initfields={k:trainer._tree_to_jax(v) for k,v in payload['state'].items()}
    initial=trainer.RLTTrainState(**initfields,actor_tx=optax.adam(acfg.actor_lr),critic_tx=optax.adam(acfg.critic_lr))
    @jax.jit
    def predict(params):return actor.sample_action(params,jax.random.PRNGKey(0),bb['z_rl'],bb['proprio'],bb['ref_chunk'],deterministic=True)
    @jax.jit
    def qphysical(params,actions):
        # Same executed fixed-gripper diagnostic for BOTH Critics, not different actions.
        normg=(data['proprio'][:,6]-adapter.stats.q01[6])/(adapter.stats.q99[6]-adapter.stats.q01[6]+1e-6)*2-1
        actions=actions.at[...,6].set(jnp.asarray(normg)[:,None])
        q1,q2=plaincritic.q_values(params,bb['z_rl'],bb['proprio'],actions)
        return jnp.stack([q1,q2,jnp.minimum(q1,q2)],-1)
    teacher=trainer._tree_to_jax(teacher_payload['actor_params'])
    snapshot=pickle.loads(paths['actor_snapshot'].read_bytes())
    for x,y in zip(jax.tree_util.tree_leaves(initial.actor_params),jax.tree_util.tree_leaves(trainer._tree_to_jax(snapshot['actor_params']))):np.testing.assert_array_equal(x,y)
    diagcritic=HeldGripperCritic(plaincritic,float(adapter.stats.q01[6]),float(adapter.stats.q99[6]))
    initial_physical=np.asarray(adapter.denormalize_to_abs_chunk(np.asarray(predict(initial.actor_params)),data['proprio']))
    def evaluate(state,label):
        pred=predict(state.actor_params);physical=np.asarray(adapter.denormalize_to_abs_chunk(np.asarray(pred),data['proprio']))
        qa=np.asarray(qphysical(state.critic_params,pred));qr=np.asarray(qphysical(state.critic_params,bb['ref_chunk']));qd=np.asarray(qphysical(state.critic_params,bb['action_chunk']))
        td=np.asarray(networks.build_td_target(actor,state.target_actor_params,diagcritic,state.target_critic_params,bb['next_z_rl'],bb['next_proprio'],bb['next_ref_chunk'],bb['rewards'],bb['done'],.99,jax.random.PRNGKey(0)))
        mixed=np.clip(.7*td+.3*np.where(np.isfinite(observed),observed,td),0.,1.)
        np.savez_compressed(out/(label+'.npz'),actor=physical,qactor=qa,qref=qr,qrecord=qd,native_td=td,mixed_target=mixed)
        result=[]
        for g,ids in groups.items():
            ids=np.array(ids);h=np.isin(data['source_chunk'][ids],[2,3]);succ=bool(data['success'][ids].any());observed_ids=observed[ids]
            kind='assisted_success' if h.any() and succ else 'autonomous_success' if succ else 'failure'
            err=np.abs(physical[ids,:,:6]-data['action_chunk'][ids,:,:6])*1000
            referr=np.abs(physical[ids,:,:6]-data['ref_chunk'][ids,:,:6])*1000
            diff=(physical[ids,:,:6]-initial_physical[ids,:,:6])*1000
            item=dict(split=str(g[0]),phase=int(g[1]),episode=int(g[2]),outcome=kind,windows=len(ids),human_mae_mrad=err[h].mean(0).tolist() if h.any() else None,reference_mae_mrad=referr[~h].mean(0).tolist() if (~h).any() else None,actor_change_mrad=np.abs(diff).mean((0,1)).tolist(),actor_signed_change_mrad=diff.mean((0,1)).tolist(),qactor=qa[ids].mean(0).tolist(),qreference=qr[ids].mean(0).tolist(),qrecord=qd[ids].mean(0).tolist(),qrecord_mse=((qd[ids]-observed_ids[:,None])**2).mean(0).tolist() if np.isfinite(observed_ids).all() else None,pure_hil_delta_q=(qd[ids][h.all(1)]-qa[ids][h.all(1)]).mean(0).tolist() if h.all(1).any() else None,native_td_mean=float(td[ids].mean()),mixed_target_mean=float(mixed[ids].mean()),human_p95_mrad=np.quantile(err[h],.95,axis=0).tolist() if h.any() else None,actor_change_p95_mrad=np.quantile(np.abs(diff).reshape(-1,6),.95,axis=0).tolist(),actor_change_max_mrad=np.abs(diff).max((0,1)).tolist())
            result.append(item)
        return result
    baseline=evaluate(initial,'baseline');save(out/'baseline.json',baseline)
    report=dict(status='running',code_sha=sha(Path(__file__)),sources=protected,devices=[str(d) for d in jax.devices()],baseline=baseline,sampling=sampling,runs=[],initial_global_step=7281,teacher_sha256=sha(paths['teacher']),single_factor=('retention_weight 50 versus 5; same packaged teacher' if a.study=='teacher' else 'teacher5 baseline versus critic clip10 OR Actor min-Q'),arms=arms,limits=['Paired single-factor continuation; not a closed-loop online success test','All DEV previously inspected; no independent TEST','Only one new HIL Episode in TRAIN and one in DEV','Q comparisons are current reconstructed alternatives, not observed counterfactual success','Evaluation TD/mixed target uses a fixed diagnostic RNG; actual training losses are in metrics logs'])
    save(out/'study.json',report)
    for seed in [41,42,43]:
      for arm in arms:
        conf=dict(cfgdict);conf.update(actor_lr=1e-5)
        cfg=RLTOnlineRLConfig(**conf);critic=HeldGripperCritic(plaincritic,float(adapter.stats.q01[6]),float(adapter.stats.q99[6]))
        state=initial.replace(actor_tx=optax.adam(cfg.actor_lr),rng=jax.random.PRNGKey(seed))
        for k in payload['state']:
            if k=='rng':continue
            for x,y in zip(jax.tree_util.tree_leaves(getattr(state,k)),jax.tree_util.tree_leaves(getattr(initial,k))):np.testing.assert_array_equal(x,y)
        np.testing.assert_array_equal(np.asarray(predict(state.actor_params)),np.asarray(predict(initial.actor_params)))
        coefficient=arms[arm]['teacher']
        from methods.openpi_rlt.experiments.actor_q_reduction import with_actor_min_q
        from methods.openpi_rlt.experiments.critic_stability import with_critic_gradient_diagnostics
        core=make_retained_train_step(.3,teacher,coefficient,target_policy='clip')
        if a.study=='critic':core=with_critic_gradient_diagnostics(core,arms[arm]['clip'])
        run=with_separate_actor_batch(with_actor_min_q(core,arms[arm]['minq']))
        ci,ai=schedules[(seed,arm)];curve=[];start=time.time();metrics_path=out/('%s_seed%d_metrics.jsonl'%(arm,seed))
        save(out/('%s_seed%d_config.json'%(arm,seed)),dict(rl_config=conf,retention_weight=coefficient,experimental_options=arms[arm],teacher_sha256=sha(paths['teacher']),initial_checkpoint_sha256=sha(paths['initial']),sampler_seed=seed,paired_training_rng_seed=seed))
        for step,(c,aidx) in enumerate(zip(ci,ai),1):
            batch={k:v[c] for k,v in bb.items()}
            batch['actor_batch']={k:v[aidx] for k,v in bb.items()}
            state,metrics=run(state,batch,actor=actor,critic=critic,rl_config=cfg,bc_weight=5.,q_weight=.1,delta_weight=10.,use_action_adapter=True,action_q01=jnp.asarray(adapter.stats.q01),action_q99=jnp.asarray(adapter.stats.q99))
            values={k:float(v) for k,v in metrics.items()};assert all(np.isfinite(list(values.values())))
            assert bool(values['did_actor_update'])==((7281+step)%2==0)
            with metrics_path.open('a') as f:f.write(json.dumps(dict(update=step,**values))+'\n')
            if step in [212,423,634,846]:
                name='%s_seed%d_u%d'%(arm,seed,step);ev=evaluate(state,name)
                curve.append(dict(update=step,evaluation=ev))
                with (out/(name+'.pkl')).open('xb') as f:pickle.dump(dict(research_only=True,rl_config=conf,state={k:trainer._tree_to_numpy(getattr(state,k)) for k in payload['state']},sampling_schedule=str(out/('%s_seed%d_batches.npz'%(arm,seed))),next_schedule_index=step,arm=arm,seed=seed,experimental_options=arms[arm],initial_checkpoint_sha256=protected[str(paths['initial'])],retention_weight=coefficient,teacher_path=str(paths['teacher']),teacher_sha256=sha(paths['teacher'])),f,pickle.HIGHEST_PROTOCOL)
                print(json.dumps(dict(arm=arm,seed=seed,update=step,elapsed_sec=time.time()-start)),flush=True)
        assert int(state.global_step)==8127 and int(state.actor_version)==4063
        report['runs'].append(dict(arm=arm,seed=seed,curve=curve,elapsed_sec=time.time()-start,final_step=int(state.global_step),actor_version=int(state.actor_version)))
        save(out/'study.json',report)
    assert protected=={f:sha(f) for f in protected}
    report['status']='complete';save(out/'study.json',report)

if __name__=='__main__':main()
