#!/usr/bin/env python3
"""Matched reference-dropout ablation using the unmodified native offline step.

Continue one retained Warmup5k state. Original data/optimizer/BC/Q/rewards/time
base unchanged; only reference dropout .5 versus 0 changes per seed pair.
"""
import argparse,dataclasses,hashlib,importlib.util,json,os,pickle,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'third_party/openpi-rlt/rlt_online_rl/src'),str(ROOT/'third_party/openpi-rlt/packages/openpi-client/src')]
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--updates',type=int,default=2000)
    a=p.parse_args()
    if a.output.exists():p.error('fresh output required')
    a.output.mkdir(parents=True)
    import numpy as np
    import jax,jax.numpy as jnp,optax
    from rlt_online_rl import trainer
    from rlt_online_rl.config import RLTOnlineRLConfig
    from rlt_online_rl.action_representation import ActionRepresentationAdapter
    source=ROOT/'third_party/openpi-rlt/rlt_online_rl/scripts/offline/offline_train_from_replay.py'
    sys.path.insert(0,str(source.parent))
    spec=importlib.util.spec_from_file_location('fixed_offline_training',source);native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
    hist=a.root/'outputs/rlt/plug_v3_yyshadow/history'
    ckpt=hist/'warmup_20260925_trials/experts120_5000/checkpoints/latest.pkl'
    train_path=ckpt.parent.parent/'replay/replay_journal.pkl'
    dev_path=hist/'warmup_20260924_v1/holdout/replay/replay_journal.pkl'
    norm=a.root/'models/rlt/plug_v3_yyshadow/warmup-5000/action_norm_stats.json'
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    hashes={str(x):sha(x) for x in [ckpt,train_path,dev_path,norm,source]}
    def read(path):
        rows=[]
        with path.open('rb') as f:
            while True:
                try:rows.append(pickle.load(f))
                except EOFError:break
        return rows
    trainrows=read(train_path);devrows=read(dev_path)
    if {int(r['episode_id']) for r in trainrows}&{int(r['episode_id']) for r in devrows}:raise ValueError('Episode leakage')
    def stack(rows):return {k:np.stack([r[k] for r in rows]) for k in rows[0] if k!='collection_phase'}
    payload=pickle.loads(ckpt.read_bytes());config=dict(payload['rl_config'],action_norm_stats_path=str(norm))
    cfg=RLTOnlineRLConfig(**config);adapter=ActionRepresentationAdapter.from_config(cfg);actor,critic=trainer._make_networks(cfg)
    initial=trainer.RLTTrainState(**{k:trainer._tree_to_jax(v) for k,v in payload['state'].items()},actor_tx=optax.adam(cfg.actor_lr),critic_tx=optax.adam(cfg.critic_lr))
    raw_train=stack(trainrows);raw_dev=stack(devrows)
    # Same retained original script's transition split. This is an internal
    # monitor, not an Episode holdout; external 20-Episode development is separate.
    train_ds,val_ds=native._split_dataset(raw_train,val_ratio=.05,seed=42)
    bt={k:jnp.asarray(v) for k,v in adapter.prepare_training_batch(train_ds).items()}
    fulltrain={k:jnp.asarray(v) for k,v in adapter.prepare_training_batch(raw_train).items()}
    bd={k:jnp.asarray(v) for k,v in adapter.prepare_training_batch(raw_dev).items()}
    @jax.jit
    def predict(ap,z,p,r):return actor.sample_action(ap,jax.random.PRNGKey(0),z,p,r,deterministic=True)
    @jax.jit
    def q(cp,z,p,actions):return jnp.stack(critic.q_values(cp,z,p,actions),-1)
    def evaluate(state,rows,raw,batch,expert_only=False):
        pred=predict(state.actor_params,batch['z_rl'],batch['proprio'],batch['ref_chunk'])
        values=adapter.denormalize_to_abs_chunk(np.asarray(pred),raw['proprio'])
        qr=np.asarray(q(state.critic_params,batch['z_rl'],batch['proprio'],batch['action_chunk']))
        qa=np.asarray(q(state.critic_params,batch['z_rl'],batch['proprio'],pred))
        groups={}
        for i,r in enumerate(rows):groups.setdefault(int(r['episode_id']),[]).append(i)
        result=[]
        for ep,ids in sorted(groups.items()):
            expert=ep<0 or ep>=100000
            if expert_only and not expert:continue
            steps={}
            for i in sorted(ids,key=lambda i:int(rows[i]['step_id'])):
                for slot in range(cfg.chunk_len):steps[int(rows[i]['step_id'])+slot]=(i,slot)
            positions=list(steps.values());ii=np.asarray([x[0] for x in positions]);jj=np.asarray([x[1] for x in positions])
            mask=np.isin(raw['source_chunk'][ii,jj],[2,3]);actual=raw['action_chunk'][ii,jj];ref=raw['ref_chunk'][ii,jj]
            errors=np.abs(values[ii,jj]-actual);hilwindow=np.isin(raw['source_chunk'][ids],[2,3]).any(1)
            result.append({'episode_id':ep,'expert':expert,'complete':bool(raw['done'][ids].any()),
                'unique_steps':len(positions),'hil_steps':int(mask.sum()),
                'human_mae_per_dim':errors[mask].mean(0).tolist() if mask.any() else None,
                'reference_human_mae_per_dim':np.abs(ref-actual)[mask].mean(0).tolist() if mask.any() else None,
                'human_p95_per_dim':np.quantile(errors[mask],.95,axis=0).tolist() if mask.any() else None,
                'step_p95_per_dim':np.quantile(np.abs(np.diff(values[ii,jj],axis=0)),.95,axis=0).tolist(),
                'hil_q1_recorded_minus_actor':float((qr[ids,0]-qa[ids,0])[hilwindow].mean()) if hilwindow.any() else None})
        return result
    def assess(state):return {'development':evaluate(state,devrows,raw_dev,bd),
                             'expert_retention':evaluate(state,trainrows,raw_train,fulltrain,True)}
    report={'source_sha256':hashes,'config':config,'initial_step':int(initial.global_step),'initial_actor':int(initial.actor_version),
        'train_journal_records':len(trainrows),'optimization_records':len(train_ds['z_rl']),'internal_transition_validation_records':len(val_ds['z_rl']),
        'train_episodes':len({int(r['episode_id']) for r in trainrows}),'development_episodes':len({int(r['episode_id']) for r in devrows}),
        'split':'External 20-Episode reused development; all expert retention Episodes previously trained. No independent test.',
        'single_factor':'reference_dropout_prob .5 versus 0; native Warmup BC10/Q.1/delta10, seed-paired indices, full optimizer/targets/init preserved.',
        'updates':a.updates,'batch':128,'native_offline_step':str(source),'baseline':assess(initial),'runs':[],
        'boundary':'Does not repair reference time base or HIL command identity. No robot performance or online-improvement claim.'}
    def save():(a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    save()
    for seed in [41,42,43]:
        indices=np.random.default_rng(seed).integers(0,len(train_ds['z_rl']),size=(a.updates,128))
        for probability in [.5,0.]:
            run_cfg=dataclasses.replace(cfg,reference_dropout_prob=probability)
            update=native._make_train_step(actor,critic,run_cfg,bc_weight=cfg.warmup_bc_weight,q_weight=cfg.warmup_q_weight,
                delta_weight=cfg.delta_weight,disable_ref_input=False,use_action_adapter=True,
                action_q01=jnp.asarray(adapter.stats.q01),action_q99=jnp.asarray(adapter.stats.q99))
            state=initial.replace(rng=jax.random.PRNGKey(seed));started=time.time();curve=[]
            for step,ids in enumerate(indices):
                state,metrics=update(state,{k:v[ids] for k,v in bt.items()})
                if (step+1)%500==0 or step+1==a.updates:
                    m={k:float(v) for k,v in metrics.items()}
                    if not np.isfinite(list(m.values())).all():raise ValueError('nonfinite training')
                    evaluation=assess(state);curve.append({'updates':step+1,'metrics':m,'evaluation':evaluation})
                    print(json.dumps({'seed':seed,'dropout':probability,'updates':step+1,'elapsed':time.time()-started,
                        'development_joint_hil_mae':float(np.mean([np.mean(x['human_mae_per_dim'][:6]) for x in evaluation['development'] if x['hil_steps']]))}),flush=True)
            label='dropout%s_seed%s'%(probability,seed);folder=a.output/label;folder.mkdir()
            path=folder/'state.pkl'
            savedcfg=dict(config,reference_dropout_prob=probability)
            with path.open('wb') as f:pickle.dump({'research_only':True,'rl_config':savedcfg,
                'effective_training':{'bc_weight':cfg.warmup_bc_weight,'q_weight':cfg.warmup_q_weight,'delta_weight':cfg.delta_weight,'reference_dropout_prob':probability},
                'state':{k:trainer._tree_to_numpy(getattr(state,k)) for k in payload['state']}},f)
            report['runs'].append({'label':label,'seed':seed,'dropout':probability,'indices_sha256':hashlib.sha256(indices.tobytes()).hexdigest(),
                'curve':curve,'evaluation':evaluation,'checkpoint':str(path),'checkpoint_sha256':sha(path),'elapsed_sec':time.time()-started})
            save()
    assert hashes=={path:sha(Path(path)) for path in hashes},'read-only source changed'
    report.update(finished_at=time.time(),sources_unchanged=True,devices=[str(d) for d in jax.devices()]);save()


if __name__=='__main__':main()
