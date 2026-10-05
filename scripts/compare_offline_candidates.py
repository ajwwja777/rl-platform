#!/usr/bin/env python3
"""Paired whole-Episode comparison of retained models on reused rollout development.

No optimization or asset modification. Most recent available anchor is used once
per logical step, so overlapping replay windows do not multiply HIL targets.
"""
import argparse,hashlib,json,os,pickle,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'third_party/openpi-rlt/rlt_online_rl/src')]
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('fresh report directory required')
    a.output.mkdir(parents=True)
    import numpy as np
    import jax,jax.numpy as jnp,optax
    from rlt_online_rl import trainer
    from rlt_online_rl.config import RLTOnlineRLConfig
    from rlt_online_rl.action_representation import ActionRepresentationAdapter
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    def read(path):
        rows=[]
        with path.open('rb') as f:
            while True:
                try:rows.append(pickle.load(f))
                except EOFError:break
        return rows
    def save(report):(a.output/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    hist=a.root/'outputs/rlt/plug_v3_yyshadow/history'
    baseline=hist/'warmup_20260925_trials/experts120_5000/checkpoints/latest.pkl'
    journal=hist/'warmup_20260924_v1/holdout/replay/replay_journal.pkl'
    trainjournal=baseline.parent.parent/'replay/replay_journal.pkl'
    payload=pickle.loads(baseline.read_bytes());cfgdict=dict(payload['rl_config'])
    norm=a.root/'models/rlt/plug_v3_yyshadow/warmup-5000/action_norm_stats.json'
    cfgdict['action_norm_stats_path']=str(norm);cfg=RLTOnlineRLConfig(**cfgdict)
    adapter=ActionRepresentationAdapter.from_config(cfg);actor,critic=trainer._make_networks(cfg)
    records=read(journal);trainrows=read(trainjournal)
    trainids={int(r['episode_id']) for r in trainrows};devids={int(r['episode_id']) for r in records}
    if trainids&devids:raise ValueError('external development leaked into baseline training journal')
    raw={k:np.stack([r[k] for r in records]) for k in records[0] if k!='collection_phase'}
    batch=adapter.prepare_training_batch(raw);b={k:jnp.asarray(v) for k,v in batch.items()}
    @jax.jit
    def predict(ap):return actor.sample_action(ap,jax.random.PRNGKey(0),b['z_rl'],b['proprio'],b['ref_chunk'],deterministic=True)
    @jax.jit
    def q(cp,action):
        q1,q2=critic.q_values(cp,b['z_rl'],b['proprio'],action);return jnp.stack([q1,q2,jnp.minimum(q1,q2)],-1)
    groups={}
    for i,r in enumerate(records):groups.setdefault(int(r['episode_id']),[]).append(i)
    # Select freshest recorded anchor once per physical logical step.
    chosen={}
    for ep,ids in groups.items():
        steps={}
        for i in sorted(ids,key=lambda i:int(records[i]['step_id'])):
            for slot in range(cfg.chunk_len):steps[int(records[i]['step_id'])+slot]=(i,slot)
        chosen[ep]=list(sorted(steps.values(),key=lambda x:int(records[x[0]]['step_id'])+x[1]))
    def measure(pred,qr=None,qa=None):
        result=[]
        for ep,ids in sorted(groups.items()):
            positions=chosen[ep];idx=np.asarray([x[0] for x in positions]);slot=np.asarray([x[1] for x in positions])
            human=np.isin(raw['source_chunk'][idx,slot],[2,3]);reference=raw['ref_chunk'][idx,slot]
            actual=raw['action_chunk'][idx,slot];values=pred[idx,slot]
            e=np.abs(values-actual);re=np.abs(reference-actual)
            correction=values-reference;desired=actual-reference
            hv=correction[human,:6];ht=desired[human,:6]
            denom=np.linalg.norm(hv,axis=1)*np.linalg.norm(ht,axis=1)
            selected=denom>1e-12
            row={'episode_id':ep,'windows':len(ids),'unique_steps':len(positions),'unique_hil_steps':int(human.sum()),
                'complete_replay_episode':bool(raw['done'][ids].any()),'success':bool(raw['success'][ids].max()),
                'assisted':bool(human.any()),'human_mae_per_dim':e[human].mean(0).tolist() if human.any() else None,
                'reference_human_mae_per_dim':re[human].mean(0).tolist() if human.any() else None,
                'human_p95_per_dim':np.quantile(e[human],.95,axis=0).tolist() if human.any() else None,
                'correction_direction_cosine':float((hv[selected]*ht[selected]).sum(1).dot(1/denom[selected])/selected.sum()) if selected.any() else None,
                'correction_rms_per_dim':np.sqrt((correction**2).mean(0)).tolist(),
                'sequence_step_p95_per_dim':np.quantile(np.abs(np.diff(values,axis=0)),.95,axis=0).tolist()}
            if qr is not None:
                row['q_recorded_mean']=qr[ids].mean(0).tolist();row['q_actor_mean']=qa[ids].mean(0).tolist()
                hmask=np.isin(raw['source_chunk'][ids],[2,3]).any(1)
                row['hil_recorded_q1_minus_actor']=float((qr[ids,0]-qa[ids,0])[hmask].mean()) if hmask.any() else None
            result.append(row)
        return result
    report={'baseline_checkpoint':str(baseline),'baseline_sha256':sha(baseline),'development_journal':str(journal),
        'development_sha256':sha(journal),'training_journal_sha256':sha(trainjournal),'norm_sha256':sha(norm),
        'actual_config':cfgdict,'development_episodes':sorted(devids),'training_overlap_episodes':[],
        'split':'20 rollout Episodes excluded from baseline optimization, repeatedly used for historical model selection; development only.',
        'reference':measure(raw['ref_chunk']),'models':[],
        'boundary':['HIL target is historical measured feedback, not verified optimal human command.',
          'Actor proposals recomputed at cached anchor states, not historical takeover proposals.',
          'Unique logical steps counted once; uncertainty bootstraps complete Episodes.',
          'Autonomous/failure reference imitation and Critic rankings do not measure robot success.',
          'Continuation runs start from one shared Warmup5k; seeds measure continuation variation only.']}
    models=[('warmup5k',baseline)]
    for folder in ['full_pool','']:
        source=a.root/'outputs/model-repair-20261005'/folder/'research_checkpoints'
        for path in sorted(source.glob('*/state.pkl')):models.append(((folder or 'selective_pool')+'/'+path.parent.name,path))
    for label,path in models:
        d=pickle.loads(path.read_bytes());state=trainer.RLTTrainState(**{k:trainer._tree_to_jax(v) for k,v in d['state'].items()},
            actor_tx=optax.adam(cfg.actor_lr),critic_tx=optax.adam(cfg.critic_lr))
        pred=predict(state.actor_params);native=adapter.denormalize_to_abs_chunk(np.asarray(pred),raw['proprio'])
        qr=np.asarray(q(state.critic_params,b['action_chunk']));qa=np.asarray(q(state.critic_params,pred))
        rows=measure(native,qr,qa)
        if any(not r['complete_replay_episode'] for r in rows):raise ValueError('incomplete development Episode')
        report['models'].append({'label':label,'path':str(path),'sha256':sha(path),'learner_step':int(state.global_step),
            'actor_version':int(state.actor_version),'effective_training':d.get('effective_training'), 'episodes':rows})
        save(report);print(json.dumps({'model':label,'hil_episodes':sum(r['assisted'] for r in rows),
            'joint_hil_mae_episode_mean':float(np.mean([np.mean(r['human_mae_per_dim'][:6]) for r in rows if r['assisted']]))}),flush=True)
    baseline_rows={r['episode_id']:r for r in report['models'][0]['episodes']}
    for model in report['models']:
        delta=np.asarray([np.asarray(r['human_mae_per_dim'])-np.asarray(baseline_rows[r['episode_id']]['human_mae_per_dim'])
            for r in model['episodes'] if r['assisted']])
        picks=np.random.default_rng(42).integers(len(delta),size=(20000,len(delta)))
        bootstrap=delta[picks].mean(1)
        model['paired_vs_warmup5k']={'hil_episodes':len(delta),'per_dim_mean_delta':delta.mean(0).tolist(),
            'per_dim_ci95':np.quantile(bootstrap,[.025,.975],axis=0).tolist(),
            'joint_mean_delta':float(delta[:,:6].mean()),'joint_ci95':np.quantile(bootstrap[:,:6].mean(1),[.025,.975]).tolist()}
    report['devices']=[str(d) for d in jax.devices()];report['finished_at']=time.time();save(report)


if __name__=='__main__':main()
