#!/usr/bin/env python3
"""Matched native TD/MC x Actor-Q continuations on immutable cached inputs.

Research-only CPU study. All evaluation partitions are reused development data;
observed assisted returns are not autonomous counterfactual action values.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'third_party/openpi-rlt/rlt_online_rl/src'), str(ROOT/'third_party/openpi-rlt/packages/openpi-client/src')]
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=2000)
    parser.add_argument('--seeds', default='41,42,43')
    parser.add_argument('--retention-weight', type=float, default=0.)
    parser.add_argument('--variants', default='0:0.1,0.3:0.1,0:0,0.3:0')
    args = parser.parse_args()
    if args.output.exists(): parser.error('Use a fresh output directory')
    if args.updates < 1: parser.error('Positive update budget required')
    args.output.mkdir(parents=True)
    import numpy as np
    import jax
    import jax.numpy as jnp
    import optax
    from rlt_online_rl import trainer, networks
    from rlt_online_rl.config import RLTOnlineRLConfig
    from rlt_online_rl.action_representation import ActionRepresentationAdapter
    from rlt_online_rl.replay import ReplayBuffer
    from methods.openpi_rlt.experiments.credit import episode_credit
    from methods.openpi_rlt.experiments.retained_actor import make_retained_train_step
    from methods.openpi_rlt.experiments.target_attribution import reconstruct_observed_returns

    def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
    def save(name, value):
        target = args.output/name
        tmp = target.with_suffix(target.suffix+'.tmp')
        tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
        tmp.replace(target)

    assets = args.assets_root
    old = assets/'outputs/model-repair-20261005'
    checkpoint = assets/'outputs/rlt/plug_v3_yyshadow/history/warmup_20260925_trials/experts120_5000/checkpoints/latest.pkl'
    journal = checkpoint.parent.parent/'replay/replay_journal.pkl'
    input_file = old/'precision_inputs.npz'
    prior_file = old/'full_pool/training_comparison.json'
    norm = old/'full_pool/action_norm_stats.json'
    sources = {str(p): sha(p) for p in [checkpoint, journal, input_file, prior_file, norm]}
    raw = dict(np.load(input_file))
    rows = []
    with journal.open('rb') as stream:
        while True:
            try: rows.append(pickle.load(stream))
            except EOFError: break
    lookup = {(int(raw['episode_id'][i]), int(raw['step_id'][i])): i for i in range(len(raw['episode_id'])) if not raw['phase_online'][i]}
    full = {k: [] for k in raw}
    for row in rows:
        match = lookup.get((int(row['episode_id']), int(row['step_id'])))
        for k in raw:
            if k == 'original_action': v = raw[k][match] if match is not None else np.asarray(row['action_chunk'], np.float32)
            elif k == 'raw_verified': v = match is not None and bool(raw[k][match])
            elif k == 'phase_online': v = False
            elif k == 'replay_index': v = -1
            elif k == 'collection_phase_id': v = 1
            else: v = row[k]
            full[k].append(v)
    for i in np.flatnonzero(raw['phase_online']):
        for k in raw: full[k].append(raw[k][i])
    data = {k: np.stack(v) for k,v in full.items()}
    assert len(data['episode_id']) == 2891
    prior = json.loads(prior_file.read_text())
    excluded = {(False,3), (False,7)}
    train_keys = {tuple(k) for k in prior['train_episodes']} - excluded
    dev_keys = {tuple(k) for k in prior['dev_episodes']} - excluded
    groups = {}
    for i in range(len(data['episode_id'])):
        groups.setdefault((bool(data['phase_online'][i]), int(data['episode_id'][i])), []).append(i)
    train = np.asarray([i for k in sorted(train_keys) for i in groups[k]], np.int64)
    assert not train_keys & dev_keys
    assert all(data['raw_verified'][groups[k]].all() for k in train_keys|dev_keys if k[0])
    payload = pickle.loads(checkpoint.read_bytes())
    config = dict(payload['rl_config'], action_norm_stats_path=str(norm))
    cfg = RLTOnlineRLConfig(**config)
    adapter = ActionRepresentationAdapter.from_config(cfg)
    actor, critic = trainer._make_networks(cfg)
    initial = trainer.RLTTrainState(**{k:trainer._tree_to_jax(v) for k,v in payload['state'].items()}, actor_tx=optax.adam(cfg.actor_lr), critic_tx=optax.adam(cfg.critic_lr))
    assert int(initial.global_step) == 5000 and int(initial.actor_version) == 2500
    observed, integrity = reconstruct_observed_returns(data, cfg.gamma)
    credit_rows = [{k:data[k][i] for k in ['collection_phase_id','episode_id','step_id','rewards','done','success']} for i in range(len(observed))]
    mc, valid = episode_credit(credit_rows, cfg.gamma)
    # The training credit seam must agree with the stricter overlap/source audit.
    valid &= np.isfinite(observed)
    np.testing.assert_allclose(mc[valid], observed[valid], rtol=2e-6, atol=1e-7)
    save('episode_integrity.json', dict(episodes=integrity, valid_mc_rows=int(valid.sum()), invalid_train_rows=int((~valid[train]).sum())))
    batch = adapter.prepare_training_batch(dict(data, action_chunk=data['original_action']))
    b = {k:jnp.asarray(v) for k,v in batch.items()}
    b.update(mc_return=jnp.asarray(mc), mc_valid=jnp.asarray(valid))
    human = np.isin(data['source_chunk'], [2,3])
    expert = (data['episode_id'] < 0) | (data['episode_id'] >= 100000)
    canonical = np.where(data['episode_id'][train]>=100000, -(data['episode_id'][train]-100000+1), data['episode_id'][train])
    phase = data['collection_phase_id'][train]
    src = data['source_chunk'][train]
    recent = (phase == 2) & (canonical >= canonical.max()-19)
    retention = np.zeros(len(data['episode_id']),bool)
    for key,ids in groups.items():
        if key in train_keys and not key[0] and (expert[ids].all() or (not human[ids].any() and data['success'][ids].any())):
            retention[ids]=True
    b['retention_mask']=jnp.asarray(retention)
    human_pool = np.asarray(data['intervention_flag'][train], bool)|np.isin(data['source'][train],[2,3])|np.isin(src,[2,3]).any(1)
    @jax.jit
    def predict(params,z,p,r): return actor.sample_action(params,jax.random.PRNGKey(0),z,p,r,deterministic=True)
    @jax.jit
    def q(params,z,p,a):
        q1,q2=critic.q_values(params,z,p,a)
        return jnp.stack([q1,q2,jnp.minimum(q1,q2)],-1)
    @jax.jit
    def td(state):
        return networks.build_td_target(actor,state.target_actor_params,critic,state.target_critic_params,b['next_z_rl'],b['next_proprio'],b['next_ref_chunk'],b['rewards'],b['done'],cfg.gamma,jax.random.PRNGKey(0))
    initial_prediction = predict(initial.actor_params,b['z_rl'],b['proprio'],b['ref_chunk'])
    initial_physical = np.asarray(adapter.denormalize_to_abs_chunk(np.asarray(initial_prediction), data['proprio']))

    def evaluate(state, name):
        pred = predict(state.actor_params,b['z_rl'],b['proprio'],b['ref_chunk'])
        physical = np.asarray(adapter.denormalize_to_abs_chunk(np.asarray(pred),data['proprio']))
        errors = np.abs(physical-data['original_action'])
        ref_error = np.abs(physical-data['ref_chunk'])
        change = np.abs(physical-initial_physical)
        qp = np.asarray(q(state.critic_params,b['z_rl'],b['proprio'],pred))
        qr = np.asarray(q(state.critic_params,b['z_rl'],b['proprio'],b['ref_chunk']))
        qrecord = np.asarray(q(state.critic_params,b['z_rl'],b['proprio'],b['action_chunk']))
        endpoint = jnp.where(jnp.asarray(human)[...,None],b['action_chunk'],pred)
        qh = np.asarray(q(state.critic_params,b['z_rl'],b['proprio'],endpoint))
        target = np.asarray(td(state))
        np.savez_compressed(args.output/(name+'.npz'), actor=physical, qactor=qp, qref=qr, qrecord=qrecord, qendpoint=qh, td=target)
        result = []
        for key,ids0 in sorted(groups.items()):
            if key in excluded: continue
            ids = np.asarray(ids0)
            mask = human[ids]
            h = mask.any(1)
            label = 'training' if key in train_keys else 'development' if key in dev_keys else 'historical_online_development'
            if label == 'historical_online_development' and not key[0]: continue
            kind = 'expert' if expert[ids].all() else 'assisted_success' if h.any() and data['success'][ids].any() else 'autonomous_success' if data['success'][ids].any() else 'failure'
            item = dict(phase='online' if key[0] else 'warmup', episode_id=key[1], split=label, outcome=kind, windows=len(ids), previously_seen_by_initial=not key[0],
                human_mae=errors[ids][mask].mean(0).tolist() if mask.any() else None,
                policy_reference_mae=ref_error[ids][~mask].mean(0).tolist() if (~mask).any() else None,
                recorded_mae=errors[ids].mean((0,1)).tolist(), change_from_initial=change[ids].mean((0,1)).tolist(),
                qrecord=qrecord[ids].mean(0).tolist(), qactor=qp[ids].mean(0).tolist(), qref=qr[ids].mean(0).tolist(),
                qrecord_minus_actor=(qrecord[ids]-qp[ids]).mean(0).tolist(),
                hil_endpoint_minus_actor=(qh[ids][h]-qp[ids][h]).mean(0).tolist() if h.any() else None,
                pure_hil_record_minus_actor=(qrecord[ids][mask.all(1)]-qp[ids][mask.all(1)]).mean(0).tolist() if mask.all(1).any() else None,
                td_mean=float(target[ids].mean()), observed_return_mean=float(observed[ids][np.isfinite(observed[ids])].mean()) if np.isfinite(observed[ids]).any() else None,
                predicted_joint_step_p95=float(np.quantile(np.abs(np.diff(physical[ids,:,:6],axis=1)),.95)))
            result.append(item)
        return result

    baseline = evaluate(initial,'baseline')
    np.savez_compressed(args.output/'data_identity.npz',episode_id=data['episode_id'],phase_online=data['phase_online'],collection_phase_id=data['collection_phase_id'],step_id=data['step_id'],success=data['success'],source_chunk=data['source_chunk'],human=human,expert=expert,observed=observed,mc_valid=valid,action=data['original_action'],reference=data['ref_chunk'])
    report = dict(status='running',source_sha256=sources,script_sha256=sha(Path(__file__)),code_head=os.popen('git -C '+str(ROOT)+' rev-parse HEAD').read().strip(),actual_config=config,devices=[str(d) for d in jax.devices()],seeds=args.seeds,updates=args.updates,batch_size=128,baseline=baseline,runs=[],
        train_episodes=[list(k) for k in sorted(train_keys)],dev_episodes=[list(k) for k in sorted(dev_keys)],training_records=len(train),development_records=sum(len(groups[k]) for k in dev_keys),excluded_wrong_prompt=[list(k) for k in sorted(excluded)],
        sampler=dict(strategy='stratified',recent_online_ratio=.4,warmup_demo_ratio=.3,human_intervention_ratio=.2,recent_episode_window=20),
        retention_weight=args.retention_weight,retention_rows=int(retention.sum()),retention_boundary='Immutable initialActor predictions on old expert/autonomous-success training Episodes only; same dropout and sample RNG. No DEV/old6 teacher data in gradients.',
        comparison='Matched TD/MC0.3 x current Actor Q0.1/0; same initial full5k state, seed, sampler indices, BC5/delta10, lr1e-4, budget. Q0 retains historical Adam momentum; it disables current Q gradients only.',
        selection='Fixed final2000 budget; intermediate500/1000 are learning-curve diagnostics, not independent selections. No production checkpoint promotion.',
        limitations=['All Warmup including development was seen by initial5k. All evaluation partitions are repeatedly reused development; independent test is absent.','Observed behavior returns include HIL. MC is off-policy observed credit, not autonomous value, corrected rewards, or proof human actions are optimal.','HIL endpoint is an unexecuted hybrid for mixed windows. Pure-HIL recorded chunks are reported separately. Actor replayed now is not the historical proposal.','Targets use verified raw feedback where available, otherwise stored archive actions. Received/executed HIL command optimality is not established.','Window slots can overlap within one Episode; confidence intervals must use complete Episode clusters, never independent windows.','Wrong-prompt3/7 excluded from continuation, but initial5k already saw them. Cached CPU feature study does not validate moved-target visual generalization or real-time execution.'])
    save('study.json',report)
    for seed in [int(x) for x in args.seeds.split(',')]:
        sampler=ReplayBuffer(len(train),seed=seed,sample_strategy='stratified',recent_episode_window=20,recent_online_ratio=.4,warmup_demo_ratio=.3,human_intervention_ratio=.2)
        sampler._size=len(train)
        sampler._storage={'collection_phase_id':phase,'episode_id':canonical,'source':data['source'][train],'source_chunk':src,'intervention_flag':data['intervention_flag'][train]}
        indices=np.stack([sampler._sample_stratified_indices(128) for _ in range(args.updates)])
        np.save(args.output/('indices_seed%d.npy'%seed),indices)
        for weight, qw in [tuple(float(v) for v in item.split(':')) for item in args.variants.split(',')]:
            variant='mc%s_q%s'%(str(weight).replace('.',''),str(qw).replace('.',''))
            state=initial.replace(rng=jax.random.PRNGKey(seed)); curve=[]; counts=np.zeros(len(train),np.int64); acounts=counts.copy()
            started=time.time(); run=make_retained_train_step(weight,initial.actor_params,args.retention_weight)
            for step, ids in enumerate(indices):
                np.add.at(counts,ids,1); before=int(state.actor_version)
                state,metrics=run(state,{k:v[train[ids]] for k,v in b.items()},actor=actor,critic=critic,rl_config=cfg,bc_weight=cfg.online_bc_weight,q_weight=qw,delta_weight=cfg.delta_weight,use_action_adapter=True,action_q01=jnp.asarray(adapter.stats.q01),action_q99=jnp.asarray(adapter.stats.q99))
                if int(state.actor_version)>before: np.add.at(acounts,ids,1)
                if (step+1)%500==0 or step+1==args.updates:
                    values={k:float(v) for k,v in metrics.items()}
                    assert all(np.isfinite(list(values.values())))
                    name='%s_seed%d_u%d'%(variant,seed,step+1)
                    curve.append(dict(updates=step+1,metrics=values,evaluation=evaluate(state,name),snapshot=name+'.npz'))
                    print(json.dumps(dict(seed=seed,variant=variant,updates=step+1,elapsed_sec=time.time()-started)),flush=True)
            folder=args.output/'research_checkpoints'/('%s_seed%d'%(variant,seed)); folder.mkdir(parents=True)
            path=folder/'state.pkl'
            with path.open('wb') as stream:
                pickle.dump(dict(research_only=True,initial_sha256=sources[str(checkpoint)],rl_config=dict(config,online_q_weight=qw),experiment=dict(mc_weight=weight,actor_q_weight=qw,retention_weight=args.retention_weight,retention_dimensions=list(range(6)),teacher_checkpoint=str(checkpoint),resume_boundary='Native state is preserved; exact continuation requires the same private MC/active6 retention helper and immutable teacher. No production/native-default resume is certified.'),state={k:trainer._tree_to_numpy(getattr(state,k)) for k in payload['state']}),stream)
            report['runs'].append(dict(seed=seed,variant=variant,mc_weight=weight,actor_q_weight=qw,learner_step=int(state.global_step),actor_version=int(state.actor_version),elapsed_sec=time.time()-started,index_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),curve=curve,evaluation=curve[-1]['evaluation'],checkpoint=str(path),checkpoint_sha256=sha(path),critic_draws=int(counts.sum()),actor_draws=int(acounts.sum()),actual_recent_ratio=float(counts[recent].sum()/counts.sum()),actual_online_ratio=float(counts[phase==2].sum()/counts.sum()),actual_warmup_ratio=float(counts[phase==1].sum()/counts.sum()),actual_hil_pool_ratio=float(counts[human_pool].sum()/counts.sum()),per_record_draws_file='indices_seed%d.npy'%seed))
            save('study.json',report)
    assert sources=={p:sha(Path(p)) for p in sources}
    report.update(status='completed',sources_unchanged=True,finished_at=time.time())
    save('study.json',report)


if __name__=='__main__': main()
