#!/usr/bin/env python3
"""Trace retained Critic targets and action preferences; cached CPU inputs only.

Optional matched Critic-only refits diagnose fixed bootstrap versus observed
behavior targets. Actor stays byte-identical; no deployable model is produced.
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
sys.path[:0] = [str(ROOT), str(ROOT/'third_party/openpi-rlt/rlt_online_rl/src')]
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--norm', type=Path, required=True)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=500)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output directory; completed experiments cannot be overwritten')
    args.output.mkdir(parents=True)
    import numpy as np
    import jax
    import jax.numpy as jnp
    import optax
    from rlt_online_rl import trainer, networks
    from rlt_online_rl.config import RLTOnlineRLConfig
    from rlt_online_rl.action_representation import ActionRepresentationAdapter
    from methods.openpi_rlt.experiments.target_attribution import reconstruct_observed_returns, episode_interval

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    source_hashes = {str(p): digest(p) for p in [args.inputs, args.checkpoint, args.norm, args.comparison]}
    def save(name, payload):
        p = args.output/name
        tmp = p.with_suffix(p.suffix+'.tmp')
        tmp.write_text(json.dumps(payload, indent=2, allow_nan=False))
        tmp.replace(p)

    raw = dict(np.load(args.inputs))
    # Reproduce the previously validated full-Warmup comparison pool exactly.
    journal = args.checkpoint.parent.parent/'replay/replay_journal.pkl'
    source_hashes[str(journal)] = digest(journal)
    records = []
    with journal.open('rb') as stream:
        while True:
            try: records.append(pickle.load(stream))
            except EOFError: break
    lookup = {(int(raw['episode_id'][i]), int(raw['step_id'][i])): i for i in range(len(raw['episode_id'])) if not raw['phase_online'][i]}
    full = {k: [] for k in raw}
    for record in records:
        match = lookup.get((int(record['episode_id']), int(record['step_id'])))
        for key in raw:
            if key == 'original_action': value = raw[key][match] if match is not None else np.asarray(record['action_chunk'], np.float32)
            elif key == 'raw_verified': value = match is not None and bool(raw[key][match])
            elif key == 'phase_online': value = False
            elif key == 'replay_index': value = -1
            elif key == 'collection_phase_id': value = 1
            else: value = record[key]
            full[key].append(value)
    for i in np.flatnonzero(raw['phase_online']):
        for key in raw: full[key].append(raw[key][i])
    data = {k: np.stack(v) for k, v in full.items()}
    comparison = json.loads(args.comparison.read_text())
    trainkeys = {tuple(k) for k in comparison['train_episodes']}
    devkeys = {tuple(k) for k in comparison['dev_episodes']}
    keys = [(bool(online), int(ep)) for online, ep in zip(data['phase_online'], data['episode_id'])]
    trainmask = np.asarray([key in trainkeys for key in keys])
    devmask = np.asarray([key in devkeys for key in keys])
    oldmask = np.asarray([key not in trainkeys and key not in devkeys for key in keys])
    payload = pickle.loads(args.checkpoint.read_bytes())
    observed, integrity = reconstruct_observed_returns(data, float(payload['rl_config']['gamma']))
    save('episode_integrity.json', dict(episodes=integrity, windows=len(keys),
        finite_fields={key: bool(np.isfinite(data[key]).all()) for key in ['z_rl','proprio','action_chunk','ref_chunk','next_z_rl','next_proprio','next_ref_chunk','rewards']},
        phase_episode_step_duplicate_count=len(keys)-len(set((key, int(step)) for key,step in zip(keys,data['step_id']))),
        zero_action_rows=int(np.all(data['action_chunk']==0, axis=-1).sum()),
        available_observed_returns=int(np.isfinite(observed).sum())))
    config = dict(payload['rl_config'], action_norm_stats_path=str(args.norm))
    cfg = RLTOnlineRLConfig(**config)
    adapter = ActionRepresentationAdapter.from_config(cfg)
    batch = adapter.prepare_training_batch(data)
    b = {k: jnp.asarray(batch[k]) for k in ['z_rl','proprio','action_chunk','ref_chunk','next_z_rl','next_proprio','next_ref_chunk','rewards','done']}
    # Frozen residuals use original stored inputs: FP32 experts, FP16 rollouts.
    actor, critic = trainer._make_networks(cfg)
    state = trainer.RLTTrainState(**{k: trainer._tree_to_jax(v) for k,v in payload['state'].items()}, actor_tx=optax.adam(cfg.actor_lr), critic_tx=optax.adam(cfg.critic_lr))
    @jax.jit
    def predict(z,p,ref):
        return actor.sample_action(state.actor_params, jax.random.PRNGKey(0),z,p,ref,deterministic=True)
    @jax.jit
    def qs(cp,z,p,a):
        q1,q2=critic.q_values(cp,z,p,a)
        return jnp.stack([q1,q2,jnp.minimum(q1,q2)],axis=-1)
    @jax.jit
    def target(rng):
        return networks.build_td_target(actor,state.target_actor_params,critic,state.target_critic_params,
            b['next_z_rl'],b['next_proprio'],b['next_ref_chunk'],b['rewards'],b['done'],cfg.gamma,rng)
    proposal = predict(b['z_rl'],b['proprio'],b['ref_chunk'])
    human = np.isin(data['source_chunk'], [2,3])
    endpoint = jnp.where(jnp.asarray(human)[...,None], b['action_chunk'], proposal)
    qrecord = np.asarray(qs(state.critic_params,b['z_rl'],b['proprio'],b['action_chunk']))
    qactor = np.asarray(qs(state.critic_params,b['z_rl'],b['proprio'],proposal))
    qendpoint = np.asarray(qs(state.critic_params,b['z_rl'],b['proprio'],endpoint))
    td = np.asarray(target(jax.random.split(state.rng)[0]))
    td_draws = np.stack([np.asarray(target(jax.random.PRNGKey(seed))) for seed in [41,42,43,44]])
    chunk_reward = np.sum(data['rewards']*cfg.gamma**np.arange(cfg.chunk_len),axis=-1)
    expert = (data['episode_id']<0) | (data['episode_id']>=100000)
    window_hil = human.any(1)&~expert
    pure_hil = human.all(1)&~expert
    mixed_hil = human.any(1)&~human.all(1)&~expert
    groups = {}
    for i,key in enumerate(keys): groups.setdefault(key,[]).append(i)
    rows = []
    for key,ids in sorted(groups.items()):
        for kind,mask in [('expert',expert),('pure_hil',pure_hil),('mixed_hil',mixed_hil),('non_hil',~human.any(1)&~expert)]:
            selected = np.asarray(ids)[mask[ids]]
            if not len(selected): continue
            valid = selected[np.isfinite(observed[selected])]
            rows.append(dict(phase='online' if key[0] else 'warmup',episode_id=key[1],
                split='training' if key in trainkeys else 'development' if key in devkeys else 'reused_online_development',
                kind=kind, windows=len(selected), terminal_windows=int(data['done'][selected].sum()),
                target_mean=float(td[selected].mean()), observed_return_mean=float(observed[valid].mean()) if len(valid) else None,
                q_recorded_mean=qrecord[selected].mean(0).tolist(), q_actor_mean=qactor[selected].mean(0).tolist(),
                q_hil_endpoint_mean=qendpoint[selected].mean(0).tolist(),
                q1_recorded_minus_target=float((qrecord[selected,0]-td[selected]).mean()),
                q1_hil_endpoint_minus_actor=float((qendpoint[selected,0]-qactor[selected,0]).mean()),
                q1_full_recorded_minus_actor=float((qrecord[selected,0]-qactor[selected,0]).mean()),
                bootstrap_fraction=float(((td-chunk_reward)[selected]).mean()),
                target_noise_std_mean=float(td_draws[:,selected].std(0).mean())))
    def interval(values,mask):
        return episode_interval(values,data['episode_id'],data['collection_phase_id'],mask)
    report = dict(source_sha256=source_hashes, actual_config=config, devices=[str(d) for d in jax.devices()],
        code_head=os.popen('git -C '+str(ROOT)+' rev-parse HEAD').read().strip(),
        windows=len(keys), train_windows=int(trainmask.sum()), development_windows=int(devmask.sum()),
        reused_online_development_windows=int(oldmask.sum()), episodes=rows,
        frozen_summary={name:dict(q_recorded_minus_target=interval(qrecord[:,0]-td,mask),
            q_recorded_minus_observed=interval(qrecord[:,0]-observed,mask),
            hil_endpoint_minus_actor=interval(qendpoint[:,0]-qactor[:,0],mask),
            full_recorded_minus_actor=interval(qrecord[:,0]-qactor[:,0],mask),
            target_minus_observed=interval(td-observed,mask)) for name,mask in [('train_hil',trainmask&window_hil),('dev_hil',devmask&window_hil),('old_dev_hil',oldmask&window_hil),('train_pure_hil',trainmask&pure_hil),('terminal_hil',window_hil&data['done'])]},
        runs=[], limitations=[
            'No historical sampled target reconstruction; targets are evaluated at retained checkpoint and explicit RNG.',
            'Observed assisted returns do not establish autonomous value or optimal human actions.',
            'Mixed-slot HIL endpoint is an unexecuted hybrid action; full recorded action is analyzed separately.',
            'Logical step discount does not verify physical HIL time. No independent test or robot success measurement.',
            'Critic-only refits diagnose mechanisms; Actor and target networks remain frozen. They are not candidate training.'])
    save('target_attribution.json',report)
    np.savez_compressed(args.output/'frozen_values.npz',episode_id=data['episode_id'],phase_id=data['collection_phase_id'],
        step_id=data['step_id'],train=trainmask,dev=devmask,old_dev=oldmask,hil=window_hil,pure_hil=pure_hil,expert=expert,
        done=data['done'],td=td,observed=observed,qrecord=qrecord,qactor=qactor,qendpoint=qendpoint)
    train = np.flatnonzero(trainmask&np.isfinite(observed))
    if not len(train): raise ValueError('No verified return targets for a matched refit')
    @jax.jit
    def update(cp,opt,z,p,a,y):
        def loss(params):
            q1,q2=critic.q_values(params,z,p,a)
            return jnp.mean((q1-y)**2)+jnp.mean((q2-y)**2)
        loss_value,grads=jax.value_and_grad(loss)(cp)
        updates,newopt=state.critic_tx.update(grads,opt,cp)
        return optax.apply_updates(cp,updates),newopt,loss_value
    for seed in [41,42,43]:
        indices=np.random.default_rng(seed).choice(train,(args.updates,128),replace=True)
        for variant,y in [('fixed_bootstrap',td),('observed_behavior_return',observed)]:
            cp=state.critic_params; opt=state.critic_opt_state; curve=[]; started=time.time()
            target_y=jnp.asarray(np.nan_to_num(y),dtype=jnp.float32)
            for step,ids in enumerate(indices):
                cp,opt,loss=update(cp,opt,b['z_rl'][ids],b['proprio'][ids],b['action_chunk'][ids],target_y[ids])
                if (step+1)%100==0 or step+1==args.updates:
                    qr=np.asarray(qs(cp,b['z_rl'],b['proprio'],b['action_chunk']))
                    qa=np.asarray(qs(cp,b['z_rl'],b['proprio'],proposal))
                    item=dict(updates=step+1,loss=float(loss),summary={name:dict(
                        recorded_minus_actor=interval(qr[:,0]-qa[:,0],mask),
                        recorded_minus_td=interval(qr[:,0]-td,mask),
                        recorded_minus_observed=interval(qr[:,0]-observed,mask))
                        for name,mask in [('train_hil',trainmask&window_hil),('dev_hil',devmask&window_hil),('old_dev_hil',oldmask&window_hil)]})
                    curve.append(item)
                    print(json.dumps(dict(seed=seed,variant=variant,updates=step+1,loss=float(loss),elapsed=time.time()-started)),flush=True)
                    if not np.isfinite(loss): raise ValueError('Nonfinite diagnostic training')
            report['runs'].append(dict(seed=seed,variant=variant,indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
                updates=args.updates,actual_train_windows=len(train),elapsed_sec=time.time()-started,curve=curve,
                saved_weights=None,actor_modified=False,target_networks_modified=False))
            save('target_attribution.json',report)
    assert source_hashes=={path:digest(Path(path)) for path in source_hashes},'Read-only source changed'
    report.update(finished_at=time.time(),source_unchanged=True)
    save('target_attribution.json',report)


if __name__=='__main__':
    main()
