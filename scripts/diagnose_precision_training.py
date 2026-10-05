"""Matched frozen and resumed diagnostics; cached features, no VLA or robot."""
import argparse, os, sys, json, pickle, hashlib, time
from pathlib import Path
import numpy as np

parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--upstream',type=Path,required=True);parser.add_argument('--checkpoint',type=Path,required=True)
parser.add_argument('--updates',type=int,default=2000);parser.add_argument('--qoff-only',action='store_true')
parser.add_argument('--full-warmup-pool',action='store_true');args=parser.parse_args()
sys.path[:0]=[str(args.upstream/'rlt_online_rl/src'),str(args.upstream/'packages/openpi-client/src')]
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import jax, jax.numpy as jnp, optax
from rlt_online_rl import trainer
from rlt_online_rl.config import RLTOnlineRLConfig
from rlt_online_rl.action_representation import ActionRepresentationAdapter

baseout=args.output;out=baseout/'full_pool' if args.full_warmup_pool else baseout
out.mkdir(exist_ok=True);data=dict(np.load(baseout/'precision_inputs.npz'))
if (out/'training_comparison.json').exists() and not args.qoff_only:
    raise FileExistsError('Use a fresh isolated output; do not overwrite completed experiments')
original_trusted_warmup=sorted(set(int(e) for e,o,v in zip(data['episode_id'],data['phase_online'],data['raw_verified']) if not o and v))
if args.full_warmup_pool:
    source=args.checkpoint.parent.parent/'replay/replay_journal.pkl'
    oldrows=[]
    with source.open('rb') as f:
        while True:
            try:oldrows.append(pickle.load(f))
            except EOFError:break
    lookup={(int(data['episode_id'][i]),int(data['step_id'][i])):i for i in range(len(data['episode_id'])) if not data['phase_online'][i]}
    combined={k:[] for k in data}
    for r in oldrows:
        matched=lookup.get((int(r['episode_id']),int(r['step_id'])))
        for k in data:
            if k=='original_action':value=data[k][matched] if matched is not None else np.asarray(r['action_chunk'],np.float32)
            elif k=='raw_verified':value=matched is not None and bool(data[k][matched])
            elif k=='phase_online':value=False
            elif k=='replay_index':value=-1  # Historical pool has its own identity.
            elif k=='collection_phase_id':value=1
            else:value=r[k]
            combined[k].append(value)
    for i in np.flatnonzero(data['phase_online']):
        for k in data:combined[k].append(data[k][i])
    data={k:np.stack(v) for k,v in combined.items()}
    (out/'pool_identity.json').write_text(json.dumps({'warmup_path':str(source),'warmup_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
      'warmup_rows':len(oldrows),'combined_rows':len(data['episode_id']),'fp32_recovered_windows':int(data['raw_verified'].sum()),
      'boundary':'Full original Warmup pool plus 10 trace-verified complete Online Episodes; raw precision recoverable for a subset only. No new assets or outcomes invented.'},indent=2))
payload=pickle.loads(args.checkpoint.read_bytes())
norm=json.loads((baseout.parent/'q-guidance-localization-20261005/data_contract_raw.json').read_text())['normalization']['stats']
(out/'action_norm_stats.json').write_text(json.dumps({'norm_stats':{'actions':norm}}))
config=dict(payload['rl_config']);config['action_norm_stats_path']=str(out/'action_norm_stats.json')
cfg=RLTOnlineRLConfig(**config);adapter=ActionRepresentationAdapter.from_config(cfg)
actor,critic=trainer._make_networks(cfg)
initial=trainer.RLTTrainState(**{k:trainer._tree_to_jax(v) for k,v in payload['state'].items()},actor_tx=optax.adam(cfg.actor_lr),critic_tx=optax.adam(cfg.critic_lr))
legacy=adapter.prepare_training_batch(data)
corrected=adapter.prepare_training_batch(dict(data,action_chunk=data['original_action']))
b={k:jnp.asarray(v) for k,v in legacy.items()};c={k:jnp.asarray(v) for k,v in corrected.items()}
groups={}
for i in range(len(data['episode_id'])):
    groups.setdefault((bool(data['phase_online'][i]),int(data['episode_id'][i])),[]).append(i)
trusted=[k for k,ids in groups.items() if data['raw_verified'][ids].all() or (args.full_warmup_pool and not k[0])]
warm=sorted(k for k in trusted if not k[0]);online=sorted(k for k in trusted if k[0])
devkeys=set([(False,ep) for ep in original_trusted_warmup[::7]]+online[::5]);trainkeys=set(trusted)-devkeys
train=np.asarray([i for k in sorted(trainkeys) for i in groups[k]])
dev=np.asarray([i for k in sorted(devkeys) for i in groups[k]])
oldcohort=np.flatnonzero(~data['raw_verified'])

def save(name,value):
    tmp=out/(name+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(out/name)

def interval(values):
    a=np.asarray(values,float)
    if len(a)<2:return None
    samples=np.random.default_rng(42).choice(a,(10000,len(a)),replace=True).mean(1)
    return {'episodes':len(a),'mean':float(a.mean()),'ci95':np.quantile(samples,[.025,.975]).tolist()}

@jax.jit
def predict(ap,z,p,r):return actor.sample_action(ap,jax.random.PRNGKey(0),z,p,r,deterministic=True)
@jax.jit
def q(cp,z,p,a):
    q1,q2=critic.q_values(cp,z,p,a);return jnp.stack([q1,q2,jnp.minimum(q1,q2)],-1)
@jax.jit
def grad_q(cp,z,p,a):return jax.grad(lambda x:critic.q_values(cp,z,p,x)[0].sum())(a)
@jax.jit
def targets(state,z,p,ref,rewards,done):
    rng=jax.random.split(state.rng)[0]
    a=actor.sample_action(state.target_actor_params,rng,z,p,ref,deterministic=False)
    v=q(state.target_critic_params,z,p,a)[...,2]
    return (rewards*jnp.power(cfg.gamma,jnp.arange(cfg.chunk_len))).sum(-1)+(1-done.astype(jnp.float32))*cfg.gamma**cfg.chunk_len*v

pred=predict(initial.actor_params,b['z_rl'],b['proprio'],b['ref_chunk'])
qlegacy=np.asarray(q(initial.critic_params,b['z_rl'],b['proprio'],b['action_chunk']))
qraw=np.asarray(q(initial.critic_params,b['z_rl'],b['proprio'],c['action_chunk']))
qa=np.asarray(q(initial.critic_params,b['z_rl'],b['proprio'],pred))
human=np.isin(data['source_chunk'],[2,3])
endpoint=pred+jnp.where(jnp.asarray(human)[...,None],c['action_chunk']-pred,0)
qe=np.asarray(q(initial.critic_params,b['z_rl'],b['proprio'],endpoint))
direction=np.asarray(grad_q(initial.critic_params,b['z_rl'],b['proprio'],pred))*np.asarray(endpoint-pred)
td=np.asarray(targets(initial,b['next_z_rl'],b['next_proprio'],b['next_ref_chunk'],b['rewards'],b['done']))
ep=[]
for key,ids in sorted(groups.items()):
    h=human[ids].any(1);hi=np.asarray(ids)[h]
    ep.append({'phase':'online' if key[0] else 'warmup','episode_id':key[1], 'windows':len(ids),
      'raw_verified':bool(data['raw_verified'][ids].all()),'split':'diagnostic_cohort' if key not in trusted else 'development' if key in devkeys else 'training',
      'success':bool(np.max(data['success'][ids])),'assisted':bool(h.any()),
      'q_quantization_change':(qraw[ids]-qlegacy[ids]).mean(0).tolist(),
      'human_endpoint_q1_change':float((qe[hi,0]-qa[hi,0]).mean()) if len(hi) else None,
      'q_gradient_toward_human':float(direction[hi].sum((1,2)).mean()) if len(hi) else None,
      'td_target_mean':float(td[ids].mean()),'recorded_q1_mean':float(qlegacy[ids,0].mean()),
      'td_rmse':float(np.sqrt(np.mean((qlegacy[ids,2]-td[ids])**2)))})
frozen={'checkpoint':str(args.checkpoint),'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
 'step':int(initial.global_step),'actor_version':int(initial.actor_version),'devices':[str(d) for d in jax.devices()],
 'config':config,'episodes':ep,'raw_precision_q1_effect':interval([e['q_quantization_change'][0] for e in ep if e['raw_verified']]),
 'raw_hil_endpoint_q1_change':interval([e['human_endpoint_q1_change'] for e in ep if e['raw_verified'] and e['assisted']]),
 'old_six_hil_endpoint_q1_change':interval([e['human_endpoint_q1_change'] for e in ep if not e['raw_verified'] and e['phase']=='online']),
 'limitations':['Targets use retained Warmup5k target networks and the next RNG split; these are diagnostic targets at this checkpoint, not reconstruction of historical sampled batches.',
 'HIL endpoint replaces only human slots in a deterministic offline Actor proposal. Historical contemporaneous proposals were not recorded.',
 'Raw HIL actions are measured feedback; full command/observation causal correspondence remains unverified.',
 'All data are historical training/development observations; no independent test or robot success measurement.']}
save('frozen_diagnosis.json',frozen)
np.savez_compressed(out/'frozen_values.npz',q_legacy=qlegacy,q_raw=qraw,q_actor=qa,q_endpoint=qe,td_target=td,q_direction=direction.sum((1,2)))
print(json.dumps({k:v for k,v in frozen.items() if k.endswith('change') or k=='raw_precision_q1_effect'}),flush=True)

def evaluate(state):
    pred=np.asarray(predict(state.actor_params,b['z_rl'],b['proprio'],b['ref_chunk']))
    native=adapter.denormalize_to_abs_chunk(pred,data['proprio'])
    errors=np.abs(native-data['original_action'])
    scores=np.asarray(q(state.critic_params,b['z_rl'],b['proprio'],jnp.asarray(pred)))
    end=jnp.asarray(pred)+jnp.where(jnp.asarray(human)[...,None],c['action_chunk']-jnp.asarray(pred),0)
    eh=np.asarray(q(state.critic_params,b['z_rl'],b['proprio'],end))
    result={}
    for split,keys in [('training',trainkeys),('development',devkeys)]:
        rows=[]
        for key in sorted(keys):
            ids=groups[key];mask=human[ids]
            rows.append({'phase':'online' if key[0] else 'warmup','episode_id':key[1],
             'human_per_dimension_mae':errors[ids][mask].mean(0).tolist() if mask.any() else None,
             'policy_reference_per_dimension_mae':np.abs(native[ids]-data['ref_chunk'][ids])[~mask].mean(0).tolist() if (~mask).any() else None,
             'human_q1_change':float((eh[ids,0]-scores[ids,0])[mask.any(1)].mean()) if mask.any() else None,
             'predicted_joint_step_p95':float(np.quantile(np.abs(np.diff(native[ids,:,:6],axis=1)),.95))})
        result[split]=rows
    diagnostic=[]
    for key,ids in sorted(groups.items()):
        if key in trusted: continue
        mask=human[ids]
        diagnostic.append({'episode_id':key[1], 'target_precision':'stored FP16; raw unavailable',
         'human_per_dimension_mae':errors[ids][mask].mean(0).tolist() if mask.any() else None,
         'human_q1_change':float((eh[ids,0]-scores[ids,0])[mask.any(1)].mean()) if mask.any() else None})
    result['historical_online_development']=diagnostic
    return result

report={'checkpoint_sha256':frozen['checkpoint_sha256'],'input_sha256':json.loads((baseout/'input_identity.json').read_text())['npz_sha256'],
 'updates':args.updates,'batch':128,'train_episodes':[list(k) for k in sorted(trainkeys)],'dev_episodes':[list(k) for k in sorted(devkeys)],
 'split_note':'Episode-separated continued-training comparison. Warmup development episodes were previously seen by the initial model. Online development episodes are repeatedly inspected historical data. No independent test.',
 'single_factor':'FP16 stored action versus trace-verified original FP32 action. Features, references, rewards, discount, initialization, optimizer, RNG, indices and update budget identical per seed.',
 'baseline':evaluate(initial),'runs':[]}
if args.qoff_only:
    report=json.loads((out/'training_comparison.json').read_text())
else:
    save('training_comparison.json',report)
for seed in [41,42,43]:
    ids=np.random.default_rng(seed).choice(train,(args.updates,128),replace=True)
    index_hash=hashlib.sha256(ids.tobytes()).hexdigest()
    variants=[('raw_fp32_qoff',c)] if args.qoff_only else [('legacy_fp16',b),('raw_fp32',c)]+([('raw_fp32_qoff',c)] if args.full_warmup_pool else [])
    for name,source in variants:
        state=initial.replace(rng=jax.random.PRNGKey(seed));start=time.time();curve=[]
        for step,indices in enumerate(ids):
            state,metrics=trainer.train_step(state,{k:v[indices] for k,v in source.items()},actor=actor,critic=critic,rl_config=cfg,
              bc_weight=cfg.online_bc_weight,q_weight=0. if name.endswith('qoff') else cfg.online_q_weight,delta_weight=cfg.delta_weight,
              use_action_adapter=True,action_q01=jnp.asarray(adapter.stats.q01),action_q99=jnp.asarray(adapter.stats.q99))
            if (step+1)%500==0:
                m={k:float(v) for k,v in metrics.items()};assert all(np.isfinite(list(m.values())))
                curve.append({'updates':step+1,'metrics':m,'evaluation':evaluate(state)})
                print(json.dumps({'variant':name,'seed':seed,'updates':step+1,'elapsed':time.time()-start}),flush=True)
        folder=out/'research_checkpoints'/('%s_seed%d'%(name,seed));folder.mkdir(parents=True,exist_ok=True)
        checkpoint=folder/'state.pkl'
        effective={'bc_weight':cfg.online_bc_weight,'q_weight':0. if name.endswith('qoff') else cfg.online_q_weight,
                   'delta_weight':cfg.delta_weight,'reference_dropout_prob':cfg.reference_dropout_prob,'fixed_std':cfg.fixed_std}
        saved_config=dict(config,online_q_weight=effective['q_weight'])
        with checkpoint.open('wb') as f:pickle.dump({'research_only':True,'rl_config':saved_config,'effective_training':effective,
          'state':{k:trainer._tree_to_numpy(getattr(state,k)) for k in payload['state']}},f)
        report['runs'].append({'variant':name,'seed':seed,'indices_sha256':index_hash,'elapsed_sec':time.time()-start,
         'learner_step':int(state.global_step),'actor_version':int(state.actor_version),'curve':curve,'evaluation':evaluate(state),
         'checkpoint':str(checkpoint),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'effective_training':effective})
        save('training_comparison.json',report)
report['finished_at']=time.time();save('training_comparison.json',report)

# Refresh all endpoints on exactly the same fixed objects, and check previous
# expert/scene fitting. These are previously trained observations, not tests.
retained_path=args.checkpoint.parent.parent/'replay/replay_journal.pkl'
retained=[]
with retained_path.open('rb') as f:
    while True:
        try:retained.append(pickle.load(f))
        except EOFError:break
oldraw={k:np.stack([r[k] for r in retained]) for k in retained[0] if k!='collection_phase'}
old=adapter.prepare_training_batch(oldraw)
oldgroups={}
for i,r in enumerate(retained):oldgroups.setdefault(int(r['episode_id']),[]).append(i)

def retention_evaluate(state):
    predictions=np.asarray(predict(state.actor_params,jnp.asarray(old['z_rl']),jnp.asarray(old['proprio']),jnp.asarray(old['ref_chunk'])))
    native=adapter.denormalize_to_abs_chunk(predictions,oldraw['proprio'])
    error=np.abs(native-oldraw['action_chunk'])
    rows=[]
    for ep,ids in sorted(oldgroups.items()):
        mask=np.isin(oldraw['source_chunk'][ids],[2,3])
        expert=ep>=100000 or ep<0
        rows.append({'episode_id':ep,'expert':expert,'previously_trained':True,
          'human_per_dimension_mae':error[ids][mask].mean(0).tolist() if mask.any() else None,
          'policy_reference_per_dimension_mae':np.abs(native[ids]-oldraw['ref_chunk'][ids])[~mask].mean(0).tolist() if (~mask).any() else None})
    return rows

fixed={'retention_journal':str(retained_path),'retention_sha256':hashlib.sha256(retained_path.read_bytes()).hexdigest(),
 'retention_note':'All Warmup episodes were seen by initial Warmup5k. Errors against original stored targets are fitting/retention proxies only. Storage is mixed: inspect dtype counts before stacking.',
 'retention_action_storage':{dtype:sum(str(np.asarray(r['action_chunk']).dtype)==dtype for r in retained)
                            for dtype in sorted({str(np.asarray(r['action_chunk']).dtype) for r in retained})},
 'initial':{'evaluation':evaluate(initial),'retention':retention_evaluate(initial)},'variants':[]}
for row in report['runs']:
    state_payload=pickle.loads(Path(row['checkpoint']).read_bytes())['state']
    state=trainer.RLTTrainState(**{k:trainer._tree_to_jax(v) for k,v in state_payload.items()},actor_tx=optax.adam(cfg.actor_lr),critic_tx=optax.adam(cfg.critic_lr))
    fixed['variants'].append({'variant':row['variant'],'seed':row['seed'],'evaluation':evaluate(state),'retention':retention_evaluate(state)})
    save('fixed_comparison.json',fixed)
save('fixed_comparison.json',fixed)
