#!/usr/bin/env python3
"""Isolated native RLT warmup with external Episode DEV and exact resume check."""
import argparse,dataclasses,hashlib,json,os,pickle,shutil,sys,time
from pathlib import Path
ap=argparse.ArgumentParser()
for key in ('assets','seed','output'):ap.add_argument('--'+key,type=Path,required=True)
ap.add_argument('--steps',type=int,default=10000);ap.add_argument('--eval-at',type=int,nargs='+',default=[500,1000,2000,5000,10000])
ap.add_argument('--delta-weight',type=float,default=10.)
a=ap.parse_args()
if a.steps<4 or a.delta_weight<0:raise ValueError('steps >=4 and nonnegative delta weight required')
root=Path(__file__).resolve().parents[1];upstream=a.assets/'third_party/openpi-rlt'
sys.path[:0]=[str(root),str(upstream/'rlt_online_rl/src'),str(upstream/'packages/openpi-client/src')]
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import numpy as np,jax,jax.numpy as jnp
from rlt_online_rl.config import RLTOnlineRLConfig,LearnerServiceConfig
from rlt_online_rl.replay import RLTTransition
from rlt_online_rl.trainer import LearnerService
from rlt_online_rl.networks import build_td_target
from methods.openpi_rlt.cobot_adapter.replay_precision import install_action_precision_patch
install_action_precision_patch(enabled=True)
if a.output.exists():raise FileExistsError('output must be new')
manifest=json.loads((a.seed/'manifest.json').read_text())
def read(split):
    path=a.seed/f'{split}_expert_journal.pkl'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['journals_sha256'][split]:raise ValueError('seed SHA mismatch')
    rows=[]
    with path.open('rb') as f:
        while True:
            try:rows.append(pickle.load(f))
            except EOFError:break
    if len(rows)!=manifest['counts'][split]:raise ValueError('seed count mismatch')
    return rows
rows={k:read(k) for k in ('train','dev')}
assert not {r['episode_uuid'] for r in rows['train']}&{r['episode_uuid'] for r in rows['dev']}
arrays={}
for split,records in rows.items():
    converted=[RLTTransition.from_mapping(r).to_numpy() for r in records]
    arrays[split]={k:np.stack([r[k] for r in converted]) for k in converted[0]}
class Source:
    def __init__(self):self.rng=np.random.default_rng(42);self.counts=np.zeros(len(rows['train']),np.int64)
    def stats(self):return dict(size=len(rows['train']),adds_total=len(rows['train']),max_episode_id=-1,recent_episode_window=20)
    def sample_batch(self,n):
        ix=self.rng.integers(len(rows['train']),size=n);np.add.at(self.counts,ix,1)
        return {k:v[ix] for k,v in arrays['train'].items()}
a.output.mkdir(parents=True)
shutil.copyfile(a.seed/'norm_stats_delta.json',a.output/'action_norm_stats.json')
config=RLTOnlineRLConfig(action_representation='delta_chunk',action_norm_stats_path=str(a.output/'action_norm_stats.json'),
    gamma=.99,fixed_std=.002,reference_dropout_prob=.5,delta_weight=a.delta_weight,warmup_bc_weight=10.,warmup_q_weight=.1,
    online_bc_weight=5.,online_q_weight=.1,actor_lr=1e-4,critic_lr=1e-4,actor_update_period=2,
    warmup_min_size=len(rows['train']),warmup_post_collect_updates=a.steps,freeze_after_warmup=True,grad_updates_per_cycle=1)
svc=LearnerServiceConfig(sample_batch_size=128,checkpoint_dir=str(a.output/'checkpoints'),
    actor_snapshot_path=str(a.output/'actor_snapshot/actor_snapshot.pkl'),checkpoint_interval_steps=1000000,push_actor_interval_steps=1000000)
source=Source();learner=LearnerService(config,svc,source,rng=jax.random.PRNGKey(42),metrics_path=str(a.output/'metrics/learner.jsonl'))
(a.output/'config.json').write_text(json.dumps(dict(rl=dataclasses.asdict(config),service=dataclasses.asdict(svc),seed_manifest=manifest,sampling='native uniform transition; observed draws saved; no DEV training',backend=str(jax.devices())),indent=2))
actor,critic=learner._actor,learner._critic
@jax.jit
def evaluate_arrays(state,b):
    pred=actor.actor_mean(state.actor_params,b['z_rl'],b['proprio'],b['ref_chunk'])
    q_actual=jnp.stack(critic.q_values(state.critic_params,b['z_rl'],b['proprio'],b['action_chunk']),axis=-1)
    q_ref=jnp.stack(critic.q_values(state.critic_params,b['z_rl'],b['proprio'],b['ref_chunk']),axis=-1)
    q_actor=jnp.stack(critic.q_values(state.critic_params,b['z_rl'],b['proprio'],pred),axis=-1)
    target=build_td_target(actor,state.target_actor_params,critic,state.target_critic_params,
        b['next_z_rl'],b['next_proprio'],b['next_ref_chunk'],b['rewards'],b['done'],config.gamma,jax.random.PRNGKey(7))
    return pred,q_actual,q_ref,q_actor,target
evaluations=[]
def evaluate(step):
    result=dict(step=step,actor_version=int(learner.state.actor_version),cohorts={})
    for split,raw in arrays.items():
        predictions=[];qa=[];qr=[];qp=[];td=[]
        for off in range(0,len(rows[split]),128):
            part={k:v[off:off+128] for k,v in raw.items()};n=len(part['done'])
            padded={k:np.concatenate([v,np.repeat(v[-1:],128-n,axis=0)]) for k,v in part.items()}
            batch=learner._action_adapter.prepare_training_batch(padded)
            pred,qactual,qref,qactor,target=jax.device_get(evaluate_arrays(learner.state,{k:jnp.asarray(v) for k,v in batch.items()}))
            predictions.append(learner._action_adapter.denormalize_to_abs_chunk(pred,padded['proprio'])[:n])
            qa.append(qactual[:n]);qr.append(qref[:n]);qp.append(qactor[:n]);td.append(target[:n])
        pred=np.concatenate(predictions);actual=np.stack([r['action_chunk'] for r in rows[split]])
        ref=np.stack([r['ref_chunk'] for r in rows[split]]);err=np.abs(pred-actual);referr=np.abs(ref-actual)
        uuids=np.array([r['episode_uuid'] for r in rows[split]]);episode=[]
        for uid in sorted(set(uuids)):
            e=err[uuids==uid];re=referr[uuids==uid]
            episode.append(dict(uuid=uid,mae_per_dim=e.mean((0,1)).tolist(),p95_per_dim=np.percentile(e.reshape(-1,7),95,axis=0).tolist(),reference_mae_per_dim=re.mean((0,1)).tolist()))
        means=np.mean([r['mae_per_dim'] for r in episode],axis=0);tails=np.mean([r['p95_per_dim'] for r in episode],axis=0)
        qa,qr,qp,td=map(np.concatenate,(qa,qr,qp,td))
        result['cohorts'][split]=dict(episodes=episode,joint_mae_rad=float(means[:6].mean()),gripper_mae_m=float(means[6]),
            joint_p95_rad=float(tails[:6].mean()),q_actual_mean=qa.mean(0).tolist(),q_ref_mean=qr.mean(0).tolist(),q_actor_mean=qp.mean(0).tolist(),
            paired_actual_minus_ref= (qa-qr).mean(0).tolist(),td_target_mean=float(td.mean()),
            q_outside_observed_return_range=float(np.mean((qa<0)|(qa>config.gamma**9))),q_max=float(qa.max()),q_min=float(qa.min()),
            boundary='complete-Episode equal imitation errors; all experts successful; no failure AUC or autonomous success inference')
        np.savez_compressed(a.output/f'{split}-step-{step}.npz',actions=pred,q_actual=qa,q_ref=qr,q_actor=qp,td_target=td,uuid=uuids)
    evaluations.append(result);(a.output/'evaluations.json').write_text(json.dumps(evaluations,indent=2))
    print('EVALUATION '+json.dumps({k:v for k,v in result.items() if k!='cohorts'})+' '+json.dumps({k:{n:v[n] for n in ('joint_mae_rad','gripper_mae_m','q_max')} for k,v in result['cohorts'].items()}),flush=True)
evaluate(0)
started=time.perf_counter()
# Check complete learner state restoration and independent sampler state before long training.
for _ in range(2):assert learner.train_once() is not None
learner.save_checkpoint();rng_state=source.rng.bit_generator.state
for _ in range(2):assert learner.train_once() is not None
expected=jax.device_get(learner.state)
restored_source=Source();restored_source.rng.bit_generator.state=rng_state
restored=LearnerService(config,svc,restored_source,rng=jax.random.PRNGKey(42),metrics_path=str(a.output/'metrics/resume-check.jsonl'))
for _ in range(2):assert restored.train_once() is not None
for x,y in zip(jax.tree.leaves(expected),jax.tree.leaves(jax.device_get(restored.state)),strict=True):
    np.testing.assert_array_equal(x,y)
(a.output/'resume-verification.json').write_text(json.dumps(dict(status='passed',comparison='4 uninterrupted vs 2 saved + 2 restored, all state leaves exact; sampler RNG restored explicitly'),indent=2))
# Continue the uninterrupted learner; restored service never publishes or receives live Replay.
del restored
for step in range(5,a.steps+1):
    metric=learner.train_once()
    if metric is None or not all(np.isfinite(v) for v in metric.values()):raise ValueError('nonfinite/missing update')
    if step in a.eval_at or step==a.steps:
        learner.save_checkpoint();learner.export_actor_snapshot(force=True)
        shutil.copyfile(a.output/'actor_snapshot/actor_snapshot.pkl',a.output/f'actor_snapshot/step_{step}.pkl')
        (a.output/'checkpoints'/f'sampler_{step}.json').write_text(json.dumps(source.rng.bit_generator.state))
        evaluate(step)
    if step%100==0:print(json.dumps(dict(step=step,elapsed_sec=time.perf_counter()-started)),flush=True)
np.save(a.output/'actual_training_draws.npy',source.counts)
(a.output/'completed.json').write_text(json.dumps(dict(status='completed',steps=a.steps,seconds=time.perf_counter()-started,
    boundary='offline warmup only, candidate selection and deployment acceptance required'),indent=2))
