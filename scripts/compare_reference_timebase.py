#!/usr/bin/env python3
"""Factorial frozen Actor/reference diagnostic: input language and action time base.

Existing cached full-model forwards only, no optimization or robot dynamics.
Explicit checkpoint/norm and held-gripper flags support isolated current-candidate diagnostics.
"""
import argparse,hashlib,json,os,pickle,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'third_party/openpi-rlt/rlt_online_rl/src')]
os.environ.setdefault('JAX_PLATFORMS','cpu')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--training-cache-directory',default='stage1_full')
    p.add_argument('--serving-cache-directory',default='stage1_serving_prompt')
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--norm',type=Path)
    p.add_argument('--held-gripper',action='store_true')
    p.add_argument('--single-cache',type=Path,help='One verified cache; no prompt factorial or serving parity claim')
    a=p.parse_args()
    if bool(a.checkpoint) != bool(a.norm): p.error('--checkpoint and --norm must be supplied together')
    if a.output.exists():p.error('fresh output required')
    a.output.mkdir(parents=True)
    import numpy as np
    import jax,jax.numpy as jnp
    from rlt_online_rl import trainer
    from rlt_online_rl.config import RLTOnlineRLConfig
    from rlt_online_rl.action_representation import ActionRepresentationAdapter
    from methods.openpi_rlt.plug_v3_yyshadow.stage1_action_metrics import interpolate_chunk,time_targets,action_summary
    out=a.root/'outputs/offline-model-selection-20261005'
    roots=[out/a.training_cache_directory,out/a.serving_cache_directory]
    labels=['training_prompt','serving_prompt']
    if a.single_cache: roots=[a.single_cache,a.single_cache]; labels=['cached_prompt']
    reports=[json.loads((d/'report.json').read_text()) for d in roots]
    if not all(r.get('complete') for r in reports):raise ValueError('complete Stage1 forwards required')
    episodes=sorted(set(r['episode'] for r in reports[0]['episodes'])&set(r['episode'] for r in reports[1]['episodes']))
    ckpt=a.checkpoint or a.root/'models/rlt/plug_v3_yyshadow/warmup-5000/checkpoints/latest.pkl'
    payload=pickle.loads(ckpt.read_bytes());norm=a.norm or a.root/'models/rlt/plug_v3_yyshadow/warmup-5000/action_norm_stats.json'
    cfg=RLTOnlineRLConfig(**dict(payload['rl_config'],action_norm_stats_path=str(norm)))
    adapter=ActionRepresentationAdapter.from_config(cfg);actor,critic=trainer._make_networks(cfg)
    if a.held_gripper:
        from methods.openpi_rlt.experiments.held_gripper import HeldGripperCritic
        critic=HeldGripperCritic(critic,float(adapter.stats.q01[6]),float(adapter.stats.q99[6]))
    ap=trainer._tree_to_jax(payload['state']['actor_params'])
    cp=trainer._tree_to_jax(payload['state']['critic_params'])
    @jax.jit
    def predict(z,proprio,ref):return actor.sample_action(ap,jax.random.PRNGKey(0),z,proprio,ref,deterministic=True)
    @jax.jit
    def paired_q(z,proprio,ref,pred):
        r1,r2=critic.q_values(cp,z,proprio,ref)
        a1,a2=critic.q_values(cp,z,proprio,pred)
        return jnp.stack([r1,r2,jnp.minimum(r1,r2),a1,a2,jnp.minimum(a1,a2)],-1)
    audit=json.loads((a.root/'outputs/full-chain-audit-20261001/stage1_audit.json').read_text())
    splits={r['release_episode_index']:r['conversion_split'] for r in audit['converted_episodes']}
    report={'tool_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'checkpoint_path':str(ckpt),'norm_path':str(norm),'checkpoint_sha256':hashlib.sha256(ckpt.read_bytes()).hexdigest(),'norm_sha256':hashlib.sha256(norm.read_bytes()).hexdigest(),
        'cache_identities':[r['identity'] for r in reports],'actual_config':dict(payload['rl_config']),'global_step':int(payload['state']['global_step']),'actor_version':int(payload['state']['actor_version']),
        'held_gripper':a.held_gripper,'single_cache':str(a.single_cache) if a.single_cache else None,'episodes':episodes,'rows':[],'prompt_token_shift':[],
        'boundary':['All observations seen by Stage1; conversion val14 were excluded only from Warmup expert materialization.',
        'Same retained Actor weights for all comparisons; new reference/time/language inputs were not part of original Actor training.',
        'No physical dynamics, contact labels, independent test or autonomous robot-success measurement.',
        'Q is the unchanged retained Critic preference at each input, not calibrated autonomous value or a release gate.']}
    for ep in episodes:
        train=dict(np.load(roots[0]/('episode_%03d.npz'%ep)))
        serve=dict(np.load(roots[1]/('episode_%03d.npz'%ep)))
        if not np.array_equal(train['proprio'],serve['proprio']):raise ValueError('prompt comparison state drift')
        tz=train['z_rl'];sz=serve['z_rl']
        relative=np.linalg.norm(sz-tz,axis=1)/np.maximum(np.linalg.norm(tz,axis=1),1e-12)
        report['prompt_token_shift'].append({'episode':ep,'relative_l2_mean':float(relative.mean()),
            'relative_l2_p95':float(np.quantile(relative,.95))})
    for label,folder in zip(labels,roots):
        for ep in episodes:
            path=folder/('episode_%03d.npz'%ep);cache=dict(np.load(path))
            expected_hash=next(r['cache_sha256'] for r in reports[roots.index(folder)]['episodes'] if r['episode']==ep)
            if hashlib.sha256(path.read_bytes()).hexdigest()!=expected_hash: raise ValueError('cache SHA mismatch')
            recorded=cache['recorded_action'];state=cache['proprio'];z=cache['z_rl'];native=cache['predicted_native']
            truth,valid=time_targets(recorded,30,20,10)
            for timing,ref in [('legacy30_at20',native[:,:10]),('retimed20',interpolate_chunk(native,30,20,10))]:
                model_ref=adapter.normalize_ref_chunk(ref,state)
                pred=predict(jnp.asarray(z),jnp.asarray(state),jnp.asarray(model_ref))
                qs=np.asarray(paired_q(jnp.asarray(z),jnp.asarray(state),jnp.asarray(model_ref),pred))
                refined=adapter.denormalize_to_abs_chunk(np.asarray(pred),state)
                if a.held_gripper: refined[...,6]=state[:,None,6]
                report['rows'].append({'prompt':label,'timing':timing,'episode':ep,'frames':len(state),
                    'stage1_training_seen':True,'warmup_expert_training_seen':splits[ep]=='train',
                    'reference':action_summary(ref,truth,valid),'actor':action_summary(refined,truth,valid),
                    'q_reference_q1_q2_minq':qs[:,:3].mean(0).tolist(),
                    'q_actor_q1_q2_minq':qs[:,3:].mean(0).tolist(),
                    'paired_actor_minus_reference_q1':float((qs[:,3]-qs[:,0]).mean()),
                    'paired_actor_minus_reference_minq':float((qs[:,5]-qs[:,2]).mean()),
                    'actor_reference_mae_per_dim':np.abs(refined-ref).mean((0,1)).tolist()})
    rng=np.random.default_rng(42);picks=rng.integers(len(episodes),size=(20000,len(episodes)))
    lookup={(r['prompt'],r['timing'],r['episode']):r for r in report['rows']}
    report['summaries']=[]
    for model in ['reference','actor']:
        baseline=np.asarray([lookup[labels[-1],'legacy30_at20',ep][model]['mae_per_dim'] for ep in episodes])
        for prompt in labels:
            for timing in ['legacy30_at20','retimed20']:
                values=np.asarray([lookup[prompt,timing,ep][model]['mae_per_dim'] for ep in episodes]);delta=values-baseline
                report['summaries'].append({'model':model,'prompt':prompt,'timing':timing,'episodes':len(episodes),
                    'mae_per_dim':values.mean(0).tolist(),'joint_mae':float(values[:,:6].mean()),
                    'paired_delta_per_dim':delta.mean(0).tolist(),'paired_delta_joint':float(delta[:,:6].mean()),
                    'paired_delta_joint_ci95':np.quantile(delta[picks][:,:,:6].mean((1,2)),[.025,.975]).tolist(),
                    'paired_delta_gripper_ci95':np.quantile(delta[picks][:,:,-1].mean(1),[.025,.975]).tolist()})
    (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report['summaries'],indent=2))


if __name__=='__main__':main()
