#!/usr/bin/env python3
"""Matched Actor-only supervision for legacy/retimed reference contracts.

Critic/target-Critic/optimizer remain unchanged. Calls the fixed native Actor
loss with BC10/Q0/delta10 and original reference dropout; no reward learning.
This is research adaptation, not a claimed faithful full Online warmup.
"""
import argparse,hashlib,importlib.util,json,os,pickle,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'third_party/openpi-rlt/rlt_online_rl/src'),str(ROOT/'third_party/openpi-rlt/packages/openpi-client/src')]
os.environ.setdefault('JAX_PLATFORMS','cpu')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=2000);a=p.parse_args()
    if a.output.exists():p.error('fresh output required')
    a.output.mkdir(parents=True)
    import numpy as np
    import jax,jax.numpy as jnp,optax
    from rlt_online_rl import trainer
    from rlt_online_rl.config import RLTOnlineRLConfig
    from rlt_online_rl.action_representation import ActionRepresentationAdapter
    from methods.openpi_rlt.plug_v3_yyshadow.stage1_action_metrics import interpolate_chunk
    source=ROOT/'third_party/openpi-rlt/rlt_online_rl/scripts/offline/offline_train_from_replay.py'
    sys.path.insert(0,str(source.parent));spec=importlib.util.spec_from_file_location('fixed_actor_loss',source)
    native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
    ckpt=a.root/'models/rlt/plug_v3_yyshadow/warmup-5000/checkpoints/latest.pkl'
    norm=a.root/'models/rlt/plug_v3_yyshadow/warmup-5000/action_norm_stats.json'
    payload=pickle.loads(ckpt.read_bytes());cfgdict=dict(payload['rl_config'],action_norm_stats_path=str(norm))
    cfg=RLTOnlineRLConfig(**cfgdict);adapter=ActionRepresentationAdapter.from_config(cfg)
    actor,critic=trainer._make_networks(cfg)
    initial=trainer.RLTTrainState(**{k:trainer._tree_to_jax(v) for k,v in payload['state'].items()},actor_tx=optax.adam(cfg.actor_lr),critic_tx=optax.adam(cfg.critic_lr))
    full=a.root/'outputs/offline-model-selection-20261005/stage1_full'
    receipt=json.loads((full/'report.json').read_text())
    if not receipt.get('complete'):raise ValueError('complete full-H50 cache required')
    dataset=Path(receipt['dataset']);split=json.loads((dataset/'splits.json').read_text())
    # Recreate exact archived expert nearest-20Hz frame grid from source times.
    import csv
    history=json.loads((a.root/'outputs/full-chain-audit-20261001/stage1_audit.json').read_text())
    # Source timestamp is frame index / 30 for the verified converted release.
    # Explicitly verify duration and step identity against all archived 120
    # expert rows, rather than assuming the same grid endpoints.
    journal=a.root/'outputs/rlt/plug_v3_yyshadow/history/warmup_20260925_trials/experts120_5000/replay/replay_journal.pkl'
    original={}
    with journal.open('rb') as f:
        while True:
            try:r=pickle.load(f)
            except EOFError:break
            if int(r['episode_id'])>=100000:original.setdefault(int(r['episode_id'])-100000,[]).append(r)
    # Timestamp float32 rounding affects nearest-grid ties. Use the authoritative
    # reconstruction positions saved by the expert time audit via a CPU helper.
    positions_file=a.root/'outputs/offline-model-selection-20261005/expert_frame_positions.json'
    positions=json.loads(positions_file.read_text())
    batches={};groups={};identities=[]
    for hz in [30,20]:
        for label,episodes in [('train',split['train']),('development',split['val'])]:
            rows=[]
            for ep in episodes:
                path=full/('episode_%03d.npz'%ep);cache=dict(np.load(path))
                item=next(e for e in receipt['episodes'] if e['episode']==ep)
                if hashlib.sha256(path.read_bytes()).hexdigest()!=item['cache_sha256']:raise ValueError('cache changed')
                pos=np.asarray(positions[str(ep)],int);n=len(pos)
                starts=sorted(set(range(0,n-9,10))|{n-10})
                for s in starts:
                    anchor=pos[s];target=cache['recorded_action'][pos[s:s+10]]
                    ref=cache['predicted_native'][anchor]
                    ref=ref[:10] if hz==30 else interpolate_chunk(ref,30,20,10)
                    state=cache['proprio'][anchor]
                    if label=='train':
                        existing=next(r for r in original[ep] if int(r['step_id'])==s)
                        if not np.array_equal(existing['action_chunk'],target) or not np.array_equal(existing['proprio'],state):raise ValueError('expert target identity mismatch')
                    rows.append({'episode':ep,'step':s,'z_rl':cache['z_rl'][anchor],'proprio':state,'ref_chunk':ref,
                        'action_chunk':target,'source_chunk':np.full(10,2,np.uint8)})
            raw={k:np.stack([r[k] for r in rows]) for k in ['z_rl','proprio','ref_chunk','action_chunk','source_chunk']}
            normalized=dict(raw,ref_chunk=adapter.normalize_ref_chunk(raw['ref_chunk'],raw['proprio']),
                action_chunk=adapter.normalize_ref_chunk(raw['action_chunk'],raw['proprio']))
            batches[hz,label]=(raw,{k:jnp.asarray(v) for k,v in normalized.items()})
            mapping={}
            for i,r in enumerate(rows):
                for slot in range(10):mapping.setdefault(r['episode'],{})[r['step']+slot]=(i,slot)
            groups[hz,label]=mapping
    if len(batches[30,'train'][0]['z_rl'])!=1186 or len(groups[30,'train'])!=120 or len(groups[30,'development'])!=14:raise ValueError('split identity')
    @jax.jit
    def predict(ap,z,p,ref):return actor.sample_action(ap,jax.random.PRNGKey(0),z,p,ref,deterministic=True)
    @jax.jit
    def update(state,batch):
        key,nextkey=jax.random.split(state.rng)
        def loss(ap):
            return native._custom_actor_loss(actor,ap,critic,state.critic_params,batch['z_rl'],batch['proprio'],batch['ref_chunk'],batch['action_chunk'],batch['source_chunk'],
                bc_weight=10.,q_weight=0.,delta_weight=cfg.delta_weight,reference_dropout_prob=cfg.reference_dropout_prob,
                disable_ref_input=False,rng=key,use_action_adapter=True,action_q01=jnp.asarray(adapter.stats.q01),action_q99=jnp.asarray(adapter.stats.q99),action_representation=cfg.action_representation)
        (value,metrics),grad=jax.value_and_grad(loss,has_aux=True)(state.actor_params)
        changes,optimizer=state.actor_tx.update(grad,state.actor_opt_state,state.actor_params)
        ap=optax.apply_updates(state.actor_params,changes)
        return state.replace(actor_params=ap,actor_opt_state=optimizer,target_actor_params=native.soft_update_targets(state.target_actor_params,ap,cfg.target_tau),actor_version=state.actor_version+1,rng=nextkey),dict(metrics,actor_loss=value)
    def evaluate(state,hz,label):
        raw,b=batches[hz,label];pred=np.asarray(predict(state.actor_params,b['z_rl'],b['proprio'],b['ref_chunk']))
        physical=adapter.denormalize_to_abs_chunk(pred,raw['proprio']);rows=[]
        for ep,mapping in sorted(groups[hz,label].items()):
            ids=np.asarray([v[0] for _,v in sorted(mapping.items())]);slots=np.asarray([v[1] for _,v in sorted(mapping.items())])
            delta=physical[ids,slots]-raw['action_chunk'][ids,slots];reference=raw['ref_chunk'][ids,slots]-raw['action_chunk'][ids,slots]
            rows.append({'episode':ep,'unique_steps':len(ids),'mae_per_dim':np.abs(delta).mean(0).tolist(),'p95_abs_per_dim':np.quantile(np.abs(delta),.95,axis=0).tolist(),
                'bias_per_dim':delta.mean(0).tolist(),'reference_mae_per_dim':np.abs(reference).mean(0).tolist()})
        return rows
    def assess(state,hz):return {'train':evaluate(state,hz,'train'),'development':evaluate(state,hz,'development')}
    def treehash(tree):
        h=hashlib.sha256()
        for leaf in jax.tree.leaves(tree):v=np.asarray(leaf);h.update(str(v.dtype).encode());h.update(str(v.shape).encode());h.update(v.tobytes())
        return h.hexdigest()
    frozen_keys=['critic_params','target_critic_params','critic_opt_state','global_step'];frozen={k:treehash(getattr(initial,k)) for k in frozen_keys}
    report={'native_loss_source':str(source),'native_loss_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'initial_checkpoint_sha256':hashlib.sha256(ckpt.read_bytes()).hexdigest(),'norm_sha256':hashlib.sha256(norm.read_bytes()).hexdigest(),
        'actual_config':cfgdict,'effective_loss':{'BC':10.,'Q':0.,'delta':cfg.delta_weight,'reference_dropout':cfg.reference_dropout_prob},
        'single_paired_factor':'reference_hz30 vs20; all data/features/targets/seed/index/budget/init and Actor-only objective identical',
        'train_episodes':120,'train_transitions':1186,'development_episodes':14,'updates':a.updates,'batch_size':128,
        'input_identity':receipt['identity'],'frozen_state_sha256':frozen,'baseline':{str(hz):assess(initial,hz) for hz in [30,20]},'runs':[],
        'boundary':['Supervised Actor-only adaptation, not native coupled Warmup/Online RL training.',
            'All 134 Episodes trained by Stage1. 14 were excluded from Warmup expert optimization but now are reused Actor-development; no independent full-pipeline test.',
            'Features and new references share recorded training-loader numeric contract; actual service cached-action parity remains a separate failed diagnostic. Three independent real-input service restores are exact.',
            'No rollout references fabricated: old rollout lacks fullH50 and image-anchor identity, so old20dev cannot evaluate this new20 deployment contract.',
            'Critic unchanged; better expert action fit cannot establish correct RL action preference or autonomous success.']}
    def save():(a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    save()
    for seed in [41,42,43]:
        indices=np.random.default_rng(seed).integers(1186,size=(a.updates,128))
        for hz in [30,20]:
            state=initial.replace(rng=jax.random.PRNGKey(seed));curve=[];started=time.time();raw,bt=batches[hz,'train']
            for step,ids in enumerate(indices):
                state,metrics=update(state,{k:v[ids] for k,v in bt.items()})
                if (step+1)%500==0 or step+1==a.updates:
                    m={k:float(v) for k,v in metrics.items()}
                    if not np.isfinite(list(m.values())).all():raise ValueError('nonfinite adaptation')
                    evaluation=assess(state,hz);curve.append({'actor_updates':step+1,'metrics':m,'evaluation':evaluation})
                    print(json.dumps({'seed':seed,'reference_hz':hz,'actor_updates':step+1,'development_joint_mae':float(np.mean([np.mean(r['mae_per_dim'][:6]) for r in evaluation['development']])), 'development_gripper_mae':float(np.mean([r['mae_per_dim'][-1] for r in evaluation['development']]))}),flush=True)
            if any(treehash(getattr(state,k))!=v for k,v in frozen.items()):raise ValueError('frozen Critic/target/optimizer/step changed')
            folder=a.output/('reference%s_seed%s'%(hz,seed));folder.mkdir();path=folder/'state.pkl'
            with path.open('wb') as f:pickle.dump({'research_only':True,'rl_config':cfgdict,'input_contract':{'reference_native_hz':hz,'actor_execution_hz':20},
                'effective_training':report['effective_loss'],'critic_frozen':True,'actual_additional_actor_updates':a.updates,
                'state':{k:trainer._tree_to_numpy(getattr(state,k)) for k in payload['state']}},f)
            report['runs'].append({'seed':seed,'reference_hz':hz,'curve':curve,'evaluation':evaluation,'critic_state_unchanged':True,
                'global_critic_step':int(state.global_step),'actor_version':int(state.actor_version),'indices_sha256':hashlib.sha256(indices.tobytes()).hexdigest(),
                'checkpoint':str(path),'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'elapsed_seconds':time.time()-started});save()
    for run in report['runs']:
        delta=np.asarray([np.asarray(r['mae_per_dim'])-np.asarray(b['mae_per_dim']) for r,b in zip(run['evaluation']['development'],report['baseline']['30']['development'])])
        picks=np.random.default_rng(42).integers(len(delta),size=(20000,len(delta)))
        run['paired_vs_legacy_5k']={'per_dim_delta':delta.mean(0).tolist(),'joint_delta':float(delta[:,:6].mean()),'joint_ci95':np.quantile(delta[picks][:,:,:6].mean((1,2)),[.025,.975]).tolist(),
            'gripper_delta':float(delta[:,-1].mean()),'gripper_ci95':np.quantile(delta[picks][:,:,-1].mean(1),[.025,.975]).tolist()}
    report['complete']=True;save()


if __name__=='__main__':main()
