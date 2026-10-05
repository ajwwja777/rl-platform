#!/usr/bin/env python3
"""Read-only full Episode Stage1 forwards, fixed serving noise and physical units.

One per-Episode receipt/NPZ allows interruption and exact identity-checked resume.
All 134 observations were trained; these are fitting/development diagnostics.
"""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def save(p,value):
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(p)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--episodes',default='0-133')
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--resume',action='store_true')
    p.add_argument('--prompt',help='Explicit input-language ablation; omitted uses recorded training task.')
    p.add_argument('--inference-path',choices=('analysis','serving'),default='analysis',
        help='serving shares the actual service compiled function and requires batch1; analysis is a separate bulk JIT.')
    args=p.parse_args()
    if args.batch_size<1:p.error('positive batch size required')
    if args.inference_path=='serving' and args.batch_size!=1:p.error('actual serving sampler requires batch-size1')
    if args.output.exists() and not args.resume:p.error('fresh output or explicit identity-checked resume required')
    args.output.mkdir(parents=True,exist_ok=True)
    # Frozen environment requires PyArrow and Torch BEFORE JAX.
    from methods.openpi_rlt.plug_v3_yyshadow import serve_stage1
    jax,jnp,nnx,bridge,RLConfig,RLModel,builder,transforms,model_api,nnx_utils,training=serve_stage1.initialize(ROOT)
    import numpy as np
    import pyarrow.parquet as pq
    from methods.openpi_rlt.plug_v3_yyshadow.evaluate_stage1 import make_loader,parse_episodes
    from methods.openpi_rlt.plug_v3_yyshadow.stage1_loading import restore_inference_params
    from methods.openpi_rlt.plug_v3_yyshadow.stage1_action_metrics import action_summary,time_targets,interpolate_chunk,recover_uint8_image
    from methods.openpi_rlt.stage1_entry import prepare_environment
    prepare_environment(args.dataset,ROOT)
    episodes=parse_episodes(args.episodes)
    manifest=args.dataset/'conversion_manifest.json'
    norm=args.checkpoint/'assets'/args.dataset.name/'norm_stats.json'
    identity={'manifest':digest(manifest),'norm':digest(norm),'params_files':{
        str(x.relative_to(args.checkpoint)):digest(x) for x in sorted((args.checkpoint/'params').rglob('*')) if x.is_file()},
        'source_sha256':{str(x.relative_to(ROOT)):digest(x) for x in [Path(__file__),ROOT/'methods/openpi_rlt/plug_v3_yyshadow/stage1_action_metrics.py']},
        'batch_size':args.batch_size,'fixed_serving_noise_seed':42,'denoising_steps':10,'prompt_override':args.prompt,
        'inference_path':args.inference_path,'serving_source_sha256':digest(ROOT/'methods/openpi_rlt/plug_v3_yyshadow/serve_stage1.py')}
    receipt=args.output/'identity.json'
    if receipt.exists() and json.loads(receipt.read_text())!=identity:raise ValueError('resume identity mismatch')
    save(receipt,identity)
    cfg=builder(dataset_repo_id=args.dataset.name,base_params=str(args.checkpoint/'params'),project_root=ROOT,
        exp_name='full-episode-offline-diagnosis',batch_size=args.batch_size,num_workers=0,fsdp_devices=1,num_train_steps=1)
    cfg=dataclasses.replace(cfg,data=dataclasses.replace(cfg.data,assets=training.AssetsConfig(assets_dir=str(args.checkpoint/'assets'))))
    dc=cfg.data.create(cfg.assets_dirs,cfg.model)
    if not dc.use_quantile_norm:raise ValueError('expected registered quantile contract')
    stats=dc.norm_stats['actions'];scale=np.asarray(stats.q99[:7])-np.asarray(stats.q01[:7])+1e-6
    offset=np.asarray(stats.q01[:7])
    tokenizer=next(t for t in dc.model_transforms.inputs if isinstance(t,transforms.TokenizePrompt))
    class Model(nnx.Module):
        def __init__(self):
            self.vla=cfg.model.create(jax.random.key(0))
            self.rlt_module=bridge.ToNNX(RLModel(config=RLConfig(num_rl_tokens=1,num_layers=2,embed_dim=2048,input_dim=2048)))
            # Infer true image-only prefix shape, without concrete random arrays.
            prefix,_=self.vla.extract_prefix_embeddings(jax.random.key(0),cfg.model.fake_obs(batch_size=1),image_only=True)
            self.rlt_module.lazy_init(jnp.zeros_like(prefix),None,rngs=nnx.Rngs(jax.random.key(1)),method='encode',train=False)
        def infer(self,observation):
            cache=self.vla.prepare_prefix_for_inference(observation)
            z=self.rlt_module(cache.image_prefix_out.astype(jnp.float32),None,method='encode',train=False)
            # Exact noise from serving's batch-1 seed42, broadcast across independent inputs.
            noise=jax.random.normal(jax.random.key(42),(1,50,32))
            noise=jnp.broadcast_to(noise,(observation.state.shape[0],50,32))
            actions=self.vla.sample_actions_from_prefix_cache(jax.random.key(42),cache,num_steps=10,noise=noise)
            return actions[...,:7],z
    if args.inference_path=='serving':
        policy=serve_stage1.load(ROOT,args.checkpoint,default_prompt=args.prompt or serve_stage1.PROMPT)
        def infer(observation):
            # Training loader normalizes in NumPy; serving normalizes uint8 on
            # the device. One GPU float32 ULP measurably changes BF16 features.
            # Recover exact uint8 pixels, then use the real service operation.
            inputs=observation.to_dict()
            inputs['image']={key:recover_uint8_image(value) for key,value in observation.images.items()}
            inputs=jax.tree.map(lambda v:None if v is None else jnp.asarray(v),inputs,is_leaf=lambda v:v is None)
            prepared=model_api.Observation.from_dict(inputs)
            actions,z=policy.sample_model_observation(prepared)
            return actions[...,:7],z
    else:
        abstract=nnx.eval_shape(Model);graph,state=nnx.split(abstract)
        loaded=restore_inference_params(args.checkpoint/'params')
        if jax.tree_util.tree_structure(state.to_pure_dict())!=jax.tree_util.tree_structure(loaded):raise ValueError('parameter tree mismatch')
        state.replace_by_pure_dict(loaded);model=nnx.merge(graph,state);infer=nnx_utils.module_jit(model.infer)
    report={'code_head':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        'devices':[str(x) for x in jax.devices()], 'requested_episodes':episodes,'identity':identity,
        'dataset':str(args.dataset),'checkpoint':str(args.checkpoint),'split':'training-seen development; no independent test',
        'units':['rad']*6+['m'],'episodes':[], 'boundary':[
        'All 134 Episodes were seen by Stage1; whole-Episode bootstrap is fitting variation, not generalization.',
        'Same seed42 initial noise as batch1 serving; batching may introduce floating-point differences.',
        'analysis uses a separate compiled graph; servicing-parity claims require inference_path=serving.',
        'No robot dynamics/contact truth; interpolation changes proposed timing only.',
        'No future targets invented: slots beyond terminal are excluded; overlapping anchors are not independent.']}
    start=time.time()
    for ep in episodes:
        er=args.output/('episode_%03d.json'%ep);cachefile=args.output/('episode_%03d.npz'%ep)
        parquet=args.dataset/'data/chunk-000'/('episode_%06d.parquet'%ep)
        parquet_hash=digest(parquet)
        if er.exists():
            row=json.loads(er.read_text())
            if row['parquet_sha256']!=parquet_hash or row['cache_sha256']!=digest(cachefile):raise ValueError('Episode receipt mismatch')
            report['episodes'].append(row);continue
        table=pq.read_table(parquet)
        raw_state=np.asarray(table['observation.state'].to_pylist(),np.float32)
        raw_actions=np.asarray(table['action'].to_pylist(),np.float32)
        loader,frames=make_loader(cfg,args.dataset,[ep],0)
        predicted=[];zs=[];norm_targets=[];started=time.time();loader_time=0.;forward_time=0.;wait=time.time()
        for observation,actions in loader:
            loader_time+=time.time()-wait
            n=int(actions.shape[0]);norm_targets.append(np.asarray(actions)[...,:7])
            if args.prompt is not None:
                # Tokenization occurs before padding in the registered pipeline.
                # Use the original seven normalized state values, never pad bins.
                tokens=[tokenizer({'state':np.asarray(observation.state[i,:7]),'prompt':args.prompt}) for i in range(n)]
                observation=dataclasses.replace(observation,
                    tokenized_prompt=jnp.asarray(np.stack([x['tokenized_prompt'] for x in tokens])),
                    tokenized_prompt_mask=jnp.asarray(np.stack([x['tokenized_prompt_mask'] for x in tokens])))
            if n<args.batch_size:
                observation=jax.tree.map(lambda x:None if x is None else jnp.concatenate([x,jnp.repeat(x[-1:],args.batch_size-n,axis=0)]),observation,is_leaf=lambda x:x is None)
            t=time.time();a,z=infer(observation);a,z=jax.device_get((a,z));forward_time+=time.time()-t
            predicted.append(np.asarray(a[:n],np.float32));zs.append(np.asarray(z[:n],np.float32).reshape(n,-1));wait=time.time()
        normalized=np.concatenate(predicted);token=np.concatenate(zs);expected=np.concatenate(norm_targets)
        native=(normalized+1)*.5*scale+offset;native[...,:6]+=raw_state[:,None,:6]
        if len(native)!=frames or frames!=len(raw_actions):raise ValueError('incomplete Episode coverage')
        if not np.isfinite(native).all() or not np.isfinite(token).all():raise ValueError('nonfinite forward')
        target30,mask30=time_targets(raw_actions,30,30,50)
        reconstructed=(expected+1)*.5*scale+offset;reconstructed[...,:6]+=raw_state[:,None,:6]
        # Directly verify actual loader targets, including its recorded tail padding.
        target_mismatch=float(np.max(np.abs(reconstructed-target30)))
        if target_mismatch>2e-5:raise ValueError('loader target/Parquet physical contract mismatch: '+str(target_mismatch))
        target20,mask20=time_targets(raw_actions,30,20,10)
        retimed=interpolate_chunk(native,30,20,10)
        hold=np.broadcast_to(raw_state[:,None,:],target20.shape)
        row={'episode':ep,'frames':frames,'samples':len(native),'complete':True,'parquet_sha256':parquet_hash,
            'loader_native_target_max_error':target_mismatch,'seconds':time.time()-started,'loader_seconds':loader_time,'forward_seconds':forward_time,
            'sample30_first10':action_summary(native[:,:10],target30[:,:10],mask30[:,:10]),
            'sample30_full50':action_summary(native,target30,mask30),
            'legacy_first10_at20':action_summary(native[:,:10],target20,mask20),
            'retimed30to20_first10':action_summary(retimed,target20,mask20),
            'hold_current_state_at20':action_summary(hold,target20,mask20),
            'token_rms':float(np.sqrt(np.mean(token**2))),
            'predicted_step_p99_per_dim':np.quantile(np.abs(np.diff(native[:,:10],axis=1)),.99,axis=(0,1)).tolist()}
        np.savez_compressed(cachefile,predicted_native=native,normalized_prediction=normalized,z_rl=token,
            proprio=raw_state,recorded_action=raw_actions)
        row['cache_sha256']=digest(cachefile);save(er,row);report['episodes'].append(row)
        report['elapsed_sec']=time.time()-start;save(args.output/'report.json',report)
        print(json.dumps({'episode':ep,'frames':frames,'elapsed_sec':report['elapsed_sec'],
            'model_joint_mae30':np.mean(row['sample30_first10']['mae_per_dim'][:6]),
            'legacy_joint_mae20':np.mean(row['legacy_first10_at20']['mae_per_dim'][:6]),
            'retimed_joint_mae20':np.mean(row['retimed30to20_first10']['mae_per_dim'][:6])}),flush=True)
    report['complete']=True;report['elapsed_sec']=time.time()-start
    assert identity['manifest']==digest(manifest) and identity['norm']==digest(norm)
    report['source_metadata_unchanged']=True;save(args.output/'report.json',report)


if __name__=='__main__':main()
