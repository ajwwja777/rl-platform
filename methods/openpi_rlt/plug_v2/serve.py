#!/usr/bin/env python3
# Matched trained-RTC Stage-1 serving, with strict weight and normalization checks.
# Importable on HPC and Cobot; this module has no robot/CAN publisher.
import argparse,dataclasses,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np

def initialize(project):
    project=Path(project).resolve()
    sys.path[:0]=[str(project),str(project/'code/openpi-rlt/src'),str(project/'code/openpi-rlt/scripts')]
    # This order is shared with the tested training/data path.
    import pyarrow
    import torch
    import jax
    import jax.numpy as jnp
    from flax import nnx
    import serve_rlt_policy as upstream
    from methods.openpi_rlt.plug_v2.rtc import install
    install(6)
    from methods.openpi_rlt.stage1_config import build_stage1_config
    from openpi import transforms
    from openpi.models import model as model_api
    from openpi.shared import nnx_utils
    from openpi.training import checkpoints
    return jax,jnp,nnx,upstream,build_stage1_config,transforms,model_api,nnx_utils,checkpoints

def load(project, checkpoint, *, num_steps=10):
    checkpoint=Path(checkpoint).resolve()
    if not (checkpoint/'params').is_dir() or not (checkpoint/'assets/lerobot/norm_stats.json').is_file():
        raise ValueError('checkpoint params/assets missing')
    stats_hash=hashlib.sha256((checkpoint/'assets/lerobot/norm_stats.json').read_bytes()).hexdigest()
    if stats_hash!='373d9a01bbcc0907dbbc27b091768774f98933cb28cdf3404a14f378ee02b49d':
        raise ValueError('normalization does not match frozen plug_v2 training dataset')
    jax,jnp,nnx,upstream,builder,transforms,model_api,nnx_utils,checkpoints=initialize(project)
    cfg=builder(exp_name='plug_v2_serve',dataset_repo_id='lerobot',base_params=str(checkpoint/'params'),
                assets_base_dir=checkpoint/'assets',checkpoint_base_dir=Path(project)/'runs/plug_v2/checkpoints',
                num_workers=0,fsdp_devices=1)
    data=cfg.data.create(checkpoint/'assets',cfg.model)
    norm=checkpoints.load_norm_stats(checkpoint/'assets','lerobot')
    if norm is None:raise ValueError('normalization is required')
    class InferenceModel(upstream.RLTInferenceModel):
        def infer_rtc(self,rng,observation,action_prefix,prefix_lengths):
            cache=self.vla.prepare_prefix_for_inference(observation)
            z=self.rlt_module(cache.image_prefix_out.astype(jnp.float32),None,method='encode',train=False)
            actions=self.vla.sample_actions_from_prefix_cache(rng,cache,num_steps=num_steps,
                                       action_prefix=action_prefix,prefix_lengths=prefix_lengths)
            return actions,z
        def evaluate_rtc(self,key,observation,actions):
            loss,embedding,mask=self.vla.compute_loss_with_prefix(
                key,observation,actions,train=False,image_only=True)
            reconstruction,details=self.rlt_module(
                embedding.astype(jnp.float32),mask,method='loss',train=False)
            return jnp.mean(loss),reconstruction
    vla=nnx.eval_shape(cfg.model.create,jax.random.key(0))
    model=InferenceModel(vla,upstream._create_rlt_config(cfg),rngs=nnx.Rngs(0),
                         prefix_seq_len=768,shared_prefix_inference=True)
    graph,state=nnx.split(model)
    loaded=model_api.restore_params(checkpoint/'params',dtype=jnp.bfloat16)
    def shapes(tree):
        pairs,_=jax.tree_util.tree_flatten_with_path(tree)
        return {str(path):tuple(value.shape) for path,value in pairs}
    expected=shapes(state.to_pure_dict());actual=shapes(loaded)
    if expected!=actual:
        raise ValueError('checkpoint parameter mismatch: missing='+str(sorted(set(expected)-set(actual)))+
                         ' extra='+str(sorted(set(actual)-set(expected))))
    state.replace_by_pure_dict(loaded);model=nnx.merge(graph,state)
    input_transform=transforms.compose([transforms.InjectDefaultPrompt('Insert the held plug into the socket.'),
            *data.data_transforms.inputs,transforms.Normalize(norm,use_quantiles=data.use_quantile_norm),
            *data.model_transforms.inputs])
    output_transform=transforms.compose([*data.model_transforms.outputs,
            transforms.Unnormalize(norm,use_quantiles=data.use_quantile_norm),*data.data_transforms.outputs])
    infer_fn=nnx_utils.module_jit(model.infer_rtc)
    evaluate_fn=nnx_utils.module_jit(model.evaluate_rtc)
    class Policy:
        def __init__(self):
            self.rng=jax.random.key(42)
            self.metadata={'cohort':'plug_v2','has_rl_token':True,'z_dim':2048,'proprio_dim':14,
               'chunk_len':50,'actor_chunk_len':10,'action_dim':14,'control_hz':30,
               'rtc_mode':'trained','rtc_max_delay':6,'denoising_steps':num_steps,
               'checkpoint':str(checkpoint),'parameter_leaves':len(expected),
               'norm_stats_sha256':hashlib.sha256((checkpoint/'assets/lerobot/norm_stats.json').read_bytes()).hexdigest(),
               'supports_batch':False,'reference_sampling':'fixed_seed_42'}
        def evaluate(self,obs,ground_truth):
            inp=input_transform({'state':np.asarray(obs['state'],np.float32).copy(),
                'images':obs['images'],'actions':np.asarray(ground_truth,np.float32).copy(),
                'prompt':obs.get('prompt','Insert the held plug into the socket.')})
            actions=inp.pop('actions')
            observation=model_api.Observation.from_dict(
                jax.tree.map(lambda v:jnp.asarray(v)[None,...],inp))
            flow,reconstruction=evaluate_fn(jax.random.key(42),observation,
                                             jnp.asarray(actions)[None,...])
            return {'heldout_flow_loss':float(flow),
                    'heldout_token_reconstruction_loss':float(reconstruction)}
        def infer(self,obs):
            state=np.asarray(obs['state'],np.float32)
            if state.shape!=(14,) or not np.isfinite(state).all():raise ValueError('invalid physical state')
            d=obs.get('prefix_length',0)
            if isinstance(d,bool) or not isinstance(d,(int,np.integer)) or not 0<=d<=6:
                raise ValueError('untrained prefix delay')
            prefix=np.asarray(obs.get('action_prefix',np.broadcast_to(state,(50,14))),np.float32).copy()
            if prefix.shape!=(50,14) or not np.isfinite(prefix).all():raise ValueError('invalid physical prefix')
            images=obs['images']
            for key in ('base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb'):
                image=np.asarray(images[key])
                if image.ndim!=3 or image.shape[-1]!=3 or image.dtype!=np.uint8:
                    raise ValueError('all three RGB cameras are required')
            inp=input_transform({'state':state.copy(),'images':images,'actions':prefix.copy(),
                       'prompt':obs.get('prompt','Insert the held plug into the socket.')})
            normalized_prefix=inp.pop('actions')
            observation=model_api.Observation.from_dict(jax.tree.map(lambda v:jnp.asarray(v)[None,...],inp))
            key=jax.random.key(42)
            start=time.perf_counter()
            actions,z=infer_fn(key,observation,jnp.asarray(normalized_prefix)[None,...],jnp.array([d],jnp.int32))
            actions=np.asarray(actions[0]);z=np.asarray(z[0],np.float32).reshape(-1)
            result=output_transform({'state':np.asarray(inp['state']),'actions':actions})['actions']
            if result.shape!=(50,14) or z.shape!=(2048,) or not np.isfinite(result).all() or not np.isfinite(z).all():
                raise ValueError('invalid model output')
            # Remove normalization roundoff from commands already committed by the client.
            result[:d]=prefix[:d]
            return {'ref_chunk':result,'z_rl':z,'proprio':state.copy(),'prefix_length':int(d),
                    'policy_timing':{'infer_ms':1000*(time.perf_counter()-start)}}
    return Policy()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project-root',type=Path,required=True)
    ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--port',type=int,default=8000)
    ap.add_argument('--num-steps',type=int,default=10);ap.add_argument('--validate-only',action='store_true')
    args=ap.parse_args()
    if not 1<=args.num_steps<=20:raise ValueError('invalid denoising steps')
    from methods.openpi_rlt.stage1_entry import prepare_environment
    dataset=args.project_root/'data/rlt/plug_v2/demonstrations/lerobot'
    if not (dataset/'meta/info.json').is_file():
        dataset=args.project_root.parent.parent/'data/rlt/plug_v2/demonstrations/lerobot'
    if not (dataset/'meta/info.json').is_file():raise ValueError('registered plug_v2 dataset missing')
    prepare_environment(dataset,args.project_root/'runs/plug_v2')
    policy=load(args.project_root,args.checkpoint,num_steps=args.num_steps)
    dummy={'images':{key:np.zeros((224,224,3),np.uint8) for key in
           ('base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb')},'state':np.zeros(14,np.float32)}
    times=[]
    for d in (0,6,6,6,0):
        obs={**dummy,'prefix_length':d,'action_prefix':np.zeros((50,14),np.float32)}
        result=policy.infer(obs);times.append(result['policy_timing']['infer_ms'])
        assert np.array_equal(result['ref_chunk'][:d],obs['action_prefix'][:d])
    receipt={'status':'passed','pid':os.getpid(),'metadata':policy.metadata,'inference_ms':times,'robot_publishers':0}
    output=args.project_root/'runs/plug_v2/model-server/validation.json';output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(receipt,indent=2));print('MODEL_VALIDATED',json.dumps(receipt),flush=True)
    if args.validate_only:return
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    print('MODEL_READY',flush=True)
    WebsocketPolicyServer(policy=policy,host='127.0.0.1',port=args.port,metadata=policy.metadata).serve_forever()

if __name__=='__main__':
    # Make project-owned namespace importable when invoked as a file.
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
    main()
