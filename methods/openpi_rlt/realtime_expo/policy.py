"""Two-wrist left-arm Stage1 inference; no robot I/O or middle-camera input."""
import dataclasses
import hashlib
from pathlib import Path
import numpy as np
from .dataset import PROMPT

CAMERAS=('left_wrist_0_rgb','right_wrist_0_rgb')

def prepare_observation(observation):
    from openpi_client.image_tools import resize_with_pad
    state=np.asarray(observation['state'],dtype=np.float32)
    if state.shape!=(7,) or not np.isfinite(state).all():raise ValueError('finite left-arm state7 required')
    images=observation['images']
    if set(images)!=set(CAMERAS):raise ValueError('exactly two wrist cameras required; middle/head camera forbidden')
    result={}
    for key in CAMERAS:
        image=np.asarray(images[key])
        if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[-1]!=3:raise ValueError('uint8 RGB required')
        result[key]=resize_with_pad(image,224,224)
    return dict(state=state.copy(),images=result,prompt=observation.get('prompt',PROMPT))

def load(checkpoint,output,*,num_steps=10):
    # Caller sets process-local upstream imports and cache before loading.
    import datasets,torch
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    import flax.nnx.bridge as bridge
    from openpi import transforms
    from openpi.models import model as model_api
    from openpi.models.rl_token import RLTokenModel
    from openpi.shared import nnx_utils
    from openpi.training import config as tc
    from .configuration import build_config
    from methods.openpi_rlt.plug_v3_yyshadow.stage1_loading import restore_inference_params
    import train_rlt
    checkpoint=Path(checkpoint); output=Path(output)
    c=build_config(output,checkpoint/'params',1,1,1)
    c=dataclasses.replace(c,data=dataclasses.replace(c.data,assets=tc.AssetsConfig(assets_dir=str(checkpoint/'assets'))))
    data=c.data.create(c.assets_dirs,c.model)
    if data.norm_stats is None:raise ValueError('checkpoint normalization required')
    cfg=train_rlt._create_rlt_config(c);prefix_len=train_rlt._infer_prefix_seq_len(c.model)
    class Model(nnx.Module):
        def __init__(self):
            self.vla=c.model.create(jax.random.key(0))
            self.rlt_module=bridge.ToNNX(RLTokenModel(config=cfg))
            self.rlt_module.lazy_init(jnp.zeros((1,prefix_len,cfg.input_dim)),jnp.ones((1,prefix_len),dtype=bool),rngs=nnx.Rngs(jax.random.key(1)),method='encode',train=False)
        def infer(self,observation):
            cache=self.vla.prepare_prefix_for_inference(observation)
            # Match one-observation deployed key42 sampling regardless of batch size.
            noise=jnp.broadcast_to(jax.random.normal(jax.random.key(42),(1,50,32)),(observation.state.shape[0],50,32))
            actions=self.vla.sample_actions_from_prefix_cache(jax.random.key(42),cache,num_steps=num_steps,noise=noise)
            token=self.rlt_module(cache.image_prefix_out.astype(jnp.float32),None,method='encode',train=False)
            return actions,token
    shape=nnx.eval_shape(Model);graph,state=nnx.split(shape)
    params=restore_inference_params(checkpoint/'params')
    if jax.tree.structure(state.to_pure_dict())!=jax.tree.structure(params):raise ValueError('wrong parameter tree')
    state.replace_by_pure_dict(params);model=nnx.merge(graph,state)
    infer=nnx_utils.module_jit(model.infer)
    input_transform=transforms.compose([*data.data_transforms.inputs,transforms.Normalize(data.norm_stats,use_quantiles=data.use_quantile_norm),*data.model_transforms.inputs])
    output_transform=transforms.compose([*data.model_transforms.outputs,transforms.Unnormalize(data.norm_stats,use_quantiles=data.use_quantile_norm),*data.data_transforms.outputs])
    class Policy:
        metadata=dict(arm='left',camera_keys=list(CAMERAS),has_rl_token=True,z_dim=2048,proprio_dim=7,action_dim=7,model_horizon=50,actor_chunk_len=10,control_hz=10,model_action_sampling_hz=10,reference_sampling_hz=10,default_prompt=PROMPT,reference_sampling='fixed_seed_42',denoising_steps=num_steps,checkpoint=str(checkpoint),parameter_asset_precision='unchanged',compute_restore_dtype='bfloat16 (existing RLT inference loader)',rtc_prefix_supported=False)
        def predict_batch(self,observations):
            transformed=[input_transform(prepare_observation(x)) for x in observations]
            batched=jax.tree.map(lambda *values:jnp.asarray(np.stack(values)),*transformed)
            obs=model_api.Observation.from_dict(batched)
            actions,z=infer(obs)
            actions=np.asarray(actions);z=np.asarray(z,dtype=np.float32).reshape(len(observations),-1)
            physical=np.stack([output_transform({'state':np.asarray(batched['state'][i]),'actions':actions[i].copy()})['actions'] for i in range(len(observations))]).astype(np.float32)
            if physical.shape!=(len(observations),50,7) or z.shape!=(len(observations),2048) or not np.isfinite(physical).all() or not np.isfinite(z).all():raise ValueError('invalid inference output')
            return physical,z
        def infer(self,observation):
            if observation.get('rtc') is not None:raise ValueError('RTC needs separate left10Hz validation; cannot silently ignore prefix')
            action,z=self.predict_batch([observation])
            return dict(ref_chunk=action[0,:10],z_rl=z[0],proprio=np.asarray(observation['state'],dtype=np.float32).copy(),rtc_used=False)
    return Policy()
