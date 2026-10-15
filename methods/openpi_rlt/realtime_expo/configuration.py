"""New left-arm cohort, fixed author RLT architecture/loss and 10 Hz input contract."""
import dataclasses
from pathlib import Path
import numpy as np
from openpi import transforms
from openpi.training import config as tc
from methods.openpi_rlt.plug_v3_yyshadow.stage1_right_arm_config import build_config as original_config
@dataclasses.dataclass(frozen=True)
class WristInputs(transforms.DataTransformFn):
 def __call__(self,d):
  left=np.asarray(d['images']['left_wrist_0_rgb']);right=np.asarray(d['images']['right_wrist_0_rgb'])
  if left.shape!=(224,224,3) or right.shape!=left.shape:raise ValueError('RGB shape')
  result=dict(state=np.asarray(d['state']).copy(),prompt=d['prompt'],image=dict(base_0_rgb=np.zeros_like(left),left_wrist_0_rgb=left,right_wrist_0_rgb=right),image_mask=dict(base_0_rgb=np.False_,left_wrist_0_rgb=np.True_,right_wrist_0_rgb=np.True_))
  if 'actions' in d:result['actions']=np.asarray(d['actions']).copy()
  return result
@dataclasses.dataclass(frozen=True)
class LeftWristData(tc.DataConfigFactory):
 def create(self,assets_dirs,model_config):
  from openpi.policies.agilexbag_image_policy import AgilexBagImageOutputs
  mask=transforms.make_bool_mask(6,-1)
  return dataclasses.replace(self.create_base_config(assets_dirs,model_config),repack_transforms=transforms.Group(),data_transforms=transforms.Group(inputs=[WristInputs(),transforms.DeltaActions(mask)],outputs=[transforms.AbsoluteActions(mask),AgilexBagImageOutputs(action_dim=7)]),model_transforms=tc.ModelTransformFactory()(model_config),prompt_from_task=False)
def build_config(output,base_params,batch_size=8,fsdp_devices=2,num_train_steps=5000):
 c=original_config(dataset_repo_id='realtime_expo_left_train',base_params=str(base_params),project_root=output,exp_name='s42',batch_size=batch_size,num_workers=0,fsdp_devices=fsdp_devices,num_train_steps=num_train_steps)
 return dataclasses.replace(c,name='rlt_realtime_expo_left_2cam_10hz',data=LeftWristData(repo_id='realtime_expo_left_train'),assets_base_dir=str(Path(output)/'assets'),checkpoint_base_dir=str(Path(output)/'checkpoints'),save_interval=1000,keep_period=1000)
