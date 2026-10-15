#!/usr/bin/env python3
"""No-ROS GPU validation for the two-wrist left-arm inference boundary."""
import argparse,json,os,sys,time
from pathlib import Path
ap=argparse.ArgumentParser()
for name in ('assets','data','checkpoint','output','tokenizer-cache','rtc-overlay'):
    ap.add_argument('--'+name,type=Path,required=True)
a=ap.parse_args()
root=Path(__file__).resolve().parents[1];u=a.assets/'third_party/openpi-rlt'
for p in (root,u/'src',u/'scripts',u/'packages/openpi-client/src'):sys.path.insert(0,str(p))
os.environ['OPENPI_DATA_HOME']=str(a.tokenizer_cache)
os.environ['JAX_COMPILATION_CACHE_DIR']=str(a.output.parent/'policy-jax-cache')
import datasets,torch,numpy as np
from methods.openpi_rlt.realtime_expo.dataset import ExpertDataset
from methods.openpi_rlt.realtime_expo.policy import load
from methods.openpi_rlt.plug_v3_yyshadow.train_stage1 import verify_upstream
verify_upstream(u)
a.output.mkdir(parents=True,exist_ok=True)
if (a.output/'validation.json').exists():raise FileExistsError('completed validation exists')
policy=load(a.checkpoint,a.output,rtc_overlay=a.rtc_overlay)
dataset=ExpertDataset(a.data,'dev')
records=[]
for index in (0,1,2,100,200):
    obs=dataset[index]
    plain=policy.infer(obs)
    assert plain['rtc_used'] is False
    repeat=policy.infer(obs)
    np.testing.assert_array_equal(plain['ref_chunk'],repeat['ref_chunk'])
    rtc=policy.infer(dict(obs,rtc=dict(previous_actions=plain['ref_chunk'][:5].copy(),delay_steps=2,execution_horizon=5)))
    assert rtc['rtc_used'] is True
    assert rtc['ref_chunk'].shape==(10,7) and np.isfinite(rtc['ref_chunk']).all()
    np.testing.assert_allclose(plain['z_rl'],rtc['z_rl'],atol=1e-6,rtol=1e-6)
    records.append(dict(frame=index,plain_ms=plain['policy_timing']['infer_ms'],repeat_ms=repeat['policy_timing']['infer_ms'],rtc_ms=rtc['policy_timing']['infer_ms'],rtc_vs_plain_max_per_dim=np.max(np.abs(rtc['ref_chunk']-plain['ref_chunk']),axis=0).tolist()))
bad=dict(obs,images={**obs['images'],'base_0_rgb':np.zeros((224,224,3),np.uint8)})
try:policy.infer(bad)
except ValueError:pass
else:raise AssertionError('middle camera accepted')
result=dict(status='passed',checkpoint=str(a.checkpoint),metadata=policy.metadata,records=records,robot_publishers=0,boundary='GPU shape, determinism, shared token and RTC finite output only; not robot success or trained-model quality')
(a.output/'validation.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
