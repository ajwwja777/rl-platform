#!/usr/bin/env python3
"""Actual batch1 serving versus cached full-Episode analysis; no network server."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cache-directory',default='stage1_serving_prompt',
        help='Explicit completed cache for matching the serving prompt and batch shape.')
    a=p.parse_args()
    if a.output.exists():p.error('fresh report required')
    a.output.mkdir(parents=True)
    import numpy as np
    from methods.openpi_rlt.plug_v3_yyshadow import serve_stage1
    # Imports frozen PyArrow/Torch before JAX through the actual serving loader.
    checkpoint=a.root/'models/rlt/plug_v3_yyshadow/stage1/4999'
    policy=serve_stage1.load(ROOT,checkpoint,reference_hz=20)
    from methods.openpi_rlt.stage1_entry import prepare_environment
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from openpi.policies.agilexbag_image_policy import _parse_image
    from methods.openpi_rlt.plug_v3_yyshadow.stage1_action_metrics import interpolate_chunk
    dataset=Path('/data/LFT-W02_data/jiaan/scratch/cobot-realworld-rl/data/rlt/plug_v3_yyshadow/plug_v3_yyshadow_demonstrations')
    prepare_environment(dataset,ROOT);base=LeRobotDataset(dataset.name,root=dataset)
    cache_root=a.root/'outputs/offline-model-selection-20261005'/a.cache_directory
    cache_receipt=json.loads((cache_root/'report.json').read_text())
    if not cache_receipt.get('complete'):raise ValueError('completed analysis cache required')
    rows=[]
    for ep,fraction in [(0,0.),(120,.5),(133,1.)]:
        cache=dict(np.load(cache_root/('episode_%03d.npz'%ep)));frame=int((len(cache['proprio'])-1)*fraction)
        index=int(base.episode_data_index['from'][ep])+frame;sample=base[index]
        images={name:_parse_image(sample[key]) for name,key in [('base_0_rgb','observation.images.cam_high'),
            ('left_wrist_0_rgb','observation.images.cam_left_wrist'),('right_wrist_0_rgb','observation.images.cam_right_wrist')]}
        observation={'state':np.asarray(sample['observation.state'],np.float32),'images':images}
        result=policy.infer(observation)
        expected=interpolate_chunk(cache['predicted_native'][frame],30,20,10)
        diff=np.max(np.abs(result['ref_chunk']-expected),axis=0)
        token=cache['z_rl'][frame];actual=result['z_rl'];relative=float(np.linalg.norm(actual-token)/max(np.linalg.norm(token),1e-12))
        rows.append({'episode':ep,'frame':frame,'ref_max_abs_delta_per_dim':diff.tolist(),
            'token_relative_l2_delta':relative,'inference_ms':result['policy_timing']['infer_ms'],
            'finite':bool(np.isfinite(result['ref_chunk']).all() and np.isfinite(actual).all())})
    report={'status':'已验证' if all(r['finite'] and max(r['ref_max_abs_delta_per_dim'][:6])<1e-4 and r['ref_max_abs_delta_per_dim'][-1]<2e-6 and r['token_relative_l2_delta']<.01 for r in rows) else '失败',
        'policy_metadata':policy.metadata,'load_seconds':policy.load_seconds,'rows':rows,
        'cache_directory':str(cache_root),'cache_identity':cache_receipt['identity'],
        'diagnostic_numeric_tolerance':{'joint_rad':1e-4,'gripper_m':2e-6,'token_relative_l2':.01},
        'boundary':'A6000 GPU0 read-only serving path with real decoded images, actual batch1 vs recorded analysis batch size. Numeric consistency only; tolerances are not hardware acceptance limits.',
        'robot_publishers':0,'network_servers':0}
    (a.output/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
