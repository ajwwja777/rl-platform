"""Isolated release construction and fail-closed runtime configuration."""
import argparse,copy,hashlib,io,json,pickle,shutil
from pathlib import Path
import numpy as np
import yaml
from integrations.cobot_runtime.online_seed import _same_tree,_journal_boundary
CONTRACT='realtime_expo_left_2cam_10hz'

class _PortablePickler(pickle.Pickler):
    """Use public NumPy constructors, readable on field NumPy 1.26 and training 2.x."""
    def reducer_override(self,obj):
        if isinstance(obj,np.ndarray):
            if obj.dtype.hasobject:raise ValueError('object arrays forbidden in model release')
            return np.ndarray,(obj.shape,obj.dtype,obj.tobytes(order='C'),0,None,'C')
        if isinstance(obj,np.generic):return type(obj),(obj.item(),)
        return NotImplemented

def portable_pickle(value):
    stream=io.BytesIO();_PortablePickler(stream,protocol=pickle.HIGHEST_PROTOCOL).dump(value)
    return stream.getvalue()

def build(source,seed,destination,installed_root,stage1,step,template):
    source,seed,destination=map(Path,(source,seed,destination));installed_root=Path(installed_root)
    if destination.exists():raise FileExistsError(destination)
    actor=pickle.loads((source/f'actor_snapshot/step_{step}.pkl').read_bytes())
    checkpoint=pickle.loads((source/f'checkpoints/step_{step}.pkl').read_bytes())
    if int(checkpoint['state']['global_step'])!=step or actor['global_step']!=step or not _same_tree(actor['actor_params'],checkpoint['state']['actor_params']):raise ValueError('Actor/checkpoint mismatch')
    boundary=_journal_boundary(seed/'train_expert_journal.pkl')
    if checkpoint['progress']['warmup_ready_adds_total']!=boundary['adds_total']:raise ValueError('Replay anchor mismatch')
    config=yaml.safe_load(Path(template).read_text())
    original=json.loads((source/'config.json').read_text())['rl']
    config['experiment']['rl']=original
    config['experiment']['rl'].update(warmup_post_collect_updates=step,freeze_after_warmup=False,grad_updates_per_cycle=1)
    destination.mkdir(parents=True)
    for mode in ('frozen','online'):
        folder=destination/mode;installed=installed_root/mode
        (folder/'actor_snapshot').mkdir(parents=True);(folder/'checkpoints').mkdir()
        shutil.copyfile(source/'action_norm_stats.json',folder/'action_norm_stats.json')
        current=copy.deepcopy(actor);current['rl_config']['action_norm_stats_path']='../action_norm_stats.json'
        (folder/'actor_snapshot/actor_snapshot.pkl').write_bytes(portable_pickle(current))
        state=copy.deepcopy(checkpoint);state['rl_config']['action_norm_stats_path']='../action_norm_stats.json'
        (folder/'checkpoints/latest.pkl').write_bytes(portable_pickle(state))
        (folder/f'checkpoints/step_{step}.pkl').write_bytes(portable_pickle(state))
        shutil.copyfile(seed/'train_expert_journal.pkl',folder/'replay_journal.pkl')
        c=copy.deepcopy(config);r=c['runtime'];c['experiment']['rl']['action_norm_stats_path']=str(installed/'action_norm_stats.json')
        c['experiment']['rl']['freeze_after_warmup']=mode=='frozen'
        r['actor_service']['snapshot_path']=str(installed/'actor_snapshot/actor_snapshot.pkl')
        r['learner_service'].update(checkpoint_dir=str(installed/'checkpoints'),actor_snapshot_path=r['actor_service']['snapshot_path'],push_actor_interval_steps=50,checkpoint_interval_steps=50,sample_batch_size=128)
        r['replay'].update(journal_path=str(installed/'replay_journal.pkl'),sample_strategy='uniform',seed=42)
        r['env_driver'].update(control_frequency_hz=10.,actor_deterministic=mode=='frozen',replay_feature_batch_size=1)
        r['monitoring']['wandb_run_name']='realtime-expo-rlt-'+mode
        r['monitoring']['wandb_dir']=str(installed/'wandb')
        (destination/f'{mode}.yaml').write_text(yaml.safe_dump(c,sort_keys=False))
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    receipt=dict(contract=CONTRACT,stage1=str(stage1),source_step=step,actor_version=actor['version'],cameras=['left_wrist_0_rgb','right_wrist_0_rgb'],logical_hz=10,prompt=__import__('methods.openpi_rlt.realtime_expo.dataset',fromlist=['PROMPT']).PROMPT,installed_root=str(installed_root),replay_seed=boundary,publication_policy='automatic every 50 learner updates; native actor history retained',update_budget='selected warmup step + new transitions * 1; initial pending 0',norm_sha256=sha(destination/'frozen/action_norm_stats.json'),frozen_actor_sha256=sha(destination/'frozen/actor_snapshot/actor_snapshot.pkl'),boundary='all-success experts; offline DEV selection, no independent TEST or autonomous Online improvement claim')
    (destination/'release.json').write_text(json.dumps(receipt,indent=2))
    return receipt

def prepare_runtime(bundle,mode,snapshot,target):
    bundle,snapshot,target=map(Path,(bundle,snapshot,target));m=json.loads((bundle/'release.json').read_text())
    if mode not in ('frozen','online') or m['contract']!=CONTRACT or m['cameras']!=['left_wrist_0_rgb','right_wrist_0_rgb']:raise ValueError('invalid task/mode/cameras')
    roots=[bundle/'frozen/actor_snapshot/actor_snapshot.pkl',bundle/'online/actor_snapshot/actor_snapshot.pkl']
    if snapshot.resolve() not in [p.resolve() for p in roots] or (mode=='online' and snapshot.resolve()!=roots[1].resolve()):raise ValueError('selected Actor outside this release')
    if not (Path(m['stage1'])/'params').is_dir():raise FileNotFoundError('Stage1 params missing')
    norm=snapshot.parent.parent/'action_norm_stats.json'
    if hashlib.sha256(norm.read_bytes()).hexdigest()!=m['norm_sha256']:raise ValueError('normalization identity mismatch')
    if snapshot.resolve()==roots[0].resolve() and hashlib.sha256(snapshot.read_bytes()).hexdigest()!=m['frozen_actor_sha256']:raise ValueError('frozen Actor changed')
    actor=pickle.loads(snapshot.read_bytes());rc=actor['rl_config']
    if (rc['action_dim'],rc['proprio_dim'],rc['z_dim'],rc['chunk_len'],rc['action_representation'])!=(7,7,2048,10,'delta_chunk'):raise ValueError('Actor contract mismatch')
    if actor.get('global_step',0)<m['source_step']:raise ValueError('Actor older than release seed')
    config=yaml.safe_load((bundle/f'{mode}.yaml').read_text());r=config['runtime']
    if mode=='frozen':r['actor_service']['snapshot_path']=str(snapshot)
    if config['experiment']['rl']['warmup_post_collect_updates']!=m['source_step']:raise ValueError('wrong warmup budget')
    if r['env_driver']['control_frequency_hz']!=10 or r['env_driver']['replay_feature_batch_size']!=1:raise ValueError('task clock/feature inference mismatch')
    if mode=='online':
        cp=pickle.loads((bundle/'online/checkpoints/latest.pkl').read_bytes())
        if cp['progress']['warmup_ready_adds_total']!=m['replay_seed']['adds_total']:raise ValueError('online budget anchor changed')
    target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_suffix('.tmp');tmp.write_text(yaml.safe_dump(config,sort_keys=False));tmp.replace(target)
    return m

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('operation',choices=['check']);ap.add_argument('bundle',type=Path);ap.add_argument('mode');ap.add_argument('snapshot',type=Path);ap.add_argument('target',type=Path)
    a=ap.parse_args();prepare_runtime(a.bundle,a.mode,a.snapshot,a.target)
