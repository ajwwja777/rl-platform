"""Predeclared offline selection; never certifies autonomous success."""
import numpy as np

def select_checkpoint(reports):
    if len(reports)<2:raise ValueError('at least two completed checkpoints required')
    identities=None;manifest=None;config=None;rows=[]
    for r in reports:
        cohorts=r['cohorts']
        if set(cohorts)!={'train','dev'}:raise ValueError('complete TRAIN and DEV required')
        ids={}
        for split in ('train','dev'):
            entries=cohorts[split]['per_episode']
            if not entries or any(x['frames']!=x['evaluated_frames'] for x in entries):
                raise ValueError('partial episodes')
            ids[split]=sorted((x['uuid'],x['frames']) for x in entries)
            if len({x[0] for x in ids[split]})!=len(ids[split]):raise ValueError('duplicate UUID')
        if set(x[0] for x in ids['train']) & set(x[0] for x in ids['dev']):raise ValueError('TRAIN/DEV leakage')
        if identities is not None and identities!=ids:raise ValueError('cohort identity mismatch')
        identities=ids
        sha=r['dataset_manifest_sha256']
        if manifest is not None and manifest!=sha:raise ValueError('dataset mismatch')
        manifest=sha
        meta=r['metadata']
        current=tuple(str(meta[k]) for k in ('arm','camera_keys','model_action_sampling_hz','denoising_steps','compute_restore_dtype','norm_stats_sha256'))
        if config is not None and config!=current:raise ValueError('inference contract mismatch')
        config=current
        if r['effective_batch_size']!=1:raise ValueError('selection must match single-observation deployment')
        means=cohorts['dev']['summary']['episode_equal_mean']
        row=dict(checkpoint=r['checkpoint'],tag=int(r['checkpoint'].rstrip('/').split('/')[-1]),
            joint_mae_rad=float(means['c10_joint_mae_rad']),gripper_mae_m=float(means['c10_gripper_mae_m']),
            joint_p95_rad=float(np.mean(means['c10_p95_per_dim'][:6])))
        if not np.isfinite([row[k] for k in ('joint_mae_rad','gripper_mae_m','joint_p95_rad')]).all():raise ValueError('nonfinite metric')
        rows.append(row)
    if len({x['checkpoint'] for x in rows})!=len(rows):raise ValueError('duplicate checkpoint')
    best_joint=min(x['joint_mae_rad'] for x in rows)
    near=[x for x in rows if x['joint_mae_rad']<=best_joint*1.01]
    selected=min(near,key=lambda x:x['tag'])
    conflicts=[k for k in ('gripper_mae_m','joint_p95_rad') if selected[k]>1.25*min(x[k] for x in rows)]
    return dict(status='metric_conflict' if conflicts else 'offline_candidate',
        selected_checkpoint=None if conflicts else selected['checkpoint'],
        conflicts=conflicts,rows=sorted(rows,key=lambda x:x['tag']),
        boundary='Repeated-use DEV imitation proxy only; no independent TEST or autonomous/Online improvement evidence')
