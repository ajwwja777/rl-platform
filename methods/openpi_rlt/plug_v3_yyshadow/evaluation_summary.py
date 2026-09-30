"""Finite-sample aggregation and complete-episode bootstrap for Stage1 audits."""
import numpy as np


def aggregate_batches(rows):
    if not rows or any(row['samples'] <= 0 for row in rows):
        raise ValueError('evaluation must contain positive-sized batches')
    weights=np.asarray([row['samples'] for row in rows],dtype=float)
    result={'samples':int(weights.sum()),'batches':len(rows)}
    keys=set(rows[0]['metrics'])
    if any(set(row['metrics']) != keys for row in rows):
        raise ValueError('inconsistent metrics')
    for key in sorted(keys):
        values=np.asarray([row['metrics'][key] for row in rows])
        if not np.isfinite(values).all():raise ValueError('nonfinite evaluation metric')
        value=np.average(values,axis=0,weights=weights)
        if key.startswith('action_mse'):
            value=np.sqrt(value)
            key=key.replace('action_mse','action_rmse',1)
        result[key]=value.tolist() if value.ndim else float(value)
    return result


def summarize_episodes(rows,seed=42,replicates=2000):
    if not rows:raise ValueError('no evaluated episodes')
    if len({row['episode'] for row in rows}) != len(rows):raise ValueError('duplicate episode identity')
    complete=all(row['samples']==row['available_frames'] for row in rows)
    keys=[key for key in rows[0] if key not in {'episode','samples','available_frames','batches','seconds'}]
    result={'episodes_evaluated':[r['episode'] for r in rows],
            'samples_evaluated':sum(r['samples'] for r in rows),
            'complete_episode_coverage':complete,'per_episode':rows,
            'uncertainty_unit':'complete Episode; no independent-window assumption',
            'episode_equal_mean':{},'episode_bootstrap_95':{}}
    rng=np.random.default_rng(seed)
    picks=rng.integers(len(rows),size=(replicates,len(rows)))
    for key in keys:
        values=np.asarray([row[key] for row in rows])
        result['episode_equal_mean'][key]=np.mean(values,axis=0).tolist()
        result['episode_bootstrap_95'][key]=(np.percentile(values[picks].mean(axis=1),[2.5,97.5],axis=0).tolist()
                                             if complete and len(rows)>1 else None)
    return result
