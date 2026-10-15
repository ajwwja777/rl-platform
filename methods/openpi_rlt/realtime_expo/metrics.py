"""Physical-unit diagnostics; complete episodes are the uncertainty unit."""
import numpy as np

def episode_metrics(predicted, target, future_valid):
    predicted=np.asarray(predicted); target=np.asarray(target); valid=np.asarray(future_valid,dtype=bool)
    if predicted.shape!=target.shape or predicted.ndim!=3 or predicted.shape[2]!=7 or valid.shape!=predicted.shape[:2]:
        raise ValueError('expected matching N,H,7 predictions and targets with N,H validity')
    if not len(predicted) or not valid[:,0].all() or not np.isfinite(predicted).all() or not np.isfinite(target).all():
        raise ValueError('nonfinite or empty/incomplete observations')
    error=np.abs(predicted-target); result={}
    for horizon in (10,50):
        mask=valid[:,:horizon]; errors=error[:,:horizon][mask]
        result[f'c{horizon}_mae_per_dim']=errors.mean(0).tolist()
        result[f'c{horizon}_p95_per_dim']=np.percentile(errors,95,axis=0).tolist()
        result[f'c{horizon}_joint_mae_rad']=float(errors[:,:6].mean())
        result[f'c{horizon}_gripper_mae_m']=float(errors[:,6].mean())
        result[f'c{horizon}_valid_action_rows']=int(mask.sum())
    pairs=valid[:,1:] & valid[:,:-1]
    second=valid[:,2:] & valid[:,1:-1] & valid[:,:-2]
    velocity=np.diff(predicted,axis=1); acceleration=np.diff(predicted,n=2,axis=1)
    result['predicted_step_abs_per_dim']=np.abs(velocity[pairs]).mean(0).tolist() if pairs.any() else None
    result['predicted_second_diff_abs_per_dim']=np.abs(acceleration[second]).mean(0).tolist() if second.any() else None
    return result

def complete_episode_summary(rows,seed=42):
    if not rows or len({r['uuid'] for r in rows})!=len(rows):raise ValueError('empty/duplicate episodes')
    if any(r['frames']!=r['evaluated_frames'] for r in rows):raise ValueError('partial episodes cannot use this summary')
    keys=('c10_joint_mae_rad','c10_gripper_mae_m','c50_joint_mae_rad','c50_gripper_mae_m','c10_mae_per_dim','c10_p95_per_dim')
    picks=np.random.default_rng(seed).integers(len(rows),size=(4000,len(rows)))
    result={'episodes':len(rows),'frames':sum(r['frames'] for r in rows),'uncertainty_unit':'complete Episode; overlapping windows are not independent','episode_equal_mean':{},'episode_bootstrap_95':{}}
    for key in keys:
        values=np.asarray([r['metrics'][key] for r in rows])
        result['episode_equal_mean'][key]=values.mean(0).tolist()
        result['episode_bootstrap_95'][key]=np.percentile(values[picks].mean(1),[2.5,97.5],axis=0).tolist() if len(rows)>1 else None
    return result
