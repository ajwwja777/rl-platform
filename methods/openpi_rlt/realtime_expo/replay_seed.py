"""Expert-only, complete C10 windows with task-local UUID and time contracts."""
import numpy as np

def episode_transitions(identity, prediction):
    if identity['outcome']!='success' or identity['outcome_source']!='user_confirmation_2026-10-15':
        raise ValueError('explicit demonstration outcome provenance required')
    # This cohort has one continuous valid segment per Episode. Do not invent
    # success at the ends of multiple disconnected segments.
    if len(identity['segments'])!=1:raise ValueError('multiple segments require separate terminal review')
    start,end=identity['segments'][0]
    frames=np.asarray(prediction['source_frame_indices'])
    if not np.array_equal(frames,np.arange(start,end)):raise ValueError('missing or reordered frames')
    n=len(frames)
    if n<10:raise ValueError('complete C10 required')
    keys=('predicted','target','z_rl','state')
    if not all(np.isfinite(prediction[k]).all() for k in keys):raise ValueError('nonfinite features')
    if prediction['predicted'].shape!=(n,50,7) or prediction['target'].shape!=(n,50,7) or prediction['z_rl'].shape!=(n,2048) or prediction['state'].shape!=(n,7):
        raise ValueError('feature shape mismatch')
    if identity['native_id']<=0:raise ValueError('native IDs must be positive')
    starts=sorted(set(range(0,n-9,10))|{n-10})
    rows=[]
    for t in starts:
        if not prediction['future_valid'][t,:10].all():raise ValueError('padded actions cannot enter expert Replay')
        terminal=t+10==n;nt=min(t+10,n-1)
        rewards=np.zeros(10,np.float32)
        if terminal:rewards[-1]=1
        rows.append(dict(z_rl=prediction['z_rl'][t],proprio=prediction['state'][t],ref_chunk=prediction['predicted'][t,:10],
            action_chunk=prediction['target'][t,:10],rewards=rewards,done=terminal,
            next_z_rl=prediction['z_rl'][nt],next_proprio=prediction['state'][nt],next_ref_chunk=prediction['predicted'][nt,:10],
            source=2,source_chunk=np.full(10,2,np.uint8),collection_phase='warmup',success=int(terminal),intervention_flag=True,
            episode_id=-int(identity['native_id']),episode_uuid=identity['uuid'],step_id=int(frames[t]),
            data_source='human_demonstration',autonomous_success=False,outcome_source=identity['outcome_source'],
            terminal_next_state='last_recorded_state_bootstrap_masked' if terminal else 'observed_C10_successor'))
    return rows

def normalization(rows):
    values=[]
    for r in rows:
        delta=np.asarray(r['action_chunk'],np.float64).copy()
        delta[:,:6]-=np.asarray(r['proprio'])[:6]
        values.append(delta)
    values=np.concatenate(values)
    return dict(norm_stats=dict(actions=dict(mean=values.mean(0).tolist(),std=values.std(0).tolist(),
        q01=np.quantile(values,.01,axis=0).tolist(),q99=np.quantile(values,.99,axis=0).tolist())),
        provenance=dict(scope='TRAIN experts only',chunk_len=10,model_hz=10,window_stride=10,
            final_complete_window=True,joint_delta_indices=list(range(6)),absolute_gripper_indices=[6],windows=len(rows)))
