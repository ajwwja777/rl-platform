# Durable fresh-cohort replay extraction. No robot publisher, no automatic warmup.
import argparse,json,os,time
from pathlib import Path
import numpy as np
os.environ['NO_PROXY']=','.join(filter(None,[os.environ.get('NO_PROXY',''),'127.0.0.1','localhost','::1']))
os.environ['no_proxy']=os.environ['NO_PROXY']
ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'runs/plug_v2'
BASE=ROOT.parent.parent/'data/rlt/plug_v2'
from methods.openpi_rlt.plug_v2.storage import validate_root,releases_for_phase
def atomic_npz(path,**arrays):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    with temp.open('wb') as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)
def extract_episode(root,index):
    root=validate_root(root)
    meta=json.loads((root/f'episode_{index:06d}.rlt.json').read_text())
    if meta['cohort']!='plug_v2' or meta['shadow'] or not meta['recording_enabled']:return
    with np.load(meta['trace'],allow_pickle=False) as trace:
        frames=json.loads(str(trace['frames_json']));plans=json.loads(str(trace['plans_json']))
    out=RUN/'replay'/meta['episode_uuid']
    # Keep factual traces available for the explicit preparation step. No GPU/RPC contention.
    atomic_npz(out.with_suffix('.source.npz'),
       metadata=np.array(json.dumps(meta)),frames_json=np.array(json.dumps(frames)),
       plans_json=np.array(json.dumps(plans)))
    print('REPLAY_SOURCE_READY',meta['episode_uuid'],len(plans),flush=True)
REPLAY_VERSION=2

def replay_folder():
    return RUN/'replay/v2'

def _sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_episode_rows(dataset_root,*,group,valid=None,generation=None,intervention=None,
                      terminal_success=True,episode_index=0,factual_trace=None,terminal_available=True):
    """CPU-only layout and factual features. Teacher features are deliberately not invented."""
    import pyarrow.parquet as pq
    from methods.openpi_rlt.plug_v2.replay_contract import teacher_rows,policy_rows
    root=Path(dataset_root)
    table=pq.read_table(root/f'data/chunk-{episode_index//1000:03d}/episode_{episode_index:06d}.parquet')
    state=np.stack(table['observation.state'].to_pylist()).astype(np.float32)
    actions=np.stack(table['action'].to_pylist()).astype(np.float32);n=len(actions)
    if state.shape!=actions.shape or state.shape!=(n,14):
        raise ValueError('invalid episode state/action')
    valid=np.ones(n,bool) if valid is None else np.asarray(valid,bool)
    generation=np.zeros(n,np.int64) if generation is None else np.asarray(generation,np.int64)
    if valid.shape!=(n,) or generation.shape!=(n,):raise ValueError('wrong mask length')
    valid=valid & np.isfinite(state).all(axis=1) & np.isfinite(actions).all(axis=1)
    terminal_frame=int(np.where(valid)[0][-1]) if valid.any() and terminal_available else -1
    if group=='expert':
        rows=teacher_rows(actions,valid,generation,terminal_frame=terminal_frame,
                          terminal_success=terminal_success,bc_allowed=True)
        return state,rows
    if factual_trace is None:raise ValueError('recorded policy/HIL episodes require their factual trace')
    with np.load(factual_trace,allow_pickle=False) as trace:
        frames=json.loads(str(trace['frames_json']));plans=json.loads(str(trace['plans_json']))
    with np.load(root/'source_facts.npz',allow_pickle=False) as facts:
        # The actor operates RIGHT joints; a left-only takeover is not an expert right action.
        right_hil=np.asarray(facts['rollout/is_intervention_right'],bool)
        stamps=np.maximum(facts['rollout/topic_timestamp/front_left'],facts['rollout/topic_timestamp/front_right'])
    if len(right_hil)!=n or len(stamps)!=n:raise ValueError('source fact length mismatch')
    trace_stamps=np.asarray([f['ros_timestamp'] for f in frames],np.float64)
    trace_order=np.argsort(trace_stamps);ordered_trace=trace_stamps[trace_order]
    if not len(ordered_trace):raise ValueError('recorded episode has no factual control frames')
    at=np.searchsorted(ordered_trace,stamps)
    before=np.maximum(at-1,0);after=np.minimum(at,len(ordered_trace)-1)
    at=np.where(abs(stamps-ordered_trace[before])<=abs(stamps-ordered_trace[after]),before,after)
    matched=trace_order[at]
    # Snapshot intervention flags can lag the joint timestamps at HIL release.
    # Match the actual handover mode/phase from the SAME generation as valid control.
    right_hil=np.asarray([frames[i]['valid_for_training'] and frames[i]['phase']=='hil'
        and 'right' in frames[i]['mode'].removeprefix('manual:').split('+')
        and frames[i]['mode'].startswith('manual:') for i in matched],bool)
    right_hil &= (abs(stamps-trace_stamps[matched])<=.05)
    right_hil &= generation==np.asarray([frames[i]['generation'] for i in matched])
    hil_valid=valid & right_hil
    rows=teacher_rows(actions,hil_valid,generation,terminal_frame=terminal_frame,
                      terminal_success=terminal_success,bc_allowed=(group=='success'))
    order=np.argsort(stamps);ordered=stamps[order]
    selected=np.searchsorted(ordered,trace_stamps)
    left=np.maximum(selected-1,0);right=np.minimum(selected,n-1)
    selected=np.where(abs(trace_stamps-ordered[left])<=abs(trace_stamps-ordered[right]),left,right)
    idx=order[selected]
    allowed=valid[idx] & (abs(trace_stamps-stamps[idx])<=.05)
    allowed &= generation[idx]==np.asarray([f['generation'] for f in frames])
    # Policy rows already require an actually sent policy command; snapshot HIL flags must not veto it.
    policy=policy_rows(frames,plans,terminal_success=terminal_success,
                      allow_bc=group=='success' and not any(f['valid_for_training'] and f['phase']=='hil' for f in frames),frame_allowed=allowed)
    if not terminal_available:
        for row in policy:row['done']=False;row['reward'][:]=0
    rows.extend(policy)
    return state,rows

def prepare_episode(dataset_root,*,uuid,group,split,model_rpc,valid=None,generation=None,
                    intervention=None,terminal_success=True,episode_index=0,factual_trace=None,
                    terminal_available=True):
    import av,hashlib
    from methods.openpi_rlt.plug_v2.replay_contract import pack_rows,validate_arrays
    root=Path(dataset_root)
    checkpoint=model_rpc.get_server_metadata()['checkpoint']
    source_paths=[root/f'data/chunk-{episode_index//1000:03d}/episode_{episode_index:06d}.parquet']
    source_paths += [root/p for p in ('checksums.json','artifact_checksums.json','selection_manifest.json','training_mask.npz','source_facts.npz') if (root/p).exists()]
    if factual_trace:source_paths.append(Path(factual_trace))
    signature=hashlib.sha256(json.dumps({'files':{str(p):_sha(p) for p in source_paths},
        'checkpoint':checkpoint,'version':REPLAY_VERSION,'group':group,'split':split,
        'terminal_available':terminal_available,'terminal_success':terminal_success,
        'valid':None if valid is None else np.asarray(valid,bool).tolist(),
        'generation':None if generation is None else np.asarray(generation).tolist()},sort_keys=True).encode()).hexdigest()
    target=replay_folder()/f'{uuid}_{episode_index:06d}.npz'
    if target.exists():
        with np.load(target,allow_pickle=False) as f:
            meta=json.loads(str(f['metadata']))
            if meta.get('source_signature')!=signature:raise ValueError('replay source changed; preserve prior cache and audit before rebuilding')
            validate_arrays({k:f[k] for k in f.files if k!='metadata'})
        return target
    state,rows=load_episode_rows(root,group=group,valid=valid,generation=generation,
        intervention=intervention,terminal_success=terminal_success,episode_index=episode_index,
        factual_trace=factual_trace,terminal_available=terminal_available)
    if not rows:return None
    anchors={row['frame'] for row in rows if row['stream']==1}
    images={t:{} for t in anchors}
    if anchors:
        for source,dest in [('cam_high','base_0_rgb'),('cam_left_wrist','left_wrist_0_rgb'),
                            ('cam_right_wrist','right_wrist_0_rgb')]:
            video_path=root/f'videos/chunk-{episode_index//1000:03d}/observation.images.{source}/episode_{episode_index:06d}.mp4'
            with av.open(str(video_path)) as video:
                for i,frame in enumerate(video.decode(video.streams.video[0])):
                    if i in anchors:images[i][dest]=frame.to_ndarray(format='rgb24')
    for row in rows:
        if row['stream']==0:continue  # The accepted rollout plan already contains real z/ref/context.
        t=row['frame']
        if len(images[t])!=3:raise ValueError('teacher image frame missing')
        prefix=np.zeros((50,14),np.float32)
        result=model_rpc.infer({'state':state[t],'images':images[t],
                                'action_prefix':prefix,'prefix_length':0})
        if result.get('prefix_length')!=0:raise RuntimeError('teacher cold-context mismatch')
        row.update(z=np.asarray(result['z_rl'],np.float32),
                   context=np.r_[state[t],np.zeros(84),0].astype(np.float32),
                   ref=np.asarray(result['ref_chunk'][:10],np.float32))
    arrays=pack_rows(rows)
    arrays['metadata']=np.array(json.dumps({'cohort':'plug_v2','source_uuid':uuid,'group':group,'split':split,
        'model_checkpoint':checkpoint,'format':'factual_policy_and_cold_teacher_v2','replay_version':REPLAY_VERSION,
        'source_signature':signature,'teacher_stride':2,'policy_stride':10,
        'policy_rows':sum(r['stream']==0 for r in rows),'teacher_rows':sum(r['stream']==1 for r in rows)}))
    atomic_npz(target,**arrays)
    print('REPLAY_PREPARED_V2',uuid,len(rows),group,split,flush=True)
    return target

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare']);args=p.parse_args()
    from methods.openpi_rlt.plug_v2.rpc import ModelClient
    from methods.openpi_rlt.plug_v2.replay_contract import assign_splits
    rpc=ModelClient(host='127.0.0.1',port=8020)
    try:
        server=rpc.get_server_metadata()
        manifest=json.loads((ROOT/'deployments/plug_v2/manifest.json').read_text())
        if server.get('cohort')!='plug_v2' or server['checkpoint']!=manifest['checkpoint']:
            raise RuntimeError('wrong frozen Stage-1 checkpoint/cohort')
        from methods.openpi_rlt.plug_v2.training_flow import session_phase
        if session_phase() not in ('offline','disarmed','ready','paused','terminal_pending','waiting_scene','stopped','fault'):
            raise RuntimeError('end/pause Session before replay preparation')
        items=[]
        for phase in ('warmup','online'):
            for release in releases_for_phase(phase):
                item=json.loads(release.read_text())
                if item.get('status')!='validated' or item.get('training_frames',0)<7:continue
                if item['metadata']['outcome'] not in ('success','failure'):continue
                if str(item['source_attrs']['checkpoint_id'])!=Path(manifest['checkpoint']).name:
                    raise ValueError('recording from another Stage-1 checkpoint')
                items.append((release,item,item['metadata']['outcome']))
        folder=replay_folder();folder.mkdir(parents=True,exist_ok=True)
        registry_path=RUN/'replay/split_registry.json'
        previous=json.loads(registry_path.read_text()) if registry_path.exists() else {}
        registry=assign_splits([(i['source_uuid'],g) for _,i,g in items],previous)
        tmp=registry_path.with_suffix('.'+str(os.getpid())+'.tmp')
        with tmp.open('w') as f:json.dump(registry,f,indent=2);f.flush();os.fsync(f.fileno())
        os.replace(tmp,registry_path)
        expert=BASE/'demonstrations/lerobot'
        selection=json.loads((expert/'selection_manifest.json').read_text())['episodes']
        last_parts={}
        for e in selection:last_parts[e['source_uuid']]=max(last_parts.get(e['source_uuid'],-1),e['episode_index'])
        for e in selection:
            prepare_episode(expert,uuid=e['source_uuid'],group='expert',split=e['split'],model_rpc=rpc,
                episode_index=e['episode_index'],terminal_available=e['episode_index']==last_parts[e['source_uuid']])
        for release,item,group in items:
            with np.load(release.parent/'training_mask.npz',allow_pickle=False) as mask:
                prepare_episode(release.parent,uuid=item['source_uuid'],group=group,
                    split=registry[item['source_uuid']],model_rpc=rpc,valid=mask['valid'],
                    generation=mask['generation'],intervention=mask['intervention'],
                    terminal_success=group=='success',factual_trace=item['metadata']['trace'])
    finally:rpc.close()
if __name__=='__main__':main()
