"""Export user-confirmed TRAIN demonstrations to an independent online seed.

Never changes HDF5 labels or mixes DEV. Requires an explicit confirmation string
and exact hashes from the immutable SFT conversion manifest.
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    for flag in ('project','manifest','raw-root','output'):p.add_argument('--'+flag,type=Path,required=True)
    p.add_argument('--num-data',type=int,default=25)
    p.add_argument('--success-confirmation',required=True)
    a=p.parse_args()
    if not a.success_confirmation.strip():raise ValueError('Successful outcomes require operator confirmation')
    sys.path.insert(0,str(a.project))
    from expo_ft.env.cobot_dataset import iter_segment,episode_metadata
    original=json.loads(a.manifest.read_text())
    assert original['split']=='train' and original['active_arm']=='left' and original['fps']==10
    selected=sorted(set(e['source_episode_id'] for e in original['episodes']))[:a.num_data]
    assert len(selected)==a.num_data
    a.output.mkdir(parents=True,exist_ok=False)
    result=dict(schema=1,source_manifest=str(a.manifest),source_manifest_sha256=sha256(a.manifest),
        split='train',source_episode_ids=selected,confirmation=a.success_confirmation,
        confirmed_outcome='success',episodes=[],frames=0)
    for identity in selected:
        segments=[e for e in original['episodes'] if e['source_episode_id']==identity]
        # Do not invent success at the end of an earlier interrupted segment.
        assert len(segments)==1, 'Multiple segments require an explicit terminal audit'
        e=segments[0];source=a.raw_root/Path(e['source_path']).name
        assert sha256(source)==e['source_sha256'],str(source)+' differs from SFT source'
        metadata=episode_metadata(source)
        assert metadata['episode_uuid']==e['episode_uuid']
        assert [e['start'],e['end']] in metadata['segments']
        tail=metadata['frames']-e['end']
        terminal_audit={'discarded_tail_frames':tail,'success_at_last_accepted_human_frame':True}
        if tail:
            import h5py
            from expo_ft.env.cobot_dataset import REQUIRED_VALIDITY
            assert tail==1 and e['start']==0 and metadata['termination_reason']=='operator_stop'
            with h5py.File(source,'r') as raw:
                assert raw['rollout/control_source_left'][e['end']-1:e['end']+1].tolist()==[2,1]
                assert all(bool(raw['rollout/valid_mask/'+key][-1]) for key in REQUIRED_VALIDITY)
                state=np.asarray(raw['observations/qpos'][-2:,:7])
                command=np.asarray(raw['rollout/coordinator_command'][-2:,:7])
                assert np.isfinite(state).all() and np.isfinite(command).all()
                terminal_audit.update(discarded_control_source='POLICY',
                    last_state_delta=(state[-1]-state[-2]).tolist(),
                    last_command_delta=(command[-1]-command[-2]).tolist(),
                    reason='Same accepted human segment as SFT; user confirmed entire demonstration successful')
        rows=list(iter_segment(source,e['start'],e['end']))
        data={k:np.stack([row[k] for row in rows]) for k in rows[0]}
        path=a.output/('episode_%06d.npz'%identity)
        np.savez_compressed(path,**data)
        result['episodes'].append(dict(source_episode_id=identity,source=str(source),
            source_sha256=e['source_sha256'],episode_uuid=e['episode_uuid'],start=e['start'],end=e['end'],
            file=path.name,sha256=sha256(path),frames=len(rows),terminal_audit=terminal_audit))
        result['frames']+=len(rows)
        print('verified and exported',identity,len(rows),flush=True)
    (a.output/'manifest.json').write_text(json.dumps(result,indent=2))
    print('complete',len(result['episodes']),result['frames'],flush=True)

if __name__=='__main__':main()
