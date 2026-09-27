# Transactional per-episode LeRobot conversion; HDF5 is deleted only after verification.
import argparse,hashlib,json,os,sys,subprocess,fcntl,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT.parent.parent/'data/rlt/plug_v2'
CAMERAS=('cam_high','cam_left_wrist','cam_right_wrist')
def atomic(path,value):
    temp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    with temp.open('w') as f:
        json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def aligned_training_mask(facts,frames):
    # Match recorder source time with the runtime's ROS time. Pause waits are false.
    stamps=np.maximum(facts['rollout/topic_timestamp/front_left'],
                      facts['rollout/topic_timestamp/front_right'])
    trace_stamps=np.asarray([f['ros_timestamp'] for f in frames],np.float64)
    if not len(trace_stamps):return np.zeros(len(stamps),bool),np.full(len(stamps),-1,np.int64)
    order=np.argsort(trace_stamps);trace_stamps=trace_stamps[order]
    index=np.searchsorted(trace_stamps,stamps)
    left=np.maximum(index-1,0);right=np.minimum(index,len(trace_stamps)-1)
    selected=np.where(abs(stamps-trace_stamps[left])<=abs(stamps-trace_stamps[right]),left,right)
    chosen=order[selected]
    valid=np.asarray([frames[i]['valid_for_training'] for i in chosen],bool)
    valid &= np.abs(stamps-trace_stamps[selected])<=.05
    generation=np.asarray([frames[i]['generation'] for i in chosen],np.int64)
    return valid,generation
def convert(root,index):
    import h5py,pyarrow.parquet as pq,av
    from importlib.metadata import version
    assert version('lerobot')=='0.1.0'
    from lerobot.common.datasets import lerobot_dataset
    from lerobot.common.datasets.video_utils import encode_video_frames
    from functools import partial
    root=Path(root).resolve(strict=True);root.relative_to(BASE.resolve(strict=True))
    raw=root/f'episode_{index:06d}.hdf5'
    meta=json.loads((root/f'episode_{index:06d}.rlt.json').read_text())
    assert meta['cohort']=='plug_v2' and meta['episode_index']==index
    uuid=meta['episode_uuid'];assert uuid and '/' not in uuid
    releases=root/'lerobot';releases.mkdir(exist_ok=True)
    out=releases/uuid
    with (root/'.conversion.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if out.exists():
            receipt=json.loads((out/'conversion.json').read_text())
            assert receipt['status']=='validated' and receipt['source_uuid']==uuid
            assert receipt['source_index']==index and receipt['cohort']=='plug_v2'
            checks=json.loads((out/'checksums.json').read_text())
            actual={str(p.relative_to(out)) for p in out.rglob('*') if p.is_file() and p.name!='checksums.json'}
            assert actual==set(checks),'converted output file set changed'
            for rel,digest in checks.items():
                path=out/rel;path.resolve(strict=True).relative_to(out.resolve(strict=True))
                assert not path.is_symlink() and sha(path)==digest,'converted output checksum changed: '+rel
            if raw.exists():
                assert sha(raw)==receipt['source_sha256'];raw.unlink()
            return out
        staging=releases/('.'+uuid+'.converting')
        if staging.exists():raise RuntimeError('partial conversion retained for inspection: '+str(staging))
        checksum=sha(raw);facts={}
        with h5py.File(raw,'r') as h:
            assert h.attrs['completion_state']=='complete' and h.attrs['episode_uuid']==uuid
            assert float(h.attrs['fps'])==30
            state=h['observations/qpos'][:];actions=h['action'][:];n=len(actions)
            assert n>0 and state.shape==actions.shape==(n,14)
            def visitor(name,ds):
                if isinstance(ds,h5py.Dataset) and not name.startswith('observations/images/'):
                    facts[name]=np.asarray(ds.asstr()[:],dtype=str) if h5py.check_string_dtype(ds.dtype) else ds[:]
            h.visititems(visitor)
            attrs={k:(v.item() if isinstance(v,np.generic) else v) for k,v in h.attrs.items()}
            with np.load(meta['trace'],allow_pickle=False) as trace:
                frames=json.loads(str(trace['frames_json']))
            valid,generation=aligned_training_mask(facts,frames)
            for name in ('action','qpos','camera_high','camera_left','camera_right'):
                valid &= np.asarray(facts['rollout/valid_mask/'+name],bool)
            valid &= np.isfinite(state).all(axis=1)&np.isfinite(actions).all(axis=1)
            intervention=np.asarray(facts['rollout/is_intervention_left'],bool)|np.asarray(facts['rollout/is_intervention_right'],bool)
            lerobot_dataset.encode_video_frames=partial(encode_video_frames,vcodec='h264',crf=10)
            features={k:{'dtype':'float32','shape':(14,),'names':None} for k in ('observation.state','action')}
            features.update({'observation.images.'+c:{'dtype':'video','shape':(480,640,3),'names':['height','width','channels']} for c in CAMERAS})
            dataset=lerobot_dataset.LeRobotDataset.create(repo_id='jiaan/plug_v2_'+uuid,
                fps=30,root=staging,robot_type='cobot',features=features,
                use_videos=True,image_writer_threads=4)
            try:
                for i in range(n):
                    frame={'observation.state':state[i],'action':actions[i],
                           'task':'Insert the held plug into the socket.'}
                    frame.update({'observation.images.'+c:h['observations/images/'+c][i] for c in CAMERAS})
                    dataset.add_frame(frame)
                dataset.save_episode()
            finally:dataset.stop_image_writer()
            table=pq.read_table(staging/'data/chunk-000/episode_000000.parquet')
            np.testing.assert_array_equal(np.stack(table['action'].to_pylist()).astype(np.float32),actions)
            np.testing.assert_array_equal(np.stack(table['observation.state'].to_pylist()).astype(np.float32),state)
            maximum=0.;decoded=0
            for c in CAMERAS:
                p=staging/f'videos/chunk-000/observation.images.{c}/episode_000000.mp4'
                with av.open(str(p)) as video:
                    count=0;stream=video.streams.video[0]
                    assert stream.width==640 and stream.height==480 and float(stream.average_rate)==30
                    for image in video.decode(stream):
                        if count in (0,n//2,n-1):
                            mae=np.abs(image.to_ndarray(format='rgb24').astype(np.float32)-h['observations/images/'+c][count]).mean()
                            maximum=max(maximum,float(mae));assert mae<=5.
                        count+=1
                    assert count==n;decoded+=count
        np.savez_compressed(staging/'source_facts.npz',**facts)
        with np.load(staging/'source_facts.npz',allow_pickle=False) as archived:
            assert set(archived.files)==set(facts)
            for key in facts:np.testing.assert_array_equal(archived[key],facts[key])
        np.savez_compressed(staging/'training_mask.npz',valid=valid,generation=generation,intervention=intervention)
        receipt={'status':'validated','cohort':'plug_v2','source_uuid':uuid,'source_index':index,
                 'source_sha256':checksum,'source_attrs':attrs,'metadata':meta,'frames':n,
                 'training_frames':int(valid.sum()),'intervention_frames':int((valid&intervention).sum()),
                 'decoded_video_frames':decoded,'max_sample_rgb_mae':maximum,
                 'raw_policy':'delete only after numeric/video/checksum verification'}
        atomic(staging/'conversion.json',receipt)
        checks={str(p.relative_to(staging)):sha(p) for p in staging.rglob('*') if p.is_file()}
        atomic(staging/'checksums.json',checks)
        assert sha(raw)==checksum
        os.rename(staging,out)
        directory=os.open(releases,os.O_RDONLY);os.fsync(directory);os.close(directory)
        for rel,digest in checks.items():assert sha(out/rel)==digest
        raw.unlink()
        # Counts survive raw deletion. Labels and small traces remain.
        summary={'cohort':'plug_v2','converted_episodes':[]}
        for p in sorted(releases.glob('*/conversion.json')):
            item=json.loads(p.read_text());summary['converted_episodes'].append(
                {k:item[k] for k in ('source_uuid','source_index','frames','training_frames','intervention_frames')})
        atomic(root/'recording_summary.json',summary)
        print('EPISODE_CONVERTED',uuid,n,int(valid.sum()),flush=True)
        return out
def discard(root,index):
    import h5py
    root=Path(root).resolve(strict=True);root.relative_to(BASE.resolve(strict=True))
    raw=root/f'episode_{index:06d}.hdf5'
    meta=json.loads((root/f'episode_{index:06d}.rlt.json').read_text())
    assert meta['cohort']=='plug_v2' and meta['outcome']=='aborted'
    folder=root/'discarded';folder.mkdir(exist_ok=True)
    uuid=meta['episode_uuid'];assert uuid and '/' not in uuid
    receipt=folder/(uuid+'.json')
    if receipt.exists() and not raw.exists():return
    digest=sha(raw);facts={}
    with h5py.File(raw,'r') as h:
        assert h.attrs['completion_state']=='complete' and h.attrs['episode_uuid']==uuid
        def visit(name,ds):
            if isinstance(ds,h5py.Dataset) and not name.startswith('observations/images/'):
                facts[name]=np.asarray(ds.asstr()[:],dtype=str) if h5py.check_string_dtype(ds.dtype) else ds[:]
        h.visititems(visit)
    archive=folder/(uuid+'.npz')
    with archive.open('wb') as f:
        np.savez_compressed(f,**facts);f.flush();os.fsync(f.fileno())
    with np.load(archive,allow_pickle=False) as loaded:
        for key in facts:np.testing.assert_array_equal(loaded[key],facts[key])
    atomic(receipt,{'status':'discarded','source_sha256':digest,'metadata':meta,'facts_sha256':sha(archive)})
    assert sha(raw)==digest
    raw.unlink()
    print('ABORTED_RAW_DISCARDED',uuid,flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--episode-index',type=int,required=True);p.add_argument('--convert-only',action='store_true');p.add_argument('--discard',action='store_true')
    args=p.parse_args()
    if args.convert_only:
        if args.discard:discard(args.root,args.episode_index)
        else:convert(args.root,args.episode_index)
        return
    env=os.environ.copy()
    env['PYTHONPATH']=':'.join([str(ROOT),str(ROOT/'envs/machine-a-py311-overlay'),
                              str(ROOT/'code/openpi-rlt/src')])
    env['CUDA_VISIBLE_DEVICES']='';env['JAX_PLATFORMS']='cpu';env['PYTHONDONTWRITEBYTECODE']='1'
    command=['/home/agilex/junfeng/workspace/pi05_cobot/.venv-server/bin/python',
             '-m','methods.openpi_rlt.plug_v2.data_worker','--root',str(args.root),
             '--episode-index',str(args.episode_index),'--convert-only']
    if args.discard:command+=['--discard']
    subprocess.run(command,env=env,check=True)
    if args.discard:return
    from methods.openpi_rlt.plug_v2.replay import extract_episode
    extract_episode(args.root,args.episode_index)
if __name__=='__main__':main()
