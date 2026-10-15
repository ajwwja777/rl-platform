#!/usr/bin/env python3
"""Read-only Cobot HDF5 -> streaming tar of RLT-sized RGB and physical arrays.
Run with fixed upstream openpi-client on PYTHONPATH. No files written on source host.
"""
import argparse,hashlib,io,json,sys,tarfile
from pathlib import Path
import h5py,numpy as np
from openpi_client.image_tools import resize_with_pad
ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);a=ap.parse_args()
root=a.source.resolve(strict=True)
records=[]
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as tar:
 def add(name,data):
  ti=tarfile.TarInfo(name);ti.size=len(data);tar.addfile(ti,io.BytesIO(data))
 for path in sorted(root.glob('episode_*.hdf5')):
  before=path.stat();digest=hashlib.sha256()
  with path.open('rb') as f:
   for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
  with h5py.File(path,'r') as h:
   assert h.attrs['completion_state']=='complete' and float(h.attrs['fps'])==10
   q=h['observations/qpos'][:,:7].astype(np.float32);cmd=h['rollout/coordinator_command'][:,:7].astype(np.float32)
   mask=np.isfinite(q).all(1)&np.isfinite(cmd).all(1)&(h['rollout/control_source_left'][:]==2)
   for key in ['coordinator_left','rear_left','camera_left','camera_right']:mask &= h['rollout/valid_mask/'+key][:]
   idx=np.flatnonzero(mask);segments=np.split(idx,np.flatnonzero(np.diff(idx)!=1)+1)
   chunks=[x for x in segments if len(x)>=2];assert chunks
   rec=dict(file=path.name,source_bytes=before.st_size,source_mtime_ns=before.st_mtime_ns,source_sha256=digest.hexdigest(),uuid=str(h.attrs['episode_uuid']),native_id=int(h.attrs['episode_index']),frames=len(q),accepted_frames=sum(map(len,chunks)),segments=[[int(x[0]),int(x[-1])+1] for x in chunks],outcome='success',outcome_source='user_confirmation_2026-10-15',fps=10,active_arm='left')
   arrays=dict(state=q,action=cmd,sample_timestamp=h['rollout/sample_timestamp'][:],valid=mask)
   for camera in ['left','right']:
    arrays[camera]=np.stack([resize_with_pad(h['observations/images/cam_'+camera+'_wrist'][i],224,224) for i in range(len(q))])
    arrays[camera+'_camera_timestamp']=h['rollout/topic_timestamp/camera_'+camera][:]
   mem=io.BytesIO();np.savez_compressed(mem,**arrays);payload=mem.getvalue();rec['payload_sha256']=hashlib.sha256(payload).hexdigest();rec['payload_bytes']=len(payload)
   add(path.stem+'.npz',payload);records.append(rec)
  after=path.stat();assert before.st_size==after.st_size and before.st_mtime_ns==after.st_mtime_ns
  print(json.dumps({'exported':path.name,'accepted':rec['accepted_frames']}),file=sys.stderr,flush=True)
 add('manifest.json',json.dumps(dict(source=str(root),status='export_complete',image_transform='fixed upstream openpi_client.resize_with_pad RGB 224x224 PIL bilinear; lossless NPZ',state_source='observations/qpos[:,0:7]',action_source='rollout/coordinator_command[:,0:7]',records=records),indent=2).encode())
