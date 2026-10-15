"""Immutable, Episode-split left-arm expert loader; no production Replay writes."""
import hashlib,json
from pathlib import Path
import numpy as np
PROMPT='Unplug the black plug from the bottom socket of the power strip and insert it into the top socket.'
DEV_NATIVE_IDS={5,12,20,24,26} # Existing EXPO development cohort, never independent TEST.
class ExpertDataset:
 def __init__(self,root,split,horizon=50):
  if split not in ('train','dev'):raise ValueError(split)
  self.root=Path(root);self.horizon=horizon;self.rows=[];self.data={};self.identities=[]
  manifest=json.loads((self.root/'manifest.json').read_text());seen=set()
  for rec in manifest['records']:
   if rec['uuid'] in seen:raise ValueError('duplicate UUID')
   seen.add(rec['uuid'])
   actual='dev' if rec['native_id'] in DEV_NATIVE_IDS else 'train'
   if actual!=split:continue
   f=self.root/(Path(rec['file']).stem+'.npz')
   if hashlib.sha256(f.read_bytes()).hexdigest()!=rec['payload_sha256']:raise ValueError('payload SHA mismatch')
   with np.load(f,allow_pickle=False) as z:self.data[rec['uuid']]={k:z[k] for k in ('state','action','left','right','valid')}
   d=self.data[rec['uuid']]
   for start,end in rec['segments']:
    if not d['valid'][start:end].all():raise ValueError('invalid segment')
    for t in range(start,end):self.rows.append((rec['uuid'],t,end))
   self.identities.append(dict(uuid=rec['uuid'],native_id=rec['native_id'],frames=rec['accepted_frames']))
 def __len__(self):return len(self.rows)
 def __getitem__(self,index):
  uuid,t,end=self.rows[int(index)];d=self.data[uuid];future=np.minimum(np.arange(t,t+self.horizon),end-1)
  return dict(state=d['state'][t].copy(),actions=d['action'][future].copy(),images={'left_wrist_0_rgb':d['left'][t],'right_wrist_0_rgb':d['right'][t]},prompt=PROMPT)
