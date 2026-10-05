#!/usr/bin/env python3
"""Read-only registered Cobot Replay structural audit. No model imports."""
from pathlib import Path
import collections
import hashlib
import json
import pickle
import numpy as np

path=Path('/media/agilex/Getea1/jiaan/data/datasets/plug_insertion/derived/rl-platform/rlt/replay_clean_v1/replay_journal.pkl')
raw=path.read_bytes();rows=[]
import io
stream=io.BytesIO(raw)
while stream.tell()<len(raw):rows.append(pickle.load(stream))
groups=collections.defaultdict(list)
for i,r in enumerate(rows):groups[(r['collection_phase'],int(r['episode_id']))].append(i)
identities=[(r['collection_phase'],int(r['episode_id']),int(r['step_id'])) for r in rows]
finite={k:sum(not np.isfinite(np.asarray(r[k])).all() for r in rows) for k in ['z_rl','proprio','action_chunk','ref_chunk','next_z_rl','next_proprio','next_ref_chunk','rewards']}
episodes=[]
for key,indices in sorted(groups.items()):
 lookup={int(rows[i]['step_id']):i for i in indices}
 terminals=[rows[i] for i in indices if rows[i]['done']]
 outcomes=set(int(r['success']) for r in terminals)
 matched=[];mismatched=[]
 for i in indices:
  r=rows[i];next_start=int(r['step_id'])+len(r['action_chunk'])
  if next_start in lookup and not r['done']:
   n=rows[lookup[next_start]]
   err=float(np.max(np.abs(np.asarray(r['next_proprio'],np.float32)-np.asarray(n['proprio'],np.float32))))
   (matched if err==0 else mismatched).append(dict(step_id=int(r['step_id']),next_step_id=next_start,max_error_rad_or_m=err))
 episode_hil=any(np.isin(rows[i]['source_chunk'],[2,3]).any() for i in indices)
 expert=key[1]<0 or key[1]>=100000
 episodes.append(dict(phase=key[0],episode_id=key[1],transitions=len(indices),terminal_windows=len(terminals),
  terminal_outcomes=sorted(outcomes),expert=expert,episode_hil=episode_hil,
  exact_adjacent_state_links=len(matched),mismatched_adjacent_state_links=mismatched,
  first_step=min(int(rows[i]['step_id']) for i in indices),last_step=max(int(rows[i]['step_id']) for i in indices)))
assert path.read_bytes()==raw,'Replay changed during read-only audit'
report=dict(journal=str(path),journal_sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),records=len(rows),episodes=len(groups),
 duplicate_identities=len(rows)-len(set(identities)),nonfinite_records=finite,episode_rows=episodes,
 precision=collections.Counter(str(np.asarray(r['action_chunk']).dtype) for r in rows),
 source_unchanged=True,boundary='State-link comparisons require a stored anchor at step+C; gaps are unknown. Structural consistency is not human command/feedback timing or semantic insertion success.')
print(json.dumps(report,allow_nan=False))
