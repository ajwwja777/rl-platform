"""Read-only selective cached-feature export; no images or model inference."""
from pathlib import Path
import io, json, pickle, hashlib, collections, base64
import numpy as np

root = Path('/media/agilex/Getea1/jiaan/data/datasets/plug_insertion/derived/rl-platform/rlt')
journal = root / 'replay_clean_v1/replay_journal.pkl'
data = journal.read_bytes()
stream = io.BytesIO(data)
rows = []
while stream.tell() < len(data):
    rows.append(pickle.load(stream))
index = collections.defaultdict(list)
for i,r in enumerate(rows):
    index[np.asarray(r['action_chunk'],np.float16).tobytes()].append(i)
matches = collections.defaultdict(list)
traces = {}
for f in sorted((root/'traces').rglob('*.jsonl')):
    raw = f.read_bytes()
    t = [json.loads(l) for l in raw.decode().splitlines() if l.strip()]
    if not t: continue
    a = np.asarray([r['action'][-7:] for r in t],np.float32)
    for start in range(len(t)-9):
        for i in index.get(a[start:start+10].astype(np.float16).tobytes(),[]):
            r=rows[i]
            if (np.array_equal(r['source_chunk'],[x['source'] for x in t[start:start+10]])
                and np.array_equal(r['proprio'],np.asarray(t[start]['observation']['state'][-7:],np.float32))
                and np.array_equal(r['next_proprio'],np.asarray(t[start+9]['next_observation']['state'][-7:],np.float32))):
                matches[i].append((str(f),start,a[start:start+10].copy()))
                traces[str(f)] = hashlib.sha256(raw).hexdigest()
groups=collections.defaultdict(list)
for i,r in enumerate(rows): groups[(r['collection_phase'],int(r['episode_id']))].append(i)
complete=[k for k,ids in groups.items() if all(len(matches[i])==1 for i in ids)]
selected=sorted(i for k in complete for i in groups[k])
cohort=[i for i,r in enumerate(rows) if r['collection_phase']=='online' and int(r['episode_id']) in [184,193,212,215,217,233]]
ids=sorted(set(selected+cohort))
arrays={k:np.stack([rows[i][k] for i in ids]) for k in rows[0] if k!='collection_phase'}
arrays['original_action']=np.stack([matches[i][0][2] if i in selected else np.asarray(rows[i]['action_chunk'],np.float32) for i in ids])
arrays['raw_verified']=np.asarray([i in selected for i in ids])
arrays['replay_index']=np.asarray(ids)
arrays['phase_online']=np.asarray([rows[i]['collection_phase']=='online' for i in ids])
buf=io.BytesIO();np.savez_compressed(buf,**arrays)
assert hashlib.sha256(journal.read_bytes()).hexdigest()==hashlib.sha256(data).hexdigest()
print(json.dumps({'journal':str(journal),'journal_sha256':hashlib.sha256(data).hexdigest(),
 'journal_records':len(rows),'selected_windows':len(ids),'raw_verified_windows':len(selected),
 'complete_raw_verified_episodes':[list(k) for k in complete],
 'excluded_partial_or_ambiguous_episodes':[list(k) for k in groups if k not in complete],
 'traces':traces,'row_origins':[{'replay_index':i,'path':matches[i][0][0],'trace_start':matches[i][0][1]} for i in selected],
 'semantics':'Complete retained Episodes only; historical trained/development data, no independent test. FP32 action is observed feedback in HIL, published command in policy slots.',
 'npz_sha256':hashlib.sha256(buf.getvalue()).hexdigest(),'npz_base64':base64.b64encode(buf.getvalue()).decode()}))
