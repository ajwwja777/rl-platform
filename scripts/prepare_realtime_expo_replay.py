#!/usr/bin/env python3
"""Create an independent expert seed after complete selected-model evaluation."""
import argparse,hashlib,json,pickle,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from methods.openpi_rlt.realtime_expo.dataset import DEV_NATIVE_IDS
from methods.openpi_rlt.realtime_expo.replay_seed import episode_transitions,normalization
ap=argparse.ArgumentParser()
for key in ('data','evaluation','output'):ap.add_argument('--'+key,type=Path,required=True)
a=ap.parse_args()
if a.output.exists():raise FileExistsError('seed destination must be new')
manifest=json.loads((a.data/'manifest.json').read_text())
evaluation=json.loads((a.evaluation/'metrics.json').read_text())
sha=hashlib.sha256((a.data/'manifest.json').read_bytes()).hexdigest()
if evaluation['dataset_manifest_sha256']!=sha:raise ValueError('dataset identity mismatch')
if evaluation['effective_batch_size']!=1:raise ValueError('features must use deployed single-observation inference')
groups={'train':[],'dev':[]};identity_rows=[]
for identity in manifest['records']:
    split='dev' if identity['native_id'] in DEV_NATIVE_IDS else 'train'
    found=[r for r in evaluation['cohorts'][split]['per_episode'] if r['uuid']==identity['uuid']]
    if len(found)!=1 or found[0]['evaluated_frames']!=identity['accepted_frames']:raise ValueError('incomplete evaluation')
    file=a.evaluation/f"{split}_{identity['native_id']:06d}.npz"
    with np.load(file,allow_pickle=False) as values:
        rows=episode_transitions(identity,{k:values[k] for k in values.files})
    groups[split].extend(rows)
    identity_rows.append(dict(uuid=identity['uuid'],native_id=identity['native_id'],split=split,transitions=len(rows),feature_sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
train_uuid={r['episode_uuid'] for r in groups['train']}
if train_uuid & {r['episode_uuid'] for r in groups['dev']}:raise ValueError('development leakage')
a.output.mkdir(parents=True)
for split,rows in groups.items():
    with (a.output/f'{split}_expert_journal.pkl').open('xb') as f:
        for r in rows:pickle.dump(r,f,protocol=4)
(a.output/'norm_stats_delta.json').write_text(json.dumps(normalization(groups['train']),indent=2))
receipt=dict(stage1_checkpoint=evaluation['checkpoint'],stage1_metadata=evaluation['metadata'],dataset_manifest_sha256=sha,
    episodes=identity_rows,counts={k:len(v) for k,v in groups.items()},source='all human demonstrations; no autonomous success/failure observations',
    boundary='Only TRAIN journal can seed learner. DEV is repeated-use evaluation, never independent TEST. Terminal reward is user-confirmed episode outcome, not per-frame semantic annotation.',
    journals_sha256={k:hashlib.sha256((a.output/f'{k}_expert_journal.pkl').read_bytes()).hexdigest() for k in groups})
(a.output/'manifest.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt['counts']),flush=True)
