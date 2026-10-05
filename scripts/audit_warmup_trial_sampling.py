#!/usr/bin/env python3
"""Recover only new trial batches whose complete index hash was recorded.

Historical production batches cannot be recovered this way. This script verifies
the newly generated indices against each completed experiment's receipt first.
"""
import argparse,hashlib,json,pickle,random
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    report=json.loads(a.report.read_text());assert len(report['runs'])==6 and report['sources_unchanged']
    paths=[Path(k) for k in report['source_sha256'] if '/experts120_5000/replay/' in k]
    assert len(paths)==1
    path=paths[0];assert hashlib.sha256(path.read_bytes()).hexdigest()==report['source_sha256'][str(path)]
    rows=[]
    with path.open('rb') as f:
        while True:
            try:rows.append(pickle.load(f))
            except EOFError:break
    ids=list(range(len(rows)));random.Random(42).shuffle(ids);size=max(1,int(len(rows)*.05))
    optimization=[rows[i] for i in ids[size:]]
    episodes=np.asarray([r['episode_id'] for r in optimization]);excluded=set(report['excluded_episode_ids'])
    eligible=np.flatnonzero(~np.isin(episodes,list(excluded)))
    source=np.stack([r['source_chunk'] for r in optimization]);success=np.asarray([r['success'] for r in optimization])
    source_terminal=np.asarray([r['done'] for r in optimization])
    output={'source_journal_sha256':report['source_sha256'][str(path)],'optimization_records':len(optimization),
        'internal_transition_monitor_records':size,'runs':[],
        'label_overlap':'expert/rollout partition Episodes; success, HIL-window and terminal are overlapping labels, not mutually exclusive strata.',
        'boundary':'Only six new CPU trials with complete batch index SHA. Does not recover missing historical Online/Warmup batches. Sampling counts are not independent outcome observations.'}
    for run in report['runs']:
        indices=np.random.default_rng(run['seed']).integers(0,len(optimization),size=(report['updates'],report['batch']))
        if run['condition']=='exclude_wrong_task':
            mask=np.isin(episodes[indices],list(excluded))
            indices[mask]=np.random.default_rng(run['seed']+10000).choice(eligible,size=int(mask.sum()))
        actual_hash=hashlib.sha256(indices.tobytes()).hexdigest();assert actual_hash==run['indices_sha256']
        frequency=np.bincount(indices.ravel(),minlength=len(optimization));draws=int(frequency.sum())
        human=np.isin(source,[2,3]);ep_draws={str(int(ep)):int(frequency[episodes==ep].sum()) for ep in np.unique(episodes)}
        expert=np.logical_or(episodes<0,episodes>=100000)
        output['runs'].append({'label':run['label'],'verified_indices_sha256':actual_hash,
            'batch_count':len(indices),'transition_draws':draws,'expert_transition_draw_ratio':float(frequency[expert].sum()/draws),
            'rollout_transition_draw_ratio':float(frequency[~expert].sum()/draws),
            'hil_window_draw_ratio':float(frequency[human.any(1)].sum()/draws),
            'human_action_slot_draw_ratio':float((frequency*human.sum(1)).sum()/(draws*source.shape[1])),
            'success_transition_draw_ratio':float(frequency[success==1].sum()/draws),
            'terminal_transition_draw_ratio':float(frequency[source_terminal].sum()/draws),
            'unique_optimization_records_drawn':int(np.count_nonzero(frequency)),'episode_draw_counts':ep_draws,
            'record_draw_count_range':[int(frequency[frequency>0].min()),int(frequency.max())]})
    (a.report.parent/'actual_sampling.json').write_text(json.dumps(output,indent=2,allow_nan=False))

if __name__=='__main__':main()
