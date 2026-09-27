"""Read-only audit of accepted decisions and factual command coverage."""
import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
def audit(path):
    with np.load(path) as f:
        meta=json.loads(str(f['metadata']));frames=json.loads(str(f['frames_json']));plans=json.loads(str(f['plans_json']))
    by={(int(f['generation']),int(f['tick'])-1):f for f in frames
        if f['valid_for_training'] and f['phase']=='rollout' and f['command'] is not None}
    rows=[]
    for p in plans:
        g=int(p['generation']);t=int(p['tick']);d=int(p['prefix_length'])
        available=[(g,k) in by for k in range(t+d,t+d+10)]
        prefix=[by[(g,k)]['command'] for k in range(t,t+d) if (g,k) in by]
        mismatch=bool(prefix and not np.array_equal(np.asarray(prefix,dtype=np.float32),np.asarray(p['prefix'],np.float32)[:len(prefix)]))
        rows.append({'tick':t,'generation':g,'delay':d,'full_executed_tail':all(available),
                     'executed_tail_count':sum(available),'prefix_mismatch':mismatch,
                     'has_full_decision_evidence':all(k in p for k in ('raw_action_plan','conditioned_plan','reference_plan'))})
    valid=[f for f in frames if f['valid_for_training']]
    terminal_policy=bool(valid and valid[-1]['phase']=='rollout' and valid[-1]['command'] is not None)
    terminal_gap=None
    if terminal_policy:
        last=valid[-1];g=int(last['generation']);tick=int(last['tick'])-1
        eligible=[int(p['tick']) for p in plans if int(p['generation'])==g and int(p['tick'])<=tick]
        if eligible:terminal_gap=tick-max(eligible)
    return {'trace':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'uuid':meta.get('episode_uuid'),'outcome':meta.get('outcome'),
            'actor_version':meta.get('actor_version'),'decisions':len(rows),
            'full_tail_decisions':sum(x['full_executed_tail'] for x in rows),
            'partial_tail_decisions':sum(not x['full_executed_tail'] for x in rows),
            'missing_full_decision_evidence':sum(not x['has_full_decision_evidence'] for x in rows),
            'prefix_mismatches':sum(x['prefix_mismatch'] for x in rows),
            'terminal_policy':terminal_policy,'terminal_offset_from_last_decision':terminal_gap,
            'terminal_outside_nominal_C10':terminal_gap is not None and terminal_gap>=10}
def main():
    from .storage import releases_for_phase
    paths=set()
    for phase in ('warmup','online'):
        for release in releases_for_phase(phase):
            d=json.loads(release.read_text())
            if d.get('status')=='validated' and d['metadata'].get('outcome') in ('success','failure'):
                paths.add(Path(d['metadata']['trace']))
    rows=[audit(p) for p in sorted(paths)]
    keys=('decisions','full_tail_decisions','partial_tail_decisions','missing_full_decision_evidence','prefix_mismatches','terminal_outside_nominal_C10')
    out=ROOT/'runs/plug_v2/diagnostics/upstream-audit-20260919/rtc-decision-coverage.json'
    result={'episodes':len(rows),'totals':{k:sum(x[k] for x in rows) for k in keys},'rows':rows,
            'note':'Delayed action tails are valid only with augmented prefix state; this audit does not infer unobserved planned actions or move rewards.'}
    out.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
