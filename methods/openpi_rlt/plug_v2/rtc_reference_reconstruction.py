"""Recover reference-only decisions from factual z/ref/prefix and verify sent commands.
Never fabricates labels, overwrites replay, or publishes actions.
"""
import json
from pathlib import Path
import numpy as np
from .rtc_queue import CommandFilter
from .rtc_history_audit import ROOT

def main():
    coverage=json.loads((ROOT/'runs/plug_v2/diagnostics/upstream-audit-20260919/rtc-decision-coverage.json').read_text())
    profiles={'legacy':CommandFilter(), 'bounded':CommandFilter(velocity=.1,acceleration=.9)}
    historical=CommandFilter();historical.lower[:]=-np.inf;historical.upper[:]=np.inf
    profiles['historical_legacy_without_clamp']=historical
    summaries=[]
    for entry in coverage['rows']:
        if entry['actor_version']!=-1:continue
        with np.load(entry['trace']) as f:
            frames=json.loads(str(f['frames_json']));plans=json.loads(str(f['plans_json']))
        latest={}
        for plan in plans:
            key=(int(plan['generation']),int(plan['tick']))
            if key not in latest or int(plan['sequence'])>int(latest[key]['sequence']):latest[key]=plan
        plans=list(latest.values())
        sent={(int(f['generation']),int(f['tick'])-1):np.asarray(f['command'],np.float32) for f in frames if f['valid_for_training'] and f['phase']=='rollout' and f['command'] is not None}
        errors={k:[] for k in profiles};counts=[]
        for p in plans:
            g=int(p['generation']);t=int(p['tick']);d=int(p['prefix_length']);state=np.asarray(p['state'],np.float32)
            prefix=np.zeros((50,14),np.float32);prefix[:6]=p['prefix']
            raw=np.tile(state,(50,1));raw[:d]=prefix[:d];raw[d:d+10]=p['ref']
            available=[i for i in range(10) if (g,t+d+i) in sent];counts.append(len(available))
            for name,conditioner in profiles.items():
                try:prediction=conditioner.plan(raw,state,prefix,d)[d:d+10]
                except ValueError:
                    errors[name].append(1e30);continue
                errors[name].append(max((float(abs(prediction[i]-sent[g,t+d+i]).max()) for i in available),default=None))
        valid_errors={k:[x for x in v if x is not None] for k,v in errors.items()}
        matched=[k for k,v in valid_errors.items() if v and max(v)<1e-6]
        summaries.append({'uuid':entry['uuid'],'trace':entry['trace'],'decisions':len(plans),
                          'partial_decisions':sum(0<c<10 for c in counts),'unobserved_decisions':sum(c==0 for c in counts),
                          'matching_profiles':matched,'profile_max_command_error':{k:max(v) if v else None for k,v in valid_errors.items()},
                          'eligible_for_reconstruction':len(matched)==1 and all(c>0 for c in counts)})
    out=ROOT/'runs/plug_v2/diagnostics/upstream-audit-20260919/rtc-reference-reconstruction.json'
    result={'episodes':len(summaries),'exact_profile_episodes':sum(x['eligible_for_reconstruction'] for x in summaries),
            'rows':summaries,'status':'diagnostic_only; matching observed commands does not recover missing termination observations'}
    out.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
    print('unmatched',[(x['uuid'],x['profile_max_command_error']) for x in summaries if not x['eligible_for_reconstruction']][:10])
if __name__=='__main__':main()
