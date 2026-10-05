#!/usr/bin/env python3
"""Summarize uniquely joined trace times with complete-Episode uncertainty."""
import argparse,collections,json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True);p.add_argument('--expected',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    raw=json.loads(a.report.read_text());expected=json.loads(a.expected.read_text())
    identities=[r for values in expected.values() for r in values];groups=collections.defaultdict(list);episodes=collections.defaultdict(list)
    for r in raw['matches']:groups[(r['split'],r['episode'],r['step'])].append(r)
    for r in identities:episodes[(r['split'],r['episode'])].append(r)
    report={'status':'已验证','finding':'Recorded completed-step spans differ from a common20Hz physical-time target. Hardware command/execution timing not identified.',
        'expected_rollout_transitions':len(identities),'unique_exact_matched_windows':sum(len(v)==1 for v in groups.values()),
        'ambiguous_identity_windows':sum(len(v)>1 for v in groups.values()),
        'unmatched_windows':sum((r['split'],r['episode'],r['step']) not in groups for r in identities),'episodes':[],'summaries':[],
        'boundary':raw['boundary']+' Bootstrap includes only fully covered complete Episodes; partial/ambiguous identities retained explicitly. Recorded step spans do not identify which latency stage caused slowness.',
        'identity_rule':'Exact full serializedFP16 C10 actions + uint8 sources + FP32 current and next state, unique match only',
        'actual_archive_expert_id_rule':'>=100000 (120 expert Episodes /1186 rows); production Replay uses remapped negative expert IDs, not silently assumed identical.'}
    assert report['unique_exact_matched_windows']+report['ambiguous_identity_windows']+report['unmatched_windows']==len(identities)
    for (split,ep),rows in sorted(episodes.items()):
        matched=[groups[(split,ep,r['step'])][0] for r in rows if len(groups.get((split,ep,r['step']),[]))==1]
        complete=any(r['done'] for r in rows);fully=complete and len(matched)==len(rows)
        item={'split':split,'episode':ep,'expected_windows':len(rows),'uniquely_matched_windows':len(matched),'source_episode_terminal':complete,'fully_covered_complete_episode':fully}
        for label,select in [('policy_only',lambda r:not any(s in [2,3] for s in r['trace_sources'])),('human_only',lambda r:all(s in [2,3] for s in r['trace_sources'])),('mixed',lambda r:any(s in [2,3] for s in r['trace_sources']) and not all(s in [2,3] for s in r['trace_sources']))]:
            selected=[r['recorded_first_to_last_seconds'] for r in matched if select(r)];item[label]={'matched_windows':len(selected),'mean_seconds':float(np.mean(selected)) if selected else None,'median_seconds':float(np.median(selected)) if selected else None}
        report['episodes'].append(item)
    for split in ['train','development']:
        for label in ['policy_only','human_only','mixed']:
            es=[r for r in report['episodes'] if r['split']==split and r['fully_covered_complete_episode'] and r[label]['matched_windows']];values=np.asarray([r[label]['mean_seconds'] for r in es]);ci=None
            if len(values)>1:
                picks=np.random.default_rng(42).integers(len(values),size=(10000,len(values)));ci=np.quantile(values[picks].mean(1),[.025,.975]).tolist()
            report['summaries'].append({'split':split,'label':label,'fully_covered_complete_episodes':len(es),'equal_episode_mean_seconds':float(values.mean()) if len(values) else None,'episode_bootstrap_ci95':ci})
    a.output.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

if __name__=='__main__':main()
