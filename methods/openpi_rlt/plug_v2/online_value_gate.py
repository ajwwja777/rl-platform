"""Pure value-evaluation and fixed release-gate helpers."""
from __future__ import annotations
import math
from collections import defaultdict
from typing import Iterable, Mapping
import numpy as np

VALUE_FIELDS=('q_data','q_data_early','q_terminal','q_reference','q_actor')
GROUPS=('all_online','autonomous','policy_terminal','hil','warmup')

def _metric(rows,field):
    positive=[float(r[field]) for r in rows if r.get('success') and r.get(field) is not None and math.isfinite(float(r[field]))]
    negative=[float(r[field]) for r in rows if not r.get('success') and r.get(field) is not None and math.isfinite(float(r[field]))]
    auc=float(np.mean([1. if a>b else .5 if a==b else 0. for a in positive for b in negative])) if positive and negative else None
    return {'success_n':len(positive),'failure_n':len(negative),
            'success_mean':float(np.mean(positive)) if positive else None,
            'failure_mean':float(np.mean(negative)) if negative else None,
            'separation':float(np.mean(positive)-np.mean(negative)) if positive and negative else None,
            'auc':auc}

def summarize_value_episodes(rows):
    result={}
    for group in GROUPS:
        chosen=[row for row in rows if group in row.get('groups',())]
        result[group]={field:_metric(chosen,field) for field in VALUE_FIELDS}
    return result

def fixed_value_gate(report, *, min_count=3, min_auc=None,
                     early_auc=.50, terminal_auc=.60, actor_auc=.55):
    reasons=[];summary=report.get('value_summary',{})
    thresholds={'q_data_early':early_auc,'q_terminal':terminal_auc,'q_actor':actor_auc}
    if min_auc is not None:thresholds={key:float(min_auc) for key in thresholds}
    for group in ('autonomous','policy_terminal'):
        for field,threshold in thresholds.items():
            metric=summary.get(group,{}).get(field,{})
            if int(metric.get('success_n') or 0)<min_count or int(metric.get('failure_n') or 0)<min_count:
                reasons.append('value:'+group+':'+field+':count');continue
            auc=metric.get('auc');separation=metric.get('separation')
            if auc is None or not math.isfinite(float(auc)) or float(auc)<threshold:
                reasons.append('value:'+group+':'+field+':auc')
            if separation is None or not math.isfinite(float(separation)) or float(separation)<=0:
                reasons.append('value:'+group+':'+field+':separation')
    return reasons

def validate_episode_splits(episodes):
    seen={}
    for meta,_data in episodes:
        uuid=str(meta['uuid']); split=str(meta['split'])
        previous=seen.setdefault(uuid,split)
        if previous!=split: raise ValueError('episode split leakage for '+uuid)
    return seen

def aggregate_metric_window(metrics):
    if not metrics: return {}
    actor=[row for row in metrics if int(row.get('did_actor_update',0))==1]
    result=dict(metrics[-1])
    critic_fields=('critic_loss','q1_mean','q2_mean','target_q_mean','td_error')
    actor_fields=('actor_loss','actor_q','bc_penalty','bc_human_penalty','bc_ref_penalty','delta_penalty','weighted_bc','weighted_q','weighted_delta','human_mask_ratio','policy_mask_ratio','actor_q_mask_ratio')
    sample_fields=tuple(key for key in metrics[-1] if key.startswith('sample_'))
    def mean(rows,key):
        values=[float(row[key]) for row in rows if key in row and isinstance(row[key],(int,float)) and math.isfinite(float(row[key]))]
        return float(np.mean(values)) if values else None
    for key in critic_fields+sample_fields:
        value=mean(metrics,key)
        if value is not None: result[key]=value
    for key in actor_fields:
        value=mean(actor,key)
        if value is not None: result[key]=value
        elif key in result: result.pop(key)
    result['actor_update_count']=len(actor)
    result['did_actor_update']=1 if actor else 0
    return result

def episode_q_payload(rows, points=20):
    groups=[]
    definitions=(('autonomous_success','autonomous',True),('autonomous_failure','autonomous',False),('hil_success','hil',True),('hil_failure','hil',False))
    target=np.linspace(0.,1.,int(points))
    for name,group,success in definitions:
        curves=[]
        for row in rows:
            curve=row.get('q_curve')
            if group not in row.get('groups',()) or bool(row.get('success'))!=success or not isinstance(curve,(list,tuple)) or not curve: continue
            values=np.asarray(curve,dtype=np.float64)
            if not np.isfinite(values).all(): continue
            source=np.linspace(0.,1.,len(values)); curves.append(np.interp(target,source,values))
        if curves:
            stack=np.stack(curves)
            groups.append({'name':name,'median':np.median(stack,axis=0).tolist(),'q25':np.quantile(stack,.25,axis=0).tolist(),'q75':np.quantile(stack,.75,axis=0).tolist(),'count':len(curves)})
    metric=summarize_value_episodes(rows)['autonomous']['q_data']
    return {'groups':groups,'autonomous_separation':metric['separation'],'auc':metric['auc'],'success_n':metric['success_n'],'failure_n':metric['failure_n']}

def build_value_episode(meta, q_values, source_chunk, done):
    q=np.asarray(q_values,dtype=np.float64); source=np.asarray(source_chunk); terminal=np.asarray(done,dtype=bool)
    transition_human=np.any(source==2,axis=1); policy=~transition_human; groups=[]
    phase=str(meta.get('phase',''))
    if phase=='online': groups.append('all_online')
    if phase=='online' and not bool(transition_human.any()): groups.append('autonomous')
    if phase=='online' and bool(transition_human.any()) and not bool(meta.get('expert',False)): groups.append('hil')
    if phase=='warmup': groups.append('warmup')
    terminal_policy=terminal&policy
    if phase=='online' and terminal_policy.any(): groups.append('policy_terminal')
    if not policy.any() or not groups: return None
    indices=np.flatnonzero(policy)
    return {'uuid':str(meta['uuid']),'success':bool(meta['success']),'groups':groups,
            'q_data':float(q[policy,0].mean()),'q_data_early':float(q[indices[:3],0].mean()),
            'q_terminal':float(q[terminal_policy,0].mean()) if terminal_policy.any() else None,
            'q_reference':float(q[policy,1].mean()),'q_actor':float(q[policy,2].mean()),
            'q_curve':[float(x) for x in q[policy,0]]}
