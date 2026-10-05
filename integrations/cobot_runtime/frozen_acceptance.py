"""Assess complete-Episode frozen trials; never deploy or start learning.

This tool assesses supplied provenance, not its physical truth. A reused
development cohort or missing runtime identity cannot authorize Online learning.
"""
from __future__ import annotations
import collections
import numpy as np


def assess(payload, *, min_blocks=30):
    gaps, failures = [], []
    for field in ['no_motion_consistency_verified','actors_frozen','learning_disabled','no_replay_commits','ground_truth_outcomes_verified']:
        if payload.get(field) is not True:gaps.append(field)
    if payload.get('data_role') != 'independent_test':gaps.append('independent_test_required')
    if payload.get('used_for_selection') is not False:gaps.append('selection_history_unknown_or_reused')
    if payload.get('independence_verified') is not True:gaps.append('independence_not_verified')
    expected=payload.get('expected',{})
    for arm in ['baseline','candidate']:
        if not expected.get(arm,{}).get('actor_sha256') or not expected.get(arm,{}).get('code_head') or not expected.get(arm,{}).get('execution_profile'):
            gaps.append(arm+'_runtime_identity_missing')
    excluded=set(payload.get('excluded_uuids',[]))
    rows=payload.get('episodes',[]);by_block=collections.defaultdict(dict);uuids=set();counts=collections.defaultdict(collections.Counter)
    for row in rows:
        uid=row.get('uuid');arm=row.get('arm');block=row.get('block_id')
        if not uid or block is None or arm not in ['baseline','candidate']:
            gaps.append('episode_identity_missing');continue
        if uid in uuids:failures.append('duplicate_episode_uuid')
        if uid in excluded:failures.append('training_or_development_leakage')
        uuids.add(uid)
        if arm in by_block[block]:failures.append('duplicate_arm_in_block')
        by_block[block][arm]=row
        if row.get('complete_episode') is not True:gaps.append('incomplete_episode')
        target=expected.get(arm,{})
        for field in ['actor_sha256','code_head','execution_profile']:
            if not row.get(field):gaps.append('episode_'+field+'_missing')
            elif row[field]!=target.get(field):failures.append('episode_'+field+'_mismatch')
        if row.get('actor_changed') is not False:gaps.append('actor_pin_unverified')
        result=row.get('outcome')
        if result not in ['autonomous_success','assisted_success','failure']:gaps.append('unknown_outcome')
        else:counts[arm][result]+=1
        safety=row.get('safety_violation')
        if safety is True:failures.append('safety_violation')
        elif safety is not False:gaps.append('safety_evidence_missing')
        if not isinstance(row.get('deadline_misses'),int) or isinstance(row.get('deadline_misses'),bool) or row['deadline_misses']<0:
            gaps.append('deadline_evidence_missing')
        elif row['deadline_misses']>payload.get('max_deadline_misses_per_episode',0):
            failures.append('deadline_limit_exceeded')
        limits=payload.get('tracking_error_limits',{})
        for field in ['joint_p95_rad','gripper_p95_m']:
            value=row.get('tracking',{}).get(field);limit=limits.get(field)
            numeric=lambda x:isinstance(x,(int,float)) and not isinstance(x,bool) and np.isfinite(x) and x>=0
            if not numeric(value) or not numeric(limit):
                gaps.append('tracking_'+field+'_missing')
            elif value>limit:failures.append('tracking_'+field+'_limit_exceeded')
    complete=[v for v in by_block.values() if set(v)=={'baseline','candidate'}]
    if len(complete)!=len(by_block):gaps.append('incomplete_paired_blocks')
    if len(complete)<min_blocks:gaps.append('too_few_complete_blocks')
    effect=None;ci=None
    if complete and not gaps and not failures:
        delta=np.array([float(v['candidate']['outcome']=='autonomous_success')-float(v['baseline']['outcome']=='autonomous_success') for v in complete])
        effect=float(delta.mean())
        samples=np.random.default_rng(42).choice(delta,(20000,len(delta)),replace=True).mean(1)
        ci=np.quantile(samples,[.025,.975]).tolist()
        if ci[0]<=0:gaps.append('autonomous_improvement_not_established')
    status='失败' if failures else '证据不足' if gaps else '已验证'
    return dict(status=status,online_learning_release_supported=status=='已验证',
        failures=sorted(set(failures)),evidence_gaps=sorted(set(gaps)),complete_blocks=len(complete),
        outcomes={arm:dict(count) for arm,count in counts.items()},autonomous_success_difference=effect,
        paired_episode_block_bootstrap_ci95=ci,
        boundary='Supplied complete Episode pairs only; block resampling retains pairing. No overlapping windows. Evidence is conditional on provided provenance and sampled scenes; not a guarantee of continued improvement, publication, motion or deployment authorization.')
