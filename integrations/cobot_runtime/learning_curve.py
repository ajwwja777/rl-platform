"""Success-rate learning-curve AUC, distinct from Critic ROC-AUC.

Calculate supplied complete-Episode evidence only; no Online release decision.
Training interaction budgets are explicit, not learner updates or Q values.
"""
from __future__ import annotations
import math


def assess_learning_curve(payload):
    gaps, failures, seen, points = [], [], set(), []
    role = payload.get('data_role')
    if role not in ['training', 'reused_development', 'independent_test']:
        gaps.append('data_role_unknown')
    horizon = payload.get('interaction_budget')
    start = payload.get('evaluation_start_interactions', 0)
    denominator = payload.get('success_denominator')
    if denominator not in ['all_frozen_trials', 'scheduled_autonomous_rollouts']:
        gaps.append('success_denominator_not_declared')
    if not isinstance(start, (int,float)) or isinstance(start, bool) or not math.isfinite(start) or start < 0:
        gaps.append('evaluation_start_invalid')
    if not isinstance(horizon, (int, float)) or isinstance(horizon, bool) or not math.isfinite(horizon) or horizon <= 0:
        gaps.append('interaction_budget_missing')
    for evaluation in payload.get('evaluations', []):
        x = evaluation.get('interactions')
        if not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x) or x < 0:
            failures.append('invalid_interaction_coordinate');continue
        trials = evaluation.get('episodes', [])
        counts = dict(autonomous_success=0, assisted_success=0, failure=0)
        scheduled_autonomous = 0
        scheduled_autonomous_successes = 0
        if not trials:
            gaps.append('missing_evaluation_episodes')
        for row in trials:
            uid = row.get('uuid')
            if not uid:
                gaps.append('missing_episode_uuid')
            elif uid in seen:
                failures.append('duplicate_episode_uuid')
            else:
                seen.add(uid)
            if row.get('complete_episode') is not True:
                gaps.append('incomplete_episode')
            if row.get('outcome') not in counts:
                gaps.append('missing_or_unknown_outcome')
            else:
                counts[row['outcome']] += 1
            if not isinstance(row.get('hil_used'), bool):
                gaps.append('hil_identity_missing')
            if denominator == 'scheduled_autonomous_rollouts':
                kind = row.get('collection_kind')
                if kind not in ['autonomous', 'assisted']:
                    gaps.append('scheduled_collection_kind_missing')
                if kind == 'autonomous':
                    scheduled_autonomous += 1
                    scheduled_autonomous_successes += int(row.get('outcome') == 'autonomous_success')
            if row.get('outcome') == 'autonomous_success' and row.get('hil_used') is not False:
                failures.append('autonomous_success_has_hil')
            if row.get('outcome') == 'assisted_success' and row.get('hil_used') is not True:
                failures.append('assisted_success_without_hil')
            if row.get('actor_frozen') is not True:
                gaps.append('actor_not_verified_frozen_during_evaluation')
            for field in ['actor_sha256', 'code_head', 'execution_profile']:
                expected = evaluation.get(field)
                if not expected or not row.get(field):
                    gaps.append(field+'_missing')
                elif row[field] != expected:
                    failures.append(field+'_mismatch')
        known = bool(trials) and all(row.get('complete_episode') is True and row.get('outcome') in counts for row in trials)
        success_count, success_den = counts['autonomous_success'], len(trials)
        if denominator == 'scheduled_autonomous_rollouts':
            success_count, success_den = scheduled_autonomous_successes, scheduled_autonomous
            if not success_den:gaps.append('no_scheduled_autonomous_trials')
        points.append({'interactions':x, 'episodes':len(trials), 'outcomes':counts,
                       'autonomous_success_denominator':success_den,
                       'autonomous_success_rate':success_count/success_den if known and success_den else None,
                       'assisted_success_rate':counts['assisted_success']/len(trials) if known else None,
                       'total_success_rate':(counts['autonomous_success']+counts['assisted_success'])/len(trials) if known else None})
    if len(points) < 2:
        gaps.append('need_two_or_more_measured_evaluations')
    if points:
        if any(b['interactions'] <= a['interactions'] for a,b in zip(points,points[1:])):
            failures.append('interaction_coordinates_not_strictly_increasing')
        if points[0]['interactions'] != start or points[-1]['interactions'] != horizon:
            gaps.append('full_budget_endpoints_not_measured')
        if isinstance(start,(int,float)) and isinstance(horizon,(int,float)) and horizon <= start:
            gaps.append('empty_evaluation_interval')
    result = {'status':'失败' if failures else '证据不足' if gaps else '已验证',
              'data_role':role, 'success_denominator':denominator,
              'evaluation_interval':[start,horizon], 'points':points, 'failures':sorted(set(failures)),
              'evidence_gaps':sorted(set(gaps)), 'autonomous_success_curve_auc':None,
              'assisted_success_curve_auc':None, 'total_success_curve_auc':None,
              'confidence_interval':None, 'online_release_supported':False,
              'boundary':'Trapezoidal integration of measured frozen Episode success rates over a declared interaction budget. Not ROC-AUC, no extrapolation or missing-outcome imputation. A single supplied run has no across-run uncertainty; no continued-improvement or deployment guarantee.'}
    if not failures and not gaps:
        for key, rate in [('autonomous_success_curve_auc','autonomous_success_rate'),
                          ('assisted_success_curve_auc','assisted_success_rate'),
                          ('total_success_curve_auc','total_success_rate')]:
            result[key] = sum((b['interactions']-a['interactions'])*(a[rate]+b[rate])/2
                              for a,b in zip(points,points[1:]))/(horizon-start)
    return result
