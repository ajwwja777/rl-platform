from integrations.cobot_runtime.learning_curve import assess_learning_curve


def payload(outcomes):
    evaluations=[]
    for i, group in enumerate(outcomes):
        identities=dict(actor_sha256='actor'+str(i),code_head='code',execution_profile='faithful20')
        episodes=[dict(uuid=f'{i}-{j}',complete_episode=True,outcome=o,
                       hil_used=o=='assisted_success',actor_frozen=True,**identities)
                  for j,o in enumerate(group)]
        evaluations.append(dict(interactions=i*10,episodes=episodes,**identities))
    return dict(data_role='independent_test',success_denominator='all_frozen_trials',interaction_budget=(len(outcomes)-1)*10,evaluations=evaluations)


def test_auc_is_budget_average_success_not_critic_ranking():
    p=payload([['failure','failure'],['autonomous_success','failure'],['autonomous_success','autonomous_success']])
    r=assess_learning_curve(p)
    assert r['status']=='已验证'
    assert r['autonomous_success_curve_auc']==.5
    assert r['assisted_success_curve_auc']==0
    assert r['online_release_supported'] is False


def test_assistance_cannot_raise_autonomous_auc():
    p=payload([['assisted_success'],['assisted_success']])
    r=assess_learning_curve(p)
    assert r['autonomous_success_curve_auc']==0
    assert r['total_success_curve_auc']==1


def test_unknown_outcomes_and_unmeasured_endpoint_stay_empty():
    p=payload([['failure'],['unlabeled']]);p['interaction_budget']=20
    r=assess_learning_curve(p)
    assert r['status']=='证据不足'
    assert r['autonomous_success_curve_auc'] is None
    assert r['points'][-1]['autonomous_success_rate'] is None
    assert 'full_budget_endpoints_not_measured' in r['evidence_gaps']


def test_hil_success_mislabel_and_reused_episode_fail():
    p=payload([['autonomous_success'],['autonomous_success']])
    p['evaluations'][1]['episodes'][0]['hil_used']=True
    p['evaluations'][1]['episodes'][0]['uuid']='0-0'
    r=assess_learning_curve(p)
    assert r['status']=='失败'
    assert r['autonomous_success_curve_auc'] is None


def test_assisted_collection_consumes_budget_but_not_autonomous_denominator():
    p=payload([['autonomous_success','assisted_success'],['autonomous_success','assisted_success']])
    p['success_denominator']='scheduled_autonomous_rollouts'
    for evaluation in p['evaluations']:
        for row in evaluation['episodes']:
            row['collection_kind']='assisted' if row['hil_used'] else 'autonomous'
    r=assess_learning_curve(p)
    assert r['autonomous_success_curve_auc']==1
    assert r['points'][0]['autonomous_success_denominator']==1
    p['success_denominator']='all_frozen_trials'
    assert assess_learning_curve(p)['autonomous_success_curve_auc']==.5


def test_declared_nonzero_start_uses_measured_interval_without_extrapolation():
    p=payload([['failure'],['autonomous_success']])
    p['evaluations'][0]['interactions']=10
    p['evaluations'][1]['interactions']=20
    p['evaluation_start_interactions']=10;p['interaction_budget']=20
    assert assess_learning_curve(p)['autonomous_success_curve_auc']==.5
