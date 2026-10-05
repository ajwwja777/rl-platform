import copy
import pytest
from integrations.cobot_runtime.frozen_acceptance import assess


def fixture():
    expected={arm:dict(actor_sha256=arm,code_head='code',execution_profile='faithful20') for arm in ['baseline','candidate']}
    p=dict(no_motion_consistency_verified=True,actors_frozen=True,learning_disabled=True,no_replay_commits=True,
        ground_truth_outcomes_verified=True,independence_verified=True,data_role='independent_test',used_for_selection=False,
        expected=expected,excluded_uuids=['trained'],tracking_error_limits=dict(joint_p95_rad=.01,gripper_p95_m=.001),episodes=[])
    for i in range(30):
        for arm in expected:
            p['episodes'].append(dict(uuid=arm+str(i),arm=arm,block_id=str(i),complete_episode=True,actor_changed=False,
                outcome='failure' if arm=='baseline' else 'autonomous_success',safety_violation=False,deadline_misses=0,
                tracking=dict(joint_p95_rad=.001,gripper_p95_m=.0001),**expected[arm]))
    return p


def test_complete_independent_episode_evidence_supports_release_without_deploying():
    r=assess(fixture());assert r['status']=='已验证' and r['autonomous_success_difference']==1.


def test_assisted_success_does_not_establish_autonomous_gain():
    p=fixture()
    for row in p['episodes']:
        if row['arm']=='candidate':row['outcome']='assisted_success'
    r=assess(p);assert not r['online_learning_release_supported'] and r['autonomous_success_difference']==0.


@pytest.mark.parametrize('change',[dict(data_role='development'),dict(used_for_selection=True),dict(learning_disabled=False)])
def test_reused_or_learning_trials_are_not_independent_frozen_evidence(change):
    p=fixture();p.update(change);assert assess(p)['status']=='证据不足'


def test_duplicate_uuid_and_training_leakage_fail():
    p=fixture();p['episodes'][0]['uuid']='trained';p['episodes'][1]['uuid']='trained'
    r=assess(p);assert r['status']=='失败' and 'training_or_development_leakage' in r['failures']


def test_identity_mismatch_and_safety_violation_fail():
    p=fixture();p['episodes'][0]['actor_sha256']='wrong';p['episodes'][1]['safety_violation']=True
    assert assess(p)['status']=='失败'


def test_missing_evidence_does_not_get_synthetic_zero_metrics():
    r=assess({});assert r['status']=='证据不足'
    assert r['autonomous_success_difference'] is None and r['paired_episode_block_bootstrap_ci95'] is None
