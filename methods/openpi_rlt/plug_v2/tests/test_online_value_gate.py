import numpy as np
import pytest
from methods.openpi_rlt.plug_v2.online_value_gate import aggregate_metric_window, build_value_episode, episode_q_payload, fixed_value_gate, summarize_value_episodes, validate_episode_splits

def row(uuid,success,q,groups):
    return {'uuid':uuid,'success':success,'groups':groups,'q_data':q,'q_data_early':q,'q_terminal':q}

def test_online_autonomous_is_separate_from_hil_and_warmup():
    rows=[row('s1',True,.8,['all_online','autonomous','policy_terminal']),row('s2',True,.7,['all_online','autonomous','policy_terminal']),
          row('f1',False,.2,['all_online','autonomous','policy_terminal']),row('f2',False,.3,['all_online','autonomous','policy_terminal']),
          row('h1',True,.95,['all_online','hil']),row('w1',True,.99,['warmup'])]
    summary=summarize_value_episodes(rows)
    auto=summary['autonomous']['q_data_early']
    assert auto['success_n']==2 and auto['failure_n']==2
    assert auto['auc']==1.0 and auto['separation']==pytest.approx(.5)
    assert summary['hil']['q_data_early']['success_n']==1

def test_hil_success_cannot_make_autonomous_reversed_q_pass():
    rows=[row('s',True,.1,['all_online','autonomous','policy_terminal']),
          row('f',False,.8,['all_online','autonomous','policy_terminal']),
          row('h',True,1.0,['all_online','hil'])]
    report={'value_summary':summarize_value_episodes(rows)}
    reasons=fixed_value_gate(report,min_count=1,min_auc=.6)
    assert 'value:autonomous:q_data_early:auc' in reasons
    assert 'value:autonomous:q_data_early:separation' in reasons
    assert 'value:policy_terminal:q_data_early:auc' in reasons

def test_missing_autonomous_success_is_rejected_even_with_hil_success():
    report={'value_summary':summarize_value_episodes([row('f',False,.2,['all_online','autonomous']),row('h',True,.9,['all_online','hil'])])}
    reasons=fixed_value_gate(report,min_count=1)
    assert 'value:autonomous:q_data_early:count' in reasons

def test_episode_uuid_cannot_cross_train_and_validation():
    validate_episode_splits([({'uuid':'a','split':'train'},{}),({'uuid':'b','split':'val'}, {})])
    with pytest.raises(ValueError,match='split leakage'):
        validate_episode_splits([({'uuid':'a','split':'train'},{}),({'uuid':'a','split':'val'}, {})])

def test_actor_metrics_are_aggregated_from_real_actor_update_rows():
    metrics=[{'global_step':1,'critic_loss':.3,'did_actor_update':0,'actor_loss':0,'weighted_bc':0},
             {'global_step':2,'critic_loss':.2,'did_actor_update':1,'actor_loss':.5,'weighted_bc':.4,'weighted_q':-.02,'weighted_delta':.03},
             {'global_step':3,'critic_loss':.1,'did_actor_update':1,'actor_loss':.3,'weighted_bc':.2,'weighted_q':-.04,'weighted_delta':.01}]
    result=aggregate_metric_window(metrics)
    assert result['global_step']==3
    assert result['critic_loss']==pytest.approx(.2)
    assert result['actor_update_count']==2
    assert result['actor_loss']==pytest.approx(.4)
    assert result['weighted_bc']==pytest.approx(.3)
    assert result['weighted_q']==pytest.approx(-.03)
    assert result['weighted_delta']==pytest.approx(.02)


def test_episode_q_payload_has_position_median_iqr_and_autonomous_gate_values():
    rows=[dict(row('s',True,.8,['all_online','autonomous']),q_curve=[.1,.4,.8]),dict(row('f',False,.2,['all_online','autonomous']),q_curve=[.1,.15,.2])]
    payload=episode_q_payload(rows,points=3)
    groups={item['name']:item for item in payload['groups']}
    assert groups['autonomous_success']['median']==[.1,.4,.8]
    assert groups['autonomous_failure']['median']==[.1,.15,.2]
    assert payload['autonomous_separation']==pytest.approx(.6)
    assert payload['auc']==1.0


def test_build_value_episode_classifies_pure_autonomous_and_hil():
    q=np.asarray([[.5,.4,.6],[.8,.4,.7]])
    autonomous=build_value_episode({'uuid':'a','phase':'online','success':True,'expert':False},q,np.zeros((2,10),dtype=np.uint8),np.asarray([False,True]))
    assert set(autonomous['groups'])=={'all_online','autonomous','policy_terminal'}
    assert autonomous['q_data_early']==pytest.approx(.65)
    source=np.zeros((2,10),dtype=np.uint8);source[1,0]=2
    hil=build_value_episode({'uuid':'h','phase':'online','success':True,'expert':False},q,source,np.asarray([False,True]))
    assert set(hil['groups'])=={'all_online','hil'}
    assert hil['q_terminal'] is None


def test_default_gate_uses_stage_specific_thresholds():
    rows=[]
    for i in range(3):
        rows.append({'uuid':'s'+str(i),'success':True,'groups':['autonomous','policy_terminal'],
          'q_data_early':.51+i*.01,'q_terminal':.8+i*.01,'q_actor':.7+i*.01,'q_data':.6})
        rows.append({'uuid':'f'+str(i),'success':False,'groups':['autonomous','policy_terminal'],
          'q_data_early':.49-i*.01,'q_terminal':.2-i*.01,'q_actor':.3-i*.01,'q_data':.4})
    assert fixed_value_gate({'value_summary':summarize_value_episodes(rows)})==[]
