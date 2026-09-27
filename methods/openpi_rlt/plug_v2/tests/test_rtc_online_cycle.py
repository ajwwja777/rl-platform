import copy
from methods.openpi_rlt.plug_v2.rtc_online_cycle import burnin_for_release,pending,regression_gate

def report():
    metric=lambda auc,sep:{'auc':auc,'separation':sep,'success_n':3,'failure_n':3}
    groups={group:{'q_data_early':metric(.9,.2),'q_terminal':metric(1.,.8),'q_actor':metric(.8,.3)} for group in ('autonomous','policy_terminal')}
    return {'summary':{k:{'deployment_matched_mse':.1,'deployment_accel_rms':.01,'ref_matched_conditioner_mse':.2} for k in ('0','6')},'value_summary':groups}

def test_batch_excludes_consumed_and_waits_for_five_new():
    assert pending(range(8),range(4),[],5)==[]
    assert pending(range(9),range(4),[],5)==list(range(4,9))
    assert pending(range(10),range(4),range(9),5)==[]
    assert pending(range(14),range(4),range(9),5)==list(range(9,14))
    assert pending(range(16),range(4),range(9),5)==list(range(9,14))

def test_reject_worse_or_nonfinite_no_loss_proxy():
    a=report();b=copy.deepcopy(a)
    assert regression_gate(a,b)==[]
    b['summary']['6']['deployment_accel_rms']=.02
    assert '6:deployment_accel_rms' in regression_gate(a,b)
    b=copy.deepcopy(a);b['summary']['0']['deployment_matched_mse']=float('nan')
    assert regression_gate(a,b)

def test_reject_value_direction_regression():
    a=report();b=copy.deepcopy(a);b['value_summary']['autonomous']['q_data_early']['auc']=.49
    assert 'value:autonomous:q_data_early:auc' in regression_gate(a,b)
    b=copy.deepcopy(a);b['value_summary']['autonomous']['q_data_early']['separation']=-.01
    assert 'value:autonomous:q_data_early:separation' in regression_gate(a,b)


def test_rollout_does_not_scan_hash_or_write(tmp_path,monkeypatch):
    from methods.openpi_rlt.plug_v2 import rtc_online_cycle as c
    monkeypatch.setattr(c,'RUN',tmp_path)
    monkeypatch.setattr(c,'session_phase',lambda:'rollout')
    def forbidden(*a,**k):raise AssertionError('background scan during rollout')
    monkeypatch.setattr(c,'selected',forbidden);monkeypatch.setattr(c,'available_uuids',forbidden);monkeypatch.setattr(c,'atomic',forbidden)
    def stop(_):raise StopIteration
    monkeypatch.setattr(c.time,'sleep',stop)
    import pytest
    with pytest.raises(StopIteration):c.cycle(5)


def test_critic_burnin_is_carried_across_releases_then_stays_off():
    assert burnin_for_release({},400)==400
    assert burnin_for_release({'critic_burnin_updates':400},500)==500
    assert burnin_for_release({'critic_burnin_updates':900},500)==100
    assert burnin_for_release({'critic_burnin_updates':1000,'critic_calibrated':True},500)==0
