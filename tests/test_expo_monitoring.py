import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'methods/expo_ft/cobot_adapter'))
from monitoring import LearnerHistory, install_selection_observer, infer_with_q, episode_summary, success_series, RolloutHistory


def test_update_points_do_not_follow_heartbeats_or_partial_critic_progress(tmp_path):
    history=LearnerHistory(tmp_path,0)
    metrics=dict(q=.1,target_q_mean=.12,critic_loss=.005)
    value=dict(critic_updates=20,actor_version=1,updated_at=1)
    history.observe(value,metrics)
    history.observe({**value,'critic_updates':35,'updated_at':2},metrics)
    assert len(history.points)==1 and history.points[0]['critic_step']==20
    # Equal numerical values are a new real update when the metrics object is replaced.
    history.observe({**value,'critic_updates':40,'updated_at':3},dict(metrics))
    assert [p['critic_step'] for p in history.points]==[20,40]
    restored=LearnerHistory(tmp_path,20)
    assert [p['critic_step'] for p in restored.points]==[20]
    restored.observe({**value,'critic_updates':40},dict(q=float('nan'),target_q_mean=0,critic_loss=0))
    assert len(restored.points)==1


def test_selection_observer_preserves_exact_result_and_does_not_resample():
    output=dict(qs=np.array([.2,.5]),idx=np.array(1),selected=np.array([[1,2]]),rng=np.array([42,10]))
    calls=[]
    module=SimpleNamespace(_jitted_fast_select=lambda *a,**kw:calls.append(1) or output)
    install_selection_observer(module);install_selection_observer(module)
    def infer(actor,route,data):
        selected=module._jitted_fast_select()
        assert selected is output
        return dict(actions=selected['selected'].tolist(),actor_version=3)
    result=infer_with_q(None,infer,'/select',{})
    assert calls==[1] and result['selected_q']==.5
    assert result['actions']==[[1,2]] and output['rng'].tolist()==[42,10]
    assert 'selected_q' not in infer_with_q(None,infer,'/precache',{})
    output['qs']=None
    assert 'selected_q' not in infer_with_q(None,infer,'/select',{})


def test_hil_and_missing_data_are_not_mislabeled_autonomous(tmp_path):
    root=tmp_path/'ep';root.mkdir()
    (root/'finished.json').write_text(json.dumps(dict(episode_uuid='ep',count=2,outcome='success',finished_at=2)))
    for i,h in enumerate([False,True]):np.savez(root/('%06d.npz'%i),is_hil=np.bool_(h),image=np.zeros((4,4)))
    before={p.name:p.read_bytes() for p in root.iterdir()}
    summary=episode_summary(root)
    assert summary['hil_steps']==1
    assert before=={p.name:p.read_bytes() for p in root.iterdir()}
    rows=[summary,dict(summary,episode_uuid='ep2',hil_steps=0,finished_at=3),dict(summary,episode_uuid='unknown',hil_steps=None,finished_at=4)]
    result=success_series(rows)
    assert result['points'][-1]['rate']==50 and result['points'][-1]['total']==2
    assert result['excluded']==1
    (root/'000001.npz').unlink()
    assert episode_summary(root)['hil_steps'] is None


def test_success_window_and_excluded_outcomes():
    rows=[dict(episode_uuid=str(i),finished_at=i,count=1,hil_steps=0,outcome='success' if i<5 else 'failure') for i in range(25)]
    rows += [dict(rows[0],episode_uuid='abort',finished_at=26,outcome='aborted')]
    result=success_series(rows)
    assert result['points'][-1]['rate']==0 and result['points'][-1]['total']==20
    assert result['completed']==26 and result['excluded']==1


def test_rollout_separates_episodes_and_does_not_write_on_control_path(tmp_path):
    history=RolloutHistory(tmp_path)
    history.selected(dict(episode_uuid='a',started_at=0),dict(selected_q=.2,actor_version=1))
    assert not history.path.exists()
    history.flush();assert json.loads(history.path.read_text())['episode_uuid']=='a'
    history.selected(dict(episode_uuid='b',started_at=0),dict(selected_q=.3,actor_version=2))
    history.flush();value=json.loads(history.path.read_text())
    assert value['episode_uuid']=='b' and len(value['points'])==1 and value['actor_version']==2


def test_observed_learner_delegates_status_without_changing_training_metadata(tmp_path,monkeypatch):
    import online_learner
    from observed_learner import ObservedLearner
    value=dict(critic_updates=20,actor_version=1,updated_at=3)
    monkeypatch.setattr(online_learner.Learner,'save_status',lambda self:dict(value))
    learner=ObservedLearner.__new__(ObservedLearner)
    learner.root=tmp_path;learner.meta=dict(critic_updates=20,updates=1,credit=30)
    learner.metrics=dict(q=.1,target_q_mean=.2,critic_loss=.01)
    original=dict(learner.meta)
    result=learner.save_status()
    assert result['monitoring']['points'][0]['critic_step']==20 and learner.meta==original
    # Chart disk failure is isolated from the real learner status/control path.
    monkeypatch.setattr(learner._monitor_history,'observe',lambda *a:(_ for _ in ()).throw(OSError('disk full')))
    assert learner.save_status()==value
