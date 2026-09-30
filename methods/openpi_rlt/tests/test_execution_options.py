import json
from pathlib import Path
import pytest
from methods.openpi_rlt.cobot_adapter.execution_profiles import normalize_options, selected_profile, describe_execution
from methods.openpi_rlt.tests.test_collection_controls import _app
from methods.openpi_rlt.cobot_adapter.session import SessionConflict
from methods.openpi_rlt.cobot_adapter.trace import EpisodeOutcome

@pytest.mark.parametrize('hz', [20,30,40,50])
@pytest.mark.parametrize('rtc', [True,False])
@pytest.mark.parametrize('smoothing', [True,False])
def test_independent_options(hz,rtc,smoothing):
    options=dict(enabled=True,publish_hz=hz,rtc=rtc,smoothing=smoothing)
    name,profile=selected_profile(environ={'COBOT_RLT_EXECUTION_OPTIONS':json.dumps(options)})
    assert name=='user_options' and profile.logical_hz==20 and profile.publish_hz==hz
    assert profile.rtc is rtc and bool(profile.smoothing_tau_sec) is smoothing
    assert profile.joint_velocity_limit==.6 and profile.gripper_velocity_limit==.08

@pytest.mark.parametrize('bad', [None, {}, {'enabled':1}, {'enabled':False,'rtc':False},
    {'enabled':True,'publish_hz':60,'rtc':True,'smoothing':True},
    {'enabled':True,'publish_hz':50,'rtc':'false','smoothing':False}])
def test_invalid_options_fail_before_runtime_load(bad):
    with pytest.raises(ValueError):normalize_options(bad)

def test_unchecked_uses_model_default_and_describes_physical_rate():
    root=Path(__file__).resolve().parents[3]
    model={'execution_profile':'async_rtc50','control_hz':20,'execution_options':{'enabled':False}}
    shown=describe_execution(model,root)
    assert shown['publish_hz']==50 and shown['logical_hz']==20 and shown['rtc']
    selected=selected_profile(root, {'COBOT_RLT_EXECUTION_OPTIONS':'{"enabled":false}',
                                  'COBOT_RLT_EXECUTION_PROFILE':'async_rtc50'})
    assert selected[0]=='async_rtc50' and selected[1].publish_hz==50

def start(app):
    s=app.snapshot();return app.start(episode_id=s.episode_id,generation=s.generation)

def test_skip_retains_model_drops_trace_no_home_and_manual_next():
    app,task,hooks=_app();s=start(app)
    retained=[]
    task.defer_episode=lambda ref,identity:retained.append(ref) or {'deferred':True}
    result=app.skip(episode_id=s.episode_id,generation=s.generation)
    assert result.phase.value=='waiting_scene' and result.policy_paused and not result.replay_eligible
    assert hooks.outcomes==[EpisodeOutcome.ABORTED] and hooks.homes==0 and len(task.started)==1
    assert not task.finished and retained
    with pytest.raises(SessionConflict):app.skip(episode_id=s.episode_id,generation=s.generation)
    next_s=app.start(episode_id=result.episode_id,generation=result.generation)
    assert next_s.episode_id==s.episode_id+1 and len(task.started)==2

def test_failed_defer_stays_pending_and_never_restarts_writer():
    app,task,hooks=_app();s=start(app)
    def timeout(*args,**kwargs):raise TimeoutError('unknown result')
    task.defer_episode=timeout
    with pytest.raises(TimeoutError):app.skip(episode_id=s.episode_id,generation=s.generation)
    pending=app.snapshot()
    assert pending.phase.value=='terminal_pending' and pending.policy_paused and hooks.pauses[-1]
    assert not hooks.outcomes and len(task.started)==1
    with pytest.raises(SessionConflict):app.start(episode_id=pending.episode_id,generation=pending.generation)

def test_recording_health_error_pauses_without_second_http_or_runtime_death():
    app,task,hooks=_app();start(app)
    calls=[]
    def fail():calls.append(True);raise OSError('connection lost')
    task.status=fail
    assert app.ensure_recorder_active() is False
    s=app.snapshot();assert s.phase.value=='terminal_pending' and s.policy_paused
    assert len(calls)==1 and hooks.pauses[-1] and not hooks.outcomes and not hooks.homes

def test_non_recorder_fault_cannot_be_skipped():
    app,task,hooks=_app();start(app)
    app._controller.fail('hardware_authority_lost')
    task.defer_episode=lambda *a,**k:pytest.fail('must not defer control fault')
    s=app.snapshot()
    with pytest.raises(SessionConflict):app.skip(episode_id=s.episode_id,generation=s.generation)
