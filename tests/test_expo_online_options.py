import importlib.util
from pathlib import Path
import pytest

path = Path(__file__).parents[1] / 'methods/expo_ft/cobot_adapter/online_options.py'
spec = importlib.util.spec_from_file_location('online_options', path)
options = importlib.util.module_from_spec(spec)
spec.loader.exec_module(options)

@pytest.mark.parametrize('delay', range(1, 9))
@pytest.mark.parametrize('tick', range(1, 11))
def test_supported_pairs_have_unique_stable_identity(delay, tick):
    values = options.normalize(dict(rtc_delay=delay, edit_scale=tick / 100))
    assert values == dict(rtc_delay=delay, edit_scale=tick / 100)
    assert options.suffix(values) == 'd%d-edit%03d' % (delay, tick)

@pytest.mark.parametrize('delay', [0, 9, True, 5., '5', None])
def test_invalid_delay(delay):
    with pytest.raises(ValueError): options.normalize(dict(rtc_delay=delay, edit_scale=.01))

@pytest.mark.parametrize('scale', [0, .001, .015, .11, True, '.01', float('nan'), float('inf'), None])
def test_invalid_scale(scale):
    with pytest.raises(ValueError): options.normalize(dict(rtc_delay=5, edit_scale=scale))

def test_missing_and_extra_keys_rejected():
    for values in ({'rtc_delay': 5}, {'edit_scale': .01}, {'rtc_delay':5,'edit_scale':.01,'run_dir':'/arbitrary'}):
        with pytest.raises(ValueError): options.normalize(values)


@pytest.mark.parametrize('episodes',[1,5])
def test_browser_choices_reach_both_real_launch_commands_and_status_guard(tmp_path, monkeypatch, episodes):
    import sys, json
    from types import SimpleNamespace
    adapter = path.parent
    monkeypatch.syspath_prepend(str(adapter))
    from remote_runtime_support import RemoteSupport
    from remote_launcher import selected_profile, learner_command
    from online_runtime import OnlineRuntime
    original = dict(model_id='expo', rtc_delay=5, edit_scale=.01, allow_online_options=True,
                    run_dir=str(tmp_path/'learner'), port=18443, python='/python', project='/native',
                    bundle='/bundle', seed_dir='/seed', gpu=1, episodes_per_update=episodes)
    learner_profile=tmp_path/'learner.json';learner_profile.write_text(json.dumps(original))
    root_config={'port':18443,'profiles':{'edit001':str(learner_profile)}}
    profile={**original, 'launch_name':'edit001','identity_file':'/key','destination':'owner@host',
             'remote_port':18443,'local_port':18453,'native_project':'/native'}
    profile_path=tmp_path/'actor-profile.json';profile_path.write_text(json.dumps(profile))
    for delay, scale in [(1,.01),(3,.07),(5,.01),(8,.10)]:
        monkeypatch.setenv('COBOT_EXECUTION_OPTIONS',json.dumps(dict(enabled=True,publish_hz=50,rtc=True,smoothing=True,expo=dict(rtc_delay=delay,edit_scale=scale))))
        args=SimpleNamespace(remote_config=profile_path,run_dir=tmp_path/'actor',model_id='expo',rtc_delay=5,bundle=Path('/bundle'),port=8043)
        runtime=OnlineRuntime.__new__(OnlineRuntime);runtime.args=args;runtime.remote_init(args)
        ssh=runtime.remote_command()
        learner=selected_profile(root_config,ssh[-1])
        lc=learner_command(learner,adapter)
        runtime.start_remote=lambda:Path('/snapshot')
        ac=runtime.policy_command()
        for command in (lc,ac):
            assert int(command[command.index('--rtc-delay')+1])==delay
            assert float(command[command.index('--edit-scale')+1])==scale
        assert args.rtc_delay==delay
        default=(delay,scale)==(5,.01)
        assert (args.run_dir==tmp_path/'actor')==default
        assert (learner['run_dir']==original['run_dir'])==default
        if not default:
            assert args.run_dir.name.endswith(options.suffix(dict(rtc_delay=delay,edit_scale=scale)))
            assert learner['run_dir'].endswith(options.suffix(dict(rtc_delay=delay,edit_scale=scale)))
        info=dict(deployment_id='expo',rtc_delay=delay,edit_scale=scale,update_layout='native_full_batch',episodes_per_update=episodes,training_enabled=True)
        assert lc[lc.index('--episodes-per-update')+1]==str(episodes)
        runtime.validate_remote_status(info)
        for bad in ({**info,'rtc_delay':9},{**info,'edit_scale':.99},{**info,'episodes_per_update':6}):
            with pytest.raises(RuntimeError):runtime.validate_remote_status(bad)
    assert json.loads(learner_profile.read_text())==original


def test_restricted_command_rejects_injection_ranges_and_unapproved_profiles(tmp_path, monkeypatch):
    import json
    monkeypatch.syspath_prepend(str(path.parent))
    from remote_launcher import selected_profile
    p=tmp_path/'profile.json';p.write_text(json.dumps(dict(port=18443,rtc_delay=5,edit_scale=.01,run_dir='/fixed',allow_online_options=True)))
    c={'port':18443,'profiles':{'edit001':str(p)}}
    for command in ('run edit001 --rtc-delay 0 --edit-scale 0.01','run edit001 --rtc-delay 9 --edit-scale 0.01',
                    'run edit001 --rtc-delay 5 --edit-scale 0.11','run edit001 --rtc-delay 5 --edit-scale 0.015',
                    'run edit001 --rtc-delay 5 --edit-scale 0.01; id', 'run unknown --rtc-delay 5 --edit-scale 0.01'):
        with pytest.raises(ValueError):selected_profile(c,command)
    p.write_text(json.dumps(dict(port=18443)))
    with pytest.raises(ValueError):selected_profile(c,'run edit001 --rtc-delay 5 --edit-scale 0.01')
