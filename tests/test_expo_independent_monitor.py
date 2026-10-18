import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'methods/expo_ft/cobot_adapter'))
import remote_monitor as monitor
import remote_launcher as launcher


def config(root):
    return dict(model_id='expo', run_dir=str(root), rtc_delay=5, edit_scale=.01)


def test_monitor_missing_run_is_read_only_and_never_launches(tmp_path, monkeypatch, capsys):
    root=tmp_path/'not-created';profile=tmp_path/'profile.json';profile.write_text(json.dumps(config(root)))
    monkeypatch.setenv('SSH_ORIGINAL_COMMAND','monitor')
    monkeypatch.setattr(sys,'argv',['remote_launcher.py','--config',str(profile)])
    monkeypatch.setattr(launcher.subprocess,'Popen',lambda *a,**k:pytest.fail('must not launch learner'))
    monkeypatch.setattr(launcher.subprocess,'run',lambda *a,**k:pytest.fail('must not probe GPUs'))
    launcher.main()
    value=json.loads(capsys.readouterr().out)
    assert not value['running'] and value['training']=={} and not root.exists()


def test_snapshot_finished_process_and_identity(tmp_path, monkeypatch):
    root=tmp_path/'run';(root/'telemetry').mkdir(parents=True)
    (root/'status.json').write_text(json.dumps(dict(deployment_id='expo',rtc_delay=5,edit_scale=.01,pid=123,phase='updating',actor_version=8)))
    history=dict(points=[dict(critic_step=160,q=.3)],updated_at=1)
    (root/'telemetry/learner.json').write_text(json.dumps(history))
    monkeypatch.setattr(monitor,'process_matches',lambda *a:False)
    value=monitor.snapshot(config(root));assert value['training']==history and value['phase']=='stopped'
    monkeypatch.setattr(monitor,'process_matches',lambda *a:True)
    assert monitor.snapshot(config(root))['phase']=='updating'
    with pytest.raises(ValueError,match='identity'):monitor.snapshot({**config(root),'edit_scale':.02})


def test_process_identity_rejects_reused_pid(tmp_path):
    p=tmp_path/'123';p.mkdir();(p/'cmdline').write_bytes(b'python\0unrelated.py\0')
    assert not monitor.process_matches(123,Path('/run'),'expo',tmp_path)
    (p/'cmdline').write_bytes(b'python\0/path/observed_learner.py\0--run-dir\0/run\0--deployment-id\0expo\0')
    assert monitor.process_matches(123,Path('/run'),'expo',tmp_path)
    assert not monitor.process_matches(123,Path('/other'),'expo',tmp_path)


def test_named_monitor_preserves_bounded_option_namespace(tmp_path,monkeypatch,capsys):
    selected=tmp_path/'selected.json';selected.write_text(json.dumps({**config(tmp_path/'run'),'port':18443,'allow_online_options':True}))
    parent=tmp_path/'parent.json';parent.write_text(json.dumps(dict(port=18443,profiles={'edit001_batch5':str(selected)})))
    seen=[];monkeypatch.setattr(monitor,'snapshot',lambda c:(seen.append(c) or dict(ok=True)))
    monkeypatch.setattr(sys,'argv',['launcher','--config',str(parent)])
    monkeypatch.setenv('SSH_ORIGINAL_COMMAND','monitor edit001_batch5 --rtc-delay 3 --edit-scale 0.07')
    launcher.main();assert seen[0]['run_dir'].endswith('--d3-edit007')
    monkeypatch.setenv('SSH_ORIGINAL_COMMAND','monitor edit001_batch5; run')
    with pytest.raises(ValueError):launcher.main()


def test_client_no_tunnel_no_run_and_rejects_wrong_response(tmp_path,monkeypatch):
    profile=dict(destination='user@host',identity_file='/key',launch_name='edit001_batch5',model_id='expo',rtc_delay=5,edit_scale=.01,allow_online_options=True)
    opts=dict(rtc_delay=3,edit_scale=.07);cmd=monitor.monitor_command(profile,opts)
    assert '-L' not in cmd and cmd[-1]=='monitor edit001_batch5 --rtc-delay 3 --edit-scale 0.07'
    value=dict(schema=1,model_id='expo',options=opts,running=False,training=dict(points=[]))
    monkeypatch.setattr(monitor.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout=json.dumps(value).encode()))
    assert monitor.fetch(profile,opts)==value
    value['model_id']='other'
    with pytest.raises(ValueError):monitor.fetch(profile,opts)
    with pytest.raises(ValueError):monitor.monitor_command({**profile,'destination':'host; rm -rf /'},opts)


def test_hpc_monitor_relay_has_no_port_forward_or_lifecycle(monkeypatch):
    profile=dict(port=18443,relay=dict(destination='user@login',port=18443,
        launcher='/source/hpc_gateway.py',profile='/owned/gateway.json',monitor=True))
    value=dict(schema=1,model_id='expo',running=True,training=dict(points=[]))
    seen=[]
    def run(command,**kwargs):
        seen.append((command,kwargs))
        return SimpleNamespace(returncode=0,stdout=json.dumps(value).encode())
    monkeypatch.setattr(monitor.subprocess,'run',run)
    assert monitor.snapshot(profile)==value
    command,kwargs=seen[0]
    assert '-L' not in command and command[-1].endswith(' --monitor')
    assert kwargs['stdin']==monitor.subprocess.DEVNULL and kwargs['timeout']==9
    monkeypatch.setattr(monitor.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=1,stdout=b''))
    with pytest.raises(OSError):monitor.snapshot(profile)
    profile['relay']['monitor']=False
    with pytest.raises(ValueError):monitor.snapshot(profile)
