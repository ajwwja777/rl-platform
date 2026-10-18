"""Fixed SSH relay keeps loopback binding and preserves argv boundaries."""
import importlib.util
from pathlib import Path
import shlex
import pytest
spec=importlib.util.spec_from_file_location('expo_remote_launcher',Path(__file__).parents[1]/'methods/expo_ft/cobot_adapter/remote_launcher.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
def config():
 return {'port':18443,'relay':{'destination':'LFT-W02@10.12.0.212','port':18443,'launcher':'/data/test path/launcher.py','profile':'/data/profile.json'}}
def test_relay_is_loopback_only_and_quotes_remote_paths():
 c=config();args=module.relay_command(c)
 assert args[args.index('-L')+1]=='127.0.0.1:18443:127.0.0.1:18443'
 assert shlex.split(args[-1])==['/usr/bin/python3',c['relay']['launcher'],'--config',c['relay']['profile']]
 assert 'ExitOnForwardFailure=yes' in args and 'BatchMode=yes' in args
@pytest.mark.parametrize('field,value',[('destination','host; touch /tmp/x'),('launcher','relative.py'),('profile','relative.json'),('port',80)])
def test_invalid_relay_rejected(field,value):
 c=config();c['relay'][field]=value
 with pytest.raises(ValueError):module.relay_command(c)


def test_named_profile_requires_explicit_allowlist_and_fixed_port(tmp_path):
 import json
 profile=tmp_path/'edit001.json';profile.write_text(json.dumps({'port':18443,'edit_scale':.01}))
 c=config();c['profiles']={'edit001':str(profile)}
 assert module.selected_profile(c,'run') is c
 assert module.selected_profile(c,'run edit001')['edit_scale']==.01
 for command in ('run ../../etc/passwd','run unknown','run edit001; id','run edit001 extra'):
  with pytest.raises(ValueError):module.selected_profile(c,command)
 profile.write_text(json.dumps({'port':18444}))
 with pytest.raises(ValueError):module.selected_profile(c,'run edit001')


def test_gpu_selection_uses_only_free_candidates_and_preserves_single_default():
 assert module.select_gpus({'gpu':1},lambda _:False)==[1]
 c={'gpu':1,'gpu_candidates':[1,0],'max_devices':2}
 assert module.select_gpus(c,lambda _:False)==[1,0]
 assert module.select_gpus(c,lambda x:x==0)==[1]
 assert module.select_gpus(c,lambda x:x==1)==[0]
 with pytest.raises(RuntimeError):module.select_gpus(c,lambda _:True)

@pytest.mark.parametrize('c',[
 {'gpu':1,'gpu_candidates':[1,1],'max_devices':2},
 {'gpu':1,'gpu_candidates':['0;bad'],'max_devices':2},
 {'gpu':1,'gpu_candidates':[],'max_devices':2},
 {'gpu':1,'gpu_candidates':[0,1],'max_devices':3},
])
def test_gpu_selection_rejects_ambiguous_configuration(c):
 with pytest.raises(ValueError):module.select_gpus(c,lambda _:False)


def test_four_gpu_selection_never_returns_unsupported_three_devices():
 c={'gpu':0,'gpu_candidates':[0,1,2,3],'max_devices':4}
 assert module.select_gpus(c,lambda _:False)==[0,1,2,3]
 assert module.select_gpus(c,lambda x:x==3)==[0,1]
