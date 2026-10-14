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
