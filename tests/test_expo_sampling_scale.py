import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'methods/expo_ft/cobot_adapter'))
from sampling_config import edit_scale
from actor_snapshot import contract
from remote_runtime_support import RemoteSupport
from types import SimpleNamespace

@pytest.mark.parametrize('value',[float('nan'),float('inf'),0,-.01,True,2])
def test_invalid_scale_rejected(value):
 with pytest.raises(ValueError):edit_scale(value)

def test_default_actor_contract_preserved_and_candidate_isolated(tmp_path):
 for name in ('expo_ft','configs','scripts/dynamic_pick'):(tmp_path/name).mkdir(parents=True)
 for name in ('train_pi_robo.py','scripts/dynamic_pick/run_server.sh','deployment.json','norm_stats.json'):(tmp_path/name).write_text('{}')
 original=contract(tmp_path,tmp_path,5)
 assert contract(tmp_path,tmp_path,5,.1)==original
 candidate=contract(tmp_path,tmp_path,5,.01)
 assert candidate=={**original,'edit_scale':.01}

def test_remote_rejects_different_scale():
 remote=RemoteSupport();remote.remote={'edit_scale':.01};remote.args=SimpleNamespace(model_id='test',rtc_delay=5)
 info=dict(deployment_id='test',rtc_delay=5,update_layout='native_full_batch',edit_scale=.01)
 remote.validate_server_status(info)
 for value in (.1,None):
  with pytest.raises(RuntimeError,match='scale'):remote.validate_server_status({**info,'edit_scale':value})
