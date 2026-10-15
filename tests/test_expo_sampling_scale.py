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
 info=dict(deployment_id='test',rtc_delay=5,update_layout='native_full_batch',edit_scale=.01,training_enabled=True)
 remote.validate_remote_status(info)
 for value in (.1,None):
  with pytest.raises(RuntimeError,match='scale'):remote.validate_remote_status({**info,'edit_scale':value})


def test_concurrent_status_writes_use_distinct_temporary_files(tmp_path,monkeypatch):
 import os,threading
 from concurrent.futures import ThreadPoolExecutor
 from online_protocol import atomic_json
 barrier=threading.Barrier(2);replace=os.replace
 def overlap(source,target):
  barrier.wait(timeout=5);return replace(source,target)
 monkeypatch.setattr(os,'replace',overlap)
 path=tmp_path/'status.json'
 with ThreadPoolExecutor(max_workers=2) as pool:
  jobs=[pool.submit(atomic_json,path,{'writer':i,'values':[i]*100}) for i in range(2)]
  for job in jobs:job.result(timeout=10)
 value=json.loads(path.read_text());assert value['values']==[value['writer']]*100
 assert not list(tmp_path.glob('*.tmp'))
