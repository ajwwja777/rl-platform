import importlib.util
import numpy as np
import pytest
M="methods.openpi_rlt.cobot_adapter.online_cycle"
def api():
 assert importlib.util.find_spec(M) is not None,"online cycle module missing"
 return __import__(M,fromlist=["x"])
def rec(ep,done=False,reward=0):
 return dict(episode_id=np.array(ep),done=done,rewards=np.array([reward],np.float32),source_chunk=np.zeros(10,np.uint8))
def test_only_complete_new_episode_is_eligible():
 a=api()
 records=[rec(0),rec(0,True,1),rec(1),rec(2,True,0)]
 groups=a.complete_episodes(records)
 assert sorted(groups)==[0,2]
 assert a.pending_episodes(groups,[0])==[2]
 assert groups[0][0]["group"]=="success"
 assert groups[2][0]["group"]=="failure"
def test_duplicate_terminals_rejected():
 with pytest.raises(ValueError):api().complete_episodes([rec(1,True),rec(1,True)])
def test_nonfinite_rejected():
 r=rec(1,True);r["z_rl"]=np.array([np.nan])
 with pytest.raises(ValueError):api().complete_episodes([r])
def test_gate_rejects_regression_and_nonfinite():
 a=api();base={"score":.01,"velocity":.01,"acceleration":.01,"first":.02,"boundary":.02}
 assert a.accept_candidate(base,dict(base))
 assert not a.accept_candidate(base,{**base,"score":.02})
 assert not a.accept_candidate(base,{**base,"acceleration":.02})
 assert not a.accept_candidate(base,{**base,"velocity":float("nan")})
def test_atomic_json_replaces_without_partial_content(tmp_path):
 a=api();p=tmp_path/"current.json";a.atomic_json(p,{"step":1});a.atomic_json(p,{"step":2})
 import json
 assert json.loads(p.read_text())=={"step":2}
def test_boundary_waits_for_matching_request_and_actor_version(tmp_path):
 a=api();seen=[]
 a.atomic_json(tmp_path/"metrics/learner_status.json",{"ready_for_online":True,"phase":"ready","request_id":"old","actor_version":5000})
 def tick():
  import json
  request=json.loads((tmp_path/"update_request.json").read_text());seen.append(request)
  a.atomic_json(tmp_path/"metrics/learner_status.json",{"ready_for_online":True,"phase":"ready","request_id":request["request_id"],"actor_version":5016})
 a.request_cycle(tmp_path,sleep=lambda _:tick(),actor_version_getter=lambda:5016,timeout=2)
 assert len(seen)==1
