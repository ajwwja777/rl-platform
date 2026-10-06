import json
import pytest
from integrations.cobot_runtime.supported_selection import hold_gripper,resolve

def test_registered_task_hold_and_default_unchanged():
    assert hold_gripper({'COBOT_DEPLOYMENT_MODEL_ID':'plug-v3-supported-7k'})
    assert not hold_gripper({'COBOT_DEPLOYMENT_MODEL_ID':'plug-v3-warmup-5k'})
    with pytest.raises(ValueError):hold_gripper({'COBOT_DEPLOYMENT_MODEL_ID':'plug-v3-supported-7k','COBOT_RLT_HOLD_RIGHT_GRIPPER':'0'})
    with pytest.raises(ValueError):hold_gripper({'COBOT_RLT_HOLD_RIGHT_GRIPPER':'yes'})

def test_registered_runtime_is_fresh_and_isolated(tmp_path):
    assert resolve('plug-v3-warmup-5k',models_root=tmp_path)is None
    with pytest.raises(ValueError,match='Prepare'):resolve('plug-v3-supported-online',models_root=tmp_path)
    run=tmp_path/'history/candidates/supported_online_20261006_runtime';run.mkdir(parents=True)
    for name in ['online.yaml','profile.json']:(run/name).write_text('{}')
    assert resolve('plug-v3-supported-online',models_root=tmp_path)['run']==str(run)
