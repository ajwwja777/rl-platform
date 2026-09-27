import numpy as np
import pytest
from methods.openpi_rlt.cobot_adapter.online_cycle import project_training_action

def test_right_hil_preserved_and_passive_target_uses_reference():
    actual=np.ones((10,14),dtype=np.float16)
    reference=np.zeros((10,14),dtype=np.float16)
    reference[:,6]=.0004;actual[:,6]=.0016
    row={"action_chunk":actual,"ref_chunk":reference,"source_chunk":np.full(10,3),"rewards":np.zeros(10)}
    out=project_training_action(row,active_arm="right",hold_grippers=True)
    np.testing.assert_array_equal(out["action_chunk"][:,7:13],actual[:,7:13])
    np.testing.assert_array_equal(out["action_chunk"][:,:7],reference[:,:7])
    np.testing.assert_array_equal(out["action_chunk"][:,13],reference[:,13])
    assert out["source_chunk"] is row["source_chunk"]
    assert out["rewards"] is row["rewards"]
    assert np.all(actual[:,6]==np.float16(.0016))
    assert not np.shares_memory(out["action_chunk"],actual)

def test_both_arms_and_unheld_grippers_preserve_actual():
    row={"action_chunk":np.ones((10,14)),"ref_chunk":np.zeros((10,14))}
    np.testing.assert_array_equal(project_training_action(row,active_arm="both",hold_grippers=False)["action_chunk"],row["action_chunk"])

def test_invalid_shape_rejected():
    with pytest.raises(ValueError):
        project_training_action({"action_chunk":np.zeros((10,7)),"ref_chunk":np.zeros((10,14))},active_arm="right",hold_grippers=True)
