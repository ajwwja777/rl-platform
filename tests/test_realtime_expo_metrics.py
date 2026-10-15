import numpy as np
import pytest
from methods.openpi_rlt.realtime_expo.metrics import episode_metrics,complete_episode_summary

def test_padding_is_excluded_and_gripper_is_separate():
    p=np.zeros((2,50,7));t=p.copy();m=np.zeros((2,50),bool);m[0,:2]=True;m[1,0]=True
    p[0,2:]=100;p[1,1:]=100;p[0,0,:6]=.003;p[1,0,6]=.006
    r=episode_metrics(p,t,m)
    assert r['c10_joint_mae_rad']==pytest.approx(.001)
    assert r['c10_gripper_mae_m']==pytest.approx(.002)
    assert r['c10_valid_action_rows']==3

def test_uncertainty_weights_episodes_not_windows():
    rows=[]
    for i,n,value in [(1,100,1.),(2,10,3.)]:
        metrics=episode_metrics(np.full((n,50,7),value),np.zeros((n,50,7)),np.ones((n,50),bool))
        rows.append(dict(uuid=str(i),frames=n,evaluated_frames=n,metrics=metrics))
    r=complete_episode_summary(rows)
    assert r['episode_equal_mean']['c10_joint_mae_rad']==2.
    rows[1]['evaluated_frames']=9
    with pytest.raises(ValueError,match='partial'):complete_episode_summary(rows)
