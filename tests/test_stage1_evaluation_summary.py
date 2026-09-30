import numpy as np
import pytest
from methods.openpi_rlt.plug_v3_yyshadow.evaluation_summary import aggregate_batches,summarize_episodes


def test_partial_batch_is_weighted_and_rmse_is_not_mean_of_roots():
    result=aggregate_batches([
        {'samples':8,'metrics':{'action_mse_10_norm':1.,'action_mae_per_dim_norm':[1.,2.]}},
        {'samples':1,'metrics':{'action_mse_10_norm':9.,'action_mae_per_dim_norm':[10.,20.]}}
    ])
    assert result['samples']==9
    assert result['action_rmse_10_norm']==pytest.approx(np.sqrt(17/9))
    assert result['action_mae_per_dim_norm']==pytest.approx([2.,4.])


def test_bootstrap_unit_is_episode_not_frame_count():
    rows=[{'episode':1,'samples':100,'available_frames':100,'batches':13,'seconds':1,'loss':0.},
          {'episode':2,'samples':2,'available_frames':2,'batches':1,'seconds':1,'loss':1.}]
    result=summarize_episodes(rows)
    assert result['episode_equal_mean']['loss']==.5
    assert result['episodes_evaluated']==[1,2]
    assert result['episode_bootstrap_95']['loss']==[0.,1.]
    assert result==summarize_episodes(rows)


def test_partial_episode_has_no_complete_episode_ci():
    rows=[{'episode':1,'samples':64,'available_frames':141,'batches':8,'seconds':1,'loss':1.}]
    result=summarize_episodes(rows)
    assert not result['complete_episode_coverage']
    assert result['episode_bootstrap_95']['loss'] is None


def test_missing_duplicate_or_nonfinite_evidence_is_rejected():
    with pytest.raises(ValueError):aggregate_batches([])
    with pytest.raises(ValueError):aggregate_batches([{'samples':1,'metrics':{'x':float('nan')}}])
    row={'episode':1,'samples':1,'available_frames':1,'batches':1,'seconds':1,'loss':1.}
    with pytest.raises(ValueError):summarize_episodes([row,row])
