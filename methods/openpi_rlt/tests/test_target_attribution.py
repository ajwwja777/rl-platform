import numpy as np
from methods.openpi_rlt.experiments.target_attribution import reconstruct_observed_returns, episode_interval


def data():
    return dict(episode_id=np.array([7, 7]), collection_phase_id=np.array([2, 2]),
        step_id=np.array([0, 2]), rewards=np.array([[0., 0., 0.], [0., 0., 1.]]),
        source_chunk=np.array([[1, 2, 2], [2, 2, 2]]), done=np.array([False, True]), success=np.array([0, 1]))


def test_overlap_counted_once_and_return_is_logical_observed_behavior():
    values, rows = reconstruct_observed_returns(data(), .9)
    np.testing.assert_allclose(values, [.9**4, .9**2])
    assert rows[0]['unique_observed_steps'] == 5
    assert rows[0]['valid_return_windows'] == 2


def test_missing_middle_stays_missing():
    d = data(); d['step_id'][1] = 4
    values, rows = reconstruct_observed_returns(d, .9)
    assert np.isnan(values[0]) and np.isfinite(values[1])


def test_contradictory_reward_or_source_invalidates_episode():
    for key in ['rewards', 'source_chunk']:
        d = data(); d[key][1, 0] = 1
        values, rows = reconstruct_observed_returns(d, .9)
        assert np.isnan(values).all() and rows[0]['conflicting_overlap_steps'] == [2]


def test_conflicting_terminal_and_reward_not_used():
    d = data(); d['success'][1] = 0
    values, rows = reconstruct_observed_returns(d, .9)
    assert np.isnan(values).all() and not rows[0]['terminal_reward_consistent']


def test_uncertainty_is_equal_episode_weight_not_equal_window_weight():
    result = episode_interval(np.array([1., 1., 1., 3.]), np.array([7, 7, 7, 8]), np.ones(4), np.ones(4, bool))
    assert result['episodes'] == 2 and result['mean'] == 2.
