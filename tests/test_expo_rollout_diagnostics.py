"""Offline diagnostics tests; no policy/GPU/robot dependencies."""
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'methods/expo_ft/cobot_adapter'))
import monitoring
from monitoring import RolloutHistory, infer_with_q, install_selection_observer, selection_diagnostics


def selected(edited=True):
    base = np.zeros((2, 8, 7), dtype=np.float32)
    base[1] = 1.0
    edits = base.copy()
    edits[0, 0, 0] += .3
    edits[0, 0, 1] += .4
    edits[1] += .1
    candidates = np.concatenate((base, edits))
    index = 2 if edited else 1
    return dict(qs=np.array([-.2, .1, .5, .2]), idx=np.array(index),
                cand=candidates, selected=candidates[index], rng=np.array([42, 9]))


def test_selected_edit_is_compared_to_its_paired_base_in_normalized_units():
    output = selected()
    before = {key: value.copy() for key, value in output.items()}
    value = selection_diagnostics(output, 2)
    assert value['selected_q'] == .5
    assert value['best_unedited_q'] == .1
    assert value['edit_selected'] is True
    assert value['edit_norm'] == pytest.approx(.5)
    for key in before:
        np.testing.assert_array_equal(output[key], before[key])
    json.dumps(value, allow_nan=False)


def test_unedited_choice_has_observed_zero_edit_and_no_extra_model_call():
    output = selected(False)
    calls = []
    def select(*args, **kwargs):
        calls.append(kwargs)
        return output
    module = SimpleNamespace(_jitted_fast_select=select)
    install_selection_observer(module)
    install_selection_observer(module)
    result = {'actions': output['selected'].tolist(), 'actor_version': 9}
    def infer(actor, route, data):
        assert module._jitted_fast_select(N=2) is output
        return result
    actual = infer_with_q(None, infer, '/select', {})
    assert actual is result
    assert calls == [{'N': 2}]
    assert actual['edit_selected'] is False and actual['edit_norm'] == 0.0
    assert actual['selected_q'] == actual['best_unedited_q'] == .1
    assert actual['actions'] == output['selected'].tolist()
    np.testing.assert_array_equal(output['rng'], [42, 9])


def test_precache_does_not_report_selection_or_leak_to_next_request():
    output = selected()
    module = SimpleNamespace(_jitted_fast_select=lambda **kwargs: output)
    install_selection_observer(module)
    def precache(actor, route, data):
        module._jitted_fast_select(N=2)
        return {'precached': 'unchanged'}
    assert infer_with_q(None, precache, '/precache', {}) == {'precached': 'unchanged'}
    assert infer_with_q(None, lambda *args: {'actions': []}, '/select', {}) == {'actions': []}


def test_missing_candidates_or_unknown_layout_do_not_fabricate_diagnostics():
    output = selected()
    assert selection_diagnostics(output) == {'selected_q': .5}
    assert selection_diagnostics(output, True) == {'selected_q': .5}
    output['cand'] = None
    assert selection_diagnostics(output, 2) == dict(selected_q=.5, best_unedited_q=.1, edit_selected=True)
    output['qs'] = None
    assert selection_diagnostics(output, 2) == {}


@pytest.mark.parametrize('index', [-1, 10, 1.5, None])
def test_invalid_index_does_not_read_arbitrary_candidate(index):
    output = selected()
    output['idx'] = index
    assert selection_diagnostics(output, 2) == {}


def test_nonfinite_optional_values_remain_missing():
    output = selected()
    output['qs'][0] = float('nan')
    output['cand'][2, 0, 0] = float('nan')
    assert selection_diagnostics(output, 2) == dict(selected_q=.5, edit_selected=True)
    output['qs'][2] = float('inf')
    assert selection_diagnostics(output, 2) == {}


def test_diagnostic_failure_does_not_change_inference_result(monkeypatch):
    result = {'actions': [[1, 2]], 'actor_version': 3}
    def unavailable(*args):
        raise RuntimeError('diagnostic conversion unavailable')
    monkeypatch.setattr(monitoring, 'selection_diagnostics', unavailable)
    assert infer_with_q(None, lambda *args: result, '/select', {}) is result


def test_same_select_fields_survive_json_and_history_archive(tmp_path, monkeypatch):
    monkeypatch.setattr(monitoring.time, 'time', lambda: 101.25)
    monkeypatch.setattr(monitoring.time, 'monotonic', lambda: 50.125)
    history = RolloutHistory(tmp_path)
    payload = dict(selection_diagnostics(selected(), 2), actor_version=7, compute_seconds=.012)
    history.selected(dict(episode_uuid='first', started_at=100), payload)
    assert not history.path.exists()
    point = history.points[0]
    assert point['seconds'] == 1.25 and point['timestamp'] == 101.25
    assert point['monotonic_seconds'] == 50.125
    assert point['q'] == .5 and point['inference_ms'] == 12
    assert 'wait_ms' not in point
    history.flush()
    archive = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'integrations/cobot_runtime/rollout_archive.py'))
    _, saved = archive['history'](tmp_path, 'first')
    assert saved['actor_version'] == 7 and saved['points'] == [point]
    history.selected(dict(episode_uuid='second', started_at=101), dict(selected_q=.2, actor_version=8))
    history.flush()
    current = json.loads(history.path.read_text())
    assert current['episode_uuid'] == 'second' and current['actor_version'] == 8
    assert set(current['points'][0]) == {'seconds', 'timestamp', 'monotonic_seconds', 'q'}
    assert archive['history'](tmp_path, 'first')[1]['points'] == [point]


def test_invalid_or_missing_metrics_are_not_zero_filled():
    assert monitoring.rollout_diagnostics({}) == {}
    invalid = dict(best_unedited_q=float('nan'), edit_selected=1, edit_norm=-1,
                   inference_ms=float('inf'), wait_ms=-.1, compute_seconds=-1)
    assert monitoring.rollout_diagnostics(invalid) == {}
    assert monitoring.rollout_diagnostics(dict(compute_seconds=1e308)) == {}
    valid = dict(best_unedited_q=-2., edit_selected=False, edit_norm=0., inference_ms=3., wait_ms=4.)
    assert monitoring.rollout_diagnostics({**valid, 'compute_seconds': 1.}) == valid


def test_runtime_pairs_precache_wait_and_selection_without_changing_rpc_or_actions(monkeypatch):
    from online_runtime import OnlineRuntime
    runtime = OnlineRuntime.__new__(OnlineRuntime)
    runtime.episode = 'online-one'
    runtime.journal = SimpleNamespace(active=dict(episode_uuid='online-one', started_at=0))
    recorded = []
    runtime.monitor_history = SimpleNamespace(selected=lambda active, payload: recorded.append((active, payload)))
    observation = {'state': np.arange(7, dtype=np.float32)}
    runtime.observation = lambda: observation
    actions = np.arange(56, dtype=np.float32).reshape(8, 7)
    cache = np.zeros((2, 16, 7), np.float32)
    calls = []
    def rpc(route, **values):
        calls.append((route, values))
        if route == '/precache':
            return dict(precached=cache.tolist(), delay=5, actor_version=9, compute_seconds=.15)
        return dict(actions=actions.tolist(), actor_version=9, selected_q=.5, compute_seconds=.012)
    runtime.array_rpc = rpc
    ticks = iter([10., 10.2])
    monkeypatch.setattr(monitoring.time, 'monotonic', lambda: next(ticks))
    result = runtime.infer(observation, np.zeros((5, 7)))
    assert result['_rollout_telemetry'] == dict(precache_completed_monotonic=10., precache_ms=150.)
    output = runtime.resolve_inference(result)
    np.testing.assert_array_equal(output, actions)
    assert [route for route, values in calls] == ['/precache', '/select']
    assert all('_rollout_telemetry' not in values for route, values in calls)
    np.testing.assert_array_equal(calls[1][1]['precached'], cache)
    assert calls[1][1]['delay'] == 5 and calls[1][1]['actor_version'] == 9
    assert len(recorded) == 1 and recorded[0][0] is runtime.journal.active
    point = monitoring.rollout_diagnostics(recorded[0][1])
    assert point['wait_ms'] == pytest.approx(200.)
    assert point['inference_ms'] == 12. and point['precache_ms'] == 150.
    assert runtime.episode == 'online-one'


def test_runtime_legacy_cache_has_no_wait_value_and_keeps_error_behavior(monkeypatch):
    from online_runtime import OnlineRuntime
    runtime = OnlineRuntime.__new__(OnlineRuntime)
    runtime.episode = 'old'
    runtime.journal = SimpleNamespace(active={})
    recorded = []
    runtime.monitor_history = SimpleNamespace(selected=lambda active, payload: recorded.append(payload))
    runtime.observation = lambda: {}
    runtime.array_rpc = lambda *args, **kwargs: dict(actions=[[1., 2.]], actor_version=1, selected_q=.2)
    monkeypatch.setattr(monitoring.time, 'monotonic', lambda: 10.)
    output = runtime.resolve_inference(dict(precached=[], delay=5, actor_version=1))
    np.testing.assert_array_equal(output, np.array([[1., 2.]], np.float32))
    assert 'wait_ms' not in recorded[0] and 'precache_ms' not in recorded[0]
    runtime.episode = None
    with pytest.raises(ValueError, match='Online episode has ended'):
        runtime.resolve_inference({})


def test_wait_is_per_cache_not_previous_or_negative():
    first = {'_rollout_telemetry': monitoring.precache_timing({'compute_seconds': .1}, 10.)}
    second = {'_rollout_telemetry': monitoring.precache_timing({'compute_seconds': .2}, 20.)}
    assert monitoring.selection_timing(second, 21.) == dict(wait_ms=1000., precache_ms=200.)
    assert monitoring.selection_timing(first, 22.) == dict(wait_ms=12000., precache_ms=100.)
    assert monitoring.selection_timing(first, 9.) == dict(precache_ms=100.)
    assert monitoring.selection_timing({}, 22.) == {}
    assert monitoring.precache_timing({'compute_seconds': float('nan')}, float('inf')) == {}
    assert monitoring.selection_timing({'_rollout_telemetry': 'unknown'}, 22.) == {}


def test_continuous_runtime_preserves_version_token_and_measures_same_cache(monkeypatch):
    from continuous_runtime import ContinuousRuntime
    runtime = ContinuousRuntime.__new__(ContinuousRuntime)
    runtime.episode = 'continuous-one'
    runtime.local_actor_version = 1
    runtime.journal = SimpleNamespace(active=dict(episode_uuid='continuous-one', actor_version=1, started_at=0))
    recorded = []
    runtime.monitor_history = SimpleNamespace(selected=lambda active, payload: recorded.append(payload))
    runtime.observation = lambda: {'state': np.zeros(7)}
    actions = np.arange(56, dtype=np.float32).reshape(8, 7)
    calls = []
    def rpc(route, **values):
        calls.append((route, values))
        if route == '/precache':
            return dict(precached=np.zeros((2, 16, 7)).tolist(), delay=5, actor_version=2,
                        inference_token='same-inference', compute_seconds=.1)
        return dict(actions=actions.tolist(), actor_version=4, precache_actor_version=2,
                    selected_q=.7, best_unedited_q=.5, edit_selected=True,
                    edit_norm=.2, compute_seconds=.012)
    runtime.array_rpc = rpc
    ticks = iter([10., 10.15])
    monkeypatch.setattr(monitoring.time, 'monotonic', lambda: next(ticks))
    cached = runtime.infer(runtime.observation(), np.zeros((5, 7)))
    assert runtime.local_actor_version == 2
    output = runtime.resolve_inference(cached)
    assert runtime.local_actor_version == 4
    np.testing.assert_array_equal(output, actions)
    assert [route for route, values in calls] == ['/precache', '/select']
    assert calls[1][1]['inference_token'] == 'same-inference'
    assert calls[1][1]['actor_version'] == 2 and calls[1][1]['delay'] == 5
    assert all('_rollout_telemetry' not in values for route, values in calls)
    assert recorded[0]['actor_version'] == 4 and recorded[0]['precache_actor_version'] == 2
    assert recorded[0]['wait_ms'] == pytest.approx(150.)
    assert recorded[0]['precache_ms'] == 100.
    assert monitoring.rollout_diagnostics(recorded[0])['inference_ms'] == 12.
    assert runtime.episode == 'continuous-one'


def test_continuous_points_archive_new_fields_and_mixed_policy_provenance(tmp_path, monkeypatch):
    from continuous_monitoring import ContinuousRolloutHistory
    history = ContinuousRolloutHistory(tmp_path)
    active = dict(episode_uuid='mixed', actor_version=2, started_at=100.)
    monkeypatch.setattr(monitoring.time, 'time', lambda: 101.5)
    monkeypatch.setattr(monitoring.time, 'monotonic', lambda: 50.75)
    metrics = dict(selected_q=.5, best_unedited_q=.1, edit_selected=True, edit_norm=.2,
                   compute_seconds=.012, wait_ms=150., precache_ms=100.)
    history.selected(active, dict(metrics, actor_version=2, precache_actor_version=2))
    history.selected(active, dict(metrics, actor_version=5, precache_actor_version=2))
    history.flush()
    stored = json.loads(history.path.read_text())
    assert stored['policy_versions'] == [2, 5]
    assert stored['actor_version'] == 2 and stored['last_actor_version'] == 5
    point = stored['points'][-1]
    assert point['actor_version'] == 5 and point['precache_actor_version'] == 2
    assert point['timestamp'] == 101.5 and point['monotonic_seconds'] == 50.75 and point['seconds'] == 1.5
    assert {k: point[k] for k in ('best_unedited_q', 'edit_selected', 'edit_norm', 'inference_ms', 'wait_ms', 'precache_ms')} == dict(
        best_unedited_q=.1, edit_selected=True, edit_norm=.2, inference_ms=12., wait_ms=150., precache_ms=100.)
    archive = runpy.run_path(str(Path(__file__).resolve().parents[1]/'integrations/cobot_runtime/rollout_archive.py'))
    _, saved = archive['history'](tmp_path, 'mixed')
    assert saved['points'][-1] == point and saved['policy_versions'] == [2, 5]
    history.selected(dict(active, episode_uuid='next', actor_version=5),
                     dict(actor_version=6, precache_actor_version=5, selected_q=.1))
    history.flush()
    next_point = json.loads(history.path.read_text())['points'][-1]
    assert next_point['actor_version'] == 6 and next_point['precache_actor_version'] == 5
    assert not any(key in next_point for key in ('best_unedited_q', 'edit_selected', 'edit_norm', 'inference_ms', 'wait_ms', 'precache_ms'))
