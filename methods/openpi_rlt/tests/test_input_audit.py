import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from methods.openpi_rlt.cobot_adapter import input_audit


def observation(value):
    return {'state': np.full(7, value, np.float32), 'prompt': 'Insert the held plug into the socket.',
            'images': {'base_0_rgb': np.full((2, 2, 3), int(value), np.uint8)}}


def test_raw_pixel_prompt_and_rtc_changes_have_distinct_input_identities():
    obs = observation(1)
    before = input_audit.observation_receipt(obs)['sha256']
    obs['images']['base_0_rgb'][0, 0, 0] += 1
    assert input_audit.observation_receipt(obs)['sha256'] != before
    before = input_audit.observation_receipt(obs)['sha256']
    obs['rtc'] = {'prefix': np.zeros((2, 7), np.float32), 'delay': 1, 'schedule': 'linear'}
    assert input_audit.observation_receipt(obs)['sha256'] != before
    before = input_audit.observation_receipt(obs)['sha256']
    obs['prompt'] = 'wrong task'
    assert input_audit.observation_receipt(obs)['sha256'] != before


def driver_fixture(tmp_path):
    from methods.openpi_rlt.cobot_adapter.online_runtime import install_bimanual_runtime_patch
    install_bimanual_runtime_patch()
    from rlt_online_rl.config import RLTOnlineRLConfig
    from rlt_online_rl.inference import EnvDriver

    driver = object.__new__(EnvDriver)
    driver._rl_config = RLTOnlineRLConfig(action_dim=7, proprio_dim=7, z_dim=4, chunk_len=2)
    driver._env = SimpleNamespace(cobot_task2_contract=True,
        _io=SimpleNamespace(_trace_writer=SimpleNamespace(_root=tmp_path)))
    raw = SimpleNamespace(episode_id=3, chunk_len=2, observations=[observation(0), observation(1), observation(2)],
        steps=[SimpleNamespace(observation_idx=i, next_observation_idx=i+1,
            action=np.full(7, i+1, np.float32), reward=float(i == 1), done=i == 1,
            source=2, collection_phase='online', success=int(i == 1), intervention_flag=True,
            episode_id=3, step_id=i, actor_param_version=2500) for i in range(2)])
    payloads = {i: dict(z_rl=np.full(4, i, np.float32), proprio=np.full(7, i, np.float32),
                       ref_chunk=np.full((2, 7), i, np.float32)) for i in range(3)}
    driver._feature_payload_for_observation = lambda raw, i, cache, stats: payloads[i]
    return driver, raw, payloads


def build(driver, raw):
    return driver._build_transition_from_window(raw, SimpleNamespace(raw_indices=[0, 1]),
        SimpleNamespace(start_offset=0), {}, {})


def test_native_transition_receipt_matches_actual_storage_without_changing_arrays(tmp_path, monkeypatch):
    monkeypatch.setenv('COBOT_RLT_INPUT_AUDIT', 'off')
    driver, raw, _ = driver_fixture(tmp_path)
    baseline = build(driver, raw)
    assert list(tmp_path.iterdir()) == []
    monkeypatch.setenv('COBOT_RLT_INPUT_AUDIT', 'strict')
    audited = build(driver, raw)
    before, after = baseline.to_numpy(), audited.to_numpy()
    for key in before:
        assert np.array_equal(np.asarray(before[key]), np.asarray(after[key]))
    files = list((tmp_path/'replay_inputs').glob('*.jsonl'))
    assert len(files) == 1
    receipt = json.loads(files[0].read_text())
    assert receipt['checks_passed']
    assert receipt['raw_step_indices'] == [0, 1]
    for key in ('z_rl', 'action_chunk', 'ref_chunk', 'next_z_rl', 'next_ref_chunk'):
        expected = hashlib.sha256(np.asarray(after[key], np.float16).tobytes()).hexdigest()
        assert receipt['training_arrays_before_storage'][key]['fp16']['sha256'] == expected
    for key, value in after.items():
        assert receipt['serialized_replay_arrays'][key]['sha256'] == hashlib.sha256(value.tobytes()).hexdigest()
        assert receipt['serialized_replay_arrays'][key]['dtype'] == str(value.dtype)
    assert not list(tmp_path.glob('episode_*.jsonl'))  # Separate from executed-step traces.


def test_strict_audit_rejects_stale_feature_state_and_preserves_failed_receipt(tmp_path, monkeypatch):
    monkeypatch.setenv('COBOT_RLT_INPUT_AUDIT', 'strict')
    driver, raw, payloads = driver_fixture(tmp_path)
    payloads[0]['proprio'] = np.full(7, 99, np.float32)
    with pytest.raises(ValueError, match='Replay input audit failed'):
        build(driver, raw)
    receipt = json.loads(next((tmp_path/'replay_inputs').glob('*.jsonl')).read_text())
    assert not receipt['checks_passed']
    assert not receipt['checks']['current_state_equals_feature_proprio']


def test_receipt_uses_actual_opt_in_float32_action_serializer(tmp_path, monkeypatch):
    from rlt_online_rl.replay import RLTTransition
    from methods.openpi_rlt.cobot_adapter.replay_precision import install_action_precision_patch

    monkeypatch.setattr(RLTTransition, 'to_numpy', RLTTransition.to_numpy)
    install_action_precision_patch(enabled=True)
    monkeypatch.setenv('COBOT_RLT_INPUT_AUDIT', 'strict')
    driver, raw, _ = driver_fixture(tmp_path)
    raw.steps[0].action[0] += np.float32(.0001)
    transition = build(driver, raw)
    receipt = json.loads(next((tmp_path/'replay_inputs').glob('*.jsonl')).read_text())
    actual = transition.to_numpy()['action_chunk']
    assert actual.dtype == np.float32
    assert receipt['serialized_replay_arrays']['action_chunk']['dtype'] == 'float32'
    assert receipt['serialized_replay_arrays']['action_chunk']['sha256'] == hashlib.sha256(actual.tobytes()).hexdigest()


def test_invalid_audit_mode_fails_before_any_output(tmp_path, monkeypatch):
    monkeypatch.setenv('COBOT_RLT_INPUT_AUDIT', 'unknown')
    with pytest.raises(ValueError, match='off, record or strict'):
        driver_fixture(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_record_mode_retains_diagnostic_without_rejecting_transition(tmp_path, monkeypatch):
    monkeypatch.setenv('COBOT_RLT_INPUT_AUDIT', 'record')
    driver, raw, payloads = driver_fixture(tmp_path)
    payloads[0]['proprio'] = np.full(7, 99, np.float32)
    transition = build(driver, raw)
    assert np.array_equal(transition.proprio, payloads[0]['proprio'])
    receipt = json.loads(next((tmp_path/'replay_inputs').glob('*.jsonl')).read_text())
    assert receipt['audit_mode'] == 'record' and not receipt['checks_passed']


def test_strict_checks_actual_storage_overflow_even_if_native_action_is_finite(tmp_path, monkeypatch):
    from rlt_online_rl.replay import RLTTransition
    from methods.openpi_rlt.cobot_adapter.replay_precision import install_action_precision_patch
    monkeypatch.setattr(RLTTransition, 'to_numpy', RLTTransition.to_numpy)
    install_action_precision_patch(enabled=False)
    monkeypatch.setenv('COBOT_RLT_INPUT_AUDIT', 'strict')
    driver, raw, _ = driver_fixture(tmp_path)
    raw.steps[0].action[0] = np.float32(1e6)
    with np.errstate(over='ignore'), pytest.raises(ValueError, match='Replay input audit failed'):
        build(driver, raw)
    receipt = json.loads(next((tmp_path/'replay_inputs').glob('*.jsonl')).read_text())
    assert receipt['training_arrays_before_storage']['action_chunk']['native']['finite']
    assert not receipt['serialized_replay_arrays']['action_chunk']['finite']
    assert not receipt['checks_passed']
