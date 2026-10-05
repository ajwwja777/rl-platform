import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from methods.openpi_rlt.cobot_adapter import input_audit, input_snapshots
from methods.openpi_rlt.tests.test_input_audit import driver_fixture, build


def enabled(tmp_path, monkeypatch, count=3):
    monkeypatch.setenv('COBOT_RLT_INPUT_SNAPSHOT_COUNT', str(count))
    monkeypatch.setenv('COBOT_RLT_INPUT_SNAPSHOT_ROOT', str(tmp_path/'snapshots'))
    monkeypatch.setenv('COBOT_RLT_INPUT_AUDIT', 'strict')
    monkeypatch.setenv('COBOT_RLT_RAW_OBSERVATION_CONTRACT', 'trace')
    driver, raw, features = driver_fixture(tmp_path)
    driver._env._io._trace_writer._root = tmp_path/'trace'
    return driver, raw, features


def last_receipt(tmp_path):
    path = next((tmp_path/'trace/replay_inputs').glob('*.jsonl'))
    return json.loads(path.read_text().splitlines()[-1])


def test_default_off_preserves_native_arrays_and_creates_no_snapshot(tmp_path, monkeypatch):
    monkeypatch.delenv('COBOT_RLT_INPUT_SNAPSHOT_COUNT', raising=False)
    monkeypatch.setenv('COBOT_RLT_INPUT_AUDIT', 'strict')
    driver, raw, _ = driver_fixture(tmp_path)
    before = build(driver, raw).to_numpy()
    after = build(driver, raw).to_numpy()
    for key in before:
        np.testing.assert_array_equal(after[key], before[key])
    receipt = json.loads(next((tmp_path/'replay_inputs').glob('*.jsonl')).read_text().splitlines()[-1])
    assert 'lossless_snapshot' not in receipt
    assert not list(tmp_path.rglob('*.npz'))


def test_lossless_roundtrip_includes_rgb_prompt_rtc_features_and_actual_serializer(tmp_path, monkeypatch):
    driver, raw, _ = enabled(tmp_path, monkeypatch)
    raw.observations[0]['rtc'] = {'prefix': np.arange(14, dtype=np.float32).reshape(2, 7),
                                  'offset': np.int32(1), 'schedule': ('linear', None)}
    raw.observations[0]['images']['base_0_rgb'][0, 0] = [1, 2, 255]
    transition = build(driver, raw)
    receipt = last_receipt(tmp_path);pointer = receipt['lossless_snapshot']
    restored, manifest = input_snapshots.load_and_verify(pointer['path'], metadata_sha256=pointer['metadata_sha256'])
    obs = restored['current_input']
    np.testing.assert_array_equal(obs['images']['base_0_rgb'], raw.observations[0]['images']['base_0_rgb'])
    np.testing.assert_array_equal(obs['rtc']['prefix'], raw.observations[0]['rtc']['prefix'])
    assert isinstance(obs['rtc']['offset'], np.int32)
    assert obs['rtc']['schedule'] == ('linear', None)
    assert obs['prompt'] == raw.observations[0]['prompt']
    for key, value in transition.to_numpy().items():
        np.testing.assert_array_equal(restored['serialized_replay_arrays'][key], value)
        assert restored['serialized_replay_arrays'][key].dtype == value.dtype
    assert input_audit.observation_receipt(obs) == manifest['receipt']['current_input']


def test_budget_is_not_reset_between_episodes_and_omission_is_explicit(tmp_path, monkeypatch):
    driver, raw, _ = enabled(tmp_path, monkeypatch)
    for ep in range(5):
        raw.episode_id = 30+ep
        build(driver, raw)
    assert len(list((tmp_path/'snapshots').rglob('*.npz'))) == 3
    assert last_receipt(tmp_path)['lossless_snapshot']['reason'] == 'snapshot_count_limit'
    assert driver._cobot_input_snapshot_writer.saved == 3


@pytest.mark.parametrize('value', ['x', '-1', '4'])
def test_invalid_count_is_rejected_before_driver_creation(tmp_path, monkeypatch, value):
    monkeypatch.setenv('COBOT_RLT_INPUT_SNAPSHOT_COUNT', value)
    with pytest.raises(ValueError, match='integer from 0 to 3'):
        driver_fixture(tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('setting,value,message', [
    ('COBOT_RLT_INPUT_SNAPSHOT_ROOT', 'relative', 'absolute independent'),
    ('COBOT_RLT_INPUT_AUDIT', 'off', 'record or strict'),
    ('COBOT_RLT_RAW_OBSERVATION_CONTRACT', 'legacy', 'trace raw'),
])
def test_enabled_contract_is_rejected_early(tmp_path, monkeypatch, setting, value, message):
    monkeypatch.setenv('COBOT_RLT_INPUT_SNAPSHOT_COUNT', '1')
    monkeypatch.setenv('COBOT_RLT_INPUT_SNAPSHOT_ROOT', str(tmp_path/'snapshots'))
    monkeypatch.setenv('COBOT_RLT_INPUT_AUDIT', 'strict')
    monkeypatch.setenv('COBOT_RLT_RAW_OBSERVATION_CONTRACT', 'trace')
    monkeypatch.setenv(setting, value)
    with pytest.raises(ValueError, match=message):
        driver_fixture(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_nested_output_root_cannot_write_inside_trace_evidence(tmp_path, monkeypatch):
    driver, raw, _ = enabled(tmp_path, monkeypatch)
    monkeypatch.setenv('COBOT_RLT_INPUT_SNAPSHOT_ROOT', str(tmp_path/'trace/snapshots'))
    with pytest.raises(ValueError, match='independent of the trace'):
        build(driver, raw)
    assert not list(tmp_path.rglob('*.npz'))


def test_oversized_input_is_rejected_before_any_snapshot_artifact(tmp_path, monkeypatch):
    driver, raw, _ = enabled(tmp_path, monkeypatch)
    raw.observations[0]['images']['base_0_rgb'] = np.zeros((2048, 2048, 3), np.uint8)
    with pytest.raises(ValueError, match='8MiB bound'):
        build(driver, raw)
    assert not (tmp_path/'snapshots').exists()


def test_corrupt_archive_and_wrong_external_manifest_hash_are_rejected(tmp_path, monkeypatch):
    driver, raw, _ = enabled(tmp_path, monkeypatch)
    build(driver, raw);pointer = last_receipt(tmp_path)['lossless_snapshot']
    with pytest.raises(ValueError, match='metadata fingerprint'):
        input_snapshots.load_and_verify(pointer['path'], metadata_sha256='0'*64)
    from pathlib import Path
    archive = Path(pointer['path']).with_suffix('.npz')
    archive.write_bytes(archive.read_bytes()+b'bad')
    with pytest.raises(ValueError, match='archive fingerprint'):
        input_snapshots.load_and_verify(pointer['path'])


def test_metadata_write_failure_does_not_leave_a_partial_or_budgeted_snapshot(tmp_path, monkeypatch):
    driver, raw, _ = enabled(tmp_path, monkeypatch)
    original = input_snapshots._publish_exclusive
    def fail_metadata(temporary, target):
        if target.suffix == '.json':
            raise OSError('fixture metadata publication failure')
        original(temporary, target)
    monkeypatch.setattr(input_snapshots, '_publish_exclusive', fail_metadata)
    with pytest.raises(OSError, match='fixture metadata'):
        build(driver, raw)
    assert not list((tmp_path/'snapshots').rglob('*.npz'))
    assert not list((tmp_path/'snapshots').rglob('*.writing'))
    assert driver._cobot_input_snapshot_writer.saved == 0


def test_mid_run_config_change_cannot_reset_the_snapshot_budget(tmp_path, monkeypatch):
    driver, raw, _ = enabled(tmp_path, monkeypatch, count=1)
    build(driver, raw)
    monkeypatch.setenv('COBOT_RLT_INPUT_SNAPSHOT_COUNT', '3')
    with pytest.raises(ValueError, match='configuration changed'):
        build(driver, raw)
    assert len(list((tmp_path/'snapshots').rglob('*.npz'))) == 1
