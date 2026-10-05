from types import SimpleNamespace
import hashlib
import json

import numpy as np
import pytest


def observation(value):
    return {'state': np.full(7, value, np.float32), 'prompt': 'Insert the held plug into the socket.',
            'images': {'base_0_rgb': np.full((2, 2, 3), int(value), np.uint8)}}


def features(obs):
    return SimpleNamespace(z_rl=np.full(4, obs['images']['base_0_rgb'][0, 0, 0], np.float32),
                           proprio=obs['state'], ref_chunk=np.zeros((1, 7), np.float32))


def native_driver(tmp_path):
    from methods.openpi_rlt.cobot_adapter.online_runtime import install_bimanual_runtime_patch
    install_bimanual_runtime_patch()
    from rlt_online_rl.config import RLTOnlineRLConfig, EnvDriverConfig
    from rlt_online_rl.inference import EnvDriver
    d = object.__new__(EnvDriver)
    d._env = SimpleNamespace(cobot_task2_contract=True,
                            replay_commit_allowed=lambda: True,
                            _io=SimpleNamespace(_trace_writer=SimpleNamespace(_root=tmp_path)))
    d._rl_config = RLTOnlineRLConfig(action_dim=7, proprio_dim=7, z_dim=4, chunk_len=1)
    d._env_config = EnvDriverConfig(step_trace_stride=0)
    def get_features(obs):
        f = features(obs)
        return dict(z_rl=f.z_rl, proprio=f.proprio, ref_chunk=f.ref_chunk)
    d._feature_provider = SimpleNamespace(get_features=get_features)
    return d


def append(d, raw, start, end, *, observation_idx, chunk_id=0, anchor=False):
    from rlt_online_rl.inference import StepTraceRecord
    step = StepTraceRecord(start, np.full(7, .12345, np.float32), np.zeros(7, np.float32),
                           1., end, 1, 'online', False, True, 1, raw.episode_id, chunk_id, 2500)
    return d._append_raw_chunk(raw, observation_idx=observation_idx, trace_records=[step],
                              chunk_step_id=chunk_id, chunk_source=1, collection_phase='online',
                              done=True, success=1, drop_transition=False,
                              start_features=None if anchor else features(start),
                              policy_anchor_offsets=[0] if anchor else [],
                              policy_anchor_features=[features(start)] if anchor else [])


def raw_trace(start):
    from rlt_online_rl.replay import RawEpisodeTrace
    return RawEpisodeTrace(31, 1, [start], [], [])


def test_legacy_reproduces_fresh_input_loss_and_strict_audit_catches_it(tmp_path, monkeypatch):
    monkeypatch.setenv('COBOT_RLT_RAW_OBSERVATION_CONTRACT', 'legacy')
    monkeypatch.setenv('COBOT_RLT_INPUT_AUDIT', 'strict')
    d = native_driver(tmp_path)
    old, actual, end = observation(0), observation(5), observation(6)
    raw = raw_trace(old)
    append(d, raw, actual, end, observation_idx=0)
    assert raw.observations[raw.steps[0].observation_idx] is old
    with pytest.raises(ValueError, match='Replay input audit failed'):
        d._build_episode_replay(raw)


def test_trace_contract_keeps_actual_rgb_state_and_native_storage(tmp_path, monkeypatch):
    monkeypatch.setenv('COBOT_RLT_RAW_OBSERVATION_CONTRACT', 'trace')
    monkeypatch.setenv('COBOT_RLT_INPUT_AUDIT', 'strict')
    d = native_driver(tmp_path)
    old, actual, end = observation(0), observation(5), observation(6)
    raw = raw_trace(old)
    next_index = append(d, raw, actual, end, observation_idx=0)
    assert raw.observations[raw.steps[0].observation_idx] is actual
    assert raw.observations[next_index] is end
    assert raw.observations[0] is old
    transitions, _ = d._build_episode_replay(raw)
    assert len(transitions) == 1
    t = transitions[0]
    np.testing.assert_array_equal(t.proprio, actual['state'])
    np.testing.assert_array_equal(t.next_proprio, end['state'])
    np.testing.assert_array_equal(t.z_rl, features(actual).z_rl)
    np.testing.assert_array_equal(t.action_chunk[0], np.full(7, .12345, np.float32))
    receipt = json.loads(next((tmp_path/'replay_inputs').glob('*.jsonl')).read_text())
    assert receipt['checks_passed']
    assert receipt['current_input']['images']['base_0_rgb']['sha256'] == hashlib.sha256(actual['images']['base_0_rgb'].tobytes()).hexdigest()


def test_replan_anchor_moves_without_overwriting_previous_next_state(tmp_path, monkeypatch):
    monkeypatch.setenv('COBOT_RLT_RAW_OBSERVATION_CONTRACT', 'trace')
    monkeypatch.setenv('COBOT_RLT_INPUT_AUDIT', 'off')
    d = native_driver(tmp_path)
    first, previous_end, replanned, final = map(observation, [0, 1, 5, 6])
    raw = raw_trace(first)
    previous_index = append(d, raw, first, previous_end, observation_idx=0)
    d._record_feature_anchor(raw, previous_index, features(previous_end))
    append(d, raw, replanned, final, observation_idx=previous_index, chunk_id=1, anchor=True)
    assert raw.observations[raw.steps[0].next_observation_idx] is previous_end
    assert raw.observations[raw.steps[1].observation_idx] is replanned
    cache = d._seed_feature_cache(raw)
    np.testing.assert_array_equal(cache[previous_index]['proprio'], previous_end['state'])
    np.testing.assert_array_equal(cache[raw.steps[1].observation_idx]['proprio'], replanned['state'])
    assert raw.policy_start_steps == [1]


def test_contiguous_observation_objects_reuse_original_indices(tmp_path, monkeypatch):
    monkeypatch.setenv('COBOT_RLT_RAW_OBSERVATION_CONTRACT', 'trace')
    monkeypatch.setenv('COBOT_RLT_INPUT_AUDIT', 'off')
    d = native_driver(tmp_path)
    start, end = observation(0), observation(1)
    raw = raw_trace(start)
    append(d, raw, start, end, observation_idx=0)
    assert len(raw.observations) == 2
    assert raw.steps[0].observation_idx == 0
    assert raw.summary['cobot_observation_reindex_receipts'] == []


def test_first_step_hil_resample_discards_cached_plan_for_other_input(monkeypatch):
    from methods.openpi_rlt.plug_v3_yyshadow.right_arm_env import RightArmCobotOnlineEnv
    from methods.openpi_rlt.tests.test_cobot_online_env import FakeTask2IO
    from methods.openpi_rlt.cobot_adapter.trace import EpisodeOutcome
    monkeypatch.setenv('COBOT_RLT_RAW_OBSERVATION_CONTRACT', 'trace')
    monkeypatch.setenv('COBOT_RLT_HIL_SAMPLING', 'logical20')
    def sample(value, mode, paused=False, outcome=None):
        return SimpleNamespace(observation=observation(value), mode=mode, paused=paused,
                               outcome=outcome, timestamp=float(value))
    io = FakeTask2IO([sample(0, 'policy'), sample(1, 'manual:right', True),
                      sample(2, 'manual:right', outcome=EpisodeOutcome.SUCCESS)])
    env = RightArmCobotOnlineEnv(io, chunk_exec_horizon=1, control_frequency_hz=20.,
                                max_episode_steps=None, joint_step_limit=.03,
                                gripper_step_limit=.004, sleep=lambda _: None)
    def planner(obs, *_):
        return SimpleNamespace(start_features=features(obs), action_chunk=np.zeros((1, 7), np.float32),
                               ref_chunk=np.zeros((1, 7), np.float32), source=1, actor_param_version=2500)
    _, _, _, info = env.execute_chunk(env.reset(), planner)
    assert info['chunk_start_features'] is None
    np.testing.assert_array_equal(info['step_trace'][0]['observation']['state'], np.ones(7, np.float32))


def test_invalid_contract_is_rejected_before_driver_or_ros_creation(monkeypatch, tmp_path):
    monkeypatch.setenv('COBOT_RLT_RAW_OBSERVATION_CONTRACT', 'unknown')
    with pytest.raises(ValueError, match='legacy or trace'):
        native_driver(tmp_path)
