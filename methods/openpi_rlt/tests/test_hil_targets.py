from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from methods.openpi_rlt.cobot_adapter import hil_targets


def sample(value=0., mode='manual:right', outcome=None):
    return SimpleNamespace(observation={'state': np.full(7, value, np.float32), 'images': {}, 'prompt': 'Insert'},
                           mode=mode, paused=True, timestamp=100., outcome=outcome,
                           io_evidence={'mode': mode, 'captured_ros_time': 100., 'captured_monotonic': 200.,
                                        'right_takeover_started_monotonic': 199.98,
                                        'coordinator_commands': {
                                            'right': {'target': [.03]*6+[.004], 'valid': True,
                                                      'topic': '/master/joint_right', 'ros_stamp': 99.99,
                                                      'arrival_monotonic': 199.99},
                                            'left': {'target': None, 'valid': False}}})


def select_options(monkeypatch):
    monkeypatch.setenv('COBOT_RLT_HIL_TARGET', 'coordinator_command')
    monkeypatch.setenv('COBOT_RLT_HIL_SAMPLING', 'logical20')
    monkeypatch.setenv('COBOT_RLT_RAW_OBSERVATION_CONTRACT', 'trace')


def test_right_command_does_not_require_left_freshness_or_use_feedback():
    s = sample(value=.9)
    target, receipt = hil_targets.right_command_at_step_start(s, (False, True))
    np.testing.assert_array_equal(target, np.asarray([.03]*6+[.004], np.float32))
    assert receipt['ros_stamp'] == 99.99
    target[0] = 9.
    assert s.io_evidence['coordinator_commands']['right']['target'][0] == .03


@pytest.mark.parametrize('case', ['missing','stale_ros','stale_arrival','future_ros','future_arrival','wrong_topic','wrong_mode','nonfinite','missing_clock','pre_takeover'])
def test_incomplete_or_noncausal_command_never_falls_back_to_feedback(case):
    s = sample(value=.9); e = s.io_evidence; c = e['coordinator_commands']['right']
    if case == 'missing': c['valid'] = False
    elif case == 'stale_ros': c['ros_stamp'] = 99.
    elif case == 'stale_arrival': c['arrival_monotonic'] = 199.
    elif case == 'future_ros': c['ros_stamp'] = 101.
    elif case == 'future_arrival': c['arrival_monotonic'] = 201.
    elif case == 'wrong_topic': c['topic'] = '/rear/joint_right'
    elif case == 'wrong_mode': e['mode'] = 'policy'
    elif case == 'nonfinite': c['target'][0] = float('nan')
    elif case == 'missing_clock': c.pop('arrival_monotonic')
    elif case == 'pre_takeover': e['right_takeover_started_monotonic'] = 199.995
    with pytest.raises(ValueError):
        hil_targets.right_command_at_step_start(s, (False, True))


def test_left_only_takeover_and_bimanual_contract_are_rejected():
    with pytest.raises(ValueError, match='right-arm takeover'):
        hil_targets.right_command_at_step_start(sample(mode='manual:left'), (True, False))
    s = sample(); s.observation['state'] = np.zeros(14, np.float32)
    with pytest.raises(ValueError, match='right-arm 7D'):
        hil_targets.right_command_at_step_start(s, (True, True))


def test_invalid_option_or_missing_sampling_identity_contract_is_rejected(monkeypatch):
    monkeypatch.setenv('COBOT_RLT_HIL_TARGET', 'invalid')
    with pytest.raises(ValueError, match='feedback or coordinator_command'): hil_targets.selected_target()
    monkeypatch.setenv('COBOT_RLT_HIL_TARGET', 'coordinator_command')
    monkeypatch.setenv('COBOT_RLT_HIL_SAMPLING', 'legacy')
    with pytest.raises(ValueError, match='logical20'): hil_targets.selected_target()
    monkeypatch.setenv('COBOT_RLT_HIL_SAMPLING', 'logical20')
    monkeypatch.setenv('COBOT_RLT_RAW_OBSERVATION_CONTRACT', 'legacy')
    with pytest.raises(ValueError, match='trace observation'): hil_targets.selected_target()


def right_env(io):
    from methods.openpi_rlt.plug_v3_yyshadow.right_arm_env import RightArmCobotOnlineEnv
    return RightArmCobotOnlineEnv(io, chunk_exec_horizon=1, control_frequency_hz=20.,
                                 max_episode_steps=None, joint_step_limit=.03,
                                 gripper_step_limit=.004, sleep=lambda _: None)


def plan(obs, *_):
    chunk = np.full((1, 7), .123, np.float32)
    return SimpleNamespace(start_features=SimpleNamespace(z_rl=np.zeros(2048, np.float32),
                                                         proprio=obs['state'], ref_chunk=chunk),
                           action_chunk=chunk, ref_chunk=chunk, source=1, actor_param_version=2500)


def test_sync_hil_uses_start_command_and_release_end_keeps_next_feedback(monkeypatch):
    from methods.openpi_rlt.tests.test_cobot_online_env import FakeTask2IO
    from methods.openpi_rlt.cobot_adapter.trace import EpisodeOutcome
    select_options(monkeypatch)
    before = sample(.1)
    after = sample(.2, mode='policy', outcome=EpisodeOutcome.SUCCESS)
    after.io_evidence['coordinator_commands']['right']['target'] = [9.]*7
    io = FakeTask2IO([sample(0., 'policy'), before, after])
    env = right_env(io)
    _, _, done, info = env.execute_chunk(env.reset(), plan)
    assert done and len(info['step_trace']) == 1 and io.published == []
    r = info['step_trace'][0]
    np.testing.assert_array_equal(r['action'], np.asarray([.03]*6+[.004], np.float32))
    np.testing.assert_array_equal(r['ref_action'], r['action'])
    np.testing.assert_array_equal(r['next_observation']['state'], np.full(7, .2, np.float32))
    assert r['actor_param_version'] == -1 and r['human_controlled']
    assert r['action_semantics'] == 'coordinator_command_at_step_start'
    assert r['hil_command_receipt']['ros_stamp'] == 99.99


@pytest.mark.parametrize('published', [True, False])
def test_sync_policy_to_hil_mid_interval_is_not_fabricated_as_human_command(monkeypatch, published):
    from methods.openpi_rlt.tests.test_cobot_online_env import FakeTask2IO
    select_options(monkeypatch)
    before = sample(0., 'policy'); before.paused = False
    io = FakeTask2IO([before, before, sample(.1)])
    io.publish_policy_action = lambda _: published
    env = right_env(io)
    with pytest.raises(ValueError, match='Policy-to-HIL boundary'):
        env.execute_chunk(env.reset(), plan)
    assert io.raw_steps == [] and not env.replay_commit_allowed()


def test_async_hil_uses_same_start_command_contract(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]/'tests'))
    from test_async_execution import setup
    select_options(monkeypatch)
    engine, env, io, clock = setup(monkeypatch, state_at=lambda t: ('manual:right', True, 'failure' if t >= .05 else None))
    original = io.sample
    def with_evidence():
        s = original(); s.io_evidence = sample().io_evidence
        return s
    io.sample = with_evidence
    try:
        _, _, done, info = env.execute_chunk(io.sample().observation)
        assert done and len(info['step_trace']) == 1 and io.published == []
        r = info['step_trace'][0]
        np.testing.assert_array_equal(r['action'], np.asarray([.03]*6+[.004], np.float32))
        assert r['hil_command_receipt']['ros_stamp'] == 99.99
        assert r['actor_param_version'] == -1
    finally:
        engine.close()


def test_c10_command_targets_reach_native_replay_and_input_receipts(tmp_path, monkeypatch):
    from methods.openpi_rlt.tests.test_cobot_online_env import FakeTask2IO
    from methods.openpi_rlt.tests.test_raw_observation_contract import native_driver, raw_trace
    from methods.openpi_rlt.cobot_adapter.trace import EpisodeOutcome
    from methods.openpi_rlt.cobot_adapter.replay_precision import install_action_precision_patch
    from rlt_online_rl.replay import RLTTransition
    from rlt_online_rl.config import RLTOnlineRLConfig
    from methods.openpi_rlt.plug_v3_yyshadow.right_arm_env import RightArmCobotOnlineEnv
    select_options(monkeypatch)
    monkeypatch.setenv('COBOT_RLT_INPUT_AUDIT', 'strict')
    monkeypatch.setattr(RLTTransition, 'to_numpy', RLTTransition.to_numpy)
    install_action_precision_patch(enabled=True)
    samples = [sample(0., 'policy')]
    targets = []
    for i in range(10):
        before = sample(.001*i)
        target = np.asarray([.030001+i*.00001]*6+[.004001], np.float32)
        before.io_evidence['coordinator_commands']['right']['target'] = target.tolist()
        targets.append(target)
        after = sample(.001*i+.0001, outcome=EpisodeOutcome.SUCCESS if i == 9 else None)
        samples.extend([before, after])
    io = FakeTask2IO(samples)
    io._trace_writer = SimpleNamespace(_root=tmp_path)
    env = RightArmCobotOnlineEnv(io, chunk_exec_horizon=10, control_frequency_hz=20.,
                                max_episode_steps=None, joint_step_limit=.03,
                                gripper_step_limit=.004, sleep=lambda _: None, collection_phase='online')
    initial = env.reset()
    _, _, done, info = env.execute_chunk(initial, plan)
    assert done and len(info['step_trace']) == 10 and io.published == []
    d = native_driver(tmp_path); d._env = env
    d._rl_config = RLTOnlineRLConfig(action_dim=7, proprio_dim=7, z_dim=2048, chunk_len=10)
    def get_features(obs):
        return dict(z_rl=np.full(2048, obs['state'][0], np.float32), proprio=obs['state'],
                    ref_chunk=np.full((10, 7), .123, np.float32))
    d._feature_provider = SimpleNamespace(get_features=get_features)
    raw = raw_trace(initial); raw.chunk_len = 10
    records = d._build_trace_records(info['step_trace'], episode_id=31, start_env_step_id=0,
                                     chunk_success=1, collection_phase='online')
    d._append_raw_chunk(raw, observation_idx=0, trace_records=records, chunk_step_id=0,
                        chunk_source=info['source'], collection_phase='online', done=done,
                        success=1, drop_transition=False, start_features=info['chunk_start_features'],
                        policy_anchor_offsets=info['policy_anchor_offsets'],
                        policy_anchor_features=info['policy_anchor_features'])
    transitions, stats = d._build_episode_replay(raw)
    assert stats['replay_transition_count'] == len(transitions) == 1
    t = transitions[0]
    np.testing.assert_array_equal(t.action_chunk, np.stack(targets))
    np.testing.assert_array_equal(t.to_numpy()['action_chunk'], np.stack(targets))
    assert t.to_numpy()['action_chunk'].dtype == np.float32
    np.testing.assert_array_equal(t.ref_chunk, np.full((10, 7), .123, np.float32))  # VLA anchor, not human commands.
    assert bool(t.done) and bool(t.intervention_flag) and t.rewards[-1] == 1.
    import json
    receipt = json.loads(next((tmp_path/'replay_inputs').glob('*.jsonl')).read_text())
    assert receipt['checks_passed'] and receipt['recorded_actor_versions'] == [-1]


def test_async_sample_preserves_evidence_in_default_feedback_mode(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]/'tests'))
    from test_async_execution import setup
    monkeypatch.setenv('COBOT_RLT_HIL_TARGET', 'feedback')
    engine, env, io, _ = setup(monkeypatch)
    original = io.sample
    evidence = sample().io_evidence
    def with_evidence():
        result = original(); result.io_evidence = evidence
        return result
    io.sample = with_evidence
    try:
        assert engine.sample().io_evidence is evidence
    finally:
        engine.close()


def test_takeover_epoch_tracks_right_side_without_left_mode_reset(monkeypatch):
    import threading
    from methods.openpi_rlt.cobot_adapter import cobot_ros1
    io = object.__new__(cobot_ros1.RosTask2IO)
    io._condition = threading.Condition()
    io._session_application = None
    io._mode = 'policy'
    io._right_takeover_started_monotonic = None
    clock = [100.]
    monkeypatch.setattr(cobot_ros1.time, 'perf_counter', lambda: clock[0])
    io._mode_callback(SimpleNamespace(data='manual:right'))
    assert io._right_takeover_started_monotonic == 100.
    clock[0] = 101.
    io._mode_callback(SimpleNamespace(data='manual:left+right'))
    assert io._right_takeover_started_monotonic == 100.
    io._mode_callback(SimpleNamespace(data='policy'))
    assert io._right_takeover_started_monotonic is None
    clock[0] = 102.
    io._mode_callback(SimpleNamespace(data='manual:right'))
    assert io._right_takeover_started_monotonic == 102.
