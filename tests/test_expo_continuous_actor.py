"""CPU-only immutable doubles exercise serving concurrency and RNG transactions."""
from dataclasses import dataclass, replace
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'methods/expo_ft/cobot_adapter'))
import continuous_actor as module

EP = 'a4444444-4444-4444-8444-444444444444'
EP2 = 'b4444444-4444-4444-8444-444444444444'

@dataclass(frozen=True)
class Agent:
    rng: object
    version: int = 0
    edit_scale: float = .01
    _infer_cache: object = None
    def replace(self, **kw):
        return replace(self, **kw)

@pytest.fixture
def actor(tmp_path, monkeypatch):
    def initial(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.rtc_delay = 5
        self.active = None
        self.phase = 'ready'
        self.error = None
        self.meta = {'updates': 0}
        self.identity = {'test': True}
        self.agent = Agent(np.array([0, 42], dtype=np.uint32))
    def infer(self, route, data):
        if str(data['episode_uuid'].item()) != self.active:
            raise ValueError('active online episode')
        self.agent = self.agent.replace(rng=self.agent.rng + (1 if route == '/precache' else 10))
        if route == '/precache':
            return dict(precached=[self.agent.version], delay=0, actor_version=self.meta['updates'])
        assert int(data['actor_version']) == self.meta['updates']
        return dict(actions=[self.agent.version], actor_version=self.meta['updates'])
    monkeypatch.setattr(module.Actor, '__init__', initial)
    monkeypatch.setattr(module, 'configure_native', lambda: None)
    monkeypatch.setattr(module.Actor, 'infer', infer)
    monkeypatch.setattr(module, 'wait_ready', lambda agent: None)
    value = module.ContinuousActor(SimpleNamespace(snapshot_root=tmp_path/'actors', model_id='test'))
    value.args.snapshot_root.mkdir()
    return value


def begin(actor, identity=EP):
    return actor.begin(dict(episode_uuid=identity, actor_version=actor.meta['updates'], rollout_rng=[999, 999]))


def data(identity=EP):
    return dict(episode_uuid=np.array(identity), prefix=np.zeros((0, 7)))


def select(actor, cached, identity=EP):
    return actor.infer('/select', {**data(identity), **cached})


def snapshot(actor, version):
    path = actor.args.snapshot_root/str(version)
    path.mkdir()
    (path/'manifest.json').write_text(json.dumps({'actor_version': version}))
    return path


def test_persistent_rng_and_canceled_precache(actor):
    begin(actor)
    cached = actor.infer('/precache', data())
    assert actor.agent.rng.tolist() == [0, 42]
    assert cached['inference_token']
    receipt = actor.finish({'episode_uuid': EP})
    assert receipt['rollout_rng'] == [0, 42]
    assert actor.finish({'episode_uuid': EP}) == receipt
    begin(actor, EP2)
    assert actor.agent.rng.tolist() == [0, 42]
    with pytest.raises(ValueError, match='token'):
        select(actor, cached, EP2)
    select(actor, actor.infer('/precache', data(EP2)), EP2)
    assert actor.agent.rng.tolist() == [11, 53]


def test_restore_does_not_block_inference_and_adoption_keeps_live_rng(actor, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    def restore(path, template, identity):
        entered.set()
        assert release.wait(3)
        return template.replace(version=2), {'actor_version': 2}
    monkeypatch.setattr(module, 'restore', restore)
    begin(actor)
    actor.stage(snapshot(actor, 2))
    assert entered.wait(1)
    assert actor.lock.acquire(timeout=.1)
    try:
        select(actor, actor.infer('/precache', data()))
    finally:
        actor.lock.release()
    old = actor.infer('/precache', data())
    assert actor.agent.rng.tolist() == [11, 53]
    release.set()
    actor.stage_thread.join(2)
    assert actor.status()['stage_phase'] == 'ready'
    response = select(actor, old)
    assert response['actions'] == [2]
    assert response['actor_version'] == 2
    assert response['precache_actor_version'] == 0
    assert actor.agent.rng.tolist() == [22, 64]
    assert actor.active == EP and actor.phase == 'rollout'
    receipt = actor.finish({'episode_uuid': EP})
    assert receipt['actor_version'] == 0 and receipt['last_actor_version'] == 2
    assert receipt['policy_versions'] == [0, 2]


def test_failed_stage_keeps_serving_and_is_retryable(actor, monkeypatch):
    def restore(*args):
        raise ValueError('checksum broken')
    monkeypatch.setattr(module, 'restore', restore)
    begin(actor)
    directory = snapshot(actor, 1)
    actor.stage(directory)
    actor.stage_thread.join(2)
    status = actor.status()
    assert status['ready'] and status['stage_phase'] == 'stage_failed'
    assert 'checksum broken' in status['stage_error']
    assert select(actor, actor.infer('/precache', data()))['actor_version'] == 0
    monkeypatch.setattr(module, 'restore', lambda *args: (args[1].replace(version=1), {'actor_version': 1}))
    actor.stage(directory)
    actor.stage_thread.join(2)
    actor.infer('/precache', data())
    assert actor.meta['updates'] == 1 and actor.error is None


def test_pending_tokens_bounded_and_never_reused(actor):
    begin(actor)
    expired = actor.infer('/precache', data())
    for _ in range(actor.max_pending_tokens):
        latest = actor.infer('/precache', data())
    with pytest.raises(ValueError, match='token'):
        select(actor, expired)
    select(actor, latest)
    with pytest.raises(ValueError, match='token'):
        select(actor, latest)
    with pytest.raises(ValueError, match='token'):
        actor.infer('/select', data())
    assert len(actor.pending_tokens) == 0


def test_failed_inference_does_not_commit_rng(actor, monkeypatch):
    begin(actor)
    cached = actor.infer('/precache', data())
    def broken(self, route, data):
        self.agent = self.agent.replace(rng=self.agent.rng + 99)
        raise ValueError('inference failed')
    monkeypatch.setattr(module.Actor, 'infer', broken)
    with pytest.raises(ValueError, match='inference failed'):
        select(actor, cached)
    assert actor.agent.rng.tolist() == [0, 42]


def test_adoption_without_selection_preserves_live_rng_and_idle_install_is_exclusive(actor, monkeypatch):
    monkeypatch.setattr(module, 'restore', lambda path, template, identity:
                        (template.replace(version=3, rng=np.array([100, 200])), {'actor_version': 3}))
    begin(actor)
    select(actor, actor.infer('/precache', data()))
    live_rng = actor.agent.rng.copy()
    actor.stage(snapshot(actor, 3))
    actor.stage_thread.join(2)
    with pytest.raises(ValueError, match='staging'):
        actor.install(actor.args.snapshot_root/'3')
    actor.infer('/precache', data())
    np.testing.assert_array_equal(actor.agent.rng, live_rng)
    assert actor.meta['updates'] == 3
    assert actor.status()['stage_phase'] == 'adopted'


def test_stage_rollbacks_and_paths_rejected_without_faulting_actor(actor, monkeypatch):
    with pytest.raises(ValueError, match='outside'):
        actor.stage(actor.args.snapshot_root.parent/'outside')
    actor.meta['updates'] = 2
    actor.stage(snapshot(actor, 1))
    actor.stage_thread.join(2)
    assert actor.status()['stage_phase'] == 'stage_failed'
    assert 'rollback' in actor.status()['stage_error']
    assert actor.phase == 'ready' and actor.meta['updates'] == 2


def test_receipts_survive_reload_and_do_not_close_other_episode(actor):
    begin(actor)
    receipt = actor.finish({'episode_uuid': EP})
    begin(actor, EP2)
    assert actor.finish({'episode_uuid': EP}) == receipt
    assert actor.active == EP2
    actor.finish({'episode_uuid': EP2})
    with pytest.raises(ValueError, match='finished'):
        begin(actor)


def test_old_competing_token_cannot_rewind_committed_rng(actor):
    begin(actor)
    old = actor.infer('/precache', data())
    new = actor.infer('/precache', data())
    select(actor, new)
    rng = actor.agent.rng.copy()
    with pytest.raises(ValueError, match='token'):
        select(actor, old)
    np.testing.assert_array_equal(actor.agent.rng, rng)


def test_native_inference_wrapper_and_wire_token_roundtrip(actor, monkeypatch):
    from monitoring import infer_with_q
    from online_learner import Learner
    from wire import encode, decode
    def wrapped(self, route, payload):
        return infer_with_q(self, Learner.infer, route, payload)
    @dataclass(frozen=True)
    class NativeDouble(Agent):
        N: int = 8
        def sample_pre_cache(self, observation, prefix):
            return np.zeros((self.N, 16, 7)), self.replace(rng=self.rng + 1)
        def sample_actions(self, observation, precached, delay, only_base_actions):
            return np.full((8, 7), self.version), self.replace(rng=self.rng + 10), {}
    monkeypatch.setattr(module.Actor, 'infer', wrapped)
    actor.aligned = True
    actor.agent = NativeDouble(actor.agent.rng)
    begin(actor)
    observation = {**data(), 'state': np.zeros(7),
        'left_wrist_image': np.full((180, 320, 3), 2, dtype=np.uint8),
        'right_wrist_image': np.full((180, 320, 3), 2, dtype=np.uint8)}
    cached = actor.infer('/precache', observation)
    actor.staged = (actor.agent.replace(version=4), {'actor_version': 4})
    body, encoding = encode({**observation, **cached})
    selected = actor.infer('/select', decode(body, encoding))
    assert selected['actor_version'] == 4 and selected['precache_actor_version'] == 0
    assert np.asarray(selected['actions']).shape == (8, 7)
    assert np.asarray(selected['actions']).min() == 4
    assert actor.agent.rng.tolist() == [11, 53]


def test_idle_adopt_and_start_report_actual_version_atomically(actor):
    actor.staged = (actor.agent.replace(version=5), {'actor_version': 5})
    # Native runtime omits a potentially stale version and never supplies RNG.
    status = actor.begin({'episode_uuid': EP})
    assert status['actor_version'] == status['starting_actor_version'] == 5
    assert actor.agent.rng.tolist() == [0, 42]
    actor.staged = (actor.agent.replace(version=6), {'actor_version': 6})
    with pytest.raises(ValueError, match='no active episode'):
        actor.adopt()
    actor.finish({'episode_uuid': EP})
    assert actor.adopt()['actor_version'] == 6
    assert actor.active is None
