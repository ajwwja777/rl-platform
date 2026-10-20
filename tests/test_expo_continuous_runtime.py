import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
from uuid import uuid4
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).parents[1]/'methods/expo_ft/cobot_adapter'))
from continuous_remote import ContinuousOutbox, ContinuousRemote
from continuous_runtime import ContinuousRuntime
from online_protocol import atomic_json


def runtime(tmp_path):
    c = ContinuousRemote()
    c.args = SimpleNamespace(run_dir=tmp_path, model_id='test', port=8043, rtc_delay=5, bundle=tmp_path/'bundle')
    c.remote = dict(async_mode='native_async_v1', critic_prior='seeded', episodes_per_update=1,
                    edit_scale=.01, learner_host='HPC', python='python', native_project='native')
    c.remote_status = dict(ready=True, phase='updating', actor_version=0, pending_episode_ids=[], admission_revision=0)
    c.local_actor_version, c.ready, c.episode = 0, True, None
    c.installing, c.install_error, c.staged = False, None, None
    c.download_state, c.upload_state, c.actor_status = {}, {}, {}
    c.network_error, c.transfer = None, None
    c.stage_retry_after, c.publication_requested_at = 0., 0.
    c.publication_pending = False
    c.outbox = ContinuousOutbox(tmp_path/'episodes')
    c.control_lock, c.transfer_lock = threading.RLock(), threading.Lock()
    c.journal = SimpleNamespace(active=None, root=tmp_path/'episodes')
    c.future = None
    c.actor_root = tmp_path/'actors'
    c.actor_root.mkdir()
    c.stop = threading.Event()
    c.failures = []
    c.fail = c.failures.append
    return c


def test_no_backlog_lag_download_or_stage_failure_gate(tmp_path):
    c = runtime(tmp_path)
    c.remote_status.update(pending_episode_ids=[str(uuid4()) for _ in range(100)], published_generation=1000)
    c.install_error = 'temporary staging OOM'
    c.actor_status = dict(stage_phase='stage_failed', stage_error=c.install_error)
    c.download_state = {'state': 'downloading'}
    value = c.remote_online_status()
    assert value['ready'] and value['pending_episodes'] == 100
    assert value['collection_waiting_for'] is None
    c.remote_status['phase'] = 'fault'
    assert not c.remote_online_status()['ready']


def test_start_is_local_persistent_rng_and_uses_atomic_actor_version(tmp_path):
    c = runtime(tmp_path)
    calls = []
    c.remote_rpc = lambda *a: pytest.fail('Start cannot wait for Learner')
    c.local_rpc = lambda path, body: calls.append((path, body)) or dict(ready=True, actor_version=7)
    identity = str(uuid4())
    result = c.json_rpc('/episode/start', {'episode_uuid': identity})
    assert calls == [('/episode/start', {'episode_uuid': identity})]
    assert result['actor_version'] == c.local_actor_version == 7
    c.json_rpc('/episode/cancel-start', {'episode_uuid': identity})
    assert calls[-1][0] == '/episode/cancel-start'


def test_identity_requires_continuous_native_sampling(tmp_path):
    c = runtime(tmp_path)
    value = dict(deployment_id='test', rtc_delay=5, edit_scale=.01, update_layout='native_continuous',
                 async_mode='native_async_v1', critic_prior='seeded', episodes_per_update=1, training_enabled=True)
    c.validate_remote_status(value)
    for key, bad in [('update_layout', 'native_full_batch'), ('critic_prior', 'demo50'), ('episodes_per_update', 5)]:
        with pytest.raises(RuntimeError):
            c.validate_remote_status({**value, key: bad})
    c.validate_server_status({**value, 'training_enabled': False})


def test_publication_request_coalesces_through_download_and_active_adoption(tmp_path):
    c = runtime(tmp_path)
    requests, operations = [], []
    c.remote_rpc = lambda path, body: requests.append(path) or dict(actor_version=0, publication_pending=True)
    directory = c.actor_root/'2'
    directory.mkdir()
    (directory/'manifest.json').write_text('{}')
    c.download_actor = lambda: (directory, {'actor_version': 2})
    status = {'actor_version': 0, 'stage_phase': 'idle'}
    c.local_status = lambda: dict(status)
    def local(path, body):
        operations.append(path)
        if path == '/actor/stage':
            status.update(stage_phase='loading', stage_directory=str(directory))
        elif path == '/actor/adopt':
            status.update(actor_version=2, stage_phase='adopted')
        return dict(status)
    c.local_rpc = local
    assert not c.download_once()
    assert not c.download_once()
    # A stale false status must not create another publication request.
    c.remote_status['publication_pending'] = False
    assert not c.download_once()
    assert requests == ['/async/actor/request']
    c.remote_status['actor_version'] = 2
    assert not c.download_once()
    assert operations == ['/actor/stage']
    timings = {key: c.download_state[key] for key in ('started_at', 'downloaded_at', 'seconds', 'stage_started_monotonic')}
    assert timings['started_at'] <= timings['downloaded_at'] and timings['seconds'] >= 0
    c.episode = str(uuid4())
    status['stage_phase'] = 'ready'
    assert not c.download_once()
    assert operations == ['/actor/stage'] and len(requests) == 1
    # Next inference adopted the completed snapshot while episode remained active.
    status.update(actor_version=2, stage_phase='adopted')
    assert c.download_once()
    assert c.local_actor_version == 2 and c.staged is None
    assert all(c.download_state[key] == value for key, value in timings.items())
    assert c.download_state['state'] == 'installed'
    assert c.download_state['install_seconds'] >= 0
    assert c.download_state['installed_at'] >= timings['downloaded_at']
    assert json.loads((tmp_path/'installed.json').read_text()) == c.download_state
    assert c.episode and not c.failures


def test_idle_adoption_serializes_with_episode_start(tmp_path):
    c = runtime(tmp_path)
    directory = c.actor_root/'4'
    directory.mkdir()
    c.staged = directory, {'actor_version': 4}
    c.local_status = lambda: dict(actor_version=0, stage_phase='ready', stage_directory=str(directory))
    calls = []
    c.local_rpc = lambda path, body: calls.append(path) or dict(actor_version=4, stage_phase='adopted')
    entered, release = threading.Event(), threading.Event()
    def opening_episode():
        with c.control_lock:
            entered.set()
            assert release.wait(2)
            c.episode = str(uuid4())
    thread = threading.Thread(target=opening_episode)
    thread.start()
    assert entered.wait(1)
    assert not c.maybe_install() and calls == []
    release.set()
    thread.join(2)
    assert not c.maybe_install() and calls == []
    c.episode = None
    assert c.maybe_install() and calls == ['/actor/adopt']


def test_stage_failure_keeps_old_actor_and_retries_without_new_publication(tmp_path):
    c = runtime(tmp_path)
    directory = c.actor_root/'3'
    directory.mkdir()
    c.staged = directory, {'actor_version': 3}
    c.local_status = lambda: dict(actor_version=0, stage_phase='stage_failed', stage_directory=str(directory), stage_error='OOM')
    calls = []
    c.local_rpc = lambda path, body: calls.append(path) or dict(actor_version=0, stage_phase='loading')
    c.remote_rpc = lambda *a: pytest.fail('No publication request while staged')
    c.download_state = dict(started_at=1., downloaded_at=2., seconds=1.)
    assert not c.download_once()
    assert calls == ['/actor/stage'] and c.install_error == 'OOM'
    assert c.download_state['started_at'] == 1. and c.download_state['downloaded_at'] == 2.
    assert c.download_state['seconds'] == 1. and 'stage_started_monotonic' in c.download_state
    assert c.remote_online_status()['ready'] and not c.failures
    assert not c.download_once() and calls == ['/actor/stage']


def test_concurrent_downloader_call_does_not_duplicate_request(tmp_path):
    c = runtime(tmp_path)
    c.remote_rpc = lambda *a: pytest.fail('Concurrent downloader must skip')
    with c.transfer_lock:
        assert not c.download_once()


def finished(c, identity, rng=False):
    value = dict(actor_version=0, finished_at=1)
    if rng:
        value['rollout_rng'] = [1, 2]
    atomic_json(c.journal.root/identity/'finished.json', value)
    return value


@pytest.mark.parametrize('has_rng', [False, True])
def test_recovered_outbox_preserves_mixed_policy_audit_fields(tmp_path, has_rng):
    c = runtime(tmp_path)
    identity = str(uuid4())
    finished(c, identity, has_rng)
    receipt = dict(actor_version=0, last_actor_version=9, policy_versions=[0, 4, 9], rollout_rng=[10, 20])
    atomic_json(tmp_path/'actor_receipts'/(identity+'.json'), receipt)
    c.outbox.recover_finished()
    result = json.loads((c.journal.root/identity/'finished.json').read_text())
    assert all(result[k] == v for k, v in receipt.items())
    assert 'actor_receipt_missing_after_restart' not in result
    # Acknowledged bytes are immutable, even if a later receipt differs.
    atomic_json(c.outbox.receipts/(identity+'.json'), {})
    before = (c.journal.root/identity/'finished.json').read_bytes()
    atomic_json(tmp_path/'actor_receipts'/(identity+'.json'), {**receipt, 'last_actor_version': 99})
    assert c.outbox.pending() == []
    assert (c.journal.root/identity/'finished.json').read_bytes() == before


def test_transfer_seals_all_audit_fields_and_receipt_recovery_rejects_mismatch(tmp_path):
    c = runtime(tmp_path)
    identity = str(uuid4())
    finished(c, identity)
    receipt = dict(actor_version=0, last_actor_version=3, policy_versions=[0, 3], rollout_rng=[5, 6])
    c.local_rpc = lambda *a: receipt
    c.transfer_episode(identity)
    stored = json.loads((c.journal.root/identity/'finished.json').read_text())
    assert all(stored[k] == v for k, v in receipt.items())
    atomic_json(tmp_path/'actor_receipts'/(identity+'.json'), {**receipt, 'actor_version': 2})
    with pytest.raises(ValueError, match='policy mismatch'):
        c.outbox.pending()


def test_runtime_forwards_inference_token_and_updates_local_served_version(tmp_path):
    c = ContinuousRuntime.__new__(ContinuousRuntime)
    c.episode = str(uuid4())
    c.local_actor_version = 0
    c.observation = lambda: {'state': np.zeros(7)}
    c.journal = SimpleNamespace(active={'episode_uuid': c.episode})
    recorded, calls = [], []
    c.monitor_history = SimpleNamespace(selected=lambda active, payload: recorded.append(payload))
    def rpc(path, **values):
        calls.append((path, values))
        if path == '/precache':
            return dict(precached=np.zeros((8, 16, 7)).tolist(), delay=0, actor_version=2, inference_token='opaque')
        return dict(actions=np.ones((8, 7)).tolist(), actor_version=4, precache_actor_version=2)
    c.array_rpc = rpc
    cached = c.infer(c.observation(), [])
    assert c.local_actor_version == 2
    actions = c.resolve_inference(cached)
    assert calls[-1][1]['inference_token'] == 'opaque'
    assert calls[-1][1]['actor_version'] == 2
    assert c.local_actor_version == 4 and actions.shape == (8, 7)
    assert recorded[-1]['precache_actor_version'] == 2


def test_policy_command_uses_continuous_actor_and_preserves_field_options(tmp_path):
    c = runtime(tmp_path)
    c.start_remote = lambda: tmp_path/'actors/0'
    command = c.policy_command()
    assert Path(command[1]).name == 'continuous_actor.py'
    assert command[command.index('--rtc-delay')+1] == '5'
    assert command[command.index('--edit-scale')+1] == '0.01'


def test_q_provenance_survives_live_json_and_sqlite_episode_archive(tmp_path):
    import runpy
    from continuous_monitoring import ContinuousRolloutHistory
    history = ContinuousRolloutHistory(tmp_path)
    active = dict(episode_uuid=str(uuid4()), actor_version=2, started_at=0)
    history.selected(active, dict(actor_version=2, precache_actor_version=2, selected_q=.1))
    history.selected(active, dict(actor_version=5, precache_actor_version=2, selected_q=.2))
    history.flush()
    stored = json.loads((tmp_path/'telemetry/rollout.json').read_text())
    assert stored['actor_version'] == 2 and stored['last_actor_version'] == 5
    assert stored['policy_versions'] == [2, 5]
    assert [(p['precache_actor_version'], p['actor_version']) for p in stored['points']] == [(2, 2), (2, 5)]
    later = {**active, 'episode_uuid': str(uuid4()), 'actor_version': 5}
    history.selected(later, dict(actor_version=7, precache_actor_version=5, selected_q=.3))
    history.flush()
    archive = runpy.run_path(str(Path(__file__).parents[1]/'integrations/cobot_runtime/rollout_archive.py'))
    listing, previous = archive['history'](tmp_path, active['episode_uuid'])
    assert len(listing) == 2
    assert previous['policy_versions'] == [2, 5]
    assert previous['points'][-1]['actor_version'] == 5
    assert previous['points'][-1]['precache_actor_version'] == 2
