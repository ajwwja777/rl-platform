import pickle
import numpy as np
from methods.openpi_rlt.cobot_adapter.replay_precision import install_action_precision_patch
from rlt_online_rl.replay import RLTTransition, ReplayBuffer, ReplayManager, ReplayClient
from openpi_client import msgpack_numpy


def transition():
    action = np.full((10, 7), -1.4029337167739868, np.float32)
    return RLTTransition(z_rl=np.zeros(2), proprio=action[0], ref_chunk=action,
        action_chunk=action, rewards=np.zeros(10), done=False, next_z_rl=np.zeros(2),
        next_proprio=action[0], next_ref_chunk=action, source=2,
        source_chunk=np.full(10, 2), collection_phase='online', success=0,
        intervention_flag=True, episode_id=1, step_id=0)


def test_precision_survives_rpc_manager_buffer_and_restore(tmp_path, monkeypatch):
    original = RLTTransition.to_numpy
    monkeypatch.setattr(RLTTransition, 'to_numpy', original)
    assert not install_action_precision_patch(enabled=False)
    raw = transition()
    assert raw.to_numpy()['action_chunk'].dtype == np.float16
    assert install_action_precision_patch(enabled=True)
    assert install_action_precision_patch(enabled=True)  # Idempotent.
    captured = []
    client = ReplayClient('http://unused')
    monkeypatch.setattr(client, '_post', lambda path, body: captured.append(body))
    client.add_transitions([raw])
    wire = msgpack_numpy.unpackb(msgpack_numpy.Packer().pack(captured[0]))['transitions'][0]
    assert wire['action_chunk'].dtype == np.float32
    assert wire['ref_chunk'].dtype == np.float16
    path = str(tmp_path/'journal.pkl')
    manager = ReplayManager(8, journal_path=path)
    manager.add_transitions([RLTTransition.from_mapping(wire)])
    first = manager.sample_batch(1)
    np.testing.assert_array_equal(first['action_chunk'][0], raw.action_chunk)
    restored = ReplayManager(8, journal_path=path)
    second = restored.sample_batch(1)
    np.testing.assert_array_equal(second['action_chunk'], first['action_chunk'])
    assert second['action_chunk'].dtype == np.float32


def test_restored_legacy_action_is_not_claimed_recovered(tmp_path, monkeypatch):
    original = RLTTransition.to_numpy
    monkeypatch.setattr(RLTTransition, 'to_numpy', original)
    raw = transition()
    old = raw.to_journal_record()
    assert old['action_chunk'].dtype == np.float16
    assert install_action_precision_patch(enabled=True)
    buffer = ReplayBuffer(8)
    buffer.add(old)
    result = buffer.sample(1)['action_chunk'][0]
    np.testing.assert_array_equal(result, old['action_chunk'].astype(np.float32))
    assert not np.array_equal(result, raw.action_chunk)
