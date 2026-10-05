import functools
import multiprocessing as mp
import pickle

import numpy as np

from methods.openpi_rlt.cobot_adapter.process_bootstrap import (
    install_spawn_bootstrap, run_initialized_role,
)


def _probe(queue, stage, value=None, journal=None):
    from rlt_online_rl import replay, trainer, inference
    from openpi_client import msgpack_numpy
    from methods.openpi_rlt.tests.test_replay_precision import transition
    raw = transition()
    report = {
        'stage': stage,
        'precision': bool(getattr(replay.RLTTransition.to_numpy, '_cobot_action_float32', False)),
        'batch_audit': bool(getattr(trainer.LearnerService, '_cobot_batch_audit', False)),
        'env_patch': bool(getattr(inference.EnvDriver.run_episode, '_cobot_session_finalize', False)),
    }
    if stage == 'producer':
        wire = msgpack_numpy.Packer().pack(raw.to_numpy())
        report['wire'] = wire
    elif stage == 'manager':
        manager = replay.ReplayManager(8, journal_path=journal)
        manager.add_transition(msgpack_numpy.unpackb(value))
        action = manager.sample_batch(1)['action_chunk'][0]
        report.update(dtype=str(action.dtype), exact=bool(np.array_equal(action, raw.action_chunk)))
    elif stage == 'restore':
        manager = replay.ReplayManager(8, journal_path=journal)
        action = manager.sample_batch(1)['action_chunk'][0]
        report.update(dtype=str(action.dtype), exact=bool(np.array_equal(action, raw.action_chunk)))
    queue.put(report)


def _spawn_probe(ctx, config, *args, **kwargs):
    queue = ctx.Queue()
    process = ctx.Process(target=run_initialized_role, args=(_probe, config, queue, *args), kwargs=kwargs)
    process.start()
    result = queue.get(timeout=45)
    process.join(timeout=15)
    if process.is_alive():
        process.terminate()
        process.join()
        raise AssertionError('Offline probe did not exit')
    assert process.exitcode == 0
    queue.close()
    return result


def test_real_spawn_producer_manager_restore_and_audit(tmp_path, monkeypatch):
    # No parent patch installation: exercise three independent fresh processes.
    monkeypatch.setenv('JAX_PLATFORMS', 'cpu')
    monkeypatch.setenv('COBOT_RLT_REPLAY_ACTION_PRECISION', 'float32')
    path = str(tmp_path / 'journal.pkl')
    config = tmp_path / 'config.yaml'
    config.write_text('runtime:\n  replay:\n    journal_path: ' + path + '\n')
    ctx = mp.get_context('spawn')
    producer = _spawn_probe(ctx, str(config), 'producer')
    manager = _spawn_probe(ctx, str(config), 'manager', value=producer['wire'], journal=path)
    restored = _spawn_probe(ctx, str(config), 'restore', journal=path)
    for result in [producer, manager, restored]:
        assert result['precision'] and result['batch_audit'] and result['env_patch']
    for result in [manager, restored]:
        assert result['dtype'] == 'float32' and result['exact']


def test_spawn_legacy_remains_quantized(tmp_path, monkeypatch):
    monkeypatch.setenv('JAX_PLATFORMS', 'cpu')
    monkeypatch.setenv('COBOT_RLT_REPLAY_ACTION_PRECISION', 'legacy')
    ctx = mp.get_context('spawn')
    producer = _spawn_probe(ctx, None, 'producer')
    assert not producer['precision']
    manager = _spawn_probe(ctx, None, 'manager', value=producer['wire'], journal=str(tmp_path / 'old.pkl'))
    assert manager['dtype'] == 'float16' and not manager['exact']


def test_native_spawn_hook_is_importable_and_idempotent():
    from types import SimpleNamespace
    captured = []
    native = SimpleNamespace(_spawn_process=lambda *a, **kw: captured.append((a, kw)))
    install_spawn_bootstrap(native, 'audit.yaml')
    installed = native._spawn_process
    install_spawn_bootstrap(native, 'audit.yaml')
    assert native._spawn_process is installed
    native._spawn_process('replay_manager', _probe, 'queue', 'producer')
    args, kwargs = captured[0]
    worker = pickle.loads(pickle.dumps(args[1]))
    assert isinstance(worker, functools.partial)
    assert worker.func is run_initialized_role
    assert worker.args == (_probe, 'audit.yaml')
    assert args[2:] == ('queue', 'producer') and kwargs == {}
