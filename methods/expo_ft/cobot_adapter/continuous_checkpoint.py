"""Owned host capture before donation; background writers never retain device state.

The optimizer owner calls capture at an update boundary and may donate its agent
again once capture returns. Only one capture may be in flight (caller enforced).
write creates immutable files; the caller commits metadata and advertises a
publication only after write succeeds. It never mutates or prunes an older save.
"""
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
import uuid
import numpy as np
from online_protocol import atomic_json


def capture(agent, updater):
    """Copy all mutable leaves once, sharing exact aliases only in owned host RAM."""
    import jax
    from actor_snapshot import trees
    from device_layout import buffer_identity
    started = time.monotonic()
    frozen = {buffer_identity(x) for x in jax.tree.leaves(
        agent.actor.get_params(agent.actor_train_state).filter(updater.frozen))}
    buffers, aliases, records = {}, {}, []

    def key(value):
        return ('device', buffer_identity(value)) if isinstance(value, jax.Array) else ('host', id(value))

    for section, tree in (('agent', agent.replace(_infer_cache=None)), ('target', updater.target)):
        for index, (path, value) in enumerate(jax.tree_util.tree_flatten_with_path(tree)[0]):
            if section == 'agent' and not hasattr(value, 'dtype'):
                value = jax.numpy.asarray(value)
            record = dict(section=section, index=index, path=jax.tree_util.keystr(path),
                          shape=list(np.shape(value)), dtype=str(value.dtype) if hasattr(value, 'dtype') else str(np.asarray(value).dtype))
            if isinstance(value, jax.Array) and buffer_identity(value) in frozen:
                record['seed'] = True
            else:
                identity = key(value)
                if identity not in aliases:
                    owned = np.array(jax.device_get(value), copy=True, order='C')
                    owned.setflags(write=False)
                    name = 'array-%d' % len(buffers)
                    buffers[name] = owned
                    aliases[identity] = name
                record['buffer'] = aliases[identity]
            records.append(record)

    actor_records = []
    for section, tree in trees(agent).items():
        for index, (path, value) in enumerate(jax.tree_util.tree_flatten_with_path(tree)[0]):
            identity = key(value)
            if identity not in aliases:
                raise ValueError('Inference parameter is absent from owned checkpoint capture: ' + section + jax.tree_util.keystr(path))
            name = aliases[identity]
            owned = buffers[name]
            if owned.shape != value.shape or str(owned.dtype) != str(value.dtype):
                raise ValueError('Inference parameter differs from checkpoint capture')
            actor_records.append(dict(section=section, index=index, path=jax.tree_util.keystr(path),
                                      shape=list(value.shape), dtype=str(value.dtype), buffer=name))
    return dict(format=1, checkpoint_records=records, actor_records=actor_records,
                buffers=buffers, actor_step=int(np.asarray(agent.actor_train_state.step)),
                host_bytes=sum(value.nbytes for value in buffers.values()),
                capture_seconds=time.monotonic()-started)


def _validate(captured):
    if captured.get('format') != 1:
        raise ValueError('Unsupported host capture format')
    buffers = captured['buffers']
    for value in buffers.values():
        if not isinstance(value, np.ndarray) or not value.flags.owndata or value.flags.writeable:
            raise ValueError('Host capture requires owned read-only NumPy buffers')
    for group in ('checkpoint_records', 'actor_records'):
        seen = set()
        for record in captured[group]:
            identity = (record['section'], record['index'])
            if identity in seen:
                raise ValueError('Duplicate host capture leaf')
            seen.add(identity)
            if record.get('seed'):
                if group != 'checkpoint_records' or 'buffer' in record:
                    raise ValueError('Invalid seed capture leaf')
                continue
            value = buffers[record['buffer']]
            if list(value.shape) != record['shape'] or str(value.dtype) != record['dtype']:
                raise ValueError('Host capture leaf shape/dtype mismatch')
    if sum(value.nbytes for value in buffers.values()) != captured['host_bytes']:
        raise ValueError('Host capture byte count mismatch')


def _chunks(value):
    flat = value.reshape(-1)
    for start in range(0, flat.size, 1024*1024):
        yield flat[start:start+1024*1024].tobytes()


def _digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_sync(path):
    fd = os.open(str(path), os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _complete(temporary, destination, manifest):
    atomic_json(temporary/'manifest.json', manifest)
    temporary.rename(destination)
    _directory_sync(destination.parent)


def _write_checkpoint(root, step, captured):
    root.mkdir(parents=True, exist_ok=True)
    destination = root/str(step)
    if destination.exists():
        raise FileExistsError('Checkpoint step already exists: '+str(step))
    temporary = root/('.pending-'+str(uuid.uuid4()))
    temporary.mkdir()
    records = []
    for source in captured['checkpoint_records']:
        record = {k:v for k,v in source.items() if k != 'buffer'}
        if not record.get('seed'):
            name = '%s-%d.bin' % (record['section'], record['index'])
            digest = hashlib.sha256()
            with (temporary/name).open('xb') as stream:
                for chunk in _chunks(captured['buffers'][source['buffer']]):
                    stream.write(chunk)
                    digest.update(chunk)
                stream.flush()
                os.fsync(stream.fileno())
            record.update(file=name, sha256=digest.hexdigest())
        records.append(record)
    _complete(temporary, destination, dict(format=1, step=step, records=records))


def _write_actor(root, version, identity, captured):
    root.mkdir(parents=True, exist_ok=True)
    destination = root/str(version)
    existing = json.loads((destination/'manifest.json').read_text()) if destination.exists() else None
    if existing is not None and (existing.get('format'), existing.get('actor_version'), existing.get('actor_step'), existing.get('identity')) != (1, version, captured['actor_step'], identity):
        raise ValueError('Existing Actor identity differs from captured state')
    temporary = None if existing is not None else root/('.pending-'+str(uuid.uuid4()))
    if temporary is not None:
        temporary.mkdir()
    records = []
    for source in captured['actor_records']:
        record = {k:v for k,v in source.items() if k != 'buffer'}
        record['seed'] = version == 0 and record['section'] == 'actor'
        name = '%s-%d.gz' % (record['section'], record['index'])
        digest = hashlib.sha256()
        stream = gzip.open(temporary/name, 'wb', compresslevel=0) if temporary is not None and not record['seed'] else None
        try:
            for chunk in _chunks(captured['buffers'][source['buffer']]):
                digest.update(chunk)
                if stream is not None:
                    stream.write(chunk)
        finally:
            if stream is not None:
                stream.close()
        record['raw_sha256'] = digest.hexdigest()
        if existing is not None:
            if len(existing['records']) != len(captured['actor_records']):
                raise ValueError('Existing Actor tree differs from captured state')
            old = existing['records'][len(records)]
            if any(old.get(k) != v for k,v in record.items()):
                raise ValueError('Existing Actor differs from captured state')
            if not record['seed']:
                if old.get('file') != name or (destination/name).stat().st_size != old['bytes'] or _digest(destination/name) != old['sha256']:
                    raise ValueError('Existing Actor checksum differs')
            record = old
        elif not record['seed']:
            with (temporary/name).open('rb') as handle:
                os.fsync(handle.fileno())
            record.update(file=name, bytes=(temporary/name).stat().st_size, sha256=_digest(temporary/name))
        records.append(record)
    manifest = dict(format=1, actor_version=version, actor_step=captured['actor_step'], identity=identity, records=records)
    if temporary is not None:
        _complete(temporary, destination, manifest)
    return manifest


def write(root, step, version, identity, captured, publish=False):
    """Serialize owned data only. Caller commits/advertises after successful return."""
    if type(step) is not int or step < 0 or type(version) is not int or version < 0:
        raise ValueError('Invalid checkpoint step or Actor version')
    _validate(captured)
    root = Path(root)
    started = time.monotonic()
    _write_checkpoint(root/'checkpoints', step, captured)
    checkpoint_seconds = time.monotonic()-started
    actor_started = time.monotonic()
    manifest = _write_actor(root/'actors', version, identity, captured) if publish else None
    return dict(checkpoint=step, actor_manifest=manifest, host_bytes=captured['host_bytes'],
                timings=dict(checkpoint_write_seconds=checkpoint_seconds,
                             actor_write_seconds=time.monotonic()-actor_started,
                             write_seconds=time.monotonic()-started))
