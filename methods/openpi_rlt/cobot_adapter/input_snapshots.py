"""Opt-in bounded lossless Replay input artifacts; no model or robot operation."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import threading
import uuid
import zipfile

import numpy as np

from . import input_audit

MAX_SNAPSHOTS = 3
MAX_ARRAY_BYTES = 8 * 1024 * 1024
MAX_METADATA_BYTES = 128 * 1024
NATIVE_FIELDS = ('z_rl', 'proprio', 'ref_chunk', 'action_chunk', 'rewards',
                 'next_z_rl', 'next_proprio', 'next_ref_chunk', 'source_chunk')


def settings():
    try:
        count = int(os.environ.get('COBOT_RLT_INPUT_SNAPSHOT_COUNT', '0'))
    except ValueError as error:
        raise ValueError('Input snapshot count must be an integer from 0 to 3') from error
    if not 0 <= count <= MAX_SNAPSHOTS:
        raise ValueError('Input snapshot count must be an integer from 0 to 3')
    if not count:
        return None
    root = Path(os.environ.get('COBOT_RLT_INPUT_SNAPSHOT_ROOT', ''))
    if not root.is_absolute():
        raise ValueError('Input snapshots require an explicit absolute independent output root')
    if input_audit.audit_mode() == 'off':
        raise ValueError('Input snapshots require record or strict input audit')
    if os.environ.get('COBOT_RLT_RAW_OBSERVATION_CONTRACT', 'legacy') != 'trace':
        raise ValueError('Input snapshots require the trace raw observation contract')
    return root, count


def _pack(value, arrays, depth=0):
    if depth > 12:
        raise ValueError('Snapshot input tree is too deep')
    if isinstance(value, (np.ndarray, np.generic)):
        array = np.asarray(value)
        if array.dtype.hasobject or not np.isfinite(array).all():
            raise ValueError('Snapshot arrays must be finite numeric arrays')
        if len(arrays) >= 128:
            raise ValueError('Snapshot input has too many arrays')
        key = f'a{len(arrays)}'
        arrays[key] = array
        if sum(a.nbytes for a in arrays.values()) > MAX_ARRAY_BYTES:
            raise ValueError('Snapshot arrays exceed the 8MiB bound')
        return {'kind': 'scalar_array' if isinstance(value, np.generic) else 'array', 'key': key}
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise ValueError('Snapshot mapping keys must be strings')
        return {'kind': 'dict', 'items': [[k, _pack(v, arrays, depth+1)] for k, v in value.items()]}
    if isinstance(value, (list, tuple)):
        return {'kind': 'tuple' if isinstance(value, tuple) else 'list',
                'items': [_pack(v, arrays, depth+1) for v in value]}
    if value is None or isinstance(value, (str, bool, int, float)):
        return {'kind': 'value', 'value': value}
    raise ValueError(f'Unsupported snapshot input type: {type(value).__name__}')


def _unpack(node, arrays, depth=0):
    if depth > 12:
        raise ValueError('Snapshot tree is too deep')
    kind = node['kind']
    if kind in ('array', 'scalar_array'):
        value = arrays[node['key']]
        return value[()] if kind == 'scalar_array' else value
    if kind == 'dict':
        return {k: _unpack(v, arrays, depth+1) for k, v in node['items']}
    if kind in ('list', 'tuple'):
        items = [_unpack(v, arrays, depth+1) for v in node['items']]
        return tuple(items) if kind == 'tuple' else items
    if kind == 'value':
        return node['value']
    raise ValueError('Unknown snapshot tree type')


def _publish_exclusive(temporary, target):
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    try:
        os.replace(temporary, target)
    except Exception:
        target.unlink()
        raise


class InputSnapshotWriter:
    """At most three artifacts per EnvDriver; capture at Episode Replay build."""
    def __init__(self, root: Path, count: int):
        self.root = Path(root)
        self.limit = count
        self.saved = 0
        self.identifier = uuid.uuid4().hex
        self.lock = threading.Lock()

    def capture(self, raw_episode, raw_indices, transition, receipt):
        with self.lock:
            if self.saved >= self.limit:
                return {'status': 'not_captured', 'reason': 'snapshot_count_limit',
                        'limit': self.limit, 'writer_id': self.identifier}
            first = raw_episode.steps[raw_indices[0]]
            last = raw_episode.steps[raw_indices[-1]]
            payload = {'current_input': raw_episode.observations[first.observation_idx],
                       'next_input': raw_episode.observations[last.next_observation_idx],
                       'native_arrays': {k: getattr(transition, k) for k in NATIVE_FIELDS},
                       'serialized_replay_arrays': transition.to_numpy()}
            arrays = {}
            tree = _pack(payload, arrays)
            manifest = {'schema': 'rlt_lossless_replay_input_v1',
                        'writer_id': self.identifier, 'index': self.saved,
                        'receipt': receipt, 'tree': tree,
                        'array_receipts': {k: input_audit.array_receipt(v) for k, v in arrays.items()},
                        'raw_array_bytes': sum(v.nbytes for v in arrays.values()),
                        'boundary': 'Raw inputs at Replay build and actual returned/native/serialized arrays; not proof of historical GPU input, checkpoint load, execution or optimal actions.'}
            # Validate metadata before creating any artifact, including NaN scalars.
            encoded = json.dumps(manifest, sort_keys=True, allow_nan=False).encode('utf-8')
            if len(encoded) > MAX_METADATA_BYTES - 2048:
                raise ValueError('Snapshot metadata exceeds its bound')
            root = self.root / f'run_{input_audit.RUN_ID}_writer_{self.identifier}'
            root.mkdir(parents=True, exist_ok=True)
            base = root / f'input_{self.saved:02d}'
            archive, metadata = base.with_suffix('.npz'), base.with_suffix('.json')
            temporary = base.with_suffix('.writing')
            metadata_tmp = base.with_suffix('.json.writing')
            linked_archive = False
            try:
                descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(descriptor, 'wb') as stream:
                    np.savez_compressed(stream, **arrays)
                manifest['array_file'] = archive.name
                manifest['array_file_sha256'] = hashlib.sha256(temporary.read_bytes()).hexdigest()
                encoded = json.dumps(manifest, sort_keys=True, allow_nan=False).encode('utf-8')
                descriptor = os.open(metadata_tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(descriptor, 'wb') as stream:
                    stream.write(encoded)
                # Reserve our own destination without overwriting prior evidence,
                # then replace it atomically; no hard-link filesystem requirement.
                _publish_exclusive(temporary, archive)
                linked_archive = True
                _publish_exclusive(metadata_tmp, metadata)
            except Exception:
                if linked_archive:
                    archive.unlink()
                raise
            finally:
                temporary.unlink(missing_ok=True)
                metadata_tmp.unlink(missing_ok=True)
            self.saved += 1
            return {'status': 'captured', 'path': str(metadata),
                    'metadata_sha256': hashlib.sha256(encoded).hexdigest(),
                    'array_file_sha256': manifest['array_file_sha256'],
                    'raw_array_bytes': manifest['raw_array_bytes'],
                    'limit': self.limit, 'writer_id': self.identifier}


def capture_for_driver(driver, raw_episode, indices, transition, receipt, trace_root):
    config = settings()
    if config is None:
        return None
    root, count = config
    root, trace = root.resolve(), Path(trace_root).resolve()
    if root == trace or root in trace.parents or trace in root.parents:
        raise ValueError('Input snapshot root must be independent of the trace output tree')
    writer = getattr(driver, '_cobot_input_snapshot_writer', None)
    if writer is None:
        writer = InputSnapshotWriter(root, count)
        driver._cobot_input_snapshot_writer = writer
    elif writer.root != root or writer.limit != count:
        raise ValueError('Input snapshot configuration changed during this EnvDriver run')
    return writer.capture(raw_episode, indices, transition, receipt)


def load_and_verify(path: Path, *, metadata_sha256=None):
    """Restore the exact input tree without pickle, checking original receipts."""
    path = Path(path)
    if path.stat().st_size > MAX_METADATA_BYTES:
        raise ValueError('Snapshot metadata exceeds its bound')
    encoded = path.read_bytes()
    if metadata_sha256 and hashlib.sha256(encoded).hexdigest() != metadata_sha256:
        raise ValueError('Snapshot metadata fingerprint mismatch')
    manifest = json.loads(encoded)
    if manifest.get('schema') != 'rlt_lossless_replay_input_v1':
        raise ValueError('Unknown snapshot schema')
    archive = path.with_suffix('.npz')
    if manifest['array_file'] != archive.name:
        raise ValueError('Snapshot array filename mismatch')
    if archive.stat().st_size > MAX_ARRAY_BYTES + MAX_METADATA_BYTES:
        raise ValueError('Snapshot array archive exceeds its bound')
    if hashlib.sha256(archive.read_bytes()).hexdigest() != manifest['array_file_sha256']:
        raise ValueError('Snapshot array archive fingerprint mismatch')
    with zipfile.ZipFile(archive) as zipped:
        entries = zipped.infolist()
        if len(entries) > 128 or sum(e.file_size for e in entries) > MAX_ARRAY_BYTES + MAX_METADATA_BYTES:
            raise ValueError('Expanded snapshot exceeds its bound')
    with np.load(archive, allow_pickle=False) as stored:
        arrays = {k: stored[k] for k in stored.files}
    if {k: input_audit.array_receipt(v) for k, v in arrays.items()} != manifest['array_receipts']:
        raise ValueError('Snapshot array receipts mismatch')
    payload = _unpack(manifest['tree'], arrays)
    receipt = manifest['receipt']
    for key in ('current_input', 'next_input'):
        if input_audit.observation_receipt(payload[key]) != receipt[key]:
            raise ValueError('Snapshot original input receipt mismatch')
    actual = {k: input_audit.array_receipt(v) for k, v in payload['serialized_replay_arrays'].items()}
    if actual != receipt['serialized_replay_arrays']:
        raise ValueError('Snapshot serialized Replay receipt mismatch')
    for key, value in payload['native_arrays'].items():
        if input_audit.array_receipt(value) != receipt['training_arrays_before_storage'][key]['native']:
            raise ValueError('Snapshot native feature receipt mismatch')
    return payload, manifest
