"""Explicit namespace import for independent collection journals; never edit sources.

Episode IDs are local to their journal. A reviewed manifest orders incoming
Episodes and gives them fresh IDs above the base journal and online ID floor.
Both original namespace/ID and TRAIN/DEV partition remain in the sidecar.
"""
from collections import Counter
import copy


def remap_collection(base_rows, incoming_rows, manifest, *, floor=10000):
    def key(r):
        return tuple(int(r[k]) for k in ('collection_phase_id', 'episode_id', 'step_id'))
    for rows in (base_rows, incoming_rows):
        keys = [key(r) for r in rows]
        if len(keys) != len(set(keys)):
            raise ValueError('Duplicate identity inside source journal')
    mappings = {}
    for item in manifest:
        old = int(item['original_native_episode'])
        if old in mappings or item['split'] not in ('TRAIN', 'DEV') or not item['source_namespace']:
            raise ValueError('Ambiguous source identity or split')
        mappings[old] = item
    counts = Counter(int(r['episode_id']) for r in incoming_rows)
    if set(counts) != set(mappings):
        raise ValueError('Manifest must cover every incoming Episode exactly')
    if any(int(r['collection_phase_id']) != 2 for r in incoming_rows):
        raise ValueError('Incoming collection must be Online phase')
    new_ids = [int(x['experimental_episode']) for x in manifest]
    maximum = max([floor - 1] + [int(r['episode_id']) for r in base_rows])
    if len(set(new_ids)) != len(new_ids) or min(new_ids) <= maximum:
        raise ValueError('Imported IDs must be unique and newer than all base IDs')
    for ep, count in counts.items():
        if count != mappings[ep]['transitions']:
            raise ValueError('Source Episode window count changed')
    train, dev = [], []
    for row in incoming_rows:
        item = mappings[int(row['episode_id'])]
        result = copy.deepcopy(row)
        result['episode_id'] = int(item['experimental_episode'])
        (train if item['split'] == 'TRAIN' else dev).append(result)
    keys = [key(r) for r in base_rows + train + dev]
    if len(keys) != len(set(keys)):
        raise ValueError('Merged identity collision')
    return train, dev
