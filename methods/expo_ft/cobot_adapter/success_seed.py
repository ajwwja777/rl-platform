"""Verified, immutable offline success cohort for native online Replay."""
import hashlib,json
from pathlib import Path
import numpy as np


def seed_manifest(root):
    root=Path(root)
    value=json.loads((root/'manifest.json').read_text())
    if value['split']!='train' or value['confirmed_outcome']!='success' or not value['confirmation']:
        raise ValueError('Replay seed requires confirmed TRAIN successes')
    if len(value['episodes'])!=25:raise ValueError('Native synchronous recipe requires25seed episodes')
    for e in value['episodes']:
        path=(root/e['file']).resolve()
        if path.parent!=root.resolve():raise ValueError('Seed path escaped root')
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
        if h.hexdigest()!=e['sha256']:raise ValueError('Seed checksum mismatch: '+e['file'])
    return value


def insert_seed(buffer,root,manifest):
    for e in manifest['episodes']:
        with np.load(Path(root)/e['file'],allow_pickle=False) as archive:
            d={k:archive[k] for k in archive.files}
            for i in range(len(d['actions'])):
                done=i==len(d['actions'])-1
                buffer.insert(dict(observations={k:d[k][i] for k in ('state','left_wrist_image','right_wrist_image')},
                    actions=d['actions'][i],rewards=float(done),masks=float(not done),dones=done,
                    is_hil=True,is_success=True))
