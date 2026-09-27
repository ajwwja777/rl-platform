"""Versioned deployment selection: immutable checkpoints, atomic accepted pointer."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];RUN=ROOT/'runs/plug_v2'
CONTRACT='plug-v2-online-v4-v010-a090'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def selected():
    m=json.loads((RUN/'learning/v4/current.json').read_text())
    if m.get('learning_algorithm')!='supported_iql_residual_v1' or m.get('execution_contract')!=CONTRACT or m.get('status')!='offline_validated_onsite_pending':
        raise ValueError('v4 release is not validated')
    p=Path(m['path']).resolve(strict=True);p.relative_to((RUN/'learning').resolve())
    if digest(p)!=m['sha256']:raise ValueError('v4 checkpoint hash mismatch')
    return p,m
