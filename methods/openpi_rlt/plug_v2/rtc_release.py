"""Validated immutable RTC upstream candidate selector, separate from revoked v4."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];RUN=ROOT/'runs/plug_v2';CURRENT=RUN/'learning/rtc-v5/current.json'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def selected(path=CURRENT):
    pointer=json.loads(Path(path).read_text());release=Path(pointer['release']).resolve(strict=True)
    release.relative_to((RUN/'learning/rtc-v5/releases').resolve())
    if digest(release)!=pointer['release_sha256']:raise ValueError('release descriptor fingerprint changed')
    m=json.loads(release.read_text())
    if m.get('status')!='offline_validated_onsite_pending' or m.get('learning_algorithm')!='pinned_upstream_rlt_rtc_smdp_v1':raise ValueError('RTC candidate not validated')
    for key in ('checkpoint','norm_stats'):
        p=Path(m[key]).resolve(strict=True);p.relative_to((RUN/'learning').resolve())
        if digest(p)!=m[key+'_sha256']:raise ValueError(key+' changed')
    for name,expected in m.get('code_sha256',{}).items():
        code=(ROOT/name).resolve(strict=True);code.relative_to(ROOT)
        if digest(code)!=expected:raise ValueError('validated runtime code changed: '+name)
    stage_path=ROOT/'deployments/plug_v2/manifest.json'
    if m.get('stage1_manifest_sha256') and digest(stage_path)!=m['stage1_manifest_sha256']:raise ValueError('Stage1 manifest changed')
    stage=json.loads(stage_path.read_text())
    if m['stage1_checkpoint']!=stage['checkpoint']:raise ValueError('Stage1 mismatch')
    return release,m
