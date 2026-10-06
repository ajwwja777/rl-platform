#!/usr/bin/env python3
"""Fork only a fresh isolated runtime from the immutable offline candidate.

Does not start roles, load Stage1 or publish an Actor. Never overwrites a run.
"""
import argparse,hashlib,json,shutil
from pathlib import Path
import yaml

FILES=['profile.json','teacher.pkl','action_norm_stats.json','provenance.json','checkpoints/latest.pkl','actor_snapshot/actor_snapshot.pkl','replay/replay_journal.pkl']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def fork(source,target):
    source=Path(source).resolve();target=Path(target).resolve()
    if target.exists()or'candidates'not in target.parts or source==target or source in target.parents or target in source.parents:raise ValueError('Fresh isolated candidate runtime required')
    cfg=yaml.safe_load((source/'online.yaml').read_text())
    original_root=Path(cfg['runtime']['actor_service']['snapshot_path']).parents[1]
    def rebase(value):
        if isinstance(value,dict):return {k:rebase(v)for k,v in value.items()}
        if isinstance(value,list):return [rebase(v)for v in value]
        if isinstance(value,str)and(value==str(original_root)or value.startswith(str(original_root)+'/')):return str(target)+value[len(str(original_root)):]
        return value
    cfg=rebase(cfg);profile=json.loads((source/'profile.json').read_text())
    if profile['publication_policy']!='staged':raise ValueError('Staged publication required')
    if cfg['runtime']['actor_service']['snapshot_path']==cfg['runtime']['learner_service']['actor_snapshot_path']:raise ValueError('Served/pending Actor must differ')
    receipts={name:sha(source/name)for name in FILES}
    for name in ['teacher','normalization']:
        item=profile[name]
        if receipts[item['file']]!=item['sha256']:raise ValueError('Source candidate asset changed')
    target.mkdir(parents=True)
    for name in FILES:
        dst=target/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/name,dst)
        if sha(dst)!=receipts[name]:raise ValueError('Copied candidate identity mismatch')
    (target/'online.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    receipt=dict(source=str(source),target=str(target),copied_sha256=receipts,online_config_sha256=sha(target/'online.yaml'),publication_policy='staged',services_started=False,model_loaded=False)
    (target/'fork_receipt.json').write_text(json.dumps(receipt,indent=2));return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--target',type=Path,required=True);a=p.parse_args();print(json.dumps(fork(a.source,a.target)))
