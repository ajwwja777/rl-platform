"""Reviewed execution-only EXPO adapter migration while the owner is stopped.

Requires the exact previous adapter files and complete committed state. Never
changes data, update budgets, optimizer/checkpoint contents or Actor identities.
"""
import argparse, fcntl, hashlib, json, os, sys
from pathlib import Path

FILES=('native_online.py','sampling_config.py','memory_update.py','batch_memory.py','split_update.py','stream_sample.py','sync_schedule.py','success_seed.py','online_protocol.py','episode_journal.py','online_learner.py','stream_checkpoint.py','remote_journal.py')

def digest(adapter,new=False):
 h=hashlib.sha256()
 for name in (('device_layout.py',)+FILES if new else FILES):h.update((adapter/name).read_bytes())
 return h.hexdigest()

def migrate(root,previous,current):
 root=Path(root).resolve()
 with (root/'owner.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  status=json.loads((root/'status.json').read_text())
  if status.get('phase')!='ready' or status.get('active_episode') or status.get('error'):
   raise ValueError('Learner must finish and commit before migration')
  try:os.kill(status['pid'],0)
  except ProcessLookupError:pass
  else:raise ValueError('Stop the Learner before migration')
  meta=json.loads((root/'committed.json').read_text())
  if not meta.get('checkpoint'):raise ValueError('A complete checkpoint is required')
  if status['actor_version']!=meta['updates']:raise ValueError('Status is not committed')
  if any(p.parent.name not in meta['accepted'] for p in (root/'episodes').glob('*/finished.json')):
   raise ValueError('Uncommitted finalized episode exists')
  path=root/'contract.json';raw=path.read_bytes();contract=json.loads(raw)
  if (contract.get('recipe'),contract.get('update_layout'),contract.get('learner_only'))!=('native_sync','native_full_batch',True):
   raise ValueError('Only the native remote full-batch Learner is supported')
  before=digest(Path(previous));after=digest(Path(current),True)
  if contract.get('adapter_source_sha256')!=before:raise ValueError('Previous adapter identity mismatch')
  manifest=json.loads((root/'checkpoints'/str(meta['checkpoint'])/'manifest.json').read_text())
  for item in manifest['records']:
   if item.get('seed'):continue
   f=root/'checkpoints'/str(meta['checkpoint'])/item['file'];h=hashlib.sha256()
   with f.open('rb') as stream:
    for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
   if h.hexdigest()!=item['sha256']:raise ValueError('Checkpoint checksum mismatch')
  backup=root/('contract.before-device-layout-'+hashlib.sha256(raw).hexdigest()[:16]+'.json')
  if backup.exists() and backup.read_bytes()!=raw:raise ValueError('Migration backup differs')
  backup.write_bytes(raw)
  contract['adapter_source_sha256']=after
  sys.path.insert(0,str(current))
  from online_protocol import atomic_json
  atomic_json(path,contract)
  return dict(before=before,after=after,backup=str(backup),checkpoint=meta['checkpoint'],updates=meta['updates'])

if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in ('run-dir','previous-adapter','current-adapter'):p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();print(json.dumps(migrate(a.run_dir,a.previous_adapter,a.current_adapter)))
