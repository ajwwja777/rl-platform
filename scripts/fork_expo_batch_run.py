"""Create a separate batch-scheduled continuation of a stopped committed run.

Only immutable checkpoint/episode/Actor files are hardlinked. Metadata is copied.
The original run remains a per-episode rollback point and is never modified.
"""
import argparse,fcntl,hashlib,json,os,shutil,sys,time
from pathlib import Path
from migrate_expo_device_layout import digest

def fork(source,target,previous,current,episodes=5, critic_prior=None):
    source=Path(source).resolve();target=Path(target).resolve();current=Path(current)
    if target.exists() or target==source or source in target.parents:
        raise ValueError('A separate fresh continuation directory is required')
    if type(episodes) is not int or episodes<2:raise ValueError('Batch must contain at least two episodes')
    with (source/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        status=json.loads((source/'status.json').read_text())
        if status.get('phase')!='ready' or status.get('active_episode') or status.get('error'):
            raise ValueError('Source must be fully committed and idle')
        try:os.kill(status['pid'],0)
        except ProcessLookupError:pass
        else:raise ValueError('Stop source Learner before branching')
        meta=json.loads((source/'committed.json').read_text())
        if not meta.get('checkpoint') or status['actor_version']!=meta['updates']:
            raise ValueError('Complete committed checkpoint required')
        if critic_prior is None and meta.get('episodes_since_update',0):raise ValueError('Source already has a partial batch')
        if critic_prior not in (None,'demo50'):raise ValueError('Invalid Critic prior mode')
        if critic_prior and episodes!=contract_episodes(source):raise ValueError('Prior-only continuation must preserve schedule')
        if any(p.parent.name not in meta['accepted'] for p in (source/'episodes').glob('*/finished.json')):
            raise ValueError('Uncommitted episode exists')
        contract=json.loads((source/'contract.json').read_text())
        if (contract.get('recipe'),contract.get('update_layout'),contract.get('learner_only'))!=('native_sync','native_full_batch',True):
            raise ValueError('Only native full-batch remote runs may continue')
        if contract.get('adapter_source_sha256')not in (digest(Path(previous)),digest(Path(previous),True)):
            raise ValueError('Previous source identity mismatch')
        checkpoint=source/'checkpoints'/str(meta['checkpoint'])
        manifest=json.loads((checkpoint/'manifest.json').read_text())
        for record in manifest['records']:
            if record.get('seed'):continue
            h=hashlib.sha256()
            with (checkpoint/record['file']).open('rb') as f:
                for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
            if h.hexdigest()!=record['sha256']:raise ValueError('Checkpoint checksum mismatch')
        staging=target.with_name(target.name+'.preparing')
        if staging.exists():raise ValueError('Inspect incomplete preparation before retrying')
        staging.mkdir(parents=True)
        for directory in ('episodes','checkpoints','actors'):(staging/directory).mkdir()
        for identity in meta['accepted']:
            shutil.copytree(source/'episodes'/identity,staging/'episodes'/identity,copy_function=lambda src,dst: shutil.copy2(src,dst) if Path(src).suffix=='.json' else os.link(src,dst))
        shutil.copytree(checkpoint,staging/'checkpoints'/checkpoint.name,copy_function=os.link)
        actor=source/'actors'/str(meta['updates'])
        if actor.exists():shutil.copytree(actor,staging/'actors'/actor.name,copy_function=os.link)
        sys.path.insert(0,str(current))
        from online_protocol import atomic_json
        before=dict(contract)
        contract.update(adapter_source_sha256=digest(current,True),episodes_per_update=episodes)
        if critic_prior:
            contract.update(critic_prior=critic_prior,prior_sampling_sha256=hashlib.sha256((current/'prior_sampling.py').read_bytes()).hexdigest())
        else:
            meta['episodes_since_update']=0
        atomic_json(staging/'contract.json',contract)
        atomic_json(staging/'committed.json',meta)
        lineage=dict(source=str(source),source_contract=before,checkpoint=meta['checkpoint'],version=meta['updates'],
                     completed_episodes=meta['completed_episodes'],episodes_per_update=episodes,created_at=time.time())
        atomic_json(staging/'continuation.json',lineage)
        os.rename(staging,target)
        return lineage

def contract_episodes(root):
    return json.loads((root/'contract.json').read_text()).get('episodes_per_update',1)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('source','target','previous-adapter','current-adapter'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--episodes',type=int,default=5)
    p.add_argument('--critic-prior',choices=('demo50',))
    a=p.parse_args();print(json.dumps(fork(a.source,a.target,a.previous_adapter,a.current_adapter,a.episodes,a.critic_prior)))
