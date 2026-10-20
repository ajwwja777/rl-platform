"""Durable async admission, independent of the optimizer and its process lock."""
import hashlib
import json
from pathlib import Path
import shutil
import threading
from online_protocol import atomic_json,episode_id
from remote_journal import receive
from episode_journal import read_journal


def rollout_rng(run,identity):
    digest=hashlib.sha256(('expo-async-rollout-v1:'+run+':'+episode_id(identity)).encode()).digest()
    return [int.from_bytes(digest[:4],'little'),int.from_bytes(digest[4:8],'little')]


def validate_pending_limit(value,episodes=1):
    # Zero disables count-based backpressure, not journal validation/disk guards.
    if type(value) is not int or not 0<=value<=50:raise ValueError('Invalid async pending limit')
    if value and value<episodes:raise ValueError('Async bound smaller than episode batch')
    return value


def collection_gate(pending,lag,active,max_pending=10,max_lag=2):
    if active:return None
    if max_pending and pending>=max_pending:return 'backlog'
    if lag>=max_lag:return 'policy_lag'
    return None


class Inbox:
    def __init__(self,root,known_versions,max_pending=10):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.episodes=self.root/'episodes';self.episodes.mkdir(exist_ok=True)
        self.path=self.root/'admitted.json';self.lock=threading.RLock()
        self.known_versions=known_versions;self.max_pending=validate_pending_limit(max_pending)
        self.records=json.loads(self.path.read_text()) if self.path.exists() else {}

    def committed(self):
        path=self.root/'committed.json'
        return json.loads(path.read_text()).get('accepted',[]) if path.exists() else []

    def receipt(self,identity):
        identity=episode_id(identity)
        with self.lock:
            record=self.records.get(identity,{})
            return dict(episode_uuid=identity,received=bool(record),processed=identity in self.committed(),**record)

    def pending(self):
        with self.lock:
            done=set(self.committed())
            return [i for i in self.records if i not in done]

    def backlog(self):
        # Committed partial-batch episodes are received, but their gradient budget
        # has not been consumed. Keep them in the backpressure count.
        with self.lock:
            path=self.root/'committed.json'
            meta=json.loads(path.read_text()) if path.exists() else {}
            count=meta.get('episodes_since_update',0)
            partial=meta.get('replay',[])[-count:] if count else []
            done=set(meta.get('accepted',[]))
            outstanding=[p.name for p in self.episodes.iterdir() if p.is_dir() and p.name not in done]
            return list(dict.fromkeys(meta.get('async_untrained',partial)+outstanding))

    def transport_status(self):
        # Revision and backlog must describe the same admission point.
        with self.lock:
            pending=self.backlog()
            return dict(admission_revision=len(self.records),pending_episode_ids=pending,pending_episodes=len(pending))

    def receive(self,identity,name,payload,digest):
        identity=episode_id(identity)
        with self.lock:
            directory=self.episodes/identity
            if name=='started.json':
                info=json.loads(payload)
                version=info.get('actor_version')
                if type(version) is not int or version not in self.known_versions():
                    raise ValueError('Episode policy was not published by this run')
                if not directory.exists():
                    if self.max_pending and len(self.backlog())>=self.max_pending:raise ValueError('Remote episode backlog is full')
                    if shutil.disk_usage(self.root).free<16*1024**3:raise ValueError('Remote journal disk space low')
            elif not (directory/'started.json').is_file():raise ValueError('Missing transferred start')
            return receive(self.episodes,identity,[],identity,name,payload,digest)

    def admit(self,identity):
        identity=episode_id(identity)
        with self.lock:
            if identity not in self.records:
                if not (self.episodes/identity/'finished.json').is_file():raise ValueError('Episode is not complete')
                info,data=read_journal(self.episodes,identity)
                if info['actor_version'] not in self.known_versions():raise ValueError('Episode policy was not published by this run')
                eligible=data is not None and info.get('training_eligible',True) and info['outcome'] in ('success','failure')
                record=dict(admission_revision=len(self.records)+1,eligible=bool(eligible),actor_version=info['actor_version'],count=info['count'],
                    sha256=hashlib.sha256((self.episodes/identity/'finished.json').read_bytes()).hexdigest())
                updated=dict(self.records);updated[identity]=record
                atomic_json(self.path,updated)
                self.records=updated
            return self.receipt(identity)

    def recover(self):
        # Only a complete, immutable finished file can survive a lost admission ACK.
        done=set(self.committed())
        paths=sorted(self.episodes.glob('*/finished.json'),key=lambda p:(json.loads(p.read_text())['finished_at'],p.parent.name))
        for path in paths:
            if path.parent.name not in done:self.admit(path.parent.name)


class Outbox:
    def __init__(self,episodes):
        self.episodes=Path(episodes);self.receipts=self.episodes.parent/'upload_receipts'
        self.receipts.mkdir(parents=True,exist_ok=True)
        self.lock=threading.RLock()
        self.acknowledged={p.stem:json.loads(p.read_text()) for p in self.receipts.glob('*.json')}

    def unconfirmed(self,revision):
        with self.lock:
            return [i for i,r in self.acknowledged.items() if r.get('admission_revision',float('inf'))>revision]

    def recover_finished(self):
        # Called only after a fresh local Actor has started with no active episode.
        # Final behavior RNG is audit-only in async mode; never invent it or feed it
        # to the learner. A complete, labeled journal remains usable after a crash.
        with self.lock:
            for identity in self.pending():
                path=self.episodes/identity/'finished.json';meta=json.loads(path.read_text())
                if 'rollout_rng' not in meta and not meta.get('actor_receipt_missing_after_restart'):
                    meta['actor_receipt_missing_after_restart']=True;atomic_json(path,meta)

    def pending(self):
        with self.lock:
            paths=sorted(self.episodes.glob('*/finished.json'),key=lambda p:(json.loads(p.read_text())['finished_at'],p.parent.name))
            pending=[]
            for p in paths:
                if (self.receipts/(p.parent.name+'.json')).exists():continue
                meta=json.loads(p.read_text())
                actor_receipt=self.episodes.parent/'actor_receipts'/(p.parent.name+'.json')
                if 'rollout_rng' not in meta and not meta.get('actor_receipt_missing_after_restart') and actor_receipt.is_file():
                    receipt=json.loads(actor_receipt.read_text())
                    if receipt['actor_version']!=meta['actor_version']:raise ValueError('Recovered Actor version mismatch')
                    meta['rollout_rng']=receipt['rollout_rng'];atomic_json(p,meta)
                pending.append(p.parent.name)
            return pending

    def ack(self,identity,receipt):
        identity=episode_id(identity)
        path=self.episodes/identity/'finished.json'
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if not receipt.get('received') or receipt.get('episode_uuid')!=identity or receipt.get('sha256')!=digest:
            raise ValueError('Durable upload receipt does not match this episode')
        with self.lock:
            atomic_json(self.receipts/(identity+'.json'),receipt)
            self.acknowledged[identity]=receipt


def retire_uncommitted_snapshots(root):
    """Keep retryable training from colliding with an unpublished export after crash.

    Published versions are immutable. Only snapshots newer than durable commit and
    absent from the publication registry can be moved out of the served namespace.
    Retain bytes for audit; floating-point replay need not reproduce an orphan bitwise.
    The caller must own the learner run lock and have no active optimizer worker.
    """
    import os,uuid
    root=Path(root);path=root/'committed.json'
    if not path.exists():return []
    version=json.loads(path.read_text())['updates']
    registry=root/'publications.json'
    published=json.loads(registry.read_text()) if registry.exists() else {}
    if any(int(v)>version for v in published):raise ValueError('Published Actor is ahead of durable commit')
    retired=[]
    for path in (root/'actors').glob('*'):
        if not path.is_dir() or not path.name.isdigit() or int(path.name)<=version:continue
        target=root/'orphaned-actors'/(path.name+'-'+str(uuid.uuid4()));target.parent.mkdir(exist_ok=True)
        os.replace(path,target);retired.append(str(target))
        for directory in (path.parent,target.parent):
            fd=os.open(str(directory),os.O_DIRECTORY)
            try:os.fsync(fd)
            finally:os.close(fd)
    return retired
