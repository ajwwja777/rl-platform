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


def collection_gate(pending,lag,active,max_pending=10,max_lag=2):
    if active:return None
    if pending>=max_pending:return 'backlog'
    if lag>=max_lag:return 'policy_lag'
    return None


class Inbox:
    def __init__(self,root,known_versions,max_pending=10):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.episodes=self.root/'episodes';self.episodes.mkdir(exist_ok=True)
        self.path=self.root/'admitted.json';self.lock=threading.RLock()
        self.known_versions=known_versions;self.max_pending=max_pending
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
                    done=set(self.committed())
                    outstanding=sum(p.is_dir() and p.name not in done for p in self.episodes.iterdir())
                    if outstanding>=self.max_pending:raise ValueError('Remote episode backlog is full')
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
                record=dict(eligible=bool(eligible),actor_version=info['actor_version'],count=info['count'],
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

    def pending(self):
        with self.lock:
            paths=sorted(self.episodes.glob('*/finished.json'),key=lambda p:(json.loads(p.read_text())['finished_at'],p.parent.name))
            return [p.parent.name for p in paths if not (self.receipts/(p.parent.name+'.json')).exists()]

    def ack(self,identity,receipt):
        identity=episode_id(identity)
        path=self.episodes/identity/'finished.json'
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if not receipt.get('received') or receipt.get('episode_uuid')!=identity or receipt.get('sha256')!=digest:
            raise ValueError('Durable upload receipt does not match this episode')
        with self.lock:atomic_json(self.receipts/(identity+'.json'),receipt)
