"""Durable10Hz observations and actual coordinator commands, including HIL."""
import io
import json
import os
from pathlib import Path
import threading
import time
import numpy as np
from online_protocol import atomic_json, episode_id, validate_episode


class EpisodeJournal:
    def __init__(self, root, max_steps=3000):
        self.root = Path(root)
        self.max_steps = max_steps
        self.lock = threading.RLock()
        self.active = None
        self.recording = False
        self.previous = None
        self.segment = 0
        self.terminal = None

    def begin(self, identity, version):
        identity = episode_id(identity)
        with self.lock:
            if self.active:
                if self.active['episode_uuid'] == identity:
                    return dict(self.active)
                raise ValueError('Another EXPO episode is active')
            directory = self.root / identity
            if directory.exists():
                raise ValueError('Episode identity already exists; do not replay a start')
            directory.mkdir(parents=True)
            self.active = dict(episode_uuid=identity, actor_version=int(version), count=0, started_at=time.time())
            self.previous = None
            self.segment = 0
            self.terminal = None
            self.recording = True
            atomic_json(directory/'started.json',self.active)
            return dict(self.active)

    def pause(self, paused):
        with self.lock:
            self.recording = not paused
            self.previous = None
            self.segment += 1

    def sample(self, observation, action, stamp, is_hil, valid=True):
        with self.lock:
            if not self.active:
                return
            if not self.recording or not valid:
                if self.previous is not None:
                    self.segment += 1
                self.previous = None
                return
            self.terminal = observation
            previous = self.previous
            self.previous = (observation, stamp, bool(is_hil), self.segment)
            if previous is None:
                return
            obs, then, human, segment = previous
            if stamp-then > .16 or stamp-then < .05 or human != bool(is_hil):
                self.segment += 1
                self.previous = (observation,stamp,bool(is_hil),self.segment)
                return
            if self.active['count'] >= self.max_steps:
                raise ValueError('EXPO episode exceeds3000 logical steps; finish the episode')
            action=np.asarray(action,np.float32)
            if action.shape != (7,) or not np.isfinite(action).all():
                raise ValueError('Invalid executed coordinator command')
            directory=self.root/self.active['episode_uuid']
            path=directory/('%06d.npz'%self.active['count'])
            temporary=path.with_suffix('.tmp')
            with temporary.open('wb') as stream:
                np.savez(stream,**obs,actions=action,times=np.float64(then),is_hil=np.bool_(human),segment=np.int32(segment))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary,path)
            self.active['count']+=1

    def finish(self, identity, outcome):
        identity=episode_id(identity)
        if outcome not in ('success','failure','unknown','aborted'):
            raise ValueError('Invalid operator outcome')
        with self.lock:
            path=self.root/identity/'finished.json'
            if path.exists():
                value=json.loads(path.read_text())
                if value['outcome'] != outcome:
                    raise ValueError('Finalized online outcome is immutable')
                return value
            recovered=False
            active=self.active
            if active is None and (path.parent/'started.json').is_file():
                active=json.loads((path.parent/'started.json').read_text())
                active['count']=len(list(path.parent.glob('[0-9][0-9][0-9][0-9][0-9][0-9].npz')))
                recovered=True
            if not active or active['episode_uuid'] != identity:
                raise ValueError('Stale EXPO episode identity')
            value={**active,'outcome':outcome,'finished_at':time.time(),'training_eligible':not recovered}
            if self.terminal is not None and not recovered:
                terminal=path.parent/'terminal.npz'
                with terminal.with_suffix('.tmp').open('wb') as stream:
                    np.savez(stream,**self.terminal);stream.flush();os.fsync(stream.fileno())
                os.replace(terminal.with_suffix('.tmp'),terminal)
            atomic_json(path,value)
            self.active=None
            self.previous=None
            self.recording=False
            return value


def read_journal(root, identity):
    identity=episode_id(identity)
    directory=Path(root)/identity
    meta=json.loads((directory/'finished.json').read_text())
    if meta['episode_uuid'] != identity:
        raise ValueError('Journal identity mismatch')
    if not 0 <= meta['count'] <= 3000:
        raise ValueError('Journal frame count out of range')
    if meta['count']==0:
        return meta,None
    rows=[]
    for i in range(meta['count']):
        with np.load(directory/('%06d.npz'%i),allow_pickle=False) as z:
            rows.append({k:z[k].copy() for k in z.files})
    data={k:np.stack([row[k] for row in rows]) for k in rows[0]}
    segment=data.pop('segment')
    data.update(episode_uuid=np.asarray(identity),actor_version=np.asarray(meta['actor_version']),outcome=np.asarray(meta['outcome']))
    data['dones']=np.r_[segment[:-1]!=segment[1:],True].astype(bool)
    validate_episode(data)
    if (directory/'terminal.npz').is_file():
        with np.load(directory/'terminal.npz',allow_pickle=False) as z:
            data.update({'terminal_'+k:z[k].copy() for k in z.files})
    return meta,data
