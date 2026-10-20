"""Durable sampleable Replay admission, independent from optimizer commits."""
import json,time
from async_state import Inbox
from episode_journal import read_journal
from native_online import insert_episode
from online_protocol import atomic_json

REPLAY_KEYS=('accepted','replay','steps','completed_episodes','last_episode_at','last_replay_insert_at')

class Activity:
    def __init__(self,timeout=120,now=None):
        self.timeout=timeout;self.last=time.time() if now is None else now;self.pending=False
    def episode(self,now):self.pending=True
    def reason(self,size,now):
        if self.pending:self.last=now;self.pending=False
        if size<64:return 'waiting_replay'
        if self.timeout>0 and now-self.last>self.timeout:return 'waiting_episode'
        return None

class ContinuousInbox(Inbox):
    def committed(self):
        path=self.root/'replay-state.json'
        return json.loads(path.read_text()).get('accepted',[]) if path.exists() else super().committed()
    def backlog(self):return self.pending()
    def admit(self,identity):
        path=self.episodes/identity/'finished.json'
        if path.exists():
            info=json.loads(path.read_text());versions=info.get('policy_versions',[info.get('actor_version')])
            if any(type(v) is not int or v not in self.known_versions() for v in versions):
                raise ValueError('Episode contains an unpublished behavior policy')
        return super().admit(identity)

def ingest(learner,identity):
    info,data=read_journal(learner.root/'episodes',identity)
    with learner.replay_lock:
        if identity in learner.meta['accepted']:return False
        eligible=data is not None and info.get('training_eligible',True) and info['outcome'] in ('success','failure')
        if eligible:
            learner.meta['replay'].append(identity)
            sizes={i:json.loads((learner.root/'episodes'/i/'finished.json').read_text())['count'] for i in learner.meta['replay']}
            evicted=False
            while sum(sizes[i] for i in learner.meta['replay'])+learner.seed['frames']>learner.args.capacity:
                learner.meta['replay'].pop(0);evicted=True
            if evicted:learner.rebuild()
            else:insert_episode(learner.buffer,data,terminal_observation=False)
            learner.meta['steps']+=len(data['actions']);learner.meta['completed_episodes']+=1
            now=time.time();learner.meta['last_episode_at']=now;learner.meta['last_replay_insert_at']=now
            learner.activity.episode(now)
            learner.resume_learning=True
        learner.meta['accepted'].append(identity)
        atomic_json(learner.root/'replay-state.json',{k:learner.meta[k] for k in REPLAY_KEYS if k in learner.meta})
    learner.wake.set();learner.save_status();return True
