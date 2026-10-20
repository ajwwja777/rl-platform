"""Native batch recipe with independent behavior RNG and durable async publication.

Training order/losses match online_learner.learn_episode; only admission, behavior
version/RNG, publication retention and timing differ. Signed sync source stays intact.
"""
import json
import time
import numpy as np
from online_protocol import atomic_json
from episode_journal import read_journal
from native_online import insert_episode,valid_batch

def learn_episode(self,identity):
    import jax
    from stream_checkpoint import save
    if identity in self.meta['accepted']:return
    previous_checkpoint=self.meta['checkpoint']
    previous_updates=self.meta['updates']
    info,data=read_journal(self.root/'episodes',identity)
    if info['actor_version'] not in self.known_versions():
        raise ValueError('Episode policy was not published by this run')
    # Behavior RNG is recorded for audit, never assigned to the independent learner.
    accepted=data is not None and info.get('training_eligible',True) and info['outcome'] in ('success','failure')
    if accepted:
        partial=self.meta.get('episodes_since_update',0)
        self.meta.setdefault('async_untrained',self.meta['replay'][-partial:] if partial else [])
        if self.meta['completed_episodes']>=10:self.meta['async_untrained'].append(identity)
        self.meta['replay'].append(identity)
        sizes={x:json.loads((self.root/'episodes'/x/'finished.json').read_text())['count']+(0 if self.aligned else 1) for x in self.meta['replay']}
        evicted=False
        while sum(sizes[x] for x in self.meta['replay'])+(self.seed['frames'] if self.seed else 0)>self.args.capacity:
            self.meta['replay'].pop(0)
            evicted=True
        if self.aligned and not evicted:
            insert_episode(self.buffer,data,terminal_observation=False)
        else:self.rebuild()  # Whole episodes, chronological; never sample across ring wrap.
        if self.aligned:
            from sync_schedule import Schedule
            schedule,count=Schedule(self.meta['completed_episodes'],self.meta['steps'],self.meta['credit'],
                self.meta['episodes_since_update']).finish(len(data['actions']),episodes_per_update=self.episodes_per_update)
            if self.critic_prior=='demo50_success30_failure20':
                from outcome_sampling import deferred_schedule
                schedule,count=deferred_schedule(schedule,count,self.outcome_pools())
            self.meta.update(completed_episodes=schedule.completed_episodes,steps=schedule.env_steps,credit=schedule.credit,
                episodes_since_update=schedule.episodes_since_update)
        else:
            self.meta['steps']+=len(data['actions'])
            self.meta['credit']+=len(data['actions'])
            count=min(self.args.max_updates,self.meta['credit']//self.args.steps_per_update) if len(self.buffer)-len(self.meta['replay'])>=self.args.warmup_steps else 0
        self.pending_updates=count
        self.save_status()
        for _ in range(count):
            started=time.monotonic()
            if self.aligned:
                if self.native_batch_update:
                    if self.critic_prior=='demo50_success30_failure20':
                        from outcome_sampling import sample_critic
                        raw=sample_critic(self.buffer,64*20,self.outcome_pools())
                    elif self.critic_prior=='demo50':
                        from prior_sampling import sample_critic
                        raw=sample_critic(self.buffer,64*20,self.seed['frames'])
                    else:
                        raw=self.buffer.sample_jax(64*20)
                    batch=self.buffer._convert_to_openpi_format(raw)
                    del raw
                else:
                    from stream_sample import Batch
                    batch=Batch(self.buffer,64*20,64)
                raw_actor=self.buffer.sample_jax(64,success_only=True)
                if raw_actor is None:raise ValueError('Confirmed success seed unavailable')
                actor_batch=self.buffer._convert_to_openpi_format(raw_actor)
                del raw_actor
            else:
                batch=valid_batch(self.buffer,self.args.batch_size)
                if batch is None:break
                actor_batch=valid_batch(self.buffer,self.args.batch_size,success_only=True)
            self.timings["sampling_seconds"]=time.monotonic()-started
            started=time.monotonic()
            self.agent=self.agent.replace(train_base_actor=actor_batch is not None)
            self.agent,metrics=self.updater.update(self.agent,batch,20 if self.aligned else 1,actor_batch)
            jax.block_until_ready(self.agent)
            self.metrics={k:float(np.asarray(v)) for k,v in metrics.items() if np.asarray(v).size==1}
            if not all(np.isfinite(v) for v in self.metrics.values()):raise ValueError('Non-finite native learner metrics')
            self.agent=self.agent.replace(train_base_actor=True)
            self.meta['updates']+=1
            if not self.aligned:self.meta['credit']-=self.args.steps_per_update
            self.meta['critic_updates']=int(self.agent.critic.step)
            self.meta['actor_updates']=int(self.agent.actor_train_state.step)
            self.timings["update_seconds"]=time.monotonic()-started
            self.pending_updates-=1
            self.save_status()
        if self.meta['updates'] != previous_updates:
            self.meta['async_untrained']=[]
            if not self.learner_only:self.warm_inference()
            # Use a fresh disk step even after an interrupted uncommitted save.
            disk_step=max([int(p.name) for p in (self.root/'checkpoints').glob('*') if p.name.isdigit()] or [0])+1
            started=time.monotonic()
            save(self.root/'checkpoints',self.agent,self.updater,disk_step)
            self.timings['checkpoint_seconds']=time.monotonic()-started
            self.meta['checkpoint']=disk_step
            started=time.monotonic()
            if self.learner_only:self.publish_actor()
            self.timings['snapshot_seconds']=time.monotonic()-started
    self.meta['accepted'].append(identity)
    self.meta['agent_rng']=np.asarray(self.agent.rng).tolist()
    if hasattr(self.buffer,'rng'):self.meta['replay_rng']=np.asarray(self.buffer.rng).tolist()
    atomic_json(self.meta_path,self.meta)
    self.publish_committed()
    # Async readers may still be downloading an older immutable snapshot. Retain it.
    # Metadata is committed before deleting older complete checkpoints.
    import shutil
    retained={self.meta['checkpoint'],previous_checkpoint}
    checkpoint_root=(self.root/'checkpoints').resolve()
    for old in checkpoint_root.glob('*'):
        if old.name.isdigit() and int(old.name) not in retained and old.resolve().parent==checkpoint_root:
            shutil.rmtree(old)

