"""Native BatchProcessor and continuous updates; publication is transport-driven."""
import copy,json,time,threading,logging
import numpy as np
from continuous_state import ingest
from continuous_persistence import checkpoint,busy,drain


def sample(learner):
    with learner.replay_lock:
        batch,actor_batch,learner.combine_rng=learner.processor.next_batch(learner.combine_rng)
        return batch,actor_batch


def consume(learner):
    while not learner.stop.is_set():
        try:
            pending=learner.inbox.pending()
            for identity in pending:
                if learner.stop.is_set():break
                ingest(learner,identity)
            if not pending:learner.stop.wait(.1)
        except Exception as exc:
            learner.error=str(exc);learner.phase='fault';learner.stop.set();learner.save_status()
            logging.exception('Continuous Replay insertion failed');return


def work(learner):
    try:
        import jax
        from expo_ft.data.batch_processor import BatchProcessor
        from native_online import replay_buffer
        learner.combine_rng=jax.random.PRNGKey(142)
        if 'combine_rng' in learner.meta:learner.combine_rng=jax.numpy.asarray(learner.meta['combine_rng'],dtype=jax.numpy.uint32)
        learner.processor=BatchProcessor(replay_buffer=learner.buffer,
            offline_replay_buffer=replay_buffer(learner.config,learner.prompt,1),
            data_sharding=learner.agent.data_sharding,batch_size=64,utd_ratio=20,offline_ratio=0.,
            actor_success_only=learner.agent.actor_success_only,use_dagger_hil_sampling=False,dataset=None)
        learner.activity.episode(time.time())
        learner.consumer=threading.Thread(target=consume,args=(learner,),name='expo-replay-insert',daemon=True);learner.consumer.start()
        while not learner.stop.is_set():
            now=time.time()
            with learner.replay_lock:
                reason=learner.activity.reason(len(learner.buffer),now)
                if not learner.resume_learning:reason='waiting_episode'
            if reason:
                learner.phase=reason
                if not busy(learner) and learner.meta['updates']>learner.committed_updates:checkpoint(learner,publish=learner.publish_requested.is_set())
                elif not busy(learner) and learner.publish_requested.is_set() and learner.meta['updates']>learner.published_version:checkpoint(learner,publish=True)
                if learner.published_version>=learner.meta['updates']:learner.publish_requested.clear()
                learner.save_status();learner.wake.wait(.25);learner.wake.clear();continue
            if learner.stop.is_set() or learner.error is not None:break
            learner.phase='sampling';learner.save_status()
            started=time.monotonic();batch,actor_batch=sample(learner)
            learner.timings['sampling_seconds']=time.monotonic()-started
            learner.phase='updating';learner.save_status()
            started=time.monotonic()
            learner.agent=learner.agent.replace(train_base_actor=actor_batch is not None)
            learner.agent,metrics=learner.updater.update(learner.agent,batch,20,actor_batch)
            jax.block_until_ready(learner.agent)
            learner.metrics={k:float(np.asarray(v)) for k,v in metrics.items() if np.asarray(v).size==1}
            if not all(np.isfinite(v) for v in learner.metrics.values()):raise ValueError('Non-finite native learner metrics')
            if learner.error is not None:
                learner.phase='fault'
                break
            with learner.replay_lock:
                learner.meta['updates']+=1;learner.meta['critic_updates']=int(learner.agent.critic.step)
                learner.meta['actor_updates']=int(learner.agent.actor_train_state.step)
            learner.timings['update_seconds']=time.monotonic()-started;learner.last_update_at=time.time()
            requested=learner.publish_requested.is_set()
            if not busy(learner) and (requested or time.time()-learner.last_checkpoint_at>=learner.args.checkpoint_interval_secs):
                checkpoint(learner,publish=requested)
            learner.save_status()
    except Exception as exc:
        learner.error=str(exc);learner.phase='fault';learner.stop.set();logging.exception('Continuous learner stopped')
    finally:
        try:
            drain(learner)
        except Exception as exc:
            learner.error=str(exc);learner.phase='fault';learner.stop.set()
            logging.exception('Continuous final checkpoint failed')
        learner.save_status()
