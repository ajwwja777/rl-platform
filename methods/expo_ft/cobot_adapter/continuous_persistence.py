"""One owned host snapshot at a time; optimizer never waits for filesystem writes."""
import copy
import logging
import shutil
import threading
import time
import numpy as np
from online_protocol import atomic_json


def busy(learner):
    writer=getattr(learner,'checkpoint_writer',None)
    return writer is not None and writer.is_alive()


def checkpoint(learner,publish=False,wait=False):
    if busy(learner):
        if not wait:return False
        learner.checkpoint_writer.join()
    if learner.error is not None:raise RuntimeError(learner.error)
    from continuous_checkpoint import capture,write
    from continuous_snapshot import contract
    # Optimizer owner captures metadata and all donated leaves before next update.
    with learner.replay_lock:
        meta=copy.deepcopy(learner.meta)
        meta['agent_rng']=np.asarray(learner.agent.rng).tolist()
        meta['replay_rng']=np.asarray(learner.buffer.rng).tolist()
        meta['combine_rng']=np.asarray(learner.combine_rng).tolist()
    previous=meta['checkpoint']
    step=max([int(p.name) for p in (learner.root/'checkpoints').glob('*') if p.name.isdigit()] or [0])+1
    learner.checkpoint_publish=publish
    learner.checkpoint_phase='capturing'
    captured=capture(learner.agent,learner.updater)
    learner.timings.update(checkpoint_capture_seconds=captured['capture_seconds'],checkpoint_host_bytes=captured['host_bytes'])
    identity=contract(learner.args.bundle,learner.args.project,learner.rtc_delay,learner.edit_scale) if publish else None
    meta['checkpoint']=step
    if publish:learner.publish_requested.clear()
    learner.last_checkpoint_at=time.time()
    learner.checkpoint_phase='writing'
    def persist():
        try:
            result=write(learner.root,step,meta['updates'],identity,captured,publish)
            atomic_json(learner.meta_path,meta)
            with learner.replay_lock:
                learner.meta['checkpoint']=step
                learner.committed_updates=meta['updates']
            learner.timings.update(result['timings'])
            learner.timings['checkpoint_seconds']=captured['capture_seconds']+result['timings']['checkpoint_write_seconds']
            if publish:
                learner.timings['snapshot_seconds']=result['timings']['actor_write_seconds']
                learner.publish_committed(meta['updates'])
            for path in (learner.root/'checkpoints').glob('*'):
                if path.is_dir() and path.name.isdigit() and int(path.name) not in (step,previous):shutil.rmtree(path)
            learner.checkpoint_phase='idle'
        except Exception as exc:
            learner.error=str(exc);learner.phase='fault';learner.checkpoint_phase='fault';learner.stop.set()
            logging.exception('Background continuous checkpoint failed')
        finally:
            learner.wake.set();learner.save_status()
    learner.checkpoint_writer=threading.Thread(target=persist,name='expo-checkpoint-write',daemon=True)
    learner.checkpoint_writer.start()
    if wait:
        learner.checkpoint_writer.join()
        if learner.error is not None:raise RuntimeError(learner.error)
    return True


def drain(learner):
    if busy(learner):learner.checkpoint_writer.join()
    if learner.error is not None:return
    pending=learner.publish_requested.is_set() and learner.meta['updates']>learner.published_version
    if learner.error is None and (learner.meta['updates']>learner.committed_updates or pending):
        checkpoint(learner,publish=pending,wait=True)
