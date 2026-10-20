"""Cobot transport workers; only idle-boundary installation touches Actor GPU."""
import hashlib
import json
import subprocess
import threading
import time
from urllib.request import Request
from online_protocol import atomic_json
from remote_journal import files
from remote_runtime_support import RemoteSupport
from async_state import Outbox,collection_gate,rollout_rng


class AsyncRemote(RemoteSupport):
    def remote_init(self,args):
        super().remote_init(args)
        if not self.remote or self.remote.get('async_mode')!='bounded_async_v1':raise ValueError('Async profile required')
        self.transfer=None
        self.outbox=Outbox(args.run_dir/'episodes')
        self.installing=False;self.install_error=None;self.staged=None
        self.download_state={};self.upload_state={};self.network_error=None
        self.upload_busy=threading.Event()

    def validate_remote_status(self,info):
        super().validate_remote_status(info)
        if info.get('async_mode')!='bounded_async_v1':raise RuntimeError('Remote learner is not asynchronous')
        for key,default in [('async_max_pending',10),('async_max_policy_lag',2)]:
            if info.get(key)!=self.remote.get(key,default):raise RuntimeError('Remote async bounds mismatch')

    def start_server(self):
        super().start_server()
        with self.control_lock:self.outbox.recover_finished()
        for target in (self.upload_loop,self.download_loop):
            threading.Thread(target=target,daemon=True).start()

    def remote_poll(self):
        while not self.stop.wait(.5):
            try:
                with self.rpc.open('http://127.0.0.1:%d/status'%self.remote.get('local_port',18453),timeout=2) as response:value=json.load(response)
                self.validate_remote_status(value)
                monitoring=value.pop('monitoring',None)
                if monitoring is not None:
                    from monitoring import write_json
                    write_json(self.args.run_dir/'telemetry'/'learner.json',monitoring)
                self.remote_status=value;self.remote_seen=time.monotonic();self.network_error=None
            except RuntimeError as exc:
                self.remote_status={'ready':False,'phase':'fault','error':str(exc)};self.fail(str(exc))
            except (OSError,ValueError) as exc:
                self.network_error=str(exc)
                # Existing local episode survives a WAN outage; starts are bounded by the outbox.
                with self.remote_process_lock:
                    if self.remote_process is not None and self.remote_process.poll() is not None and time.monotonic()>=self.reconnect_after:
                        self.remote_process=subprocess.Popen(self.remote_command(),stdin=subprocess.PIPE)
                        self.reconnect_after=time.monotonic()+5

    def remote_online_status(self):
        value=dict(self.remote_status)
        local_pending=self.outbox.pending()
        pending=set(value.get('pending_episode_ids',[]))|set(local_pending)|set(self.outbox.unconfirmed(value.get('admission_revision',0)))
        publications=value.get('publications',{})
        served_generation=publications.get(str(self.local_actor_version),{}).get('generation',0)
        lag=max(0,value.get('published_generation',0)-served_generation)
        reason=collection_gate(len(pending),lag,bool(self.episode),self.remote.get('async_max_pending',10),self.remote.get('async_max_policy_lag',2))
        if self.installing:reason='installing'
        if getattr(self,'install_error',None):reason='install_failed'
        if value.get('phase')=='fault' and not self.episode:reason='learner_fault'
        ready=bool(self.ready and self.local_actor_version>=0 and reason is None)
        value.update(ready=ready,phase='rollout' if self.episode else 'ready' if ready else reason or 'loading',
            learner_phase=self.remote_status.get('phase'),learner_active_episode=value.get('active_episode'),
            active_episode=self.episode,served_actor_version=self.local_actor_version,
            published_actor_version=self.remote_status.get('actor_version',-1),
            downloaded_actor_version=self.staged[1]['actor_version'] if self.staged else self.local_actor_version,
            pending_upload_episodes=len(local_pending),pending_episodes=len(pending),policy_generation_lag=lag,
            collection_waiting_for=reason,download=dict(self.download_state),upload=dict(self.upload_state),
            network_error=self.network_error,install_error=getattr(self,'install_error',None),
            compute_host='Learner %s / Actor Cobot'%self.remote.get('learner_host','HPC'))
        return value

    def json_rpc(self,path,body):
        if path=='/episode/start':
            value=self.remote_online_status()
            if not value['ready']:raise ValueError('Async collection waiting: '+str(value['collection_waiting_for']))
            data=dict(body,actor_version=self.local_actor_version,rollout_rng=rollout_rng(self.args.model_id,body['episode_uuid']))
            return self.local_rpc(path,data)
        if path=='/episode/cancel-start':return self.local_rpc(path,body)
        return super().json_rpc(path,body)

    def transfer_episode(self,identity):
        path=self.journal.root/identity/'finished.json';meta=json.loads(path.read_text())
        if meta.get('actor_receipt_missing_after_restart'):return self.remote_online_status()
        receipt=self.local_rpc('/episode/end',{'episode_uuid':identity})
        if receipt['actor_version']!=meta['actor_version']:raise ValueError('Actor receipt policy mismatch')
        meta['rollout_rng']=receipt['rollout_rng'];atomic_json(path,meta)
        return self.remote_online_status()

    def upload_one(self,identity):
        # The control handler seals the RNG receipt before upload; finished.json alone
        # can briefly exist while /episode/end is still running.
        meta=json.loads((self.outbox.episodes/identity/'finished.json').read_text())
        if 'rollout_rng' not in meta and not meta.get('actor_receipt_missing_after_restart'):return False
        receipt=self.remote_rpc('/async/episode/receipt',{'episode_uuid':identity})
        if not receipt.get('received'):
            paths=files(self.outbox.episodes,identity);started=time.monotonic();total=0
            for index,path in enumerate(paths):
                if self.stop.is_set():return False
                payload=path.read_bytes();digest=hashlib.sha256(payload).hexdigest()
                request=Request('http://127.0.0.1:%d/async/journal/%s/%s'%(self.remote.get('local_port',18453),identity,path.name),data=payload,
                    headers={'Content-Type':'application/octet-stream','X-Content-SHA256':digest})
                with self.rpc.open(request,timeout=15) as response:result=json.load(response)
                if result.get('sha256')!=digest:raise ValueError('Async upload checksum mismatch')
                total+=len(payload)
                self.upload_state=dict(episode_uuid=identity,files_sent=index+1,files_total=len(paths),bytes=total,seconds=time.monotonic()-started)
            receipt=self.remote_rpc('/async/episode/end',{'episode_uuid':identity})
        self.outbox.ack(identity,receipt)
        self.upload_state=dict(self.upload_state,state='received',episode_uuid=identity)
        return True

    def upload_loop(self):
        while not self.stop.wait(.2):
            pending=self.outbox.pending()
            if not pending:continue
            self.upload_busy.set()
            try:self.upload_one(pending[0])
            except (OSError,ValueError) as exc:self.upload_state={'state':'retrying','error':str(exc)};self.stop.wait(2)
            finally:self.upload_busy.clear()

    def download_loop(self):
        while not self.stop.wait(.5):
            if self.staged:
                self.maybe_install()
                continue
            latest=self.remote_status.get('actor_version',-1)
            if latest<=self.local_actor_version or self.outbox.pending():continue
            try:
                started=time.monotonic();self.download_state={'state':'downloading','actor_version':latest}
                directory,manifest=self.download_actor()
                self.staged=(directory,manifest)
                self.download_state=dict(state='staged',actor_version=manifest['actor_version'],seconds=time.monotonic()-started)
                atomic_json(self.args.run_dir/'downloaded.json',dict(directory=str(directory),actor_version=manifest['actor_version']))
            except (OSError,ValueError) as exc:self.download_state={'state':'retrying','error':str(exc)};self.stop.wait(2)

    def maybe_install(self):
        if not self.staged or getattr(self,'install_error',None):return False
        if not self.control_lock.acquire(blocking=False):return False
        try:
            if self.episode or self.journal.active or not self.gate.paused:return False
            if self.future is not None and not self.future.done():return False
            self.installing=True;directory,manifest=self.staged;started=time.monotonic()
            try:
                value=self.local_rpc('/actor/install',{'directory':str(directory)},timeout=600)
                if not value.get('ready') or value.get('actor_version')!=manifest['actor_version']:raise ValueError('Installed Actor identity mismatch')
                self.local_actor_version=value['actor_version'];self.staged=None
                self.download_state=dict(state='installed',actor_version=self.local_actor_version,install_seconds=time.monotonic()-started)
                atomic_json(self.args.run_dir/'installed.json',self.download_state)
                # No in-flight episode references an old local snapshot at this boundary.
                from actor_snapshot import prune
                prune(self.actor_root,self.local_actor_version)
                return True
            except Exception as exc:
                self.install_error=str(exc);self.fail('Async Actor install failed: '+str(exc));return False
            finally:self.installing=False
        finally:self.control_lock.release()
