"""Cobot-side SSH lifecycle, cached status, and retryable finalized uploads."""
import hashlib
import json
from pathlib import Path
import threading
import subprocess
import socket
import os
import re
import time
from urllib.request import Request
from online_protocol import atomic_json
from remote_journal import files


class RemoteSupport:
    def remote_init(self, args):
        self.remote = json.loads(args.remote_config.read_text()) if getattr(args, "remote_config", None) else None
        if self.remote and (self.remote["model_id"] != args.model_id or self.remote["rtc_delay"] != args.rtc_delay):
            raise ValueError("Remote profile model identity mismatch")
        raw = json.loads(os.environ.get('COBOT_EXECUTION_OPTIONS', '{"enabled":false}'))
        if 'expo' in raw:
            if not raw.get('enabled') or not self.remote or not self.remote.get('allow_online_options'):
                raise ValueError('This runtime does not support configurable EXPO Online options')
            from online_options import configured_run, normalize
            selected = normalize(raw['expo'])
            configured = configured_run({**self.remote, 'run_dir': str(args.run_dir)}, selected)
            args.run_dir = Path(configured.pop('run_dir'))
            args.rtc_delay = selected['rtc_delay']
            self.remote = configured
            self.remote['selected_online_options'] = selected
        self.remote_status = {"ready": False, "phase": "loading"}
        self.remote_seen = 0.
        self.remote_process = None
        self.remote_process_lock = threading.Lock()
        self.reconnect_after = 0.
        self.local_actor_version = -1
        self.actor_root = args.run_dir / "actors"
        self.transfer = None
        self.transfer_lock = threading.Lock()
        self.pending_upload = args.run_dir / "pending_upload.json"
        if self.remote and self.pending_upload.exists():
            self.transfer = json.loads(self.pending_upload.read_text())

    def remote_command(self):
        c = self.remote
        launch=c.get('launch_name')
        if launch is not None and not re.fullmatch(r'[a-z][a-z0-9_]{0,31}',launch):raise ValueError('Invalid configured Learner name')
        command='run' if launch is None else 'run '+launch
        if 'selected_online_options' in c:
            if launch is None: raise ValueError('Configurable Online requires a named Learner profile')
            from online_options import normalize
            selected = normalize(c['selected_online_options'])
            command += ' --rtc-delay %d --edit-scale %.2f' % (selected['rtc_delay'], selected['edit_scale'])
        return ["ssh", "-T", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=3", "-o", "ServerAliveCountMax=2", "-o", "ConnectTimeout=8", "-i", c["identity_file"], "-L", "127.0.0.1:%d:127.0.0.1:%d" % (c.get("local_port",18453), c["remote_port"]), c["destination"], command]

    def validate_identity(self, info, role):
        if info.get("deployment_id") != self.args.model_id or info.get("rtc_delay") != self.args.rtc_delay or info.get("update_layout") != "native_full_batch":
            raise RuntimeError("%s identity mismatch" % role)
        if 'edit_scale' in self.remote and info.get('edit_scale') != self.remote['edit_scale']:
            raise RuntimeError('%s edit scale mismatch' % role)

    def validate_server_status(self, info):
        # Runtime.start_server polls the local inference-only Actor, not the Learner.
        if self.remote:
            self.validate_identity(info, 'Local Actor')
            if info.get('training_enabled') is not False:
                raise RuntimeError('Local Actor role mismatch')

    def validate_remote_status(self, info):
        self.validate_identity(info, 'Remote learner')
        if info.get('training_enabled') is not True:
            raise RuntimeError('Remote learner role mismatch')
        if info.get('episodes_per_update', 1) != self.remote.get('episodes_per_update', 1):
            raise RuntimeError('Remote learner episode batch size mismatch')

    def remote_poll(self):
        while not self.stop.wait(.5):
            try:
                with self.rpc.open("http://127.0.0.1:%d/status" % self.remote.get("local_port",18453), timeout=5 if self.transfer else 1) as response:
                    value = json.load(response)
                self.validate_remote_status(value)
                monitoring = value.pop('monitoring', None)
                revision = (monitoring or {}).get('updated_at')
                if monitoring is not None and revision != getattr(self, '_monitoring_revision', object()):
                    try:
                        from monitoring import write_json
                        write_json(self.args.run_dir / 'telemetry' / 'learner.json', monitoring)
                        self._monitoring_revision = revision
                    except (OSError, ValueError):
                        pass  # Observability must never pause the robot or learner.
                self.remote_status = value
                self.remote_seen = time.monotonic()
            except RuntimeError as exc:
                self.remote_status={"ready":False,"phase":"fault","error":str(exc)}
                self.fail(str(exc))
            except (OSError,ValueError):
                if self.ready and time.monotonic() - self.remote_seen > 3:
                    if self.transfer:
                        self.remote_status={**self.remote_status,"ready":False,"phase":"reconnecting","error":"Waiting for remote transfer connection; local episode retained"}
                        with self.remote_process_lock:
                            if (not self.stop.is_set() and self.remote_process is not None and self.remote_process.poll() is not None and time.monotonic()>=self.reconnect_after):
                                self.remote_process=subprocess.Popen(self.remote_command(),stdin=subprocess.PIPE)
                                self.reconnect_after=time.monotonic()+5
                    else:
                        self.remote_status = {"ready": False, "phase": "fault", "error": "Remote learner connection lost"}
                        self.fail("Remote learner connection lost; episode retained locally")

    def remote_online_status(self):
        value = dict(self.remote_status)
        host=self.remote.get("learner_host", "A6000")
        if value.get('learner_devices') and value.get('visible_gpus'):
            host=re.sub(r'\s*GPU[0-9, ]+$','',host)+' GPU '+value['visible_gpus']
        value["compute_host"] = "Learner %s / Actor Cobot" % host
        value["served_actor_version"] = self.local_actor_version
        if value.get("ready") and value.get("actor_version") != self.local_actor_version:
            value.update(ready=False,phase="publishing")
        if self.transfer and value.get("phase") != "fault":
            value.update(ready=False, phase="publishing" if self.transfer.get("state")=="publishing" else "updating" if self.transfer.get("state")=="waiting_commit" else "transferring", transfer=dict(self.transfer))
        return value

    def transfer_episode(self, identity):
        if self.transfer and self.transfer["episode_uuid"] != identity:
            raise ValueError("Previous episode transfer is pending")
        receipt=self.local_rpc("/episode/end",{"episode_uuid":identity})
        finished=self.journal.root/identity/"finished.json"
        meta=json.loads(finished.read_text())
        if receipt["actor_version"]!=meta["actor_version"]:raise ValueError("Local Actor receipt version mismatch")
        meta["rollout_rng"]=receipt["rollout_rng"]
        atomic_json(finished,meta)
        self.transfer = {"episode_uuid": identity, "state": "pending"}
        atomic_json(self.pending_upload, self.transfer)
        threading.Thread(target=self.upload_episode, args=(identity,), daemon=True).start()
        return self.remote_online_status()

    def upload_episode(self, identity):
        if not self.transfer_lock.acquire(blocking=False):
            return
        try:
            paths = files(self.journal.root, identity)
            deadline = time.monotonic() + 2400
            while not self.stop.is_set() and time.monotonic() < deadline:
                try:
                    receipt = self.remote_rpc("/episode/receipt", {"episode_uuid": identity})
                    if not receipt["accepted"]:
                        self.remote_rpc("/episode/start", {"episode_uuid": identity})
                        for index, path in enumerate(paths):
                            if self.stop.is_set():
                                return
                            payload = path.read_bytes()
                            digest = hashlib.sha256(payload).hexdigest()
                            request = Request("http://127.0.0.1:%d/journal/%s/%s" % (self.remote.get("local_port",18453), identity, path.name), data=payload, headers={"Content-Type": "application/octet-stream", "X-Content-SHA256": digest})
                            with self.rpc.open(request, timeout=15) as response:
                                response_value = json.load(response)
                            if not response_value.get("accepted") and response_value.get("sha256") != digest:
                                raise ValueError("Remote file acknowledgement mismatch")
                            self.transfer = {"episode_uuid": identity, "state": "uploading", "files_sent": index + 1, "files_total": len(paths)}
                        self.remote_rpc("/episode/end", {"episode_uuid": identity})
                        # Wait for durable learner commit, including checkpoint save.
                        self.transfer = {"episode_uuid": identity, "state": "waiting_commit"}
                        last_progress=self.remote_status.get("updated_at")
                        while not self.stop.wait(1) and time.monotonic() < deadline:
                            progress=self.remote_status.get("updated_at")
                            if self.remote_status.get("phase")=="updating" and progress!=last_progress:
                                deadline=time.monotonic()+2400
                                last_progress=progress
                            receipt = self.remote_rpc("/episode/receipt", {"episode_uuid": identity})
                            if self.remote_status.get("phase") == "fault":
                                self.fail(self.remote_status.get("error", "Remote learning failed"))
                                return
                            if receipt["accepted"]:
                                break
                        if not receipt["accepted"]:
                            raise TimeoutError("Remote episode not committed")
                    self.transfer={"episode_uuid":identity,"state":"publishing"}
                    directory,manifest=self.download_actor()
                    if self.local_actor_version!=manifest["actor_version"]:
                        result=self.local_rpc("/actor/install",{"directory":str(directory)},timeout=600)
                        if result["actor_version"]!=manifest["actor_version"] or not result["ready"]:raise ValueError("Local Actor publication failed")
                        self.local_actor_version=result["actor_version"]
                    from actor_snapshot import prune
                    prune(self.actor_root,self.local_actor_version)
                    self.pending_upload.unlink(missing_ok=True)
                    self.transfer = None
                    return
                except Exception as exc:
                    self.transfer = {"episode_uuid": identity, "state": "retrying", "error": str(exc)}
                    atomic_json(self.pending_upload, self.transfer)
                    if self.stop.wait(3):
                        return
            if not self.stop.is_set():
                self.fail("Remote episode transfer/commit timed out; local journal retained")
        finally:
            self.transfer_lock.release()

    def remote_rpc(self,path,body):
        request=Request("http://127.0.0.1:%d%s"%(self.remote.get("local_port",18453),path),data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
        with self.rpc.open(request,timeout=15) as response:return json.load(response)

    def local_rpc(self,path,body,timeout=15):
        request=Request("http://127.0.0.1:%d%s"%(self.args.port,path),data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
        with self.rpc.open(request,timeout=timeout) as response:return json.load(response)

    def download_actor(self):
        from actor_snapshot import digest_file
        base="http://127.0.0.1:%d"%self.remote.get("local_port",18453)
        with self.rpc.open(base+"/actor/manifest",timeout=15) as response:manifest=json.load(response)
        version=manifest["actor_version"]
        if type(version) is not int or version<0:raise ValueError("Invalid Actor version")
        directory=self.actor_root/str(version);directory.mkdir(parents=True,exist_ok=True)
        if version==getattr(self,'local_actor_version',-1):
            cached=json.loads((directory/'manifest.json').read_text())
            if cached!=manifest:raise ValueError('Already installed Actor manifest changed')
            return directory,manifest
        records=[record for record in manifest["records"] if not record.get("seed")]
        if sum(record["bytes"] for record in records)>3*1024**3:raise ValueError("Actor snapshot exceeds budget")
        for record in records:
            name=record["file"]
            if not re.fullmatch(r"(?:actor|encoder|edit|critic)-[0-9]+\.gz",name):raise ValueError("Invalid Actor filename")
            destination=directory/name
            if destination.exists() and destination.stat().st_size==record["bytes"] and digest_file(destination)==record["sha256"]:continue
            temporary=destination.with_suffix('.download')
            size=temporary.stat().st_size if temporary.exists() else 0
            if size>record['bytes']:temporary.unlink();size=0
            digest=hashlib.sha256()
            if size:
                with temporary.open('rb') as stream:
                    for chunk in iter(lambda:stream.read(4*1024*1024),b''):digest.update(chunk)
            if size==record['bytes']:
                if digest.hexdigest()==record['sha256']:os.replace(temporary,destination);continue
                temporary.unlink();raise ValueError('Partial Actor checksum mismatch; restarting file')
            request=Request(base+'/actor/%d/%s'%(version,name),headers={'Range':'bytes=%d-'%size} if size else {})
            with self.rpc.open(request,timeout=30) as response:
                if size and response.status==200:
                    # Compatibility with an earlier non-range server.
                    size=0;digest=hashlib.sha256()
                elif size and (response.status!=206 or response.headers.get('Content-Range')!='bytes %d-%d/%d'%(size,record['bytes']-1,record['bytes'])):
                    raise ValueError('Invalid Actor range response')
                with temporary.open('ab' if size else 'wb') as stream:
                    while True:
                        chunk=response.read(1024*1024)
                        if not chunk:break
                        size+=len(chunk)
                        if size>record['bytes']:raise ValueError('Actor download size exceeded')
                        stream.write(chunk);digest.update(chunk)
                    stream.flush();os.fsync(stream.fileno())
            if size!=record['bytes'] or digest.hexdigest()!=record['sha256']:
                temporary.unlink();raise ValueError('Actor download checksum mismatch')
            os.replace(temporary,destination)
        atomic_json(directory/'manifest.json',manifest)
        return directory,manifest

    def start_remote(self):
        port=self.remote.get("local_port",18453)
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1",port))==0:raise RuntimeError("Remote Learner tunnel port is occupied")
        with self.remote_process_lock:
            self.remote_process=subprocess.Popen(self.remote_command(),stdin=subprocess.PIPE)
        deadline=time.monotonic()+1200
        while not self.stop.wait(.5):
            if self.remote_process.poll() is not None:raise RuntimeError('Remote Learner exited during load')
            if time.monotonic()>deadline:raise TimeoutError('Remote Learner load timed out')
            try:
                with self.rpc.open('http://127.0.0.1:%d/status'%port,timeout=2) as response:value=json.load(response)
                self.validate_remote_status(value)
                if value['ready']:
                    self.remote_status=value
                    return self.download_actor()[0]
            except (OSError,ValueError):pass
        raise RuntimeError('Remote load interrupted')

    def close_remote(self):
        with self.remote_process_lock:
            if self.remote_process is not None and self.remote_process.poll() is None:
                self.remote_process.terminate()
                try:self.remote_process.wait(timeout=10)
                except subprocess.TimeoutExpired:self.remote_process.kill();self.remote_process.wait()
