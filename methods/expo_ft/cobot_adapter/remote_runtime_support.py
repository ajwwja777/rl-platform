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
        self.remote_status = {"ready": False, "phase": "loading"}
        self.remote_seen = 0.
        self.remote_process = None
        self.local_actor_version = -1
        self.actor_root = args.run_dir / "actors"
        self.transfer = None
        self.transfer_lock = threading.Lock()
        self.pending_upload = args.run_dir / "pending_upload.json"
        if self.remote and self.pending_upload.exists():
            self.transfer = json.loads(self.pending_upload.read_text())

    def remote_command(self):
        c = self.remote
        return ["ssh", "-T", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=3", "-o", "ServerAliveCountMax=2", "-o", "ConnectTimeout=8", "-i", c["identity_file"], "-L", "127.0.0.1:%d:127.0.0.1:%d" % (c.get("local_port",18453), c["remote_port"]), c["destination"], "run"]

    def validate_server_status(self, info):
        if self.remote and (info.get("deployment_id") != self.args.model_id or info.get("rtc_delay") != self.args.rtc_delay or info.get("update_layout") != "native_full_batch"):
            raise RuntimeError("Remote learner identity mismatch")

    def remote_poll(self):
        while not self.stop.wait(.5):
            try:
                with self.rpc.open("http://127.0.0.1:%d/status" % self.remote.get("local_port",18453), timeout=1) as response:
                    value = json.load(response)
                self.validate_server_status(value)
                self.remote_status = value
                self.remote_seen = time.monotonic()
            except Exception:
                if self.ready and time.monotonic() - self.remote_seen > 3:
                    self.remote_status = {"ready": False, "phase": "fault", "error": "Remote learner connection lost"}
                    self.fail("Remote learner connection lost; episode retained locally")

    def remote_online_status(self):
        value = dict(self.remote_status)
        value["compute_host"] = "Learner A6000 / Actor Cobot"
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
        records=[record for record in manifest["records"] if not record.get("seed")]
        if sum(record["bytes"] for record in records)>3*1024**3:raise ValueError("Actor snapshot exceeds budget")
        for record in records:
            name=record["file"]
            if not re.fullmatch(r"(?:actor|encoder|edit|critic)-[0-9]+\.gz",name):raise ValueError("Invalid Actor filename")
            destination=directory/name
            if destination.exists() and destination.stat().st_size==record["bytes"] and digest_file(destination)==record["sha256"]:continue
            temporary=destination.with_suffix('.download')
            digest=hashlib.sha256();size=0
            with self.rpc.open(base+'/actor/%d/%s'%(version,name),timeout=30) as response,temporary.open('wb') as stream:
                while True:
                    chunk=response.read(1024*1024)
                    if not chunk:break
                    size+=len(chunk)
                    if size>record['bytes']:raise ValueError('Actor download size exceeded')
                    stream.write(chunk);digest.update(chunk)
                stream.flush();os.fsync(stream.fileno())
            if size!=record['bytes'] or digest.hexdigest()!=record['sha256']:raise ValueError('Actor download checksum mismatch')
            os.replace(temporary,destination)
        atomic_json(directory/'manifest.json',manifest)
        return directory,manifest

    def start_remote(self):
        port=self.remote.get("local_port",18453)
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1",port))==0:raise RuntimeError("Remote Learner tunnel port is occupied")
        self.remote_process=subprocess.Popen(self.remote_command(),stdin=subprocess.PIPE)
        deadline=time.monotonic()+1200
        while not self.stop.wait(.5):
            if self.remote_process.poll() is not None:raise RuntimeError('Remote Learner exited during load')
            if time.monotonic()>deadline:raise TimeoutError('Remote Learner load timed out')
            try:
                with self.rpc.open('http://127.0.0.1:%d/status'%port,timeout=2) as response:value=json.load(response)
                self.validate_server_status(value)
                if value['ready']:
                    self.remote_status=value
                    return self.download_actor()[0]
            except (OSError,ValueError):pass
        raise RuntimeError('Remote load interrupted')

    def close_remote(self):
        if self.remote_process is not None and self.remote_process.poll() is None:
            self.remote_process.terminate()
            try:self.remote_process.wait(timeout=10)
            except subprocess.TimeoutExpired:self.remote_process.kill();self.remote_process.wait()
