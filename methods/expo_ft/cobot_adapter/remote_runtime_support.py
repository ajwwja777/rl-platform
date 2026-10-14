"""Cobot-side SSH lifecycle, cached status, and retryable finalized uploads."""
import hashlib
import json
from pathlib import Path
import threading
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
        self.transfer = None
        self.transfer_lock = threading.Lock()
        self.pending_upload = args.run_dir / "pending_upload.json"
        if self.remote and self.pending_upload.exists():
            self.transfer = json.loads(self.pending_upload.read_text())

    def remote_command(self):
        c = self.remote
        return ["ssh", "-T", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=3", "-o", "ServerAliveCountMax=2", "-o", "ConnectTimeout=8", "-i", c["identity_file"], "-L", "127.0.0.1:%d:127.0.0.1:%d" % (self.args.port, c["remote_port"]), c["destination"], "run"]

    def validate_server_status(self, info):
        if self.remote and (info.get("deployment_id") != self.args.model_id or info.get("rtc_delay") != self.args.rtc_delay):
            raise RuntimeError("Remote learner identity mismatch")

    def remote_poll(self):
        while not self.stop.wait(.5):
            try:
                with self.rpc.open("http://127.0.0.1:%d/status" % self.args.port, timeout=1) as response:
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
        value["compute_host"] = "A6000"
        if self.transfer and value.get("phase") != "fault":
            value.update(ready=False, phase="updating" if self.transfer.get("state")=="waiting_commit" else "transferring", transfer=dict(self.transfer))
        return value

    def transfer_episode(self, identity):
        if self.transfer and self.transfer["episode_uuid"] != identity:
            raise ValueError("Previous episode transfer is pending")
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
                    receipt = self.json_rpc("/episode/receipt", {"episode_uuid": identity})
                    if not receipt["accepted"]:
                        self.json_rpc("/episode/start", {"episode_uuid": identity})
                        for index, path in enumerate(paths):
                            if self.stop.is_set():
                                return
                            payload = path.read_bytes()
                            digest = hashlib.sha256(payload).hexdigest()
                            request = Request("http://127.0.0.1:%d/journal/%s/%s" % (self.args.port, identity, path.name), data=payload, headers={"Content-Type": "application/octet-stream", "X-Content-SHA256": digest})
                            with self.rpc.open(request, timeout=15) as response:
                                response_value = json.load(response)
                            if not response_value.get("accepted") and response_value.get("sha256") != digest:
                                raise ValueError("Remote file acknowledgement mismatch")
                            self.transfer = {"episode_uuid": identity, "state": "uploading", "files_sent": index + 1, "files_total": len(paths)}
                        self.json_rpc("/episode/end", {"episode_uuid": identity})
                        # Wait for durable learner commit, including checkpoint save.
                        self.transfer = {"episode_uuid": identity, "state": "waiting_commit"}
                        while not self.stop.wait(1) and time.monotonic() < deadline:
                            receipt = self.json_rpc("/episode/receipt", {"episode_uuid": identity})
                            if self.remote_status.get("phase") == "fault":
                                self.fail(self.remote_status.get("error", "Remote learning failed"))
                                return
                            if receipt["accepted"]:
                                break
                        if not receipt["accepted"]:
                            raise TimeoutError("Remote episode not committed")
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
