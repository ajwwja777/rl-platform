"""Immutable episode transfer over the SSH-forwarded learner HTTP endpoint."""
import hashlib
import json
import os
from pathlib import Path
import re
import numpy as np
from online_protocol import episode_id


def receive(root, active, accepted, identity, name, payload, digest):
    identity = episode_id(identity)
    if identity in accepted:
        return {"accepted": True}
    if identity != active:
        raise ValueError("Transfer requires the active episode")
    if not re.fullmatch(r"(?:started|finished)\.json|terminal\.npz|[0-9]{6}\.npz", name):
        raise ValueError("Invalid journal filename")
    if not payload or len(payload) > 2000000:
        raise ValueError("Invalid journal payload size")
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError("Journal transfer checksum mismatch")
    directory = Path(root) / identity
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / name
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
            raise ValueError("Transferred journal file is immutable")
        return {"sha256": digest, "duplicate": True}
    if (directory / "finished.json").exists():
        raise ValueError("Finished journal is immutable")
    if name.endswith(".json"):
        meta = json.loads(payload)
        if meta["episode_uuid"] != identity:
            raise ValueError("Transferred identity mismatch")
        if name == "finished.json":
            if not (directory / "started.json").exists():
                raise ValueError("Missing episode start")
            count = meta["count"]
            if not isinstance(count, int) or not 0 <= count <= 3000:
                raise ValueError("Invalid transferred frame count")
            if meta["outcome"] not in ("success", "failure", "unknown", "aborted"):
                raise ValueError("Invalid transferred outcome")
            for index in range(count):
                if not (directory / ("%06d.npz" % index)).exists():
                    raise ValueError("Incomplete journal transfer")
            if count and meta.get("training_eligible", True) and not (directory / "terminal.npz").exists():
                raise ValueError("Missing terminal observation")
            started = json.loads((directory / "started.json").read_text())
            if meta["actor_version"] != started["actor_version"]:
                raise ValueError("Transferred policy version changed")
    temporary = destination.with_suffix(destination.suffix + ".upload")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    fd = os.open(str(directory), os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return {"sha256": digest}


def files(root, identity):
    directory = Path(root) / episode_id(identity)
    meta = json.loads((directory / "finished.json").read_text())
    if meta["episode_uuid"] != identity or not 0 <= meta["count"] <= 3000:
        raise ValueError("Invalid finished local journal")
    names = ["started.json"] + ["%06d.npz" % i for i in range(meta["count"])]
    if (directory / "terminal.npz").exists():
        names.append("terminal.npz")
    names.append("finished.json")
    return [directory / name for name in names]
