"""Strict durable episode transport shared by ROS Python3.8 and native learner."""
import hashlib
import io
import json
import os
from pathlib import Path
import uuid
import numpy as np

KEYS = ("state", "left_wrist_image", "right_wrist_image")


def episode_id(value):
    return str(uuid.UUID(str(value)))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory_fd=os.open(str(path.parent),os.O_DIRECTORY)
    try:os.fsync(directory_fd)
    finally:os.close(directory_fd)


def validate_episode(data):
    identity = episode_id(data["episode_uuid"].item())
    outcome = str(data["outcome"].item())
    if outcome not in ("success", "failure", "unknown", "aborted"):
        raise ValueError("invalid episode outcome")
    n = len(data["actions"])
    if not 1 <= n <= 12000:
        raise ValueError("episode must contain1..12000 logical transitions")
    for key in ("state", "actions"):
        if data[key].shape != (n,7) or not np.isfinite(data[key]).all():
            raise ValueError("invalid physical left-arm7D " + key)
    for key in KEYS[1:]:
        if data[key].shape != (n,180,320,3) or data[key].dtype != np.uint8:
            raise ValueError("invalid two-view RGB " + key)
    for key in ("is_hil", "dones"):
        if data[key].shape != (n,) or data[key].dtype != np.bool_:
            raise ValueError("invalid flags " + key)
    times = data["times"]
    if times.shape != (n,) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("non-monotonic logical steps")
    if not data["dones"][-1]:
        raise ValueError("missing durable episode end")
    # Gaps are truncations, never continuations or fabricated actions.
    for i, dt in enumerate(np.diff(times)):
        if dt > .16 and not data["dones"][i]:
            raise ValueError("recording gap without truncation")
    version = int(data["actor_version"].item())
    if version < 0:
        raise ValueError("invalid actor version")
    return identity, outcome, n, version


def transitions(data):
    _, outcome, n, _ = validate_episode(data)
    terminal = outcome in ("success", "failure")
    # Sparse reward belongs only to the final transition. Native success-only
    # BC can use valid actions throughout the operator-labelled successful episode.
    for i in range(n):
        done = bool(data["dones"][i])
        final = i == n - 1 and terminal
        yield dict(observations={key: data[key][i] for key in KEYS},
                   actions=data["actions"][i], rewards=float(final and outcome == "success"),
                   masks=float(not final), dones=done, is_hil=bool(data["is_hil"][i]),
                   is_success=bool(outcome == "success"))


def load_episode(path):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    validate_episode(data)
    return data


def durable_bytes(path, body):
    path = Path(path)
    digest = hashlib.sha256(body).hexdigest()
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("episode UUID already exists with different content")
        return digest
    with np.load(io.BytesIO(body), allow_pickle=False) as archive:
        validate_episode({key: archive[key] for key in archive.files})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory_fd=os.open(str(path.parent),os.O_DIRECTORY)
    try:os.fsync(directory_fd)
    finally:os.close(directory_fd)
    return digest
