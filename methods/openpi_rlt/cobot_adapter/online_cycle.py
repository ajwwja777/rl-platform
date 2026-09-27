"""Disk-backed episode-boundary update protocol; no ROS imports or commands."""
import json,os,pickle,time,uuid
from pathlib import Path
import numpy as np

def atomic_json(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+".tmp")
    with temp.open("w") as f:
        json.dump(payload,f,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)

def read_optional_json(path):
    """Poll an optional IPC file without an exists/open race.

    A missing file means no message on this poll. Invalid JSON and other
    I/O errors remain fatal; never hide corruption or permission failures.
    """
    try:
        text = Path(path).read_text()
    except FileNotFoundError:
        return {}
    return json.loads(text)

def load_journal(path):
    records=[]
    if not Path(path).exists():return records
    with Path(path).open("rb") as f:
        while True:
            try:records.append(pickle.load(f))
            except EOFError:break
    return records

def complete_episodes(records):
    grouped={}
    for r in records:
        for v in r.values():
            if isinstance(v,np.ndarray) and np.issubdtype(v.dtype,np.number) and not np.isfinite(v).all():
                raise ValueError("nonfinite replay")
        grouped.setdefault(int(r["episode_id"]),[]).append(dict(r))
    complete={}
    for ep,rs in grouped.items():
        terminals=sum(bool(r["done"]) for r in rs)
        if terminals==0:continue
        if terminals!=1 or not bool(rs[-1]["done"]):raise ValueError("invalid replay terminal order/count")
        reward=sum(float(np.sum(r["rewards"])) for r in rs)
        if reward not in (0.,1.):raise ValueError("invalid sparse terminal reward")
        if any(r.get("shadow",False) or r.get("outcome")=="aborted" for r in rs):continue
        for r in rs:r["group"]="success" if reward else "failure"
        complete[ep]=rs
    return complete

def pending_episodes(complete,processed):
    return sorted(set(complete)-set(int(x) for x in processed))

def accept_candidate(base,candidate):
    if not all(np.isfinite(v) for v in candidate.values()):return False
    # Relative gates compare against the accepted actor, not random initialization.
    return all(candidate[k]<=base[k]*1.10+1e-4 for k in ["score","velocity","acceleration","first","boundary"])

def request_cycle(run,*,actor_version_getter=None,sleep=time.sleep,timeout=180):
    run=Path(run);request_id=uuid.uuid4().hex
    atomic_json(run/"update_request.json",dict(request_id=request_id,time=time.time()))
    deadline=time.monotonic()+timeout
    if actor_version_getter is None:
        from urllib.request import build_opener,ProxyHandler
        op=build_opener(ProxyHandler({}))
        def actor_version_getter():
            with op.open(os.environ["COBOT_RLT_ACTOR_URL"]+"/version",timeout=2) as f:
                from openpi_client import msgpack_numpy
                value=msgpack_numpy.unpackb(f.read())
            return int(value["actor_param_version"])
    while time.monotonic()<deadline:
        p=run/"metrics/learner_status.json"
        status=read_optional_json(p)
        if status.get("phase")=="error":raise RuntimeError("online learner: "+str(status.get("error")))
        if status.get("request_id")==request_id and status.get("ready_for_online"):
            if actor_version_getter()>=int(status["actor_version"]):return status
        sleep(.1)
    raise RuntimeError("online update/actor publication readiness timeout")

def project_training_action(record, *, active_arm, hold_grippers):
    """Project only the learner's action input onto the controlled subspace.

    Preserve the raw measured replay record and all source/reward metadata.
    Inactive dimensions follow reference, consistent with the policy BC target,
    rather than training against uncontrollable offsets amplified by tiny scales.
    """
    from methods.openpi_rlt.cobot_adapter.action_conditioning import active_action_mask
    mask = active_action_mask(active_arm, explore_gripper=not hold_grippers).astype(bool)
    actual = np.asarray(record["action_chunk"], dtype=np.float32)
    reference = np.asarray(record["ref_chunk"], dtype=np.float32)
    if actual.ndim != 2 or actual.shape != reference.shape or actual.shape[-1] != 14:
        raise ValueError("training actions and reference must have matching [T,14] shape")
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError("nonfinite training action")
    projected = actual.copy()
    projected[:, ~mask] = reference[:, ~mask]
    return {**record, "action_chunk": projected}
