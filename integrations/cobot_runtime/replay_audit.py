"""Replay provenance and real-batch composition, separate from the training algorithm."""
import collections
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

def identity(row):
    return (str(row.get("collection_phase", row.get("phase", {1: "warmup", 2: "online"}.get(
        int(row.get("collection_phase_id", 0)), "unknown")))),
        int(row.get("episode_id", -1)), int(row.get("step_id", -1)))

def metadata(row):
    phase, episode, step = identity(row)
    source = int(row.get("source", -1))
    chunks = np.asarray(row.get("source_chunk", [source])).reshape(-1)
    return {"phase": phase, "episode_id": episode, "step_id": step,
        "source": source, "hil": bool(row.get("intervention_flag", False)) or bool(np.isin(chunks, [2, 3]).any()),
        "human_steps": int(np.isin(chunks, [2, 3]).sum()), "chunk_steps": len(chunks),
        "done": bool(row.get("done", False)), "terminal_success": int(row.get("success", -1))}

def annotate(rows):
    """Only a recorded terminal outcome labels an entire episode; unknown stays unknown."""
    groups = collections.defaultdict(list)
    for i, row in enumerate(rows):
        groups[(row["phase"], row["episode_id"])].append(i)
    out = [dict(row) for row in rows]
    recent_start = max((r["episode_id"] for r in rows if r["phase"] == "online"), default=-1) - 19
    for key, indices in groups.items():
        indices.sort(key=lambda i: rows[i]["step_id"])
        terminals = {rows[i]["terminal_success"] for i in indices if rows[i]["done"] and rows[i]["terminal_success"] in (0, 1)}
        outcome = "success" if terminals == {1} else "failure" if terminals == {0} else "unknown"
        for rank, i in enumerate(indices):
            out[i].update(outcome=outcome,
                episode_hil=any(rows[j]["hil"] for j in indices),
                portion=("early", "middle", "late")[min(2, rank * 3 // len(indices))],
                age=("warmup" if key[0] == "warmup" else
                     "recent_online" if key[0] == "online" and key[1] >= recent_start else
                     "older_online" if key[0] == "online" else "unknown"))
    return out

def composition(rows):
    dimensions = {}
    for key in ("outcome", "phase", "age", "portion", "source", "hil", "episode_hil"):
        count = collections.Counter(str(r.get(key, "unknown")) for r in rows)
        dimensions[key] = [{"label": k, "count": n, "ratio": n / len(rows)} for k, n in sorted(count.items())] if rows else []
    cross = collections.Counter((r.get("outcome", "unknown"), r.get("age", "unknown"),
                                 "HIL" if r.get("hil") else "autonomous", r.get("portion", "unknown")) for r in rows)
    return {"count": len(rows), "dimensions": dimensions,
        "human_control_step_ratio": sum(r.get("human_steps", 0) for r in rows) / max(1, sum(r.get("chunk_steps", 0) for r in rows)),
        "cross": [dict(zip(("outcome", "age", "control", "portion"), k), count=n, ratio=n / len(rows))
                  for k, n in sorted(cross.items())]}

class JournalIndex:
    """Incremental trusted-local pickle reader. No HTTP-supplied paths. No RNG calls."""
    def __init__(self, path):
        self.path = Path(path)
        self.offset = 0
        self.raw = []
        self.rows = []
        self.by_id = {}
        self.duplicates = 0
        self.signature = None

    def refresh(self):
        st = self.path.stat()
        signature = (st.st_dev, st.st_ino)
        if signature != self.signature or st.st_size < self.offset:
            self.offset, self.raw = 0, []
            self.signature = signature
        size = st.st_size
        if size == self.offset:
            return
        with self.path.open("rb") as f:
            f.seek(self.offset)
            while f.tell() < size:
                start = f.tell()
                try:
                    row = pickle.load(f)
                except (EOFError, pickle.UnpicklingError):
                    f.seek(start)
                    break
                if f.tell() > size:
                    f.seek(start)
                    break
                self.raw.append(metadata(row))
                self.offset = f.tell()
        self.rows = annotate(self.raw)
        self.by_id = {identity(r): r for r in self.rows}
        self.duplicates = len(self.rows) - len(self.by_id)

    def batch(self, batch):
        self.refresh()
        count = len(batch["episode_id"])
        result, ids = [], []
        for i in range(count):
            phase_id = int(batch.get("collection_phase_id", np.zeros(count))[i])
            key = ({1: "warmup", 2: "online"}.get(phase_id, "unknown"),
                   int(batch["episode_id"][i]), int(batch["step_id"][i]))
            row = self.by_id.get(key)
            if row is None or self.duplicates:
                row = dict(phase=key[0], episode_id=key[1], step_id=key[2],
                           outcome="unknown", age="unknown", portion="unknown",
                           source=int(batch["source"][i]), hil=bool(batch.get("intervention_flag", np.zeros(count))[i]))
            result.append(row)
            ids.append(list(key))
        return dict(composition(result), identities=ids,
                    unique_transitions=len(set(tuple(i) for i in ids)),
                    identity_resolution="ambiguous" if self.duplicates else "phase+episode+step")

def report_from_index(index):
    """Build a metadata snapshot without reopening the large journal."""
    episodes = {}
    for row in index.rows:
        key = (row["phase"], row["episode_id"])
        episodes.setdefault(key, dict(row, transitions=0))["transitions"] += 1
    episode_composition = composition(list(episodes.values()))
    # A first stored window is not an Episode's source, HIL fraction or portion.
    # Keep complete-Episode outcome/HIL marginals, and derive control-step ratio
    # from all windows rather than each Episode's first window.
    for key in ("portion", "source", "hil"):
        episode_composition["dimensions"].pop(key, None)
    episode_composition["human_control_step_ratio"] = composition(index.rows)["human_control_step_ratio"]
    episode_composition["cross"] = []
    return {"schema": 2, "generated_at": time.time(), "journal": str(index.path),
        "bytes_read": index.offset, "duplicate_identities": index.duplicates,
        "transitions": composition(index.rows), "episodes": episode_composition,
        "episode_rows": list(episodes.values()),
        "definitions": {
            "outcome": "Complete episode terminal label; not transition.success.",
            "hil": "Contains HUMAN/MIXED or an intervention flag, including old human demonstrations; this does not always mean online intervention.",
            "age": "warmup / recent online episode ID window of 20 / older online; not wall-clock age.",
            "portion": "Transition rank thirds only, not semantic stages. No first-window portion/source/HIL marginals are reported as Episode distributions.",
            "human_control_step_ratio": "Fraction of stored window slots; overlapping windows repeat steps. Not elapsed HIL time.",
            "episode_rows": "First-window metadata retained for provenance; only outcome, episode_hil and transitions describe the complete Episode.",
            "sampling": "Current upstream uniform sampling is with replacement; composition does not prove data influence.",
            "identity": "phase + episode_id + step_id. Duplicate identities are reported, never silently disambiguated."
        }}

def build_report(journal):
    index = JournalIndex(journal)
    index.refresh()
    return report_from_index(index)

def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, allow_nan=False))
    os.replace(temp, path)

def install_batch_audit(config_path):
    """Decorate the project entry only. Keep upstream sampling, RNG and train_step intact."""
    import yaml
    from rlt_online_rl import trainer
    cfg = yaml.safe_load(Path(config_path).read_text())
    runtime = cfg.get("runtime", {})
    journal = runtime.get("replay", {}).get("journal_path")
    if not journal or getattr(trainer.LearnerService, "_cobot_batch_audit", False):
        return
    original = trainer.LearnerService.train_once
    run_id = "%s-%s" % (int(time.time()), os.getpid())

    def observed(service, *args, **kwargs):
        original_source = service._replay_source
        captured = {}
        class Source:
            def __getattr__(self, name):
                return getattr(original_source, name)
            def sample_batch(self, *a, **kw):
                batch = original_source.sample_batch(*a, **kw)
                # Metadata copies only; never retain or mutate the training arrays.
                captured["batch"] = {k: np.asarray(v).copy() for k, v in batch.items()
                    if k in ("episode_id", "step_id", "source", "source_chunk", "collection_phase_id", "intervention_flag")}
                return batch
        service._replay_source = Source()
        try:
            result = original(service, *args, **kwargs)
        finally:
            service._replay_source = original_source
        if result and "batch" in captured:
            try:
                if not hasattr(service, "_cobot_audit_index"):
                    service._cobot_audit_index = JournalIndex(journal)
                index = service._cobot_audit_index
                report = index.batch(captured["batch"])
                # Publish composition only when new journal metadata arrives, not every update.
                signature = (index.signature, index.offset)
                if getattr(service, "_cobot_audit_snapshot", None) != signature:
                    metrics = Path(service._metrics_path)
                    snapshot = metrics.parent.parent.parent / "analysis/replay_composition.json"
                    atomic_json(snapshot, report_from_index(index))
                    service._cobot_audit_snapshot = signature
                report.update(run_id=run_id, global_step=int(result["global_step"]),
                              actor_version=int(result["actor_version"]), timestamp=time.time())
                target = Path(service._metrics_path).with_name("batch_composition.jsonl")
                with target.open("a") as f:
                    f.write(json.dumps(report, separators=(",", ":")) + "\n")
            except Exception:
                import logging
                logging.getLogger(__name__).exception("Batch audit unavailable; training unchanged")
        return result
    trainer.LearnerService.train_once = observed
    trainer.LearnerService._cobot_batch_audit = True
