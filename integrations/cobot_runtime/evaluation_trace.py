"""Optional numeric evaluation evidence, isolated from collection and Replay.

Set COBOT_RLT_EVALUATION_TRACE_DIR to an absolute, independent output directory
before creating a runtime. This is diagnostic I/O and can affect timing; it is
not enabled by model selection and does not recover traces from old episodes.
RGB is deliberately omitted by the existing atomic trace writer.
"""
from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4


class EvaluationTraceWriter:
    def __init__(self, root):
        from methods.openpi_rlt.cobot_adapter.cobot_ros1 import AtomicEpisodeTraceWriter
        self.writer = AtomicEpisodeTraceWriter(root)

    def start_episode(self):
        self.writer.start_episode()

    def append(self, record):
        payload = dict(record)
        payload.update(trace_purpose="evaluation_diagnostic", replay_eligible=False,
                       images_retained=False)
        self.writer.append(payload)

    def discard(self):
        self.writer.discard()

    def finalize(self, outcome, *, identity=None):
        self.writer.finalize(outcome, identity=identity)


def evaluation_trace_writer():
    """None means unchanged no-output evaluation; never reuse collection files."""
    configured = os.environ.get("COBOT_RLT_EVALUATION_TRACE_DIR", "").strip()
    if not configured:
        return None
    root = Path(configured).expanduser()
    if not root.is_absolute():
        raise ValueError("Evaluation diagnostic trace directory must be absolute")
    root = root.resolve()
    collection = os.environ.get("COBOT_RLT_TRACE_DIR", "").strip()
    if collection:
        collection = Path(collection).expanduser().resolve()
        if root == collection or collection in root.parents or root in collection.parents:
            raise ValueError("Evaluation diagnostic trace must be separate from collection traces")
    # Fresh namespace per runtime; never append to historical evaluation or
    # production trace files even when the configured parent is reused.
    return EvaluationTraceWriter(root / ("evaluation-" + uuid4().hex))
