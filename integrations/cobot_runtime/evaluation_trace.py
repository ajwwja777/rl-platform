"""Optional numeric evaluation evidence, isolated from collection and Replay.

Set COBOT_RLT_EVALUATION_TRACE_DIR to an absolute, independent output directory
before creating a runtime. This is diagnostic I/O and can affect timing; it is
not enabled by model selection and does not recover traces from old episodes.
RGB is deliberately omitted by the existing atomic trace writer.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4


class EvaluationTraceWriter:
    def __init__(self, root):
        from methods.openpi_rlt.cobot_adapter.cobot_ros1 import AtomicEpisodeTraceWriter
        self.run_id = Path(root).name
        self.model_id = os.environ.get("COBOT_DEPLOYMENT_MODEL_ID")
        project = Path(__file__).resolve().parents[2]
        names = ["integrations/cobot_runtime/evaluation_trace.py",
                 "integrations/cobot_runtime/evaluation_env.py",
                 "integrations/cobot_runtime/shared_model_env.py",
                 "methods/openpi_rlt/cobot_adapter/async_execution.py",
                 "methods/openpi_rlt/cobot_adapter/cobot_ros1.py"]
        self.provenance = dict(code_files_sha256={name: hashlib.sha256((project/name).read_bytes()).hexdigest() for name in names},
                               learner_disabled_environment=os.environ.get("RLT_DISABLE_LEARNER"),
                               execution_options_environment=os.environ.get("COBOT_RLT_EXECUTION_OPTIONS"))
        self.first_row = True
        self.writer = AtomicEpisodeTraceWriter(root, retain_aborted=True)

    def start_episode(self):
        self.first_row = True
        self.writer.start_episode()

    def append(self, record):
        payload = dict(record)
        payload.update(trace_purpose="evaluation_diagnostic", replay_eligible=False,
                       images_retained=False, diagnostic_run_id=self.run_id,
                       deployment_model_id=self.model_id)
        if self.first_row:
            payload["runtime_provenance"] = self.provenance
            self.first_row = False
        self.writer.append(payload)

    def discard(self):
        self.writer.discard()

    def finalize(self, outcome, *, identity=None):
        self.writer.finalize(outcome, identity=identity)


def evaluation_trace_writer():
    """None means unchanged no-output evaluation; never reuse collection files."""
    configured = os.environ.get("COBOT_RLT_EVALUATION_TRACE_DIR", "").strip()
    if not configured:
        # Owned by RLT, read only at the next runtime construction. This lets
        # the existing web launcher opt in without restarting the web server.
        from .paths import RUNTIME_ROOT
        settings = RUNTIME_ROOT / "evaluation-diagnostic.json"
        if settings.exists():
            value = json.loads(settings.read_text(encoding="utf-8"))
            if value.get("enabled") is True:
                if value.get("schema_version") != 1:
                    raise ValueError("Unsupported evaluation diagnostic settings schema")
                if value.get("model_id") != "plug-v3-warmup-5k" or value.get("model_id") != os.environ.get("COBOT_DEPLOYMENT_MODEL_ID"):
                    raise ValueError("Evaluation diagnostic model does not match selection; disable or correct settings")
                configured = str(value["trace_root"]).strip()
                if not configured:
                    raise ValueError("Enabled evaluation diagnostic trace_root must be nonempty")
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
