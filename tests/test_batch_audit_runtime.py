"""Frozen JAX environment regression: observation must not change an update."""
import importlib.util
import json
import pickle
import sys
from pathlib import Path

import pytest
jax = pytest.importorskip("jax")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"third_party/openpi-rlt/rlt_online_rl/src")]
from integrations.cobot_runtime.replay_audit import install_batch_audit
from rlt_online_rl.trainer import LearnerService
from rlt_online_rl.config import LearnerServiceConfig

def test_real_learner_is_numerically_unchanged(tmp_path, monkeypatch):
    source=ROOT/"third_party/openpi-rlt/rlt_online_rl/tests/test_trainer.py"
    spec=importlib.util.spec_from_file_location("upstream_trainer_fixture",source)
    fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)
    cfg=fixtures._config();batch=fixtures._batch(cfg)
    batch["collection_phase_id"]=np.full(8,2,dtype=np.int32)
    batch["done"][-1]=1;batch["success"][-1]=1
    journal=tmp_path/"synthetic.pkl"
    with journal.open("wb") as f:
        for i in range(8):pickle.dump({k:v[i] for k,v in batch.items()},f)
    config=tmp_path/"config.yaml"
    config.write_text("runtime:\n  replay:\n    journal_path: "+str(journal)+"\n")
    def service(name):
        folder=tmp_path/name
        options=LearnerServiceConfig(sample_batch_size=8,checkpoint_dir=str(folder/"ckpts"),
            actor_snapshot_path=str(folder/"actor.pkl"),push_actor_interval_steps=500,checkpoint_interval_steps=1000)
        return LearnerService(cfg,options,fixtures.FakeReplay(batch),
            metrics_path=str(folder/"online/metrics/learner_metrics.jsonl"),rng=jax.random.PRNGKey(42))
    original=service("original")
    baseline=[original.train_once() for _ in range(2)]
    original_method=LearnerService.train_once
    try:
        install_batch_audit(config)
        audited=service("audited")
        observed=[audited.train_once() for _ in range(2)]
        for a,b in zip(jax.tree.leaves(original.state),jax.tree.leaves(audited.state)):
            np.testing.assert_array_equal(np.asarray(a),np.asarray(b))
        assert baseline==observed
        rows=[json.loads(line) for line in (tmp_path/"audited/online/metrics/batch_composition.jsonl").read_text().splitlines()]
        assert [r["global_step"] for r in rows]==[1,2]
        assert rows[-1]["dimensions"]["outcome"]==[dict(label="success",count=8,ratio=1.)]
        assert json.loads((tmp_path/"audited/analysis/replay_composition.json").read_text())["transitions"]["count"]==8
    finally:
        LearnerService.train_once=original_method
        if hasattr(LearnerService,"_cobot_batch_audit"):delattr(LearnerService,"_cobot_batch_audit")
