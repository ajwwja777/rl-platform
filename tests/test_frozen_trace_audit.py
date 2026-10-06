import json
import numpy as np
import pytest
from methods.openpi_rlt.experiments.frozen_trace_audit import audit_episode, audit_traces, episode_interval, sha256


def write_trace(tmp_path, rows, name="episode.jsonl"):
    path = tmp_path / name
    path.write_text("".join(json.dumps(row)+"\n" for row in rows))
    return path


def row(index=0, **kw):
    value = dict(trace_purpose="evaluation_diagnostic", replay_eligible=False,
        deployment_model_id="plug-v3-warmup-5k", actor_param_version=2500,
        session_id="s", session_episode_id=1, source=1, human_controlled=False,
        action=[.01]*7, planned_action=[.04]*7, logical_ref_action=[.02]*7,
        next_observation={"state":[0.]*7}, sample_received_monotonic=1.+index*.05+.04,
        done=False, publish_hz=50, publications=[dict(action=[.01]*7,
            scheduled_monotonic=1.+index*.05, publish_started_monotonic=1.+index*.05+.002,
            publish_finished_monotonic=1.+index*.05+.003, feedback_received_monotonic=1.+index*.05,
            actor_param_version=2500, execution_epoch=2)])
    value.update(kw)
    return value


def test_queue_command_and_feedback_are_distinct_and_hil_ref_not_counterfactual(tmp_path):
    rows = [row(),row(1, source=2,human_controlled=True,planned_action=None,
        logical_ref_action=None,ref_action=[.5]*7,action=[.5]*7,publications=[],
        done=True,outcome="success")]
    path=write_trace(tmp_path, rows)
    result=audit_episode(path, dict(sha256=sha256(path),split="development",condition="offset_A",actor_version=2500))
    assert result["outcome_class"] == "assisted_success"
    np.testing.assert_allclose(result["metrics"]["actor_reference_mae"], [20.]*7)
    np.testing.assert_allclose(result["metrics"]["command_queue_target_mae"], [30.]*7)
    np.testing.assert_allclose(result["metrics"]["endpoint_feedback_command_mae"], [10.]*7)
    assert result["series"]["logical"][1]["reference"] is None
    assert result["metrics"]["scheduled_start_lateness_ms"]["median"] == pytest.approx(2.)
    assert result["metrics"]["rejected_deadline_miss_count"] is None


def test_missing_receipts_and_aborted_do_not_become_zero_or_failed_trials(tmp_path):
    path=write_trace(tmp_path,[row(action=None,planned_action=None,publications=[],next_observation={},sample_received_monotonic=None,done=False,outcome="aborted")])
    report=audit_traces([path])
    ep=report["episodes"][0]
    assert ep["outcome_class"]=="aborted" and not ep["complete_episode"]
    assert ep["metrics"]["publication_interval_ms"] is None
    assert ep["metrics"]["actor_reference_mae"] is None
    assert all(value is None for value in next(iter(report["groups"].values()))["metrics"].values())


def test_duplicate_snapshot_rejected_and_manifest_sha_bound(tmp_path):
    rows=[row(done=True,outcome="failure")]
    a=write_trace(tmp_path,rows,"a.jsonl");b=write_trace(tmp_path,rows,"b.jsonl")
    with pytest.raises(ValueError,match="Duplicate"):
        audit_traces([a,b])
    with pytest.raises(ValueError,match="hash mismatch"):
        audit_episode(a,dict(sha256="bad"))


def test_independent_test_label_requires_provenance_and_version_mismatch_flag(tmp_path):
    path=write_trace(tmp_path,[row(done=True,outcome="success")])
    ep=audit_episode(path,dict(sha256=sha256(path),split="independent_test",actor_version=10000))
    assert ep["split"]=="unassigned"
    assert any("manifest" in x for x in ep["issues"])
    assert any("provenance" in x for x in ep["issues"])


def test_episode_balanced_uncertainty_does_not_count_overlapping_rows(tmp_path):
    a=write_trace(tmp_path,[row(i, session_episode_id=1,planned_action=[.03]*7,
        done=i==99,outcome="success" if i==99 else None) for i in range(100)],"a.jsonl")
    b=write_trace(tmp_path,[row(session_episode_id=2,planned_action=[.05]*7,done=True,outcome="success")],"b.jsonl")
    report=audit_traces([a,b])
    metric=next(iter(report["groups"].values()))["metrics"]["actor_reference_mae"]
    assert metric["n_episodes"]==2
    np.testing.assert_allclose(metric["mean"],[20.]*7)  # Not 100:1 row-weighted.
    assert episode_interval([[1.]])["episode_bootstrap95"] is None


def test_14d_uses_right_arm_and_no_sign_flip_across_publication_gap(tmp_path):
    pubs=[]
    for t,x in [(1.,0.),(1.02,.01),(2.,.02),(2.02,.01)]:
        pubs.append(dict(action=[x]*7,publish_finished_monotonic=t,execution_epoch=1))
    path=write_trace(tmp_path,[row(action=[9.]*7+[.01]*7,
        planned_action=[9.]*7+[.04]*7,logical_ref_action=[9.]*7+[.02]*7,
        publications=pubs,sample_received_monotonic=2.04,done=True,outcome="failure")])
    ep=audit_episode(path)
    np.testing.assert_allclose(ep["metrics"]["actor_reference_mae"],[20.]*7)
    assert ep["metrics"]["publication_interval_ms"]["n"]==2
    assert ep["metrics"]["command_direction_flips"]==[0]*7
