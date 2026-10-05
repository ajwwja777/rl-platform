"""Synthetic evidence contracts; no model, production assets or robot I/O."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


def run_comparison(tmp_path, commands, command_stamps=None):
    h5py = pytest.importorskip("h5py")
    frames = len(commands)
    raw = tmp_path / "episode.hdf5"
    feedback = np.concatenate([np.zeros((frames, 7)), np.ones((frames, 7))], axis=1)
    with h5py.File(raw, "w") as handle:
        handle.create_dataset("observations/qpos", data=feedback)
        handle.create_dataset("rollout/front_observation", data=feedback)
        handle.create_dataset("rollout/coordinator_command", data=np.concatenate(
            [np.full((frames, 7), np.nan), np.array(commands)], axis=1))
        for key in ["is_intervention_right", "valid_mask/coordinator_right", "valid_mask/front_right"]:
            handle.create_dataset("rollout/"+key, data=np.ones(frames, dtype=bool))
        handle.create_dataset("rollout/control_source_right", data=np.full(frames, 2, dtype=np.uint8))
        handle.create_dataset("rollout/sample_timestamp", data=np.arange(frames)+500.)
        handle.create_dataset("rollout/topic_timestamp/front_right", data=np.full(frames, 100.))
        handle.create_dataset("rollout/topic_timestamp/coordinator_right", data=
                              command_stamps if command_stamps is not None else np.full(frames, 100.))
    trace = tmp_path / "trace.jsonl"
    row = dict(action=[1.]*7, observation={"state": [0.]*7}, next_observation={"state": [1.]*7},
               source=2, timestamp=100.05, done=False)
    trace.write_text("\n".join(json.dumps(dict(row, source=2 if i == 0 else 0)) for i in range(10)))
    request = {"matches": [dict(split="train", episode=1, step=0, trace=str(trace),
                              trace_start_row=0, trace_sources=[2]+[0]*9)],
               "census": {"rows": [dict(path=str(raw), frames=frames,
                         attrs={"start_timestamp": 99., "end_timestamp": 101., "episode_uuid": "synthetic"})]}}
    script = Path(__file__).resolve().parents[3] / "scripts/collect_hil_command.py"
    result = subprocess.run([sys.executable, str(script)], input=json.dumps(request),
                            text=True, capture_output=True, check=True)
    return json.loads(result.stdout)["steps"][0]


def test_moving_hil_anchors_to_action_and_end_feedback_not_start_state(tmp_path):
    row = run_comparison(tmp_path, [[2.]*7])
    assert not row["action_equals_state"]
    assert row["action_equals_next_state"]
    assert row["exact_action_feedback_compatible_snapshots"] == 1
    assert not row["compatible_snapshots"][0]["feedback_matches_state"]
    assert row["all_exact_candidate_commands_equal"]
    assert row["command_identity_status"] == "insufficient_evidence"


def test_conflicting_command_snapshots_are_preserved_without_nearest_frame_guess(tmp_path):
    row = run_comparison(tmp_path, [[2.]*7, [3.]*7])
    assert row["exact_action_feedback_compatible_snapshots"] == 2
    assert not row["all_exact_candidate_commands_equal"]
    assert row["command_identity_status"] == "insufficient_evidence"


def test_future_command_message_is_not_a_past_action_candidate(tmp_path):
    row = run_comparison(tmp_path, [[2.]*7], [100.06])
    assert row["compatible_snapshots"] == []
    assert row["exact_action_feedback_compatible_snapshots"] == 0
