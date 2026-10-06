# Frozen target-position / jitter localization

This is an opt-in diagnostic workflow, not Online release acceptance. Original
Warmup5k Actor2500 remains the baseline. No new Actor is promoted by this workflow.

## Evidence and preparation

A6000 owns source/Git/output. The field runtime must be released before enabling
diagnostic settings. Read deployment status using GET with HTTP proxies disabled:
phase offline, process_started false, model_ready false. Historical ready_confirmed
and remembered model selection do not prove that a model is running.

After verified source synchronization, on Cobot:
~~~bash
cd /home/agilex/jiaan/project/rl-platform
./envs/online/bin/python scripts/configure_evaluation_diagnostics.py --enable \
  --trace-root /home/agilex/jiaan/project/rl-platform/runtime/diagnostics/target-offset-20261006
~~~
This writes only runtime/evaluation-diagnostic.json for the NEXT runtime. No
model/service/session/robot starts. It is model-bound to plug-v3-warmup-5k;
disable it before choosing another model. Environment-based trace opt-in remains
available. Diagnostic output must be independent of collection trace storage.

## Operator-controlled frozen diagnostic

Use the evaluation/deployment page, scene plug_insertion, family RLT, original
method, Warmup5000 (plug-v3-warmup-5k). Explicitly select 50 Hz publication,
RTC enabled and smoothing enabled before loading. Logical Actor/Replay is 20 Hz.
Keep the same Actor, prompt, start robot pose, camera mounting and execution
settings throughout; no online model, collection, training or auto-promotion.

After load, CHECK actual Actor2500, selected model ID, Learner disabled,
evaluation purpose and Replay eligibility false. Start Session/episode only after
the operator checks the scene and hardware. Verify the first trace is created in
a fresh evaluation-* namespace. A missing trace means stop and repair evidence
capture; do not silently continue and infer a diagnosis from only success labels.

Measure and record two socket/target displacements A/B. Do not invent millimetres
from pixels. Interleave nominal, A, B, nominal, A, B; two complete Episodes per
condition initially. These are diagnostic DEVELOPMENT Episodes, not independent
TEST data and not a powered success-rate comparison. Link each complete
session_id/session_episode_id or task5_episode_uuid and trace SHA to condition,
measured offset or explicit uncalibrated image identity. Keep all outcomes.
Success without any takeover is autonomous; success after takeover is assisted.
Aborted/incomplete/unlabelled are separate, not failures or automatic successes.
Manual abort retains numeric diagnostic evidence with done=false/truncated=true;
ordinary collection retains its existing abort/discard policy.

Stop for continued oscillation/contact risk, stale feedback/deadline/runtime fault,
Actor/profile change, Learner update, Replay commit, missing identity/trace or
unverified target placement. Follow the existing operator pause/abort procedure;
do not automatically restart or move/homing/recover to hide faults. Preserve the
GET status and relevant log tail for faulted/incomplete traces. Diagnostic writes
may affect timings; report that capture is enabled.

## Read-only analysis

Copy only these small numeric diagnostic traces and identity/status receipts to
a new A6000 output. Do not copy videos, datasets or weights or alter Replay.
Use a hash-bound manifest with episodes entries:
~~~json
{"episodes": [{"path": "/absolute/trace.jsonl", "sha256": "ACTUAL_SHA256",
 "split": "development", "condition": "measured_offset_A",
 "measured_offset_mm": null, "actor_version": 2500}]}
~~~
Use null for an unmeasured offset; retain the external position evidence.
Never upgrade reused development data into independent_test.

~~~bash
/usr/bin/python3 scripts/audit_frozen_execution_trace.py \
  --trace /absolute/trace.jsonl --manifest /absolute/identity.json \
  --output /absolute/new-a6000-report --plots
~~~
Repeat --trace for other Episodes. Output must be new. The existing A6000 system
Python provides matplotlib; project .venv runs CPU regressions and needs no new
installation. Reports contain raw identity, missing fields, per-Episode metrics,
Episode bootstrap (no single-Episode interval), seven joint/gripper action plots,
physical command/feedback plots and publication/request timelines. mrad and mm
are shown separately; no joint error is claimed as calibrated TCP error.

Reference vs Actor uses the queued logical pair, before interpolation/EMA/clamp.
Queued proposal is not a fresh same-state HIL counterfactual. HIL ref_action may
equal the human command/feedback and is excluded as an independent Reference.
Request anchors describe inference, not proof of queue acceptance. Physical
publication receipts and logical20 rows are distinct; gaps/epochs are never
joined for motion statistics. Endpoint feedback error has reaction delay and is
not settled tracking accuracy. Missing/rejected deadline evidence stays null,
rather than counting only emitted commands as all deadline misses.

## Decision and rollback

If the Reference fails to adapt at displaced targets, prioritize raw-image/target
coverage and Stage1 checks. If Reference adapts but Actor removes the correction,
prioritize paired Actor/Warmup regression. If targets are stable while published
commands/feedback oscillate, prioritize RTC/timing/filter/tracking diagnosis.
These patterns localize hypotheses; numeric traces alone cannot prove visual
misperception, valid insertion commands, contact dynamics or an algorithm cause.

Disable diagnostic settings after the runtime is released:
~~~bash
./envs/online/bin/python scripts/configure_evaluation_diagnostics.py --disable
~~~
This stops capture at the next runtime; it does not delete prior evidence.
Source rollback uses the recorded pre-sync backup, preserving other edits.
Do not replace 5k or begin Online from this six-Episode diagnostic. A resulting
candidate still needs old-scene retention and separately reserved independent,
frozen autonomous evaluation before controlled batches of Online learning.
