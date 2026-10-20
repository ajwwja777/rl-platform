# EXPO bounded asynchronous Online

This runbook describes the user-approved optional mode. Read STATUS.md for actual qualification and current field state; file presence alone is not acceptance.

## Web selection and operator workflow

Scene: plug_left_2cam. Model: EXPO. Method: realtime-expo-ft-532-hpc4-async. Base version:2000. Model ID:expo-plug-left-2cam-532-hpc4-async-2000.

The initial asynchronous continuation comes from the real committed Actor20, Critic400,15 eligible online episodes,1851 online transitions and25 successful demonstrations/3301 transitions. The base version2000 is the SFT origin, not the current online Actor number. Synthetic qualification journals and Actor25 are isolated and never become the production continuation.

Load and wait for readiness, start Session, then use the existing rollout/pause/HIL/success/failure workflow. An eligible finalized episode is persisted locally first, uploaded in the background and admitted durably on HPC. Upload acknowledgment is distinct from completion of training. Unknown/aborted/ineligible episodes earn no learning budget.

The local Actor keeps one version for the whole episode. The learner can update while collection continues. A complete committed inference bundle downloads in the background, with checksums and resumable HTTP Range. Installation requires no active episode, paused execution and no unfinished inference future. It uses the existing Actor process, but the CPU-to-GPU replacement and warmup still take measurable time; there is no promise of a zero-length boundary pause.

Warmup10,5-episode trigger,K30,UTD20,global batch64,Critic50/30/20,success-only VLA,edit0.01 and RTCd5 remain unchanged. Async changes execution overlap and RNG separation; it does not grant an unlimited optimizer budget. Received transitions can be reused through ordinary off-policy Replay sampling; behavior policy version remains recorded.

## What the counters mean

- Learner: optimizer update count, including work in progress.
- Published: latest complete inference bundle associated with durable committed training state.
- Downloaded: complete verified local bundle available for installation.
- Served: local policy actually used to start the next episode.
- Upload queue: finalized local journals without a durable remote admission receipt.
- Untrained: union of pending uploads, admitted/unprocessed journals and committed partial/deferred batch identities. Acknowledged IDs remain counted until a fresh admission revision confirms remote visibility.
- Lag: published generations behind the current local Actor, not the numerical difference between optimizer step counts.

New episodes are blocked at10 outstanding episodes or2 observed published generations of lag. An active episode may finish. This prevents unbounded disk/replay age growth when collection is faster than learning; it does not promise indefinitely uninterrupted collection. If the 532 success/failure pools are missing, their earned credit is retained and the explicit missing-pool/backlog condition needs resolution. This continuation already has both outcome pools.

After a WAN interruption, bounded local collection can continue; completed journals retry and duplicate admissions remain idempotent. Do not delete outbox or receipt files to clear a warning. After a crash between journal finalization and Actor receipt, a complete labeled journal is recovered with an explicit missing-final-RNG audit marker; no fake RNG is created and the learner's independent RNG is unchanged.

## Locations

A6000 main code: /data/LFT-W02_data/jiaan/jiaan/projects/rl-platform

A6000 evidence: /data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/outputs/expo-async-20261020

Cobot runtime: /home/agilex/jiaan/data/expo_ft/plug_left_2cam/native_remote_532_hpc4_async_20261020

Cobot recordings: /media/agilex/Getea1/jiaan/data/datasets/plug_left_2cam/recordings/expo-532-hpc4-sft2000-async

Cobot profile: /home/agilex/jiaan/project/rl-platform/runtime/expo-remote/profile-532-hpc4-async.json

HPC root: /data/user/jhe724/jiaan/research-workspace/projects/rl-platform/outputs/expo-async-20261020

HPC production Replay/checkpoints: the online/ directory under that root. qualification/, test-upload/ and verification/ are isolated tests and must not be imported into production.

## Resume on a future owned allocation

Retain the HPC root, native assets, seed demonstrations, checkpoints and JAX cache. On the HPC login host, use an owned RUNNING single-node four-GPU allocation with sufficient CPU/RAM; the launcher checks scheduler ownership and four visible GPUs. Never reuse someone else's job ID.

```bash
H=/data/user/jhe724/jiaan/research-workspace/projects/rl-platform/outputs/expo-async-20261020
JOB=YOUR_OWNED_FOUR_GPU_JOB_ID
python3 "$H/source/scripts/resume_expo_hpc_learner.py" "$JOB" --root "$H" --check
nohup python3 "$H/source/scripts/resume_expo_hpc_learner.py" "$JOB" --root "$H" > "$H/resume-${JOB}.log" 2>&1 < /dev/null &
```

Wait for resume-ready.json to name that job and the expected async model/committed Actor. The launcher updates learner.json and gateway.json to the new allocation. The Cobot named route expo532_hpc4_async stays fixed. Update its human-readable learner_host job label if needed. A stale readiness marker or old Web label is not proof that a new allocation is running. Load the model through Web only after checking the live remote status. Robot motion remains an operator action.

## Rollback and release

Finish/pause the current episode and end Session. Wait for any desired training commit before stopping the task-owned learner; never treat an in-progress optimizer step as a committed checkpoint. The original synchronous run and catalogue entry are retained. To roll back, unload the async local runtime, stop only its owned learner orchestrator, resume the original synchronous root on the owned allocation and load realtime-expo-ft-532-hpc4. It resumes the original Actor20 state, not later asynchronous updates. A continuation back from later async learning requires an explicit audited copy.

HPC allocation expiry stops learning; durable journals/checkpoints remain. Release a finished reservation through its scheduler job ID after confirming the field is no longer using it. No automatic robot restart or continuation is performed.
