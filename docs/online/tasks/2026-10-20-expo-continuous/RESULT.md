# EXPO native continuous asynchronous delivery, 2026-10-20

## Delivered behavior

Optional Web model `expo-plug-left-2cam-native-async-hpc4-2000`, selected as `plug_left_2cam / EXPO / realtime-expo-ft-native-async-hpc4 / 2000`. Cobot runs inference; owned HPC allocation740494 runs the learner on four H100s at ACD1-67 until2026-10-20 22:34:20 CST. Session remains closed, model paused, robot commands published by this runtime0. Operators start Session and rollout/HIL through the existing Web controls.

Each eligible finished episode is independently uploaded and inserted into durable sampleable Replay. Learning uses the supplied native asynchronous BatchProcessor: uniform Critic, success-only Actor, native N8/n_edit8/filter_N8/filter_n_edit8, batch64/UTD20. No five-episode trigger, K30 quota, artificial10-episode backlog stop or two-version stop. Native120-second finished-event timeout is retained. Persistent local RNG and per-chunk/precache policy identities are preserved. Cross-machine upload/publication runs concurrently with inference; verified weights stage and adopt without restarting Actor. Inference/publication data movement is the hardware integration, not a claim of byte-identical single-process execution.

Cobot execution remains explicit50Hz/RTC/filtering, d5/Edit0.01 and existing gripper-feedback/HIL path. First formal CLI load used default10Hz/no filter; this was detected before Session or any command and corrected by paused reload using the normal model API with explicit execution_options. Learner/data remained intact. Future Web loads send the chosen checkbox settings; bare console model-load has different defaults.

## Production state and provenance

Continuation starts at original committed Actor20/Critic400/Replay5152:25 confirmed successful demonstrations3301 frames plus15 eligible online episodes1851 frames. Previous updates used custom532 sampling. Only subsequent updates follow supplied native async sampling. Original synchronous/bounded-async roots and checkpoints remain intact. The qualification synthetic30-frame episode is absent from production; Replay stays5152 through no-motion acceptance. Actor21/22 are real production updates on original Replay; qualificationActor25 is never installed into the formal run.

Core qualified adapter source7ef725f; publication timing6dea568; catalogue qualification metadata0d440da. Preserved separately authored rollout diagnostics974e06b/e4476f0 on Cobot;30 combined CPU integration checks passed after installation. HPC retains qualified7ef725f with no diagnostics-only compiler flags. Web continuous status6c1036b plus concurrent metrics/review changes is served via versioned static assets. Exact delivered Cobot continuous file SHA256 list is in outputs/expo-continuous-20261020/delivered-continuous-source-sha256.json.

HPC root: `/data/user/jhe724/jiaan/research-workspace/projects/rl-platform/outputs/expo-continuous-20261020/final`; production run`online`, test run`qualification`. Production orchestrator981213, GPU learner265023. Cobot runtime51530 and its child Actor are production; the earlier isolated probe26702 has exited. Cobot run`/home/agilex/jiaan/data/expo_ft/plug_left_2cam/native_continuous_hpc4_20261020`; profile`/home/agilex/jiaan/project/rl-platform/runtime/expo-remote/profile-native-async-hpc4.json`.

After Cobot reboot, Web was restored. By final readback ROS/CAN/three cameras/five arm nodes were already running with fresh feedback; those nodes were retained without restart/home. The user explicitly permitted startup while paused, but no redundant hardware startup was necessary. Recorder readiness=ok, no error, no active acquisition. Three camera frame ages around16-26ms at sampled readback. This is readiness evidence, not a newly recorded robot episode.

## Verification

- Broad qualified CPU suite338 passed/10 skipped/one known baseline fixture deselected. Earlier run reproduced missing previous/device_layout.py in test_expo_execution_migration; unrelated fixture was not repaired. Scalar checkpoint correction has focused tests plus full real-GPU restore qualification.
- Web115 tests passed; served HTML references native-continuous JS and HTTP returns200. No browser surface was available, so this does not claim visual browser acceptance.
- Final four-H100 isolated test: Actor20->25, Critic400->500; durable concurrent admission and sampleable insertion, duplicate handling, compatible save/restore, native continued updating all passed. Replay5182 exists only in isolated test. Scalar capture fix avoids identity aliasing of temporary Python scalar device arrays.
- Complete final updated Actor25 HTTP return:2,017,658,362 bytes,379.14s, range/checksum verification. Local real GPU same-PID stage/adoption5.35s;45 overlapping inference calls, median109.6ms/p95 149.6ms/max237.1ms. No Actor restart or stage-time rewarm, no ROS commands. These timings exclude real ROS scheduling/hardware and do not certify physical50Hz.
- Formal initial20 cache reused only after raw identity matching and exact authoritative compressed SHA256 validation for all189 files; no unverified shortcut or synthetic state.
- Formal paused load, explicit50Hz/RTC/filter settings, no fault, no pending upload,4 learner devices, recorder readiness and camera freshness verified. Final production hot-adoption receipt confirms Actor21->22 in the unchanged Actor51650 (same start_ticks); runtime51530, Learner22/Critic440, Replay5152. Explicit50Hz/RTC/filter retained, Sessioninactive, ready/paused, faultempty, published robot commands0.

## Performance and limits

Final steady native groups60.35-61.60s per20 Critic +1 VLA/edit; first/restored76.45-79.86s. Earlier new XLA compile125.76s produced217.74s group. Owned host capture~2s; checkpoint+snapshot writer17.80s overlapped subsequent learning. Native compilation, transfers, sampling and CPU EMA still create idle GPU intervals; sampled utilization averaged roughlyhalf. No claim of100% CPU/GPU/power saturation or universal fastest performance. Rejected whole-leaf EMA microbenchmark saved only~0.5s/group; candidate files are archived and not deployed.

Network remains dominant: full updated Actor is~2GB, measured6min19s on this link. Collection continues with the currently served version during transfer; intermediate publications coalesce. Web distinguishes training vs committed vs published vs downloaded vs actually served. A higher Learner number does not mean immediate local adoption. No completed episode for native120s leads to waiting; a new completed event resumes learning.

Campus DNS returned REFUSED. The task-specific A6000 alias expo-native-hpc-login uses the observed existing login peer10.120.81.30, verified as ACD-Login-1 with original strict host-key identity. No global DNS/network changes; details in RUNBOOK. Allocation expiry still applies; next session requires a valid resumed allocation, not just an old ready marker.

Physical rollout success rate, new HIL collection and50Hz motion are operator acceptance. No synthetic production episode or autonomous robot movement was used to pass delivery checks. Failed/aborted recording exclusion and retracting already admitted data remain explicit workflows; ordinary file deletion must not be assumed to retract Replay.

## Evidence

A6000 root `/data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/outputs/expo-continuous-20261020/`: final-hpc-qualification.json, full-http-return.json, local-actor.json, production-resume-ready.json, formal-paused-status.json, recorder-and-console-readback.jsonl, served-static-assets.json, formal-hot-adoption.json and final-acceptance.json. Cobot mirror`/home/agilex/jiaan/project/rl-platform/runtime/verification/expo-continuous-20261020/`. Ready gate installed only after GPU/transfer qualification and formal remote restore. Old return coordinator reports a last-receipt DNS failure; its actual download/probe passed and the receipt was later written over the verified login route. Do not mistake that intermediate failure for lost weights.

## Final observed handoff

At2026-10-20 18:32 CST, formal production Actor22 was fully downloaded and adopted with unchanged Actor PID51650/start_ticks290524; runtime51530 stayed paused and Sessionclosed. Learner22/Critic440, Replay5152,15 eligible online episodes, no pending upload or network/install fault. Production Actor22 transfer224.33s (3min44s), stage/adoption5.04s. Thus observed full-transfer range is224-379s across these tests, not a fixed379s estimate. Local publication adopted automatically; no manual model restart was needed for21->22. Final-acceptance.json asserts actual execution50Hz/RTC/filter, zero robot commands, recorderready and freshthreecamera/fivearm feedback.

Owned test GPU processes, qualification orchestration, benchmark and GPU monitors have ended; formal read-only adoption observer exited after success. Retained only production Learner/orchestrator/tunnels, Cobot Web/Actor/runtime and pre-existing hardware nodes. Field lifecycle released to user. HPC740494 still expires22:34:20; no allocation cancellation performed.
