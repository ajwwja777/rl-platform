# Native continuous EXPO operational handoff

Status: four-GPU and local hot-adoption qualification passed; production loaded paused. Final actual served version is recorded in RESULT.md. Robot motion remains operator initiated.

## Web operation

Entry: plug_left_2cam / EXPO / realtime-expo-ft-native-async-hpc4 / 2000.
Model id: expo-plug-left-2cam-native-async-hpc4-2000. Keep explicit d=5 and edit scale=0.01.
After acceptance and source gate installation, refresh Web. Load the entry, start Session, then use existing rollout/HIL and success/failure controls. A complete eligible labeled episode is uploaded, durably admitted and inserted into sampleable Replay independently of optimizer execution. Aborted/invalid recordings are not training examples. Do not assume deleting a recording retracts already ingested Replay; use the explicit exclusion workflow.

No batch-of-five trigger, no pending-ten gate, no two-version lag gate. Native BatchProcessor continues sampling existing Replay,20 Critic updates and one VLA/edit update per native optimizer group. Native120s without an observed finished-episode event pauses learning; a new complete episode resumes it. This is not a requirement to collect a full new batch. Native prefetch means insertion is not proof that an example has already been sampled.

The UI distinguishes: pending upload, admitted/sampleable Replay, sampling, training update (including compilation / parameter synchronization), live learner step, committed checkpoint step, published snapshot, downloaded snapshot, GPU staging and actual served Actor. Only Served proves local inference adoption. Transfer failure keeps the previous verified local Actor; field faults remain visible. Updated weights load in background and are atomically adopted on inference or idle boundaries, preserving live RNG and auditing per-chunk/episode versions. No Actor process restart is required.

## State and source

Local runtime: /home/agilex/jiaan/data/expo_ft/plug_left_2cam/native_continuous_hpc4_20261020
Local profile: /home/agilex/jiaan/project/rl-platform/runtime/expo-remote/profile-native-async-hpc4.json
Gate: runtime/expo-remote/native-async-hpc4-ready.json (qualification evidence installed; it does not by itself prove a current live HPC allocation)
Remote continuation starts from original Actor20 and5152 Replay frames, including25 demonstrations/3301 frames and15 eligible online episodes/1851 frames. Prior source used custom532 sampling; only subsequent training uses supplied native async settings. Qualification's synthetic30-frame episode is isolated and must never enter production.

Native identity: async script SHA256 dbf56b0413d8f10697950068b12dd10fe2d2c117317ed9a19c69206f73252d1a; native N8, n_edit8, filter_N8, filter_n_edit resolves8; uniform Critic and success-only Actor per supplied native config. Shared CPU target-Actor EMA, microbatch/rematerialization and four-GPU placement are explicit memory/hardware adaptations. Cross-machine publication requests coalesce snapshots; not every optimizer step is necessarily transmitted. No learning rate, loss, reward, sampling ratio or candidate count is reduced to improve timing.

## Checkpoint and transport

Optimizer captures owned host copies at a group boundary, then resumes while one background writer persists compatible resumable training state and inference snapshot. No live donated GPU leaves are retained by writer. Replay has an independent durable ledger, overlaid during recovery so episodes arriving after capture are not lost. Exported files use checksums, range resume, atomic visibility and committed publication identities. Initial20 local payload reuse is allowed only after authoritative raw SHA256 matching; updated qualification Actor must traverse the complete remote HTTP path.

Rollback: retain old synchronous/bounded-async entries, profiles, run directories and checkpoints. Unload new local Actor with Session stopped; stop only owned new remote orchestration; resume the recorded old profile/run using its verified source. Never start two learners on the same Replay/checkpoint root.

## HPC lifecycle

Allocation740494: ACD1-67,4H100,48CPU,384GB, expiration2026-10-20 22:34:20 CST. The learner can outlive Web/Actor unload, but cannot outlive its allocation. Next allocation must be owned, one-node four-GPU and source/profile-validated before running resume_expo_hpc_learner.py <job_id> --root <qualified-production-root>. Use --check first. Update gateway binding and readiness evidence after restore; do not reuse an old job id or stale gate as evidence of availability. Do not stop other jobs.

## Acceptance boundary

CPU/Web tests, isolated4GPU concurrent insertion/update/save/restore, full returned snapshot checksum/range transfer, local real-GPU stage+inference continuity, and final paused Web load are required. No test starts robot motion. Field rollout success rate and physical continuous50Hz execution remain a user-run acceptance step. Exact production root, commits, timing distributions and final status are recorded in RESULT.md after acceptance.

## Qualified production connection and timings

Production HPC root: /data/user/jhe724/jiaan/research-workspace/projects/rl-platform/outputs/expo-continuous-20261020/final. Learner run: final/online. Qualification run final/qualification is isolated and must not be used for production. Production transport config on A6000: projects/rl-platform/outputs/expo-continuous-20261020/relay.json. Cobot still reaches the restricted A6000 gateway using its existing identity.

On 2026-10-20 campus resolvers10.90.63.2/3 returned REFUSED for the HPC login name. A task-specific A6000 SSH alias expo-native-hpc-login uses10.120.81.30, observed from the already established login SSH socket and verified as ACD-Login-1 against the existing hpc3login.hpc.hkust-gz.edu.cn host key with StrictHostKeyChecking=yes. Global DNS and other SSH aliases were unchanged. Revalidate this route if the login endpoint changes; do not disable host-key checking.

Measured final native optimizer groups:60.35,61.19,61.60 seconds after first/restore groups76.45-79.86 seconds. An earlier run encountered a125.76-second XLA compile and a217.74-second group. Checkpoint owned-host capture about2seconds; checkpoint plus Actor file writing17.80seconds in background. These are measurements, not a bound or full-utilization guarantee. Existing native transfer/EMA/sampling/compilation can leave GPUs idle; measured mean utilization was abouthalf during sampled groups. An isolated whole-leaf EMA-copy experiment improved only~0.5seconds/group and was not adopted.

Complete updated2,017,658,362-byte HTTP transfer to Cobot:379.14seconds (6min19s) in qualification; production Actor22 later224.33seconds (3min44s). Local GPU stage/adopt:5.35seconds while45 overlapping inference calls continued, maximum237ms; same Actor PID, no rewarm/restart. Those inference measurements exclude ROS/hardware and do not certify physical50Hz. Native training can continue while transport uses the prior local policy, and the next transfer coalesces to a newer published version. A reported Learner update is not proof the robot is already using it.

Final paused handoff: runtime51530/Actor51650, served22/Learner22/Critic440, Replay5152; formal same-process21->22 adoption5.04s passed. Explicit50Hz/RTC/filter retained, Sessionclosed, zero robot commands. See RESULT.md and final-acceptance.json.
