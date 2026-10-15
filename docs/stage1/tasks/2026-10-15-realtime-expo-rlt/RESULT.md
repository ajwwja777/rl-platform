# realtime_expo two-wrist RLT delivery

Verified at 2026-10-15T19:19:07 (A6000 local time): Web frozen and Online entries installed; both no-motion loads passed. Current selected runtime is `realtime-expo-rlt-online`, ready/disarmed, policy paused, emitted_commands=0. No episode, robot motion or home command was started.

## Delivered identity

| Item | Value / verification |
|---|---|
| Web | http://10.12.166.118:8015/ ; scene `plug_left_2cam`, family RLT, method `realtime-expo-rlt` |
| Frozen | `realtime-expo-rlt-frozen`, Learner disabled, initial Actor3500 / learner7000 |
| Online | `realtime-expo-rlt-online`, CPU Learner enabled; 7000 / Actor3500; initial pending budget0 |
| Stage1 | new realtime_expo checkpoint4999; NOT old plug_v3 supported-MC30/7k |
| Input | LEFT 7D; only left/right wrist images, resized/padded224x224; middle/head input rejected |
| Actual subscriptions | `/camera_l/color/image_raw`, `/camera_r/color/image_raw`; no `/camera_f` subscription by RLT |
| Clock | demonstration/reference/Actor/Replay10Hz, H50/C10; actual selected publication50Hz + RTC + EMA(tau.08s) |
| Raw Stage1 assets | `/home/agilex/jiaan/data/rlt/realtime_expo/stage1_4999` ; 35 files / 15,428,278,977 bytes SHA matched HPC -> A6000 -> Cobot, original parameter asset dtype retained |
| Release | `/home/agilex/jiaan/data/rlt/realtime_expo/release_20261015/{frozen,online}` |
| Recordings | `/media/agilex/Getea1/jiaan/data/datasets/realtime_expo/recordings/rlt/{frozen,online}` ; model-specific directory |
| Online evidence | `/home/agilex/jiaan/project/rl-platform/outputs/realtime-expo-runtime/online/online/{logs,metrics,traces}` |
| Cobot receipt | `/home/agilex/jiaan/project/rl-platform/runtime/verification/realtime-expo-rlt-20261015/final-acceptance.json` |

The loaded Online entry is paused for the operator. The frozen entry never trains. The Online entry automatically trains when newly committed collection transitions arrive and publishes/saves every50 learner updates. No automatic homing/reset is registered for this left task. Evaluation use is latched separately and does not commit Replay.

## Data / selection / actual training

49 user-confirmed successful human demonstrations;44 TRAIN/5 repeated DEV, no independent TEST. TRAIN551 expert transitions and DEV62 isolated transitions. Initial Online Replay contains TRAIN only, all expert/human successful examples; this is not 551 independent successes or autonomous successes. Uniform transition sampling is used; future success/failure/new-data proportions are NOT fixed8:2 and must be measured from arrivals and actual batches.

Stage1 5k updates on4H100 completed55m17s; full five-checkpoint evaluation selected4999. Warmup native actor/critic architecture/TD retained, delta_chunk6 joints + absolute variable gripper; TRAIN-derived normalization. WarmupBC10/Q.1, OnlineBC5/Q.1, delta10, reference-dropout.5, actor/criticLR1e-4, batch128, gamma.99, target-tau.005, actor every2 updates, fixed_std.002. Online new-arrival UTD1, warmup boundary7000/adds551; every50 updates saves a full checkpoint and a native Actor history snapshot. These are project configuration choices, not a claim of matching every upstream default.

Warmup 500..10000 checkpoints compared with deployed float32 single-input inference. Selected7000: TRAIN joint5.0545mrad/gripper1.6742mm; reused DEV joint18.7425mrad/gripper4.6135mm. Reference DEV18.8040mrad/3.7101mm. Joint changes are small, gripper DEV retention is worse than Reference. Single-factor delta0 control did not resolve this (7k DEV18.7913mrad/4.6126mm), so it was not adopted. Selection is descriptive after observing results, not an independent-test claim. Episode-bootstrap intervals and plots are saved; overlapping windows are never independent uncertainty samples.

## Verified software and field boundaries

- 99 RL/data/async/publication tests, 4 dedicated release guards, 11 Web routing/directory tests; after the field phase fix, 11 left-task tests passed (sets overlap, not an additive test count).
- Native exact save/resume: all state leaves equal for uninterrupted4 vs2+restore2. Export matches deployed evaluator exactly for32 inputs on A6000 and Cobot CPU/NumPy1.26.4.
- Isolated copied branch: initial budget0; synthetic50 arrivals -> exactly50 updates -> step7050/Actor3525; saved checkpoint/history, observed reload, all state leaves restored exactly. No production Replay used.
- Actual Cobot Online: Replay size/adds551, global7000/Actor3500, pending0, ready_for_online=true, CPU Learner. Frozen real load explicitly reported Learner disabled.
- Actual Stage1 two-wrist GPU and RTC warmup passed. First warm load measured plain84-89ms and RTC124-125ms; this is inference-only timing, not closed-loop deadlines or success rate. All field tests stayed disarmed/paused with zero robot commands.
- Frozen assets/norms and both Replay journals stayed byte-identical. Native Online Learner initialization re-exported Actor with current online configuration/field serialization, changing file SHA; all Actor parameter leaves and version7000/3500 remain exactly equal to frozen seed. See initial-republication.json.

## Failures resolved and retained evidence

1. First export comparison differed up to448 microrad because Replay quantizes reference to float16 while serving uses float32. Re-evaluated all candidates on actual serving inputs; did not relax equality thresholds or change production Replay formats. Executed action storage remains float32.
2. NumPy2 checkpoint namespace was unreadable by field NumPy1.26.4. Portable public ndarray constructors preserve dtype/value and now load/compare exactly; no field environment upgrade.
3. First frozen field launch exited because load mode `frozen` was sent as native collection phase. Fixed to native `online` phase while retaining frozen directory, deterministic execution and disabled Learner; added a fail-closed phase guard and regression. Stage1 was retained and no robot command occurred. Subsequent frozen and Online loads passed.
4. Initial test attempts used missing test names / missing isolated dependencies; those are retained in logs and are not counted as passes. Formal successful commands and receipts supersede them.

## Evidence gaps / acceptance

Software/data identity and no-motion deployment are verified. Autonomous success, moved-target generalization, closed-loop smoothness/deadlines and sustained Online improvement are **insufficient evidence**. All expert training examples are positive; this dataset cannot establish failure-action Critic ranking. The observed gripper DEV gap is not hidden by low TD loss or rising Q. The current release is for controlled first rollouts/Online collection, not unconditional success or monotonic-improvement certification.

Prior EXPO/old RLT weights, Replay, hardware services and detached EXPO Learner return/monitor were preserved. Initial frozen branch is retained for comparison/rollback; each Online checkpoint/history remains separately addressable. No automatic rollback or new experiment runs are left queued.

## Sources / commands / records

Source base RL243ffdc + phase fix9e05361 (both pushed); Web installation manifest records84e4d73. Field SHA manifests provide exact installed files. Fixed upstream8cef77e remains unchanged. Installation first verified known historical field file identities and used the shared model-operation lock, backup and rollback; only idle Web service was refreshed.

All durable analysis: `/data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/outputs/realtime-expo-rlt-20261015/`:
- `stage1-comparison/`: source metrics/selection and two figures.
- `warmup-native-10k/`: complete config, logs, full checkpoints, actual sampling counts, repeated-DEV selection, `warmup_comparison.png`, `warmup_q.png`.
- `warmup-delta0-10k/`, `warmup-single-factor-comparison.json`: rejected single-factor alternative.
- `release-native-validation.json`, `release-bundle-cobot-portable-sha256.json`, `selected-stage1-cobot-transfer.json`.
- `field-receipts/{source-install,phase-fix,final-acceptance,initial-republication}.json` and actual frozen/Online status.

Reproducible tool entrypoints in project scripts: `train_realtime_expo_warmup.py`, `evaluate_realtime_expo_actor.py`, `select_realtime_expo_warmup.py`, `validate_realtime_expo_release.py`. Exact executed installation/verification commands and configurations are in the output root and field logs. Laptop root remains unchanged; no project artifacts stored on laptop.
