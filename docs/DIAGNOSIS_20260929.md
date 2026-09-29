# RLT learning diagnosis — 2026-09-29

## Structure and locations

~~~text
rl-platform/
├── methods/openpi_rlt/                 # Original method and runtime entry
├── third_party/openpi-rlt/             # Pinned upstream
├── integrations/cobot_runtime/
│   ├── replay_audit.py                 # Provenance and actual sampled batches
│   └── analysis.py                     # Read-only telemetry aggregation
├── scripts/
│   ├── analyze_replay.py               # Composition, PCA and clustering
│   ├── diagnose_online_learning.py     # Version audit and sampling experiments
│   ├── analyze_rl_sensitivity.py       # Gradients, input ablations and Q probe
│   └── analyze_visual_sensitivity.py   # Recorded-image occlusion
└── outputs/                           # Regenerable evidence, ignored by Git
~~~

A6000 source/docs: /data/LFT-W02_data/jiaan/jiaan/projects/rl-platform.
Cobot: /home/agilex/jiaan/project/rl-platform.
Cobot reports: outputs/rlt/plug_v3_yyshadow/analysis/:
replay_composition.json, learning_diagnosis.json, rl_sensitivity.json,
visual_sensitivity.json. Small result copies are on A6000 in
outputs/rlt-diagnosis-20260929/. No dataset or weight backup was added.

Web 8015 → Analysis presents composition, version audits, experiments,
loss/input sensitivity, camera occlusion and posture/action clusters.
Training retains progress, publication and principal loss curves.

## Recommendation

The evidence supports investigating scarce NEW AUTONOMOUS success coverage and
Critic action ranking before simply adding updates or prescribing 90% successes.
No production weights, losses, reward, UTD, Replay or control settings changed.

1. Interleave frozen Reference, Warmup5000 and one fixed Online Actor under matching
   initial conditions. Record Actor hash/version; keep it fixed within each trial.
   Ten trials per model are an initial screen, not adequate proof; expand the
   sample and report uncertainty before choosing a winner.
2. Reserve NEW diagnostic episodes that never enter training. Separate autonomous
   success, HIL rescue and failure. Cover actual contact/alignment failure modes,
   including near-success failures and recent autonomous successes, instead of
   repeatedly adding the same old demonstrations.
3. Check HIL timing, action representation and reward alignment against videos,
   then test whether Q ranks physically reasonable corrections higher.
4. Keep original sampling as baseline. 70%/90% remain candidate comparisons;
   the offline differences do not justify a production default change.
5. Audit each retained published version on the same fixed set. Actual task
   success requires separate robot trials. GPU audits are explicitly run below,
   not automatically scheduled alongside live inference.

## Replay composition

Snapshot: 3,917 transitions, 277 episodes; 194 successful and 83 failed episodes.
No duplicate phase/episode/step identities. An episode's terminal label annotates
all its windows: non-terminal transition.success=0 is NOT a failed episode.

| Dimension | Transition count / share |
|---|---|
| Outcome | success 2,669 / 68.14%; failure 1,248 / 31.86% |
| Phase | Warmup 2,567 / 65.53%; Online 1,350 / 34.47% |
| Age | Warmup 2,567; older Online 1,042; recent Online 308 / 7.86% |
| Position | early 1,391 / 35.51%; middle 1,312 / 33.50%; late 1,214 / 30.99% |
| Source | BASE 1,134; RL 896; HUMAN 1,850; MIXED 37 |
| Human actions | 1,887 chunks / 48.17%; 47.59% of stored control steps |

Recent means the last 20 Online episode-ID window, not minutes.
Position means rank thirds of stored windows, NOT semantic insertion stages.
HUMAN includes old demonstrations, not exclusively online interventions.
Different dimensions overlap; do not add them.

| Group | Episodes | Transitions |
|---|---:|---:|
| Warmup successful autonomous | 21 | 360 |
| Warmup failed autonomous | 38 | 584 |
| Warmup successful human-containing/demo | 144 | 1,609 |
| Warmup failed human-containing | 1 | 14 |
| Online failed autonomous | 44 | 650 |
| Online successful human-containing | 23 | 598 |
| Online autonomous success | 6 | 102 |

New autonomous success contributes only 2.60% of the total pool.
6/73 is the autonomous success fraction of these Online episodes. It is not
comparable to the user's earlier fixed-Reference ~40% evaluation: scene difficulty,
exploration and changing model versions differ.

### Every actual batch

Upstream samples 128 TRANSITIONS uniformly with replacement. Several windows
from one episode can appear and a transition can repeat. Success share is
expected near 68.14% if this pool stays fixed, not guaranteed in each batch.
There is no unconditional 50% success rule. Human BC applies to the corresponding
control-step mask, not automatically to every action in an assisted episode.

The project entry observes the original LearnerService.sample_batch return.
It neither samples again nor touches RNG/training arrays. Actual identity,
composition, learner step and Actor version are written after every successful
update to outputs/rlt/plug_v3_yyshadow/online/metrics/batch_composition.jsonl.
The page shows the latest 128 batches; the JSONL retains the recorded history.
Replay metadata composition refreshes when the journal grows during training;
PCA/clustering remains a separate offline job.

Historical identities were not logged and cannot be reconstructed. Production
auditing activates at the NEXT Learner launch through the project entry.
Unresolved/duplicate identities produce unknown, not invented labels.

## Completed sampling experiment

Initialization: original Warmup5000. Split Online by whole episode, stratified by
outcome and human presence; 19 episodes / 381 transitions held out, containing
1,415 HIL control steps. These Online episodes are excluded from this experiment's
optimization; Warmup remains training data. Split seed42, training seeds41/42/43.
Four variants × three seeds × 2,000 updates = 12 runs.

Original train_step, action normalization, BC5, Q0.1, delta10 and gamma are unchanged.
Only sampling composition changes. Actual shares are reported (128-sample integer
rounding can differ slightly from the requested percentage).

| Sampling | Mean HIL six-joint MAE, rad | early / middle / late Q outcome AUC |
|---|---:|---|
| Uniform | 0.002154 | 0.545 / 0.659 / 1.000 |
| 50% success | 0.002156 | 0.572 / 0.500 / 0.898 |
| 70% success | 0.002092 | 0.572 / 0.705 / 1.000 |
| 90% success | 0.002085 | 0.568 / 0.837 / 1.000 |

Reference HIL MAE=0.002233, initial Warmup Actor=0.002547.
70%/90% improve this proxy by ~2.9%/~3.2% versus uniform; their difference is small.
Three seeds and few held-out episodes do not establish robot success improvement.
Repeatedly examined holdout is now a development set; use a new final test set.
Q outcome AUC does not prove useful action ranking. 90% may reduce failure coverage.

An earlier 500-update screen did not outperform Reference; 2,000 updates did
on this proxy. This is not an argument to increase production updates indefinitely.
Experimental weights were not saved or registered for deployment.

## Retained published versions

14 Actors at Learner5000–11500, every500 steps, audited on the same Online cohort.
These Actors may already have trained on it: this is a training-seen retrospective
audit, NOT independent validation.

HIL MAE at5000/10000/10500/11500:
0.002547 / 0.002018 / 0.001973 / 0.001991.
Better imitation fit is not proof of better task execution.
Matching Critic early outcome AUC: 5000=0.909, 10000=0.477, 11000=0.534.
Only matching-step Critic checkpoints are used; unretained5500 etc are left blank.

Last observed learner11750/internalActor5875; published snapshot11500/Actor5750.
The latest retained complete checkpoint used for gradient probes is11000,
not11750. Historical timestamps do not prove a process is currently running.

## Influence and input sensitivity

Checkpoint11000; at most128 transitions per group, fixed seed. Weighted Actor
gradient norms: BC3.967, minus-Q1.051; cosine=-0.349, Online cosine=-0.410.
This is local gradient conflict, not proof of which loss to reweight.
Similar loss magnitudes do not imply equal update pressure.

Along recorded human correction: alpha0 is Actor, alpha1 is the human action.
Mean Q1 falls0.787→0.690; alpha-1 gives0.815. This mixed, training-seen local
probe motivates checking Critic/action/HIL alignment. Negative alpha can be
out-of-distribution; never execute that direction because it has higher Q.
Per-example causal contribution to final robot success has NOT been measured.

Input ablation six-joint RMS: z_rl→zero0.003685rad,
proprio→median0.000038rad, reference→zero0.001338rad.
Replacement inputs can be out-of-distribution; sensitivity does not justify
removing an input.

## Visual sensitivity, not attention

Real Stage1, recorded RGB/state, fixed seed42; full-camera and4×4 patch mean-color
replacement. Six frames,306 perturbed forwards; original/overlay toggle in web.
Online episode_000001 failure and000004 human-containing success, first/mid/last.
Recorder IDs are not assumed to equal Replay IDs.

Failure last frame whole-camera action RMS:
right-wrist0.00153rad, high0.00005, left-wrist0.00020.
Dependence changes over frames. Six frames cannot establish a useless camera
or correct/incorrect attention. Heat color encodes ACTION CHANGE, not Attention.

Small RLT Actor/Critic are MLPs; cached z_rl cannot recover backbone patch attention.
Actual layerwise attention/rollout was not implemented. A future pinned Stage1
offline exporter should first prove unchanged actions, then combine attention
with occlusion and failure strata; no costly production inference hook was added.

## Reproduce without robot motion

Use Cobot when the model is idle and GPU resources are available. Do not compete
with live inference. Existing frozen environments suffice; the Stage1 script uses
the registered machine-a-py311-overlay. On A6000/another machine, provide registered
journal, checkpoints, normalization and recorded RGB assets; Git excludes those.
This batch did not copy them to A6000.

~~~bash
cd /home/agilex/jiaan/project/rl-platform
envs/online/bin/python scripts/analyze_replay.py

envs/online/bin/python scripts/diagnose_online_learning.py \
  --updates 2000 --seeds 41,42,43 \
  --output outputs/rlt/plug_v3_yyshadow/analysis/learning_diagnosis.json

# Refresh audits on the saved episode cohort; preserve sampling experiments
envs/online/bin/python scripts/diagnose_online_learning.py \
  --versions-only \
  --output outputs/rlt/plug_v3_yyshadow/analysis/learning_diagnosis.json

envs/online/bin/python scripts/analyze_rl_sensitivity.py \
  --output outputs/rlt/plug_v3_yyshadow/analysis/rl_sensitivity.json

envs/stage1/bin/python scripts/analyze_visual_sensitivity.py \
  --checkpoint /home/agilex/jiaan/data/rlt/plug_insertion/reference_4999 \
  --recordings \
  /media/agilex/Getea1/jiaan/data/datasets/plug_insertion/recordings/rl-platform/rlt/online/episode_000001.hdf5 \
  /media/agilex/Getea1/jiaan/data/datasets/plug_insertion/recordings/rl-platform/rlt/online/episode_000004.hdf5 \
  --output outputs/rlt/plug_v3_yyshadow/analysis/visual_sensitivity.json
~~~

The report records journal/init-checkpoint SHA256, split/training seeds and version
sources. Keep JSON with repository revision. Scripts save metrics, not experimental
deployment weights; no production checkpoint/Replay writes occur.

## Primary sources and limits

- [RLT paper](https://arxiv.org/html/2604.23073v1) gives the frozen-VLA/RL-token
  method. This project uses the [openpi-RLT reproduction](https://github.com/Yyshadow/openpi-RLT)
  on pi0.5; it is not identical to the paper's model/experiments.
  No universal fixed success-majority requirement was established.
- [HIL-SERL](https://hil-serl.github.io/): demonstration/online mixing and
  intervention are relevant; demonstration ratio is not success-outcome ratio.
- [VLA-Trace](https://vla-trace.github.io/): combines representation analysis,
  attention intervention and input perturbations. We implemented recorded-image
  occlusion, not every attention technique.
- [Attention is not Explanation](https://aclanthology.org/N19-1357/) and
  [Quantifying Attention Flow](https://aclanthology.org/2020.acl-main.385/):
  raw attention alone has explanatory limits; use interventions and validate
  applicability to this VLA before treating a heatmap as a failure explanation.

Validation: 10 lightweight regressions and one frozen-JAX real-Learner equality test passed. Web release checks and verification boundaries are in cobot-web/docs/MIGRATION.md. No independent robot success test was run in this batch.
