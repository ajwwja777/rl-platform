# RLT credit and execution experiments

## Project structure

```text
rl-platform/
├── configs/experiments/credit_ablation.json       # named offline recipes
├── configs/experiments/runtime_profiles.json      # explicit runtime registrations
├── configs/deployment_models.json                # shared web / CLI model catalog
├── methods/openpi_rlt/experiments/
│   ├── credit.py                                # sampling + optional Critic target
│   ├── runtime.py                               # isolated spawned Learner adapter
│   └── rtc.py                                   # 7D bridge to existing VLA RTC
├── scripts/experiment_credit_assignment.py       # reproducible offline comparisons
├── scripts/prepare_credit_candidate.py           # separate resumable branch
├── scripts/validate_credit_candidate.py          # restore / UTD / publish / restart
├── scripts/validate_rtc_reference.py              # recorded-image RTC, no robot
└── scripts/diagnose_action_timing.py              # noise units / timing / resampling
```

Main Git/docs: /data/LFT-W02_data/jiaan/jiaan/projects/rl-platform on A6000.
Deployment: /home/agilex/jiaan/project/rl-platform on Cobot.
This batch changes no fixed upstream source, original reward journal, HIL mask,
Actor loss, UTD, training rate, or default model selection. MC profiles explicitly
change the Critic target and are NOT claims of faithful baseline reproduction.

## Findings, 2026-09-30

All 21 runs started at the original Warmup 5000 state, trained 2,000 updates with
seeds 41/42/43, and excluded the same 19 Online episodes (381 transitions).
Replay has 3,917 transitions; training pool has 3,536. This repeatedly inspected
cohort is a development set, not an untouched test set. The published online
versions may have seen these episodes and cannot be fairly compared as held-out
competitors. No robot success rate was measured.

Three-seed means:

| Profile | HIL joint MAE, rad | Early Q outcome AUC | MC return RMSE |
|---|---:|---:|---:|
| Native uniform | .00215369 | .5417 | .2618 |
| Terminal windows only | .00242109 | .4735 | .2386 |
| 50% tail mixture | .00218868 | .5644 | .4197 |
| Episode balanced | .00219356 | .5189 | .2823 |
| Uniform + MC10 | .00214354 | .5720 | .1565 |
| Uniform + MC30 | .00214182 | .6667 | .1135 |
| 25% tail mixture + MC10 | .00217725 | .5568 | .2054 |

Terminal-only sampling discards most early/middle-state learning and worsened
HIL fit by about 12%. Tail-heavy sampling also did not improve the action metric.
MC30 improves the tested early Q ranking, but action improvement versus the
matched uniform run is only about 0.55%. It is a candidate for controlled testing,
not an established better robot policy. Its HIL MAE is not better than every
previous training-seen online snapshot.

The prior ratio experiments are in docs/DIAGNOSIS_20260929.md. Overall Replay
already contains about 68% successful transitions when labeled by episode outcome;
new autonomous successful transitions were only 102/3917 (2.60%). More success
data is not equivalent to repeatedly sampling the same assisted demonstrations.
Prioritize diverse autonomous successes and informative HIL corrections, then
evaluate on newly collected episodes.

### Sparse success and credit

The native target is discounted chunk reward plus gamma^chunk_len times target
Q on nonterminal windows. A middle transition with success=0 still receives TD
credit. The loss does not directly consume transition.success.

The experiment derives G = outcome * gamma^(terminal_control_step - start_step)
from complete, continuous episodes and mixes:
target = (1 - w) * native_TD + w * G.
MC10/MC30 mean w=.1/.3, not success sampling ratios. Rewards on disk are unchanged.
Unknown outcomes, duplicate identities, gaps or inconsistent terminal rewards are
not invented. Online missing/incomplete metadata retains native TD. This is an
off-policy behavior-return regularizer: it can bias Q and must be evaluated.
Zero weight returns the original jitted training function exactly.

### RTC, horizon and noise

The shared flow sampler remains owned by VLA:
 /home/agilex/jiaan/project/vla-platform/integrations/cobot/pi05/dagger/common/rtc_overlay/rtc_openpi/sampler.py

RLT adds a 7D physical-action transform bridge. An optional prefix-cache argument
reuses the existing encoder computation. Without RTC requests, baseline actions
and RL tokens were exactly repeatable in the recorded-data test.

Three real recorded frames were tested without ROS publishers. After compilation:
baseline approximately 72 ms; guided sampling approximately 114 ms. A small
pending-prefix perturbation had RMS error .004082 before guidance and
.003960–.003990 rad afterward. This modest improvement does NOT validate smooth
closed-loop RLT: the downstream Actor may modify the Reference prefix again.
Asynchronous queue integration, HIL invalidation and actual emitted-action Replay
alignment still require implementation/acceptance together. RTC is NOT enabled
in the online candidate. Do not label interpolation as RTC.

Using the measured 72 ms Reference latency as an illustrative synchronous model:
10 executed steps at 20 Hz gives .5 s open loop and ~1.75 replans/s;
5 gives .25 s open loop and ~3.10 replans/s, but blocking rises from 12.6% to 22.4%.
These are calculations, not measured robot loops. Shorter execution means MORE
replanning. Reducing denoising iterations is a different experiment and was not
performed. The Actor outputs only 10 steps, so increasing its execution to 50
without adapting/training the Actor is invalid.

Reusable pure action utilities live in VLA's existing execution_methods package:
common/runtime_lib/execution_methods/action_processing.py.
20->40 Hz interpolation preserves the original duration and logical endpoints;
tests passed. It changes within-step timing and remains offline-only. No hardware
publication frequency was changed.

Native normalized noise std=.002 corresponds to roughly 0.000007–0.000046 rad
joint std (quantile scaling), and about 0.0000029 m gripper std. It is very small.
The diagnostic sweep compares .001/.002/.004/.01, IID vs temporally correlated
noise, with gripper excluded in the experimental utility. The utility bounds
perturbations and takes a caller-owned RNG. It does not automatically replace or
add to the Actor's existing noise. No exploration amplitude was changed live.

References:
- [RTC paper](https://arxiv.org/abs/2506.07339): asynchronous overlapping chunks
  and prefix inpainting; requires model, queue and control-time agreement.
- [Fixed RLT implementation](https://github.com/Yyshadow/openpi-RLT): native
  trainer/networks/Replay remain under third_party/openpi-rlt with their recorded version.

## Candidate: explicit selection, reversible weights

Web: choose plug_insertion -> RLT -> **MC30 candidate**.
Model ID: plug-v3-credit-mc30. It starts from Warmup5000 + 2000 experimental
updates (Learner7000 / Actor3500), NOT from the previous online11500 snapshot.
It is registered for online collection; standalone evaluation is disabled while
learning. Loading prepares a paused session; the operator starts the episode.

Cobot files:
- Candidate weights and full optimizer state:
  /media/agilex/Getea1/jiaan/model/rl-platform/rlt/plug_insertion/history/candidates/credit_20260930/mc_30/online_candidate/
- Generated machine config:
  /home/agilex/jiaan/project/rl-platform/runtime/experiments/credit_mc30/online.yaml
- Candidate runtime:
  /home/agilex/jiaan/project/rl-platform/outputs/rlt/candidates/credit_mc30/online/
- Shared Replay:
  /media/agilex/Getea1/jiaan/data/datasets/plug_insertion/derived/rl-platform/rlt/replay_clean_v1/replay_journal.pkl
- Shared Stage1:
  /home/agilex/jiaan/data/rlt/plug_insertion/reference_4999

From a terminal (hardware/recorder prerequisites are the normal RLT prerequisites):
```bash
cd /home/agilex/jiaan/project/rl-platform
COBOT_DEPLOYMENT_MODEL_ID=plug-v3-credit-mc30 ./scripts/rlt_up.sh online
```

Ctrl-C stops this Session, retaining Stage1 as usual. The experimental supervisor
stops EnvDriver, then flushes Learner while Replay remains reachable, then Actor,
then Replay. A real isolated process test found the upstream simultaneous stop
could close Replay before Learner flush; this candidate now orders shutdown.

To revert: end the episode/session, release the selected model, choose the original
online or warmup entry and load manually. Reusing Stage1 is handled by the existing
runtime. Never start two entries on the same ports. Original weights/config stay
unchanged. Shared Replay is intentionally NOT rolled back: new accepted episodes
remain available to both branches. This is weight/config rollback, not data erasure.

For another machine, deploy both Git projects and restore their frozen environments;
replace machine paths through the existing project configs. Generate the candidate
from its registered report/research state using prepare_credit_candidate.py.
The script refuses to overwrite an existing candidate. Runtime config is not a
hidden source dependency: its reproducible generator and profile registration are
versioned. Large artifacts are on Getea1, not in Git.

## Reproduce / inspect without robot motion

Run commands in the same Cobot project directory:
```bash
./envs/online/bin/python scripts/experiment_credit_assignment.py \
  --cohort outputs/rlt/plug_v3_yyshadow/analysis/learning_diagnosis.json \
  --output outputs/rlt/plug_v3_yyshadow/analysis/credit_assignment_repeat.json \
  --updates 2000 --seeds 41,42,43

./envs/online/bin/python scripts/validate_credit_candidate.py \
  --config runtime/experiments/credit_mc30/online.yaml \
  --output outputs/rlt/plug_v3_yyshadow/analysis/credit_candidate_validation.json
```

The first command trains only in memory unless --save-candidates is explicitly
given. The second uses a temporary isolated checkpoint and synthetic arrival
count; it never appends Replay, starts ROS or publishes actions.

Receipts and full 21-run results:
 /home/agilex/jiaan/project/rl-platform/outputs/rlt/plug_v3_yyshadow/analysis/
 credit_assignment.json, credit_candidate_validation.json,
 credit_spawn_validation.json, rtc_reference.json, action_timing.json.
A6000 keeps small result copies in outputs/rlt-diagnosis-20260930/.

Passed: pure sampling/credit contracts; actual JAX adapter update; real candidate
restore, exactly five updates per synthetic arrival, Actor publication, exact
restart; isolated real Replay/Learner/Actor processes with no EnvDriver; recorded
Stage1 RTC inference; action/noise array tests; web tests and responsive preview.
Still pending: actual candidate episode, true online arrival/update from a new
robot rollout, physical success/safety A/B, integrated RLT asynchronous RTC and
40 Hz command acceptance. Keep 20 Hz, execute10 and existing noise for the first
candidate check. Use new evaluation episodes with fixed initial conditions and
report autonomous success, assisted success, HIL time, action discontinuities and
timeouts separately.

## Modified documentation paths

- /data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/README.md
- /data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/docs/MIGRATION.md
- /data/LFT-W02_data/jiaan/jiaan/projects/vla-platform/docs/JIAAN.md
- /data/LFT-W02_data/jiaan/jiaan/projects/vla-platform/docs/MIGRATION.md
- /data/LFT-W02_data/jiaan/jiaan/projects/cobot-web/docs/COMMAND_LINE.md
- /data/LFT-W02_data/jiaan/jiaan/projects/cobot-web/docs/MIGRATION.md
- /data/LFT-W02_data/jiaan/jiaan/agent-guide/projects/rl-platform/README.md
- /data/LFT-W02_data/jiaan/jiaan/agent-guide/projects/vla-platform/README.md
- /data/LFT-W02_data/jiaan/jiaan/agent-guide/projects/cobot-web/README.md
- /data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/docs/EXPERIMENTS_20260930.md

Final acceptance: formal 8015 serves 21 credit experiments and seven action plots; English rendering and responsive read-only checks passed. MC30 catalog reports available, training enabled, published Learner7000 / Actor3500. UI-only reload preserved all six observed hardware process identities. Group-wide Ctrl-C also passed ordered candidate shutdown with no traceback or listening test ports. Guide records updated without committing its Git.

## 2026-09-30 follow-up: does MC30 repair useful action guidance?

The first candidate selection compared seven recipes x three seeds x 2,000 updates
from the same Warmup5000 optimizer state. MC30 is an empirical research choice,
not an upstream RLT default or an established optimum. Its target is
`0.7 * native_TD + 0.3 * recorded_discounted_episode_return` on valid rows.
Middle transitions already receive native TD bootstrap credit; their zero
`transition.success` is not proof that their success information was discarded.
MC30 changes how credit is anchored, not the success/failure sampling ratio.

New script: scripts/audit_critic_guidance.py. It loads six frozen saved states,
excludes the same complete Online episodes, and tests Q along recorded HIL actions.
The **fixed baseline Actor** path is identical for every Critic, separating action
changes from Critic changes. Q1 is used by the Actor objective; min(Q1,Q2) by TD.
It also measures per-joint/gripper errors and bootstraps paired whole episodes,
never overlapping windows as independent samples. Five uncertainty/identity
contracts and the real GPU audit passed.

### Actual new evidence

Seed42 matched Native TD vs MC30, Learner7000/Actor3500:

| Diagnostic | Native TD | MC30 |
|---|---:|---:|
| HIL joint MAE (rad) | .00213242 | .00212074 |
| Early recorded-outcome AUC | .5455 | .6818 |
| Q1 prefers HIL endpoint, fixed Actor path | 22.76% | 35.86% |
| min-Q prefers HIL endpoint, fixed Actor path | 26.90% | 42.76% |
| HIL gripper MAE (m) | .00019283 | .00020150 |

There are 145 HIL-containing windows but only **six HIL episodes** in this
development cohort. They are successful assisted episodes; this is not a diverse
test of successful and failed interventions. Human counterfactual actions can be
off-distribution and are not certified optimal. Most windows still do not get
higher Q1 toward the human endpoint. MC30 mitigates this diagnostic; it does not
resolve it. J1/J2/J4 errors are slightly worse; gripper error worsens ~4.5%.
A six-joint scalar must not hide these dimensions or imply smooth hardware tracking.

Paired episode bootstrap for early AUC improvement, seed42:
+.1364, 95% interval [-.0682, +.3182]. It includes zero.
Episode-equal HIL MAE difference: -.00001225 rad, interval
[-.00001826, -.00000560], but this is conditional on six already-inspected episodes
and one saved training seed. These are exploratory intervals, not selection-adjusted
evidence or a robot success test. Three-seed mean early AUC .5417 -> .6667 remains
below the initial Warmup5000 .9091; there is no monotonic improvement claim.
MC return RMSE is measured against the target being introduced, so its decrease
is not an independent demonstration of better task execution.

Replay contains 73 Online episodes: 44 failed without HIL, 23 successful with HIL,
six successful without HIL. These are accepted Replay counts, not an unbiased
operational success-rate estimate. Among the 19 held-out development episodes:
11 autonomous failures, six assisted successes, **only two autonomous successes**.
The pure-autonomous early AUC uses 13 episodes / two positives, far too few to
establish generalization. Of all successful Replay windows, 2,207 belong to assisted
successful episodes. 316 complete pre-HIL windows inherit positive MC returns.
This is expected behavior-return credit, not proof those actions were wrong:
later human rescue makes eventual success an ambiguous target for autonomous
decision quality. Both native TD and MC can inherit this confound.

### Actor Q-loss ablation: the new competing explanation

Ran three additional seeds 41/42/43 with exactly the same Warmup5000 initialization,
train/holdout split, sample sequence and 2,000 updates. Only offline
`online_q_weight` changes .1 -> 0; BC, delta regularization, native Critic TD,
reward, action normalization and all production configuration remain unchanged.
It is Q-loss-off, not removal of the Critic or a change to the deployed algorithm.

| Method | Three-seed HIL joint MAE (rad) | Early outcome AUC |
|---|---:|---:|
| Native TD / original Actor Q weight | .00215369 | .5417 |
| MC30 / original Actor Q weight | .00214182 | .6667 |
| Native TD / Actor Q weight zero | .00213262 | .4470 |

Turning off Actor Q pressure improves this imitation proxy by ~0.98%, versus
MC30's ~0.55%, yet yields worse Q outcome ranking. This supports checking whether
the learned Critic is useful for action improvement rather than attributing every
imitation gain to RL. It does **not** justify disabling Q in production: fitting
recorded HIL actions and autonomous task success are different objectives.
No Q-loss-off weights were saved or registered; training happened only in memory.

The next priorities are: verify observation/action/HIL timing and normalization;
stratify pre-intervention, intervention and autonomous-success data; evaluate
matched Critic action preferences and dimensions on new untouched episodes.
Only then compare registered credit/loss variants in controlled field A/B.
RTC/40 Hz/noise remain separate execution experiments; none is enabled here.

### Paths and reproduction

A6000 reports and the four-panel figure:
`/data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/outputs/rlt-diagnosis-20260930/`
contains critic_guidance.json, actor_bc_only.json, actor_bc_only.yaml,
actor_bc_only_registry.json and critic_guidance.png/.svg.
Cobot reports use:
`/home/agilex/jiaan/project/rl-platform/outputs/rlt/plug_v3_yyshadow/analysis/`.

```bash
cd /home/agilex/jiaan/project/rl-platform
./envs/online/bin/python scripts/audit_critic_guidance.py \
  --cohort outputs/rlt/plug_v3_yyshadow/analysis/credit_assignment.json \
  --states /media/agilex/Getea1/jiaan/model/rl-platform/rlt/plug_insertion/history/candidates/credit_20260930 \
  --output outputs/rlt/plug_v3_yyshadow/analysis/critic_guidance.json \
  --guard-web-url http://127.0.0.1:8015

# Inspect the saved experimental YAML: online_q_weight=0; production YAML is unchanged.
./envs/online/bin/python scripts/experiment_credit_assignment.py \
  --config outputs/rlt/plug_v3_yyshadow/analysis/actor_bc_only.yaml \
  --registry outputs/rlt/plug_v3_yyshadow/analysis/actor_bc_only_registry.json \
  --cohort outputs/rlt/plug_v3_yyshadow/analysis/credit_assignment.json \
  --output outputs/rlt/plug_v3_yyshadow/analysis/actor_bc_only.json \
  --updates 2000 --seeds 41,42,43
```

The Q-off YAML is a copy of the recorded machine online configuration with only
experiment.rl.online_q_weight set to zero; its registry has one uniform profile:
`{"schema":1,"profiles":{"actor_bc_only":{"sampling":"uniform"}}}`.
The experiment report now includes actual algorithm configuration and config hash.
The audit refuses a changed journal, occupied GPU or active web model/session.
Only audit output files are written. Final Replay SHA256 still matches
c0ac2d7a755c8055b567942d154d98102456cff3aafc54ce179fadc82fcd0092.
GPU returned idle; no production weights/config or hardware tasks were changed.
Sources: actual fixed source code, recorded reports and this dialogue, 2026-09-30.
