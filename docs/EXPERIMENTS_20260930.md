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
