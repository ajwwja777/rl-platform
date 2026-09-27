# plug_v3 episode-heldout warmup: 20k training and offline diagnosis (2026-09-24)

## Decision

The fixed upstream Yyshadow offline actor/critic training script completed 20,000 steps on A6000 GPU 1. Its products are retained as research evidence but rejected for Cobot deployment and online RL continuation. No robot command, Cobot model replacement, online replay mutation, or Git commit/push occurred. The machine-readable gate at the run root states online_eligible=false.

Run root: /data/LFT-W02_data/jiaan/scratch/cobot-realworld-rl/runs/plug_v3_yyshadow/warmup_20260924_v1. The source selection audit is 2026-09-24-plug-v3-warmup-selection.md and its JSON manifest in this directory. Split metadata, source hash, train/heldout journal hashes, command, log, checkpoints, metrics, figures, and release gate are under the run root.

## Data and exact run

- Technical selection: 104 matched decided episodes, 56 success and 48 failure. Orphan replay ID 132 and seven decided HDF5 episodes without replay were not trained.
- Seed-42 episode-stratified split: 84 episodes / 1,381 transitions in the training journal; 20 episodes / 370 transitions held out throughout optimization. Heldout outcomes: 11 success, 9 failure; 5 autonomous success, 9 autonomous failure, 6 HIL success. The single HIL failure remains in training.
- The unchanged upstream offline_train_from_replay.py has SHA-256 56640aec0f8434ccafdd04e61d44a4932126542c1990d0fc7188e698a6b4e15f. Its own 5% random transition split inside the training journal used 1,312 transitions for parameter updates and 69 for its in-run actor/ref fit monitor. This internal monitor is not the independent heldout set.
- Parameters: 20,000 gradient steps; batch 128; seed 42; warmup BC/Q 10/0.1; delta weight 10; fixed std 0.002; phase warmup; source all; uniform sampling in the fixed train journal. Actor/critic architecture and normalization came from the existing plug_v3 actor snapshot; no plug_v2 model, stats, or replay were imported.
- The one-step GPU smoke passed before the formal run. Formal status reached step 20,000 and actor version 10,000. Best step 4,500 is only the upstream script's minimum actor/reference fit criterion, not a Q or success criterion. Final actor snapshot SHA-256: 5c17ba0ff529369a04ad82154f77c298b57eef75418a3845710bded06dc6387a. Final checkpoint SHA-256: c848632b076a6b3e90a9d924eba5c6e3ae770de6084fa8e0ba348b64a6a892c4.

## Independent heldout evidence

AUC is episode-level pairwise ranking of successful versus failed trajectories by Q(s, recorded action); 0.5 is chance. Early/middle/late refer to thirds of an episode. Terminal is the last chunk.

| Model | Early | Middle | Late | Terminal |
| --- | ---: | ---: | ---: | ---: |
| One-step smoke | 0.848 | 0.222 | 0.535 | 0.747 |
| Step 4,500 | 0.424 | 0.434 | 0.525 | 0.899 |
| Step 20,000 | 0.364 | 0.253 | 0.232 | 0.778 |

The one-step early AUC is an unstable initialization on 20 episodes, not learned performance. At 20k, mean Q across the episode is 0.985 for success versus 1.377 for failure: the ordering is wrong. The step-20k training subset probe (16 episodes) yielded early/middle/late/terminal AUC 0.717/0.600/0.367/1.000, substantially better than the heldout early/middle ranking. The heldout autonomous-only AUC is also below chance in early/middle/late, so HIL-versus-autonomous mixture does not alone explain the inversion.

The terminal reward is sparse (success +1, failure 0). The critic sees the terminal better than the earlier chunks, consistent with poor or confounded backward value propagation. The actor-Q training metric rises to about 1.59 despite reward scale 0/1, while critic TD loss falls to about 0.0011. A small self-bootstrapped TD error is not evidence of calibrated Q or useful policy improvement.

On 501 heldout HIL control steps, first-six-joint MAE to the executed human action is 0.001981 rad for the Stage 1 reference and 0.002005 rad for the final actor. The final actor does not improve heldout HIL action fit. Its correction direction has mean cosine about 0.359 with the human correction, so it has partial directional alignment but inadequate magnitude/precision. Across all heldout transitions, the final actor/reference mean absolute difference is about 0.00084 in mixed 7D native units. Do not interpret this as a demonstrated insertion correction.

## Figures and interpretation

Run analysis directory contains loss_curves.png, fit_curves.png, holdout_auc_by_stage.png, holdout_q_by_progress.png, train_vs_holdout_q_auc.png, holdout_hil_action_error.png, diagnostic_summary.json, plus per-episode Q tables/profiles and actor-fit tables. User-accessible figure copies are in D:/Code/jiaan_workspace/jiaan_upload/cobot_rl/warmup_v3_20260924 on the MateBook. The HIL-error figure uses only the first six joint coordinates in radians; gripper is excluded from that plot.

The loss curves show successful numerical optimization. The Q plots show the actual blocker: failed episodes receive higher value before the terminal chunk. The actor/HIL comparison shows no heldout benefit over reference. Neither final nor step-4,500 checkpoint should be labeled online-ready.

## Next diagnosis before another warmup

1. Audit Q-target construction and magnitude under the exact upstream gamma/chunk convention; check why values above the terminal reward scale appear and whether target bootstrapping is amplifying errors.
2. Evaluate matched near-success/near-failure pairs and scene strata, especially early shared approach states, to distinguish visual ambiguity from overfitting.
3. Inspect HIL action-target alignment and timing at intervention boundaries. Compare actor-to-human error on heldout human steps rather than only actor-to-reference fit.
4. If changing sampling, reward, BC/Q weights, or warmup schedule, discuss a single controlled change and rerun with the same episode-level holdout. Do not select by training loss or the upstream best-ref-fit tag alone.

The existing Stage 1 Reference remains the only model with prior onsite evidence; this new warmup was not published to Cobot.
