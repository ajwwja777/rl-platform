# plug_v3_yyshadow faithful warmup and online release — 2026-09-26

## Scope and method

This release uses the fixed Yyshadow/openpi-RLT upstream commit `c1e40ac360185778c98cf20da2820e22d2d415e7` and its unmodified `offline_train_from_replay.py` actor/critic, reward, TD target and stratified sampler. Stage 1 remains plug_v3 step_4999, a frozen single-right-arm 7D policy with mid, left-wrist and right-wrist cameras. The chosen early-stop budget is 5,000 **critic** updates (actor version 2,500 because it updates every second step). This budget differs from the original 20k run; the objective/network/other weights do not. No alternative RL algorithm was substituted.

Training replay has 120 distinct successful Stage 1 expert demonstrations (1,186 C10 transitions) plus 84 selected rollout episodes (45 success / 39 failure, 1,381 transitions), total 2,567. Expert features and reference actions came from the frozen Stage 1 checkpoint. The same 20 rollout episodes (11 success / 9 failure, 370 transitions) were excluded from every trial and used for the identical offline evaluation. The complete 134-expert source remains unchanged. Production online replay remaps expert IDs to negative IDs without changing action, observation, reward or labels, so new robot episode IDs remain recent.

## Comparison on the independent rollout holdout

| Candidate | Training transitions | Updates | Early / middle / late / terminal Q AUC | Whole-episode Q AUC | HIL six-joint MAE, rad | Predicted within-chunk step p95, rad |
|---|---:|---:|---|---:|---:|---:|
| Stage 1 reference | — | — | — | — | 0.001981 | 0.001750 |
| Old rollout-only warmup | 1,381 | 20,000 | 0.364 / 0.253 / 0.232 / 0.778 | — | 0.002005 | — |
| 120 experts + rollout | 2,567 | 3,000 | 0.465 / 0.626 / 1.000 / 1.000 | 0.838 | 0.001996 | 0.002603 |
| **Selected: 120 experts + rollout** | **2,567** | **5,000** | **0.626 / 0.657 / 1.000 / 1.000** | **0.889** | **0.001963** | **0.002486** |
| 120 experts + rollout | 2,567 | 20,000 | 0.414 / 0.434 / 0.970 / 1.000 | 0.717 | 0.001988 | 0.001918 |

All-episode early/middle Q AUC is affected by later HIL success; in the five autonomous heldout successes versus nine failures, the selected 5k AUC is 0.933 early and 1.000 middle/late/terminal. That subgroup is too small to infer a real-world success probability. Late/terminal separation is expected with terminal labels and does not establish early causal credit assignment. The selected actor has a small HIL imitation improvement over reference, not a demonstrated insertion-success improvement. Its raw predicted step p95 and reversal fraction (0.610 versus reference 0.056) remain motion caveats; the unchanged robot command rate/acceleration/tracking guards must remain enabled. An offline sequence statistic cannot prove physical smoothness.

Five initial 3k single-factor ablations (double successful rollouts, BC weight 5, Q weight 0.3, double HIL, 40 successful experts) did not beat the selected 120-expert 5k on the combined holdout Q/HIL criteria. The older 20k trained on rollout only failed the holdout Q direction check. The 120-expert 20k variant is retained as a reversible comparison, but not selected because its critic ranking and HIL fit degraded after 5k.

## Online release and evidence

A6000 isolated trial root: `/data/LFT-W02_data/jiaan/scratch/cobot-realworld-rl/runs/plug_v3_yyshadow/warmup_20260925_trials`. Per-trial `experiment.json`, `status.json`, `metrics.jsonl`, `analysis/{q_summary,hil_fit,motion}.json`, holdout Q tables, and exact checkpoint/snapshot are retained there. The unchanged upstream training command is in `run_experts120_5000.sh`. The online replay manifest is under `experts120_20000/replay_online/manifest.json`; the 5k release uses the byte-identical replay SHA-256 `65e84ddf78482abec2c9870f77ac78251bbf137dcc3adf96fffa1f8d9fa81bd1`.

No-robot A6000 online restore smoke: `experts120_5000/online_resume_smoke/report.json`. It restored learner step 5,000 / actor 2,500 with 2,567 replay transitions, reached `ready_for_online=true`, and one simulated new transition caused exactly five finite learner updates, step 5,005 / actor 2,502. This required aligning the upstream `warmup_post_collect_updates` runtime budget to the selected 5k early-stop budget; leaving it at 20k would quietly continue warmup before online training.

Cobot installed release: `/media/agilex/Getea1/jiaan/projects/rlt/runs/plug_v3_yyshadow/online/release.json`. Checkpoint SHA-256 `be50b6cf0586864d89173a6456128bffa8b37621a6fda500f5be287223050f33`; actor snapshot `4ffdcdf6e4f710ff806118b7c98a73c094a1ec47e1d613910c74d3d83d356a32`. Previous 20k production files/config/script were backed up under `runs/plug_v3_yyshadow/backups/pre-experts120-5k-20260925`. The `rlt_v3_up.sh` online actor gate now matches actor version 2,500. Cobot zero-publisher actor inference loaded the production snapshot with `source=1`, finite 10×7 action; phase-gate smoke returned `online`. Both are under `runs/plug_v3_yyshadow/candidates/experts120_5k_20260925`. RLT phase tests: 2 passed; console model tests: 5 passed. No real arm command was sent.

Cobot data UI `http://10.7.165.64:8015/`: ordinary recording directory/history/Episode controls now use the aligned RL-style grid; switching episodes no longer fades or shifts the history cards. Training page defaults to the selected 5k run, retains 20k comparisons and displays five verified plots. Plot files also live on the laptop at `D:\Code\jiaan_workspace\jiaan_upload\cobot_rl\warmup_v3_20260925\experts120_5k`. Read-only Chrome regression verified all five images loaded and ordinary collection layout at 1600px/800px. Static UI and script syntax checks passed.

## Onsite acceptance and limits

At final read-only check, Cobot reported CAN 5/5, ROS ready, cameras 3/3, but arms offline (0/3 coordinators, no fresh arm feedback). RLT Session/model server remain stopped. Thus this is an **offline-validated, real-robot-pending** release; no claim of physical smoothness, insertion success, or monotonic online learning is made. The UI model catalog shows Stage 1 reference, frozen actor 2,500 and online actor 2,500 available, with Stage 1 remaining the current/default selection until explicitly changed.

After the onsite arm/camera/workspace check, first use `cd /media/agilex/Getea1/jiaan/projects/cobot-platform && ./scripts/rlt_v3_up.sh frozen` for a small fixed-scene, frozen-actor acceptance set, labeling true successes/failures and recording HIL/guard events. If tracking and motion are acceptable, end that Session/Ctrl-C and use `./scripts/rlt_v3_up.sh online` to collect/train. Monitor `./scripts/rlt_v3_status.sh` and the training page; each new committed transition budgets five learner updates. Actor is published every 500 learner steps, so a few episodes may update critic/actor internally without immediately changing the served snapshot. Evaluate by matched-scene success and intervention rates over batches, not by one anecdotal rollout. Ctrl-C ends the Session but leaves Stage 1 loaded; `./scripts/rlt_v3_down.sh` releases it when done. Existing safety guards and operator terminal/home flow remain in force.

Do not treat 5k as a guaranteed final winner: if the frozen trial is worse than reference or new online updates destabilize control, use the preserved release/backup for rollback and diagnose the first failing stage before more robot data collection.
