# RLT diagnostics

## Structure

- integrations/cobot_runtime/analysis.py: read-only JSONL aggregation, restart segments, actual sampling ratios and labeled outcomes.
- scripts/analyze_replay.py: offline PCA / k-means snapshot from the configured trusted Replay journal.
- outputs/rlt/plug_v3_yyshadow/online/metrics/: original telemetry.
- outputs/rlt/plug_v3_yyshadow/analysis/replay_projection.json: regenerable diagnostic artifact, ignored by Git.
- tests/test_analysis.py: statistics, restart, missing data and read-only projection regression tests.

A6000 source: /data/LFT-W02_data/jiaan/jiaan/projects/rl-platform.
Cobot deployment: /home/agilex/jiaan/project/rl-platform.
The web Training and Analysis pages use the same telemetry reader. No algorithm imports, model loads, Replay writes or robot commands are issued by this reader.

## Usage

On Cobot, from the project root, run envs/online/bin/python scripts/analyze_replay.py.
It reads runtime.replay.journal_path in configs/rlt/plug_v3_yyshadow/online_rl.yaml and atomically writes the JSON snapshot above. NumPy and PyYAML are already part of the registered online environment. On another machine, supply --config and --output; the config must reference a trusted local journal. Pickle is executable input: the offline CLI is for the registered training artifact only, never an uploaded or browser-selected file.

The journal remains in /media/agilex/Getea1/jiaan/data/datasets/plug_insertion/derived/rl-platform/rlt/replay_clean_v1/replay_journal.pkl. The snapshot does not copy weights or images. It captures the initial journal byte boundary and ignores a final incomplete append; source byte size, time and transition count are recorded. Re-run explicitly when newer data are needed; opening the webpage never rebuilds it.

## Interpretation

Training step, learner-internal Actor version and published Actor version are separate. Logs may span restarts; a non-increasing step begins a new segment. Missing or old heartbeats are shown as historical data, never as proof of a running process. YAML is the registered config, not proof of runtime overrides.

Actual sampling ratios are measured batch composition. Warmup, recent Online and HIL overlap. Source BASE/RL/HUMAN/MIXED ratios are disjoint. A batch samples transitions, not complete episodes. Actor curves exclude did_actor_update=0 placeholders. Charts decimate to 320 points per series; recent means use 100 raw batches. Logs are bounded to the last 24 MiB and truncation/invalid rows are disclosed.

Outcome denominators include only labeled rollout rows with transitions_written > 0, grouped by recorded phase and deterministic/stochastic behavior. Zero-write/unlabeled/discarded logs are excluded. Autonomous successes require no recorded intervention/human/mixed step; assisted successes are separate. A switch of Actor within an episode is labeled start→end. Wilson intervals express sampling uncertainty, not controlled experimental comparability; online exploration and changing scene difficulty remain confounders.

Projection features are proprio[7] and mean(action_chunk-reference_chunk)[7], standardized within the snapshot. PCA gives two display axes with explained variance; k-means (k=6, seed=42) operates in 14D. Points are a deterministic uniform sample of at most 2400 transitions. Cluster counts use all valid transitions. This is posture/action coverage, not visual embedding or a causal failure diagnosis. Representative chunks prefer the last intervention chunk when present. Replay episode IDs are not recorder indices. Action differences may include HIL, exploration or clamping, not only Actor corrections. No video-frame linkage is inferred.

## Verification (2026-09-29)

Four isolated tests passed: repeated-step segmentation; autonomous/assisted and zero-write denominators; partial/missing telemetry; deterministic projection with unchanged journal SHA. Live read-only results and UI release evidence are recorded in cobot-web/docs/MIGRATION.md. No motion or training parameter change is part of this feature.

## Replay/batch audit and offline experiments (2026-09-29)

See [the reproducible diagnosis report](DIAGNOSIS_20260929.md) for actual composition,
12 sampling runs, 14 Actor audits, gradients and recorded-image sensitivity.
The Learner entry observes the original sampled batch without re-sampling.
Composition refreshes on journal growth during training; PCA remains an explicit job.
Historical batch identities are not recoverable. HUMAN includes old demonstrations;
episode success is not transition.success; rank thirds are not semantic task phases.
Published-version audits are training-seen. Sampling experiments hold out complete
Online episodes from a Warmup5000 start. Production weights/parameters are unchanged.
