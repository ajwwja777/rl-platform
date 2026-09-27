# plug_v3 warmup technical selection (2026-09-24)

Selection only. No warmup, robot action, dataset deletion, or replay mutation was performed. The machine-readable selection and label/replay fingerprints are in 2026-09-24-plug-v3-warmup-selection-v1.json. Training is disallowed until the operator and assistant compare data and parameters.

## Selection

| Class | Count | Disposition |
| --- | ---: | --- |
| Matched decided recording, metric, and clean replay | 104 | Technical candidate |
| Aborted recording | 59 | Exclude |
| Decided recording without metric/replay | 7 | Hold; do not infer RL transitions from labels |
| Replay episode without matching HDF5/label | 1 | Exclude from curated replay |

The 104 candidates are 56 success and 48 failure episodes, including 30 HIL successes and one HIL failure. They contain 1,751 transitions, 301 with intervention (17.2%). The seven held raw episode indices are 114, 115, 119, 120, 122, 123, 124: five HIL successes and two autonomous failures. Replay episode ID 132 has 15 transitions and a success terminal but no matching raw recording, so its success label cannot be audited. The current unfiltered clean journal has 105 episodes and 1,766 transitions; it must not be fed directly to a curated warmup because it still contains this orphan.

A zero-transition aborted metric reused episode ID 112, which later represented decided raw episode 125 with ten failure transitions. Selection uses the later finalized metric. Never match raw and replay by episode ID alone; use the metric completion timestamp, outcome, and nonzero transitions. No two candidates share a replay ID.

For all 104 candidates, metric outcome agrees with the operator sidecar and terminal replay reward. Replay arrays checked were finite, and each episode has one final done. The 104 candidate camera masks were valid for all frames. Camera arrival-time p95 age per episode is at most 36.5 ms and three-camera arrival skew p95 is at most 31.8 ms. First/middle/last sampled images were nonblank and nonidentical. This is a recorder integrity screen, not proof of insertion outcome or scene equivalence.

## Review flags retained in the candidate set

- Raw episode 133 has one qpos frame marked invalid in its HDF5 mask. Its replay has 16 finite transitions, no reported dropped transitions, and a success terminal. Keep technically, but inspect if a strict raw-frame conversion is planned.
- Raw episode 111 is the shortest decided record at 67 frames. It has 13 replay transitions and a long HIL segment, with no technical corruption detected. The terminal snapshot cannot independently establish successful insertion; operator label remains the source of truth.

The HDF5 action dataset is NaN when its valid mask is false outside manual control; this is expected for these rollout captures. The policy/HIL replay action_chunk arrays, which warmup would consume, were finite. Compare camera arrivals with rollout/sample_timestamp (same clock), not ROS topic timestamps (different clock domain).

## Before any warmup

Discuss whether to accept the 104 candidates and how to reserve held-out episodes. Confirm the intended uniform sampler, 600-transition gate, 20,000 planned updates, batch 128, and warmup BC/Q weights 10/0.1 against the upstream AgileX reproduction. Only then create a physically filtered, checksum-backed replay snapshot excluding ID 132 and begin training. No training was started during this selection.
