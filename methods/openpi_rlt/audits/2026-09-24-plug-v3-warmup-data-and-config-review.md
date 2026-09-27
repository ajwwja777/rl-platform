# plug_v3 warmup data and configuration review (2026-09-24)

This is a read-only snapshot for discussion. No critic/actor warmup was started.

## Data snapshot

Cobot path: `/media/agilex/Getea1/jiaan/data/rlt/plug_v3_yyshadow/warmup`.
170 complete HDF5/label pairs, about 82 GiB total. The 111 success/failure records are 61 success and 50 failure; 59 aborted records are excluded from replay. Of the decided records, 26 autonomous successes and 49 autonomous failures contain no HIL, while 35 HIL successes and 1 HIL failure contain intervention frames. Thus autonomous success is 26/75 among decided autonomous trials, not a controlled model success-rate estimate. Recent 30 decided records: 5 autonomous success, 11 autonomous failure, 13 HIL success, 1 HIL failure.

The label-sidecar HIL flag agrees with the HDF5 intervention arrays for all 170 recordings. The 104 technically eligible records have valid camera masks for all recorded frames. Two invalid camera masks occur in aborted episode 9. Valid masks alone do not prove visual synchronization or terminal-label correctness. A three-episode terminal-frame spot check found visible images in all views. Median episode length is 152.5 recorded frames; 41 timestamp gaps over 0.3 s across the dataset are compatible with operator pauses and need not be treated as action frames.

Current clean replay journal: 1,766 transitions from 105 episodes, 57 successful and 48 failed; 301/1,766 transitions (17.0%) carry intervention, 1,465 do not. This exceeds the 600-transition gate. Seven finalized decided HDF5 records were not time-matched to rollout metrics around the earlier terminal/replay crashes: episode indices 114, 115, 119, 120, 122, 123, 124 (five HIL successes, two autonomous failures). This is a timing audit, not a validated import list. Reconcile against journal IDs/traces before deciding whether to import them. Failures have `keep_for_training=false` in Task5 labels but 48 failures are present in the RLT replay; a future rebuild must not silently filter failures using that label flag.

## Configuration comparison

| Parameter | Current plug_v3 | Upstream AgileX task config |
| --- | ---: | ---: |
| action/proprio dim, chunk | 7/7, 10 | same |
| z dimension | 2048 | same |
| gamma, fixed std | 0.99, 0.002 | same |
| reference dropout, delta weight | 0.5, 10 | same |
| warmup BC/Q weights | 10 / 0.1 | same |
| actor/critic LR, target tau | 1e-4 / 1e-4, 0.005 | same |
| actor update period, batch | 2, 128 | same |
| warmup gate / planned updates | 600 / 20,000 | same |
| replay sampling | uniform (default) | uniform (default) |

Cobot-specific paths, ports, GPU memory fractions, replay seed 42, and robot reset/home behavior differ for the local deployment. Reference rollout explicitly disables the learner; the learner status is still global_step 0. Stage 1 reference is loaded, Session disarmed, policy paused.

## Decision before warmup

Agree which of the 111 decided episodes to use, review the seven journal gaps and any questionable terminal labels, then freeze a checksum-backed replay snapshot. Preserve both success and failure and retain HIL provenance. If selected data remain near the observed 55/45 success/failure episode balance and 17% intervention-transition share, the upstream uniform sampler and original 20,000-update settings are the simplest faithful first run. Compare training and held-out Q separation plus autonomous deployment only after the data set is fixed. No training should start before the operator and assistant discuss this choice.
