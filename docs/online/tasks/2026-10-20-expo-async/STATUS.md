# EXPO asynchronous Online acceptance

2026-10-20. User approved implementation and four-GPU throughput qualification, then confirmed collection ended and allowed the idle field switch. Isolated returned-Actor acceptance passed. Formal continuation is loaded and paused; see the delivery addendum below for actual field state.

## Implemented behavior

Optional bounded_async_v1 mode with durable local outbox, independent remote admission, sequential native learning, committed-version publication, background resumable download and idle episode-boundary installation. Original synchronous run, catalogue and signed learner source are retained. Initial production continuation is the real Actor20/Critic400/15 eligible online episodes/1851 transitions with25 demonstrations/3301 transitions. Isolated qualification data are excluded from it.

Same warmup10,batch5,K30,UTD20,global batch64,Critic50/30/20,success-only VLA,RTC5/edit0.01. Local rollout RNG is independent; old published behavior versions are admissible. Outstanding uploads/admissions/partial and deferred batches are bounded at10, with2 observed published generations of lag. Starts can pause at the bounds; this is not unlimited continuous collection.

## CPU and Web validation

EXPO suite:289 passed,8 skipped,1 known pre-existing continuation fixture deselected (missing device_layout in that fixture). Initial repository-wide collection hit unrelated RLT import errors; these are not reported as EXPO test results. Web suite:113 Node tests passed with the repository jsdom dependency and backend working directory. Actual DOM test keeps Start enabled while async learning continues and disables it at the backlog bound.

Independent review found and fixed: completed journal missing Actor receipt after crash; ACK disappearing from the local count before remote status catches up; deferred 532 credit losing its outstanding episode identities; configured queue smaller than batch size. Additional checks separate the local rollout UUID from the learner's active job and retire unpublished snapshots left beyond a durable commit into an audit directory before retry. Published-ahead-of-commit state fails closed. The latter startup hygiene and immediate pending-update status are CPU-tested additions after the core GPU run; final production startup uses them.

## Real four-H100 acceptance

Owned allocation740494,ACD1-67,48 CPUs,384GB; expires2026-10-20 22:34:20 CST. Cancelled owned pending741226 and reused the idle existing allocation after user confirmation. The old learner was stopped only at durable Actor20; exact committed metadata remained unchanged.

Isolated run updated Actor20 to25 and Critic400 to500. A sixth 30-transition journal (~10.77MB) was durably admitted in2.01 seconds while the optimizer was updating; receipt correctly reported received=true,processed=false. An old-Actor20 journal remained admissible after publication25. Duplicate admission did not change committed metadata. Process restart restored exact metadata, Actor25 and2/5 partial-batch episodes. Initial load111.65 seconds; restore105.13 seconds. No ROS publishers or robot actions were involved.

Evidence: /data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/outputs/expo-async-20261020/qualification-result.json

## Throughput result

Six-update comparisons from identical isolated Actor20 checkpoints retained the original pooled allocator/native mapping4 implementation. Stable sample+update mean34.60 seconds; sampling approximately0.2 seconds, native update approximately34.4 seconds, of which CPU target EMA approximately1.20 seconds. One update means one VLA update and20 Critic updates, not one five-episode batch.

Stable training GPU utilization means:95.2%,94.8%,96.3%,96.1%; power approximately308-323W per card; peak memory during those samples approximately54.2GiB on the first device and34.7GiB on each other device. Power need not reach the700W cap. The first three updates had distinct input-sharding compilation variants; do not confuse cold compilation or checkpoint I/O with steady compute throughput. Native update code/precision/minibatch/UTD were not altered to inflate utilization.

Bulk target transfer candidate:34.85 seconds, CPU target EMA1.42 seconds; all-metric/2229-parameter-probe comparisons were not bitwise identical. Rejected as slower and unqualified. Prior mapping16 candidate was also not adopted. The selected configuration is the fastest qualified option among these measured comparisons; no claim of a global hardware optimum or zero idle time.

Isolated final save: checkpoint18.39 seconds; inference bundle export6.94 seconds. Training and transfer can overlap local rollout, but zero-work queues, compilation, checkpointing and episode-boundary installation can still idle some resources. At about600 new10Hz transitions per five episodes, the unchanged K30 budget gives about20 update groups: roughly11.5 minutes of steady update work plus save/export and transport. Actual episode lengths and cache state determine the total.

## Field transfer status

Both source repositories were submitted/pushed before SHA-guarded Cobot installation. New entry remains gated until GPU return and final paused load pass. The HTTP acceptance wrapper initially treated an accepted asynchronous Web unload as completed and collided with the old SSH forward; it was corrected to wait for phase=offline and operation=null. No production journals/weights were changed by that failed attempt. Complete2,017,658,362-byte Actor25 HTTP transfer/SHA and Range verification passed in256.39 seconds. One Cobot process loaded Actor20 and installed25 in5.75 seconds without restart; active-episode install was rejected. Local load/warmup32.36 seconds; subsequent complete pre-cache/selection samples approximately0.147-0.149 seconds. No ROS publication or production data was involved. The formal continuation keeps Actor20; its exact original snapshot is retained on both hosts to reuse the verified initial cache.

Related: DESIGN.md,PLAN.md,RUNBOOK.md. Detailed artifacts under the evidence directory; current field owner recorded in scratch/cobot-web/coordination/cobot-live.md. Physical rollout/HIL acceptance remains operator-owned.

## Final field delivery (2026-10-20 15:25 CST)

Formal optional entry realtime-expo-ft-532-hpc4-async loaded successfully after the user ended collection. Cobot runtime1309486, local Actor20, published20, Learner20/Critic400, Replay5152, completed15, seed25/3301 and online1851; batch0/5, pending uploads0, untrained0, lag0. All four learner GPUs are configured. Web phase=paused/model_ready=true/Session inactive, no runtime/network/install fault. Local8043 and control8044 confirm the same identity. 50Hz publication/logical10Hz/RTC/filtering and d5/edit0.01 are active. No episode, recording or robot command was started. Operator owns the next Session/rollout/HIL.

Production committed metadata is byte-identical to the real Actor20 baseline; qualification UUIDs are absent. Initial connection-refused messages occurred during multi-hop SSH startup; the final live chain and model readiness succeeded without restart or source change. Evidence: outputs/expo-async-20261020/final-acceptance.json. Runtime sourcecad6dc6, catalogue2a8ce93, Web runtime0bc6192; documentation commits follow separately. Allocation740494 remains owned/running until22:34:20 CST; no automatic extension. Test GPU workers and the temporary return wrapper exited, pending reservation741226 was cancelled. Future allocation recovery is documented in RUNBOOK.md.

Eight isolated before/after inference samples were recorded, not a statistically supported real-robot p95 under continuous upload/training load. Physical continuous rollout/HIL and autonomous success rate remain operator acceptance; do not equate transport/update qualification with learned policy improvement.
