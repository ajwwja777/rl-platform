# Bounded asynchronous EXPO

User approved 2026-10-20: implement continuous local rollout with remote learning and measured HPC throughput optimization.

## Contract
Independent optional run and model; preserve synchronous code and rollback. Keep warmup10, batch5, K30, UTD20, global batch64, demo/success/failure50/30/20, edit0.01, RTCd5, success-only base actor. No robot movement by agent. No production synthetic data. Isolated GPU validation before switching an idle field service.

## Architecture
Cobot persists finalized journals and uploads through a durable outbox. Remote inbox accepts immutable files independently of the training lock and acknowledges durable admission separately from training commit. One learner owns Replay and weights, consuming episodes chronologically at update boundaries. Old published policy versions are accepted; behavior version stays immutable. Actor and Learner RNG streams are independent. Existing batch credit controls updates; no unbounded free-running optimizer.

Snapshots are complete inference bundles, published only after durable training commit. Background download uses SHA and HTTP Range. Remote versions are retained during transfer; first implementation retains snapshots with a free-space guard, avoiding unsafe pruning. Local installation is serialized with episode controls, only while idle and paused; service does not restart. Active episodes keep their version. No double GPU copy preparation during rollout.

Boundaries: at most10 pending uploads, admissions, partial-batch or deferred-learning episodes plus one in-progress episode per Cobot runtime, and at most2 published generations of policy lag before new starts are blocked. Active episodes may finish. Both bounds are explicit optional-profile settings. Unknown/aborted journals never add training credit. Received, processed and eligible counters are distinct. Network outage allows bounded local collection; errors never discard local journals. A failed parameter installation remains faulted, never claims old weights are usable without verification.

## Performance
Measure sampling, device placement/update, CPU target EMA, snapshot export, checkpoint save, upload/download and install. Retain exact precision and update recipe. Test cache reuse and numerical equality before changing compilation/allocator/CPU settings. Compression only if measured total time improves. Empty work queues may idle; full utilization is not a correctness criterion.

## Acceptance
CPU: duplicate/lost receipts, partial files, restarts, stale/future versions, distinct RNG, budget/count preservation, concurrent upload/training, snapshot completeness, disconnect, backlog gates, active-episode install rejection. GPU: full four-device native update, durable restore, receive during training, snapshot return/local inference and hot install without restart. Preserve production run. Web displays served/published/downloaded versions and queue counts. Operator owns physical rollout/HIL acceptance.

## Recovery and admission visibility
A monotonic durable admission revision accompanies each upload receipt. Local ACKed episodes remain counted until a status snapshot includes that revision, preventing an outage from underreporting the combined backlog. Explicit async_untrained identities remain committed across missing-outcome deferral; no training credit is silently dropped. A configured bound below batch size is rejected. If an outcome pool is absent at the bound, collection blocks with the sampling-wait status; curate/import the missing confirmed outcome through the existing controlled data process rather than silently alter 532 sampling.

On process restart, completed journals with a surviving Actor receipt recover its final RNG. If the crash preceded that receipt, they retain the complete labeled data and explicitly record actor_receipt_missing_after_restart. Final behavior RNG cannot be recovered and is not fabricated. Async learner RNG is independent, so these eligible journals can still be trained. Recovery runs only after a fresh idle Actor starts, before upload workers; ordinary active finalization must still seal its receipt before upload.

Unpublished numeric snapshots newer than committed optimizer state are moved to orphaned-actors/ on a fresh owned learner startup. They were never advertised and may differ after a floating-point retry; retaining them in the served namespace could otherwise block re-export. Published versions are never retired this way; a publication ahead of commit is an explicit corruption error.
