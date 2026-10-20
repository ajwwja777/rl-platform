# Native continuous EXPO implementation plan

Goal: implement the user-approved native async script semantics with remote HPC learning and local Cobot inference, informative Web state. User approved the concrete alignment design and requested implementation/delivery. Root executes learner/runtime/deployment inline; independent native audit, Actor implementation and Web status use disjoint files, followed by integration review.

Architecture: independent durable episode insertion under the native sampling buffer lock; native BatchProcessor (including its prefetch/RNG order) and continuous native updates; native120s episode inactivity pause, no warmup10/K30/batch5 or count/version-lag gate. Snapshot publication is independently requested/coalesced by the remote consumer; durable checkpoint/export runs only at optimizer boundaries. Actor prepares verified GPU inference params off the control path and adopts on later RPC operations, retaining live behavior RNG and native old-precache/new-selection behavior. Source checkpoint continuation is explicit; existing production run remains rollback.

Constraints: no robot motion/synthetic production journals; prior user-selected d5/edit.01 and Cobot IO retained. Algorithm config uses native defaults (N8/filter8 rather than sync-run overrides), uniform native replay with success-only actor per native config; remove532 quotas. Shared memory optimizations preserve native losses but must be documented and measured; no precision changes. Every complete labeled episode can enter sampleable Replay while optimizer computes; native prefetch can delay actual use. No claim identical stochastic path across distributed processes.

Tasks:
- [x] Audit supplied native source/config/batch processor, record exact source differences.
- [x] TDD continuous learner ingestion/timeout/publication/restart; independent Replay durability and checkpoint consistency; native BatchProcessor, no new data-trigger budget.
- [x] TDD concurrent Actor stage/adopt, persistent RNG, transactional pre-cache tokens, bad-transfer rejection without breaking old policy; version audit on journals.
- [x] New optional runtime/profile/catalogue; background request/download/stage, no lag/backlog starts cap; clear Web counters/timestamps/phase.
- [ ] Independent code review; CPU/Web regressions; isolated4GPU update/concurrent admission/restore and Cobot stage+inference concurrency test.
- [ ] Push verified code before SHA installation; explicit real-state continuation; final no-motion paused load and user handoff with actual limits/timings.

Review focus: new data insertion during optimizer update and checkpointsave; no loss/double count after crash; previous snapshot remains available during download; RNG never rewound atstageadoption; stage error keeps known-good policy; no metadata claims sampleable before buffer insertion; source-faithful idle behavior across restarts; no production quota/config drift concealed by UI.

Implementation status: native audit/review findings addressed; source includes one owned host checkpoint capture and background disk/export worker, preserving native update and sampling. CPU suite338 passed/10skipped; one known baseline migration fixture missing previous/device_layout.py excluded. Web115 passed. Initial isolated4H100 proved immediate sampleable insertion during optimizer; first cold update278.73s, blocking save18.30s/export7.32s motivates owned-host background writer. GPU background-writer and local continuous-adoption acceptance still pending; no ready marker or production switch claimed.
