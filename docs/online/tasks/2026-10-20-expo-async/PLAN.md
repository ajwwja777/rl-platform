# Bounded asynchronous EXPO Implementation Plan

> Implement inline using executing-plans, test-driven-development and isolated worktree. User explicitly requested implementation and self-testing.

**Goal:** Deliver optional local-Actor/HPC-Learner asynchronous EXPO with bounded lag and measured throughput.
**Architecture:** Durable inbox/outbox decouple transport from the single-owner native learner. Background complete snapshot download and idle-boundary installation decouple rollout from publication.
**Tech Stack:** Python3.8 Cobot; Python3.11/JAX HPC; existing HTTP/SSH/Slurm and Web adapter.
**Spec:** DESIGN.md

## Global Constraints
Preserve synchronous signed learner files and existing production run. Keep warmup10/batch5/K30/UTD20/batch64/532/edit0.01/RTCd5. No robot commands or synthetic production data. Default limits10 pending episodes/2 published generations.

## Review Focus
Lost ACKs must not duplicate budget. Restart during uncommitted updates must restore committed optimizer/RNG/Replay. Concurrent begin/install must never mix policies. Disconnected downloads must retain their source snapshots. Profile mismatch must not attach to another learner.

### Task1: Durable transport and admission
Files async_state.py; tests/test_expo_async_state.py.
Interface Inbox(root,max_pending=10).receive/receipt/admit/pending; Outbox(root).enqueue/pending/ack. Validate immutable episode IDs, checksums, version and terminal before acknowledgment.
- [x] RED tests duplicate admission/restart/partial transfer/invalid policy/outbox recovery and bounds.
- [x] Implement; run targeted tests and sync journal regression; commit.

### Task2: Single-owner asynchronous learner and HTTP
Files async_learner.py, async_http.py; tests/test_expo_async_learner.py.
Interface AsyncLearner(args) reuses signed native initialization but has independent ingestion and learning worker; snapshot() returns immutable cached status. HTTP admission never takes training lock. Derived mode contract prevents accidental sync restore; copy continuation explicitly.
- [x] RED tests real scheduling/old-policy/RNG/commit ordering/HTTP during blocked update.
- [x] Implement preserving native losses, update count and save ordering; observe stage timings.
- [x] Test restart + fallback sync suite; commit.

### Task3: Local asynchronous coordinator
Files async_remote.py, async_runtime.py; minimal online_runtime.py control serialization; tests/test_expo_async_runtime.py.
Interface AsyncRemote mixes into OnlineRuntime, uploads durable queues, downloads complete candidates, installs under episode lock at idle only. Existing sync remote support remains default.
- [x] RED tests offline/lag/queue gates, no mid-episode install, duplicated/restarted uploads, version distinctions.
- [x] Implement and run sync/runtime regressions; commit.

### Task4: Optional launch/profile and isolated qualification
Files remote_launcher.py, scripts/expo_532_async_hpc4_up.sh, deployment_models.json; preparation/qualification scripts under scripts and task outputs. Cross-project Web modifications only for status rendering if necessary.
- [x] RED launcher mode/profile checks; implement optional entry.
- [x] Isolated continuation from committed Actor20 (or latest stable snapshot), explicit source/hash provenance; never alter source.
- [x] Four-H100 concurrency/update/restore benchmarks, compare options from same state; verify no numerical/recipe drift.
- [x] Return and hot-install isolated Actor on Cobot only with available resources/idle window; record bounded inference samples and install latency (real-robot contention/p95 remains field acceptance).
- [x] Whole-branch review and regression; commit/push and SHA installation; keep default sync available.
- [x] Update task status and guide summary; hand off actual available mode and performance limits.
