# EXPO progress in Cobot Web

User requires learner/Replay sampling/transfer/deployment visualization in the existing Web, with final-return-before-release as the next shutdown procedure. Scope: independent read-only observer, static UI, scoped tests and static deployment. No learning schedule, Replay membership, robot lifecycle, hardware or GPU reservation changes.

The five stages show learner steps trained, committed, published, downloaded/verified, and served. Byte progress uses local file sizes and a manifest matched to the exact version; absent total is unknown, 100% bytes is not installation. Interrupted upstream states retain their original timestamp and are labeled stale/disconnected. SQLite keeps first-observed version milestones separately from remote publication timestamps. Old unrecorded milestones are not reconstructed.

Sampling shows configured batch64/UTD20 for native continuous, uniform Critic with replacement and success-only VLA; inventory percentages and locally acknowledged episode/source-version counts are explicitly not actual sampled batch composition. Exact per-batch episode IDs/composition are not present in existing telemetry; UI says unrecorded. The partial local audit does not claim to cover inherited remote Replay.

Implementation: scripts/expo_metrics_observer.py and new scripts/expo_progress.py; Web segmented_frontend/expo_progress.{js,css}, native_expo_metrics.js and rl_monitoring.js/index.html. User-visible overview outside collapsed training diagnostics; existing metrics/history remain.

Verification so far: 30 scoped CPU tests and137 frontend tests passed. Deployment/visual acceptance pending. Current final102 checkpoint and interrupted98 return are preserved; this feature does not claim final102 was transferred.
