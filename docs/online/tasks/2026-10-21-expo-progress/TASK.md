# EXPO progress in Cobot Web

User requires learner/Replay sampling/transfer/deployment visualization in the existing Web, with final-return-before-release as the next shutdown procedure. Scope: independent read-only observer, static UI, scoped tests and static deployment. No learning schedule, Replay membership, robot lifecycle, hardware or GPU reservation changes.

The five stages show learner steps trained, committed, published, downloaded/verified, and served. Byte progress uses local file sizes and a manifest matched to the exact version; absent total is unknown, 100% bytes is not installation. Interrupted upstream states retain their original timestamp and are labeled stale/disconnected. SQLite keeps first-observed version milestones separately from remote publication timestamps. Old unrecorded milestones are not reconstructed.

Sampling shows configured batch64/UTD20 for native continuous, uniform Critic with replacement and success-only VLA; inventory percentages and locally acknowledged episode/source-version counts are explicitly not actual sampled batch composition. Exact per-batch episode IDs/composition are not present in existing telemetry; UI says unrecorded. The partial local audit does not claim to cover inherited remote Replay.

Implementation: scripts/expo_metrics_observer.py and new scripts/expo_progress.py; Web segmented_frontend/expo_progress.{js,css}, native_expo_metrics.js and rl_monitoring.js/index.html. User-visible overview outside collapsed training diagnostics; existing metrics/history remain.

Verification: 31 scoped CPU tests passed. Final Web suite: 139 tests (including installed-receipt timestamp precedence). Static files deployed with SHA256 verification; only expo-metrics-observer.service restarted. Production browser GET/render and timeline expansion passed without page errors; desktop view visually inspected, isolated narrow-width component has no overflow. Existing global camera dock can overlap tall mobile screenshots; this task does not redesign the global workspace. Current final102 checkpoint and interrupted98 return are preserved; this feature does not claim final102 was transferred.


Production acceptance (2026-10-21): refreshed Data collection with Use model enabled and realtime-expo-ft-native-async-hpc4 selected. Five stages = trained102 / committed102 / published98 / downloaded89 / served89. Version98 matched manifest payload2,017,658,362 bytes; local cache1,802,682,930 bytes (89.345%), retrying, no fabricated ETA. Remote timestamp remains original; disconnected status is explicit. Replay8960 = seed3301 + online5659; online45eps. Local receipt coverage30 qualified episodes is cumulative/partial, not a current-inventory claim. Native sampling1280 Critic draws (20x64), VLA64 success-only; actual draw identities unavailable.

Before/after PIDs10750 Web,51530 runtime,51650 Actor,1037914 arms,1039034 cameras and Session paused/active were unchanged. No HPC allocation/training/robot commands. Scoped backups: Cobot rl-platform/runtime/verification/expo-progress-20261021/backup. Evidence: outputs/expo-progress-20261021/{deploy-before.json,deploy-after.json,deploy-files.json,browser-live.json,live-wide.png,live-timeline.png,frontend-final.log,tests-0.log}. Browser test's fresh profile set only local Use model preference; no model/control POST.

Shutdown procedure updated in docs/online/tasks/2026-10-20-expo-continuous/RUNBOOK.md. Latest102 export/return is still outstanding and is not claimed completed by this observability delivery.
