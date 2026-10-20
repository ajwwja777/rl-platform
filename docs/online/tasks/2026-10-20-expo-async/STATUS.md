# EXPO async implementation status

2026-10-20: user approved implementation and HPC performance tests. Baseline7bd83b5; branch feature/expo-async-20261020. Production synchronous Actor20/HPC740494 retained; local Session still active. No product edits yet. Design/plan above capture approved scope. Next: durable transport tests.

Ruling: execute the approved design inline without another procedural approval round; user explicitly requested implementation. Existing signed sync learner remains unchanged; asynchronous modules are separate. Isolated continuation rather than new SFT start preserves learned model and permits rollback.

Tasks1/2 initial implementation:17 targeted tests passed. Real HTTP ingestion remains responsive while optimizer lock is held; old behavior versions accepted without resetting learner RNG; native batch credit/count regression passed. GPU and restart qualification pending.

2026-10-20 14:50: independent review identified four boundary defects (unfinished Actor receipt on restart, ACK visibility race, deferred credit omitted from backlog, undersized queue/batch deadlock). Regression tests reproduced failures, fixes now pass. EXPO CPU suite:286 passed,8 skipped,1 existing continuation fixture deselected. Earlier repository-wide collection included unrelated RLT missing-import errors; scoped EXPO suite is the reported result. Web112 Node tests passed with repository jsdom dependency and backend cwd; initial wrong test environment failures were corrected.

User confirmed collection ended and permits switch after qualification. Ended idle Session through Web CLI. Stopped owned idle HPC orchestrator2524389, preserving exact committed Actor20 metadata and Replay. Reusing existing4H100 allocation740494; cancelled owned pending741226 and its waiter. GPU benchmark now running on an isolated Actor20 snapshot; async HTTP/update/restart follows. No robot start, no synthetic production data, no source deployed yet.
