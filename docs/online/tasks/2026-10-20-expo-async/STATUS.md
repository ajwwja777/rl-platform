# EXPO async implementation status

2026-10-20: user approved implementation and HPC performance tests. Baseline7bd83b5; branch feature/expo-async-20261020. Production synchronous Actor20/HPC740494 retained; local Session still active. No product edits yet. Design/plan above capture approved scope. Next: durable transport tests.

Ruling: execute the approved design inline without another procedural approval round; user explicitly requested implementation. Existing signed sync learner remains unchanged; asynchronous modules are separate. Isolated continuation rather than new SFT start preserves learned model and permits rollback.

Tasks1/2 initial implementation:17 targeted tests passed. Real HTTP ingestion remains responsive while optimizer lock is held; old behavior versions accepted without resetting learner RNG; native batch credit/count regression passed. GPU and restart qualification pending.
