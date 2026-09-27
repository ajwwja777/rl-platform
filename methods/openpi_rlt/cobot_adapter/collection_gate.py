"""Explicit collection-only gate; never latch warmup or execute an optimizer step."""
import time

def install_collection_gate(learner_class=None, *, write_json=None):
    if learner_class is None:
        from rlt_online_rl.trainer import LearnerService
        learner_class = LearnerService
    if write_json is None:
        from rlt_online_rl.trainer import _atomic_write_json
        write_json = _atomic_write_json
    if getattr(learner_class, "_cobot_collection_gate", False):
        return

    def refresh(self, stats):
        progress = dict(
            replay_size=int(stats["size"]),
            adds_total=int(stats.get("adds_total", stats["size"])),
            global_step=int(self._state.global_step),
            actor_version=int(self._state.actor_version),
            pending_update_budget=0,
            warmup_required_updates=0,
            ready_for_online=False,
        )
        self._write_status(progress)
        return progress

    def train_once(self, *, stop_event=None):
        self._refresh_progress(self._replay_source.stats())
        return None

    def write_status(self, progress):
        write_json(self._status_path, dict(
            **progress,
            warmup_ready_adds_total=self._warmup_ready_adds_total,
            warmup_post_collect_updates=self._rl_config.warmup_post_collect_updates,
            training_frozen=True,
            training_enabled=False,
            collection_only=True,
            update_ratio=int(self._rl_config.grad_updates_per_cycle),
            timestamp=time.time(),
        ))

    learner_class._refresh_progress = refresh
    learner_class.train_once = train_once
    learner_class._write_status = write_status
    learner_class._cobot_collection_gate = True
