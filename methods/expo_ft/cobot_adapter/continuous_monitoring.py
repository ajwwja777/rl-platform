"""Per-selection policy provenance for mixed-version continuous episodes."""
from pathlib import Path
import runpy
import time
from monitoring import RolloutHistory, finite, write_json


class ContinuousRolloutHistory(RolloutHistory):
    def __init__(self, root):
        super().__init__(root)
        self.policy_versions = []
        self.last_version = None

    def payload(self):
        return dict(schema=2, episode_uuid=self.identity, actor_version=self.version,
                    last_actor_version=self.last_version, policy_versions=list(self.policy_versions),
                    points=list(self.points))

    def selected(self, active, payload):
        if not active or not finite(payload.get('selected_q')):
            return
        with self.lock:
            if active['episode_uuid'] != self.identity:
                if self.identity and self.points:
                    self.pending.append(self.payload())
                self.identity = active['episode_uuid']
                self.version = active.get('actor_version', payload['actor_version'])
                self.policy_versions = [self.version]
                self.points.clear()
            version = payload['actor_version']
            precache = payload['precache_actor_version']
            for used in (precache, version):
                if used not in self.policy_versions:
                    self.policy_versions.append(used)
            self.last_version = version
            self.points.append(dict(seconds=max(0., time.time()-active['started_at']),
                q=payload['selected_q'], timestamp=time.time(),
                actor_version=version, precache_actor_version=precache))
            self.dirty = True

    def flush(self):
        with self.lock:
            if not self.dirty:
                return
            value = self.payload()
            self.dirty = False
            pending, self.pending = self.pending, []
        try:
            archive = runpy.run_path(str(Path(__file__).resolve().parents[3]/'integrations/cobot_runtime/rollout_archive.py'))
            for snapshot in [*pending, value]:
                archive['save_snapshot'](self.path.parent.parent, snapshot)
        except Exception:
            with self.lock:
                self.pending = pending + [value] + self.pending
                self.dirty = True
        write_json(self.path, value)
