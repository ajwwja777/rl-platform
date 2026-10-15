"""Native episode/batch scheduling from the supplied train_pi_robo.py.

Warmup tests the previous episode count (first eligible episode is 11).
Warmup credit is discarded; batch mode carries credit across episode commits.
"""
from dataclasses import dataclass

@dataclass(frozen=True)
class Schedule:
    completed_episodes: int = 0
    env_steps: int = 0
    credit: int = 0
    episodes_since_update: int = 0

    def finish(self, transitions, *, batch_size=64, step_interval=30, episodes_per_update=1):
        if (transitions < 0 or batch_size <= 0 or step_interval <= 0
                or type(episodes_per_update) is not int or episodes_per_update < 1):
            raise ValueError('Invalid synchronous schedule parameters')
        steps = self.env_steps + transitions
        credit = self.credit + transitions
        eligible = self.completed_episodes >= 10 and steps >= batch_size
        pending = self.episodes_since_update + 1 if eligible else 0
        updates = 0
        if not eligible:
            credit = 0
        elif pending >= episodes_per_update:
            updates, credit = divmod(credit, step_interval)
            pending = 0
        return Schedule(self.completed_episodes + 1, steps, credit, pending), updates
