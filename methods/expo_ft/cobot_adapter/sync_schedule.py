"""Synchronous scheduling from the supplied train_pi_robo.py episode branch.

Episode count is tested before increment, so a fresh run first trains at the
end of episode 11. Warmup credits are discarded. No per-episode update cap.
"""
from dataclasses import dataclass

@dataclass(frozen=True)
class Schedule:
    completed_episodes: int = 0
    env_steps: int = 0
    credit: int = 0

    def finish(self, transitions, *, batch_size=64, step_interval=30):
        if transitions < 0 or batch_size <= 0 or step_interval <= 0:
            raise ValueError('Invalid synchronous schedule parameters')
        steps = self.env_steps + transitions
        credit = self.credit + transitions
        eligible = self.completed_episodes >= 10 and steps >= batch_size
        updates, remainder = divmod(credit, step_interval) if eligible else (0, 0)
        return Schedule(self.completed_episodes + 1, steps, remainder), updates
