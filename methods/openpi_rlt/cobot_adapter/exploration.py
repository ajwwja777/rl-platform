"""Bounded, per-plan exploration for the explicit frozen collection entry.

Adapted from the uncommitted online-review-20261011 prototype. Sampling has no
filesystem I/O. Evidence travels with queue rows through the existing async trace.
Base Actor is deterministic; exploration is enabled only for latched collection.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ExplorationConfig:
    sigma_fraction: float = .15
    rho: float = .5
    clip_sigma: float = 2.
    seed: int | None = None

    def __post_init__(self):
        if not 0 < self.sigma_fraction <= .15 or not 0 <= self.rho < 1 or not 0 < self.clip_sigma <= 2:
            raise ValueError("Exploration parameters exceed the reviewed bounds")


class ChunkExploration:
    def __init__(self, config, q01, q99):
        self.config = config
        half = (np.asarray(q99, np.float64)[:6] - np.asarray(q01, np.float64)[:6]) / 2
        if half.shape != (6,) or not np.all(np.isfinite(half)) or np.any(half <= 0):
            raise ValueError("Six finite positive joint normalization ranges required")
        self.sigma = config.sigma_fraction * half
        # A replayable session seed, without repeating the same trajectory on reload.
        self.seed = int(config.seed if config.seed is not None else np.random.SeedSequence().entropy)
        self.rng = np.random.default_rng(self.seed)
        self.context = None
        self.offset = None
        self.index = 0

    def apply(self, actions, *, context):
        actions = np.asarray(actions, np.float32)
        if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
            raise ValueError("Exploration needs finite right-arm absolute targets (H,7)")
        if context != self.context:
            self.context, self.offset, self.index = context, None, 0
        noise = self.rng.standard_normal(6) * self.sigma
        if self.offset is not None:
            noise = self.config.rho * self.offset + np.sqrt(1-self.config.rho**2) * noise
        self.offset = np.clip(noise, -self.config.clip_sigma*self.sigma, self.config.clip_sigma*self.sigma)
        out = actions.copy()
        out[:, :6] += self.offset.astype(np.float32)
        receipt = dict(offset_rad=self.offset.astype(np.float32).tolist(), sigma_rad=self.sigma.tolist(),
                       plan_index=self.index, seed=str(self.seed), rho=self.config.rho,
                       clip_sigma=self.config.clip_sigma)
        self.index += 1
        return out, receipt


def attach(engine, driver):
    from integrations.cobot_runtime.exploration_collection import selected, ACTOR_SHA256, STATS_SHA256
    if not selected():
        return
    import os
    import hashlib
    from pathlib import Path
    if os.environ.get("RLT_DISABLE_LEARNER") != "1" or not driver._env_config.actor_deterministic:
        raise ValueError("Exploration requires frozen deterministic Actor and disabled Learner")
    from rlt_online_rl import inference
    stats = inference.ActionRepresentationAdapter.from_config(driver._rl_config).stats
    engine.explorer = ChunkExploration(ExplorationConfig(), stats.q01, stats.q99)
    application = getattr(driver._env._io, "_session_application", None)
    lifecycle = getattr(application, "_task5", None)
    if lifecycle is None or not hasattr(lifecycle, "collecting"):
        raise ValueError("Exploration requires latched shared collection lifecycle")
    engine.exploration_collecting = lambda: lifecycle.collecting
    code_files = [Path(__file__), Path(__file__).with_name("async_execution.py"),
                  Path(__file__).with_name("execution_runtime.py")]
    if engine.config.smoothing_order == 2:
        code_files.append(Path(__file__).with_name("cascade_filter.py"))
    engine.exploration_identity = dict(model_id=os.environ["COBOT_DEPLOYMENT_MODEL_ID"],
                                      code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in code_files},
                                      actor_sha256=ACTOR_SHA256, stats_sha256=STATS_SHA256,
                                      learner_enabled=False, base_actor_deterministic=True)
    engine.stats.update(exploration_collection=True, exploration_sigma_rad=engine.explorer.sigma.tolist(),
                        exploration_bound_rad=(2*engine.explorer.sigma).tolist(), **engine.exploration_identity)
