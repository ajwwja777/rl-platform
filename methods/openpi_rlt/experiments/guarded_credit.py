"""Episode-aware sampling and optional MC credit, independent of ROS and the UI.

Sparse recorded rewards are immutable. MC is an experimental off-policy target,
not a corrected ground truth or a claim of faithful baseline reproduction.
"""
from dataclasses import dataclass
from collections import defaultdict
import types
import numpy as np

@dataclass(frozen=True)
class Profile:
    sampling: str = "uniform"
    tail_fraction: float = 0.25
    mc_weight: float = 0.0

    def __post_init__(self):
        if self.sampling not in {"uniform", "terminal", "tail_mix", "episode_balanced"}:
            raise ValueError("Unknown sampling method")
        if not 0 <= self.tail_fraction <= 1 or not 0 <= self.mc_weight <= 1:
            raise ValueError("Fractions must be between zero and one")

def episode_credit(rows, gamma):
    """Derive a terminal-outcome return only for complete, consistent episodes.

    step_id is the raw control-step offset, not the stored-window rank. Reject
    nonterminal rewards, duplicate identities, conflicting outcomes and gaps.
    Overlapping windows are valid if they agree about reward and terminal time.
    """
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        phase = r.get("collection_phase", int(r.get("collection_phase_id", 0)))
        groups[(str(phase), int(r["episode_id"]))].append(i)
    target = np.zeros(len(rows), np.float32)
    valid = np.zeros(len(rows), bool)
    for ids in groups.values():
        starts = [int(rows[i]["step_id"]) for i in ids]
        if len(set(starts)) != len(starts): continue
        terminals = [i for i in ids if bool(rows[i]["done"])]
        if not terminals: continue
        endpoints = {int(rows[i]["step_id"]) + len(rows[i]["rewards"]) - 1 for i in terminals}
        labels = {int(rows[i]["success"]) for i in terminals}
        if len(endpoints) != 1 or len(labels) != 1 or not labels <= {0, 1}: continue
        end, label = next(iter(endpoints)), next(iter(labels))
        rewards_by_step = {}
        consistent = True
        for i in ids:
            start = int(rows[i]["step_id"])
            for offset, reward in enumerate(rows[i]["rewards"]):
                t = start + offset
                value = float(reward)
                if t > end or not np.isfinite(value) or value != (label if t == end else 0):
                    consistent = False
                if t in rewards_by_step and rewards_by_step[t] != value: consistent = False
                rewards_by_step[t] = value
        # A prefix may be missing after ring eviction; require continuity from
        # the first retained step, never fill internal trace gaps.
        first = min(starts)
        if not consistent or len(rewards_by_step) != end-first+1: continue
        for i in ids:
            target[i] = label * gamma ** (end-int(rows[i]["step_id"]))
            valid[i] = True
    return target, valid

def sample_indices(rng, indices, meta, profile, size=128):
    """Return indices with replacement. All pools are restricted to train IDs."""
    indices = np.asarray(indices, dtype=np.int64)
    if not len(indices): raise ValueError("Empty training pool")
    if profile.sampling == "uniform":
        return rng.choice(indices, size, replace=True)
    if profile.sampling == "terminal":
        pool = indices[[bool(meta[i]["done"]) for i in indices]]
        if not len(pool): raise ValueError("No terminal transitions")
        return rng.choice(pool, size, replace=True)
    if profile.sampling == "tail_mix":
        pool = indices[[meta[i]["portion"] == "late" for i in indices]]
        count = round(size * profile.tail_fraction)
        if count and not len(pool): raise ValueError("No tail transitions")
        selected = np.concatenate([rng.choice(indices, size-count, replace=True),
                                   rng.choice(pool, count, replace=True)])
        rng.shuffle(selected)
        return selected
    groups = defaultdict(list)
    for i in indices:
        groups[(meta[i]["phase"], meta[i]["episode_id"])].append(i)
    pools = list(groups.values())
    return np.array([rng.choice(pools[k]) for k in rng.integers(len(pools), size=size)])

def make_train_step(mc_weight=0.0, *, target_policy="native"):
    """Keep the upstream loop/Actor unchanged; replace only its Critic seam.

    No global monkeypatch. Each function owns a private globals dictionary, so
    multiple experiment variants can coexist in one process. Weight zero returns
    the exact original jitted function. Unknown/incomplete rows keep native TD.
    """
    from rlt_online_rl import trainer
    if not 0 <= mc_weight <= 1: raise ValueError("Invalid MC weight")
    if target_policy not in {"native", "clip", "takeover"}:
        raise ValueError("Unknown target policy")
    if mc_weight == 0 and target_policy == "native": return trainer.train_step
    import jax
    import jax.numpy as jnp
    import optax
    from rlt_online_rl.networks import build_td_target

    def update_critic(state, batch, actor, critic, rl_config):
        critic_rng, next_rng = jax.random.split(state.rng)
        td = build_td_target(actor, state.target_actor_params, critic,
            state.target_critic_params, batch["next_z_rl"], batch["next_proprio"],
            batch["next_ref_chunk"], batch["rewards"], batch["done"],
            rl_config.gamma, critic_rng)
        weight = mc_weight * batch["mc_valid"].astype(td.dtype)
        mc = batch["mc_return"]
        if target_policy == "takeover":
            # Different, explicitly censored objective. Replay labels stay intact.
            td = jnp.where(batch["bootstrap_cut"], 0., td)
            mc = jnp.where(batch["future_takeover"], 0., mc)
        target = (1-weight)*td + weight*mc
        if target_policy == "clip":
            target = jnp.clip(target, 0., 1.)
        target = jax.lax.stop_gradient(target)
        def loss(params):
            q1, q2 = critic.q_values(params, batch["z_rl"], batch["proprio"], batch["action_chunk"])
            value = jnp.mean((q1-target)**2) + jnp.mean((q2-target)**2)
            return value, {"critic_loss": value, "q1_mean": jnp.mean(q1),
                "q2_mean": jnp.mean(q2), "target_q_mean": jnp.mean(target),
                "mc_effective_weight": jnp.mean(weight)}
        (_, metrics), grads = jax.value_and_grad(loss, has_aux=True)(state.critic_params)
        updates, opt_state = state.critic_tx.update(grads, state.critic_opt_state, state.critic_params)
        return state.replace(critic_params=optax.apply_updates(state.critic_params, updates),
                             critic_opt_state=opt_state, rng=next_rng), metrics

    original = trainer.train_step.__wrapped__
    if "update_critic" not in original.__code__.co_names:
        raise RuntimeError("Upstream trainer changed; review the experiment adapter")
    namespace = dict(original.__globals__, update_critic=update_critic)
    run = types.FunctionType(original.__code__, namespace, original.__name__,
                             original.__defaults__, original.__closure__)
    run.__kwdefaults__ = original.__kwdefaults__
    return jax.jit(run, static_argnames=("actor", "critic", "rl_config", "use_action_adapter"))
