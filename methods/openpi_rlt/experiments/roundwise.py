"""Opt-in complete-Episode round data and disjoint sampling; no Replay mutation.

Tail windows are time proxies, not semantic insertion stages. Takeover cuts define
an experimental autonomous-return objective, not proof every pre-HIL action failed.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib

import numpy as np

POOLS = ("expert", "human_correction", "auto_success", "failure")
HUMAN = (2, 3)


def episode_key(row, namespace="plug-v3"):
    phase = str(row.get("collection_phase", {1: "warmup", 2: "online"}.get(
        int(row.get("collection_phase_id", 0)), "unknown")))
    uid = row.get("task5_episode_uuid") or row.get("episode_uuid")
    identity = "uuid:" + str(uid) if uid else "legacy:" + str(int(row["episode_id"]))
    return namespace + ":" + phase + ":" + identity


def heldout(key, fraction=.2, salt="roundwise-v1"):
    if not 0 <= fraction < 1:
        raise ValueError("heldout fraction must be in [0,1)")
    value = int(hashlib.sha256((salt + ":" + key).encode()).hexdigest()[:16], 16)
    return value / 2**64 < fraction


def annotate(rows, *, gamma=.99, tail_steps=60, takeover_before=20,
             takeover_after=10, heldout_fraction=.2, namespace="plug-v3", rounds=None, legacy_expert_id_base=None):
    """Validate sparse success rewards against logical timestamps and split Episodes.

    Unknown, conflicting, missing-prefix/gapped Episodes and mixed-source windows
    are excluded rather than assigned invented outcomes. Legacy keys lack UUID
    proof; a repeated step identity is rejected. Original rows stay unchanged.
    """
    if not 0 < gamma <= 1 or min(tail_steps, takeover_before, takeover_after) < 0:
        raise ValueError("invalid time/discount settings")
    rounds = rounds or {}
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[episode_key(row, namespace)].append(i)
    result = [dict(index=i, eligible=False, reason="unvalidated") for i in range(len(rows))]
    for key, ids in groups.items():
        ids.sort(key=lambda i: int(rows[i]["step_id"]))
        def reject(reason):
            for i in ids:
                result[i].update(episode_key=key, reason=reason)
        starts = [int(rows[i]["step_id"]) for i in ids]
        if len(set(starts)) != len(starts):
            reject("duplicate_episode_step_identity"); continue
        terminals = [i for i in ids if bool(rows[i]["done"])]
        if not terminals:
            reject("no_terminal_outcome"); continue
        ends = {int(rows[i]["step_id"]) + len(rows[i]["rewards"]) - 1 for i in terminals}
        outcomes = {int(rows[i].get("success", -1)) for i in terminals}
        if len(ends) != 1 or len(outcomes) != 1 or not outcomes <= {0, 1}:
            reject("conflicting_terminal_identity"); continue
        end, success = next(iter(ends)), next(iter(outcomes))
        timeline, human_times, human_by_step = {}, set(), {}
        valid = True
        for i in ids:
            start = int(rows[i]["step_id"])
            rewards = np.asarray(rows[i]["rewards"]).reshape(-1)
            sources = np.asarray(rows[i]["source_chunk"]).reshape(-1)
            if not len(rewards) or len(rewards) != len(sources) or not np.isin(sources, [0, 1, 2, 3]).all():
                valid = False; break
            for slot, reward in enumerate(rewards):
                t = start + slot
                expected = float(success) if t == end else 0.
                if t > end or not np.isfinite(reward) or reward != expected or (t in timeline and timeline[t] != reward):
                    valid = False
                timeline[t] = float(reward)
                is_human = sources[slot] in HUMAN
                if t in human_by_step and human_by_step[t] != is_human: valid = False
                human_by_step[t] = is_human
                if is_human:
                    human_times.add(t)
        if not valid or starts[0] != 0 or len(timeline) != end + 1:
            reject("inconsistent_sparse_reward_or_incomplete_timeline"); continue
        takeover_steps = sorted(t for t in human_times if t-1 not in human_times)
        first_human = takeover_steps[0] if takeover_steps else None
        expert = all(int(rows[i]["episode_id"]) < 0 for i in ids)
        if legacy_expert_id_base is not None and all(rows[i].get("collection_phase") == "warmup"
                and int(rows[i]["episode_id"]) >= legacy_expert_id_base for i in ids):
            expert = True
        if expert and len(human_times) != end + 1:
            reject("expert_identity_without_all_human_source"); continue
        development = not expert and heldout(key, heldout_fraction)
        for i in ids:
            row = rows[i]; start = int(row["step_id"])
            source = np.asarray(row["source_chunk"])
            human = np.isin(source, HUMAN)
            mixed = human.any() and not human.all()
            next_human = next((t for t in takeover_steps if t >= start), None)
            pre = next_human is not None and not human.any() and not expert
            pool = ("expert" if expert else "human_correction" if human.all()
                    else "auto_success" if success and first_human is None else "failure")
            critical = start >= max(0, end + 1 - tail_steps)
            if first_human is not None and not expert:
                critical |= any(t - takeover_before <= start <= t + takeover_after for t in takeover_steps)
            autonomous_cut = pre and start + len(source) >= next_human
            # Use raw logical time, including the reward slot in the final
            # chunk. Assisted pre-HIL zero is an explicitly different objective.
            mc = 0. if pre else float(success) * gamma ** (end - start)
            result[i].update(episode_key=key, eligible=bool(critical and not mixed),
                reason="mixed_source_window" if mixed else "outside_tail_time_proxy" if not critical else "eligible",
                pool=pool, episode_success=bool(success), episode_assisted=first_human is not None and not expert,
                outcome="expert_success" if expert and success else "assisted_success" if success and first_human is not None
                    else "auto_success" if success else "failure",
                split="development" if development else "train", logical_start=start, logical_end=end,
                reward_bound=[0., 1.], mc_return=mc, autonomy_cut=autonomous_cut,
                pre_takeover=pre, first_human_step=first_human, next_human_step=next_human, takeover_steps=takeover_steps, round_id=rounds.get(key),
                uuid_verified=bool(row.get("task5_episode_uuid") or row.get("episode_uuid")))
    return result


@dataclass(frozen=True)
class SamplingProfile:
    # Weights normalize to 100%; the review's 20/20/20/30 sums to 90%.
    pool_weights: tuple = (.2, .2, .2, .3)
    recent_round_weight: float = 2.
    recent_round: str | None = None

    def __post_init__(self):
        if len(self.pool_weights) != len(POOLS) or any(not np.isfinite(x) or x < 0 for x in self.pool_weights):
            raise ValueError("four finite nonnegative weights required")
        if sum(self.pool_weights) <= 0 or not np.isfinite(self.recent_round_weight) or self.recent_round_weight <= 0:
            raise ValueError("positive sampling weights required")


class RoundSampler:
    """Quota by disjoint category, then weighted Episode, then a row in that Episode."""
    def __init__(self, metadata, *, profile=SamplingProfile(), indices=None):
        self.metadata, self.profile = metadata, profile
        allowed = set(range(len(metadata))) if indices is None else set(map(int, indices))
        self.groups = {p: defaultdict(list) for p in POOLS}
        for i, m in enumerate(metadata):
            if i in allowed and m.get("eligible") and m.get("split") == "train":
                self.groups[m["pool"]][m["episode_key"]].append(i)
        self.draws = np.zeros(len(metadata), np.int64)
        self.pool_draws = dict.fromkeys(POOLS, 0)
        self.total = self.recent = 0

    def sample(self, rng, size):
        if size <= 0:
            raise ValueError("positive batch size required")
        weights = np.array([w if self.groups[p] else 0. for p, w in zip(POOLS, self.profile.pool_weights)])
        if weights.sum() <= 0:
            raise ValueError("empty eligible training pools")
        exact = size * weights / weights.sum()
        counts = np.floor(exact).astype(int)
        for i in np.argsort(-(exact - counts), kind="stable")[:size - counts.sum()]:
            counts[i] += 1
        parts = []
        for pool, count in zip(POOLS, counts):
            if not count: continue
            groups = self.groups[pool]; keys = list(groups)
            ep_weights = np.array([self.profile.recent_round_weight if self.profile.recent_round is not None
                and self.metadata[groups[k][0]].get("round_id") == self.profile.recent_round else 1. for k in keys])
            episodes = rng.choice(len(keys), int(count), p=ep_weights / ep_weights.sum())
            parts.extend(int(rng.choice(groups[keys[e]])) for e in episodes)
            self.pool_draws[pool] += int(count)
        selected = np.asarray(parts, np.int64); rng.shuffle(selected)
        np.add.at(self.draws, selected, 1)
        self.total += len(selected)
        self.recent += sum(self.profile.recent_round is not None and self.metadata[i].get("round_id") == self.profile.recent_round for i in selected)
        return selected

    def receipt(self):
        return {"draws": self.total, "pool_draws": self.pool_draws,
            "actual_pool_ratio": {p: n / self.total if self.total else None for p, n in self.pool_draws.items()},
            "recent_ratio": self.recent / self.total if self.total and self.profile.recent_round is not None else None,
            "recent_identity_available": self.profile.recent_round is not None,
            "unique_rows_drawn": int((self.draws > 0).sum()),
            "per_pool_mean_row_reuse": {p: float(self.draws[[i for ids in g.values() for i in ids]].mean()) if g else None
                for p, g in self.groups.items()}}


def derive_batch(raw, metadata, indices):
    """Return private target fields without changing rewards/source/reference assets."""
    out = {k: np.asarray(v)[indices].copy() for k, v in raw.items()}
    metas = [metadata[int(i)] for i in indices]
    cut = np.asarray([m["autonomy_cut"] for m in metas])
    out["done"] = np.logical_or(out["done"], cut)
    out["rewards"][cut] = 0.
    out["mc_return"] = np.asarray([m["mc_return"] for m in metas], np.float32)
    out["autonomous_success"] = np.asarray([m["pool"] == "auto_success" for m in metas])
    out["autonomy_cut"] = cut
    return out
