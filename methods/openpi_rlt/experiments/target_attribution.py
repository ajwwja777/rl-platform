"""Validate chunk evidence and reconstruct observed (not optimal) returns.

Overlapping windows are one trajectory. Missing steps, contradictory overlaps
or conflicting terminal labels must not silently become a return target.
"""
from __future__ import annotations

import collections
import numpy as np


def reconstruct_observed_returns(data, gamma):
    count = len(data['episode_id'])
    values = np.full(count, np.nan, dtype=np.float64)
    groups = collections.defaultdict(list)
    for i in range(count):
        groups[(int(data['collection_phase_id'][i]), int(data['episode_id'][i]))].append(i)
    reports = []
    for key, ids in sorted(groups.items()):
        rewards, sources, terminals, conflicts = {}, {}, set(), set()
        starts = [int(data['step_id'][i]) for i in ids]
        duplicate_starts = len(starts) - len(set(starts))
        for i in ids:
            start = int(data['step_id'][i])
            chunk_rewards = np.asarray(data['rewards'][i])
            for j, reward in enumerate(chunk_rewards):
                step = start + j
                if step in rewards and rewards[step] != float(reward):
                    conflicts.add(step)
                rewards[step] = float(reward)
                source = int(data['source_chunk'][i, j])
                if step in sources and sources[step] != source:
                    conflicts.add(step)
                sources[step] = source
            if bool(data['done'][i]):
                # This audit's contract is a full C-step terminal-aligned window.
                terminals.add((start + len(chunk_rewards) - 1, int(data['success'][i])))
        valid_terminal = len(terminals) == 1
        terminal = next(iter(terminals)) if valid_terminal else None
        for i in ids:
            start = int(data['step_id'][i])
            if terminal is None or conflicts or duplicate_starts:
                continue
            end, success = terminal
            if success not in (0, 1) or start > end:
                continue
            if any(step not in rewards for step in range(start, end + 1)):
                continue
            observed = np.asarray([rewards[step] for step in range(start, end + 1)])
            if not np.isfinite(observed).all():
                continue
            values[i] = np.dot(observed, gamma ** np.arange(len(observed)))
        nonzero = sorted(step for step, reward in rewards.items() if reward != 0)
        terminal_reward_consistent = bool(terminal is not None and nonzero == ([terminal[0]] if terminal[1] else [])
                                          and rewards.get(terminal[0]) == float(terminal[1]))
        # A malformed reward/terminal association is not usable ground truth.
        if not terminal_reward_consistent:
            values[ids] = np.nan
        reports.append(dict(phase_id=key[0], episode_id=key[1], windows=len(ids),
            unique_observed_steps=len(rewards), terminal=list(terminal) if terminal else None,
            duplicate_starts=duplicate_starts, conflicting_overlap_steps=sorted(conflicts),
            terminal_reward_consistent=terminal_reward_consistent,
            valid_return_windows=int(np.isfinite(values[ids]).sum()),
            boundary='Discounted observed behavior, including assistance; not autonomous or alternative-action value. Logical indices do not verify physical elapsed time.'))
    return values, reports


def episode_interval(values, episode_ids, phases, mask, seed=42):
    groups = collections.defaultdict(list)
    for value, ep, phase, keep in zip(values, episode_ids, phases, mask):
        if keep and np.isfinite(value):
            groups[(int(phase), int(ep))].append(float(value))
    means = np.asarray([np.mean(v) for _, v in sorted(groups.items())])
    if not len(means):
        return dict(episodes=0, mean=None, ci95=None)
    ci = None
    if len(means) >= 2:
        samples = np.random.default_rng(seed).choice(means, (10000, len(means)), replace=True).mean(1)
        ci = np.quantile(samples, [.025, .975]).tolist()
    return dict(episodes=len(means), mean=float(means.mean()), ci95=ci)
