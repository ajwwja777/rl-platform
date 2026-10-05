"""Opt-in preservation of actual Cobot trace observations in native Replay."""
from __future__ import annotations

import os

import numpy as np


def selected_contract() -> str:
    mode = os.environ.get('COBOT_RLT_RAW_OBSERVATION_CONTRACT', 'legacy')
    if mode not in {'legacy', 'trace'}:
        raise ValueError('COBOT_RLT_RAW_OBSERVATION_CONTRACT must be legacy or trace')
    return mode


def preserve_trace_observations(driver, raw_episode, kwargs, step_start, anchors_before) -> None:
    """Reindex new steps without rewriting the previous action's next-state.

    Native append assumes every step begins at the preceding next-observation.
    Cobot can resample after pauses or fresh planning. Preserve both observations
    and move feature anchors with their actual step; no invented gap transition.
    """
    records = kwargs['trace_records']
    if not records:
        return
    reindexed = []
    for offset, record in enumerate(records):
        step = raw_episode.steps[step_start+offset]
        previous_index = step.observation_idx
        if raw_episode.observations[previous_index] is not record.observation:
            step.observation_idx = len(raw_episode.observations)
            raw_episode.observations.append(record.observation)
            reindexed.append({'step': step_start+offset,
                              'previous_observation_index': previous_index,
                              'actual_observation_index': step.observation_idx})
    chunk = raw_episode.chunks[-1]
    chunk.observation_idx = raw_episode.steps[step_start].observation_idx
    # A Cobot first-step takeover can change observation after the initial plan.
    # Such cached features must not be attached to that new input.
    state = raw_episode.observations[chunk.observation_idx]['state']
    if chunk.start_proprio is not None and not np.array_equal(
            np.asarray(state, np.float32), np.asarray(chunk.start_proprio, np.float32)):
        chunk.start_z_rl = chunk.start_proprio = chunk.start_ref_chunk = None
    # Native append placed newly cached policy features at the old indices.
    # Restore pre-existing anchors (including the previous action's next-state)
    # and reattach this append's anchors using corrected step identities.
    raw_episode.summary['feature_anchors'] = anchors_before
    for offset, features in zip(kwargs['policy_anchor_offsets'], kwargs['policy_anchor_features']):
        absolute = step_start+int(offset)
        if step_start <= absolute < len(raw_episode.steps):
            driver._record_feature_anchor(raw_episode,
                                         raw_episode.steps[absolute].observation_idx, features)
    raw_episode.summary.setdefault('cobot_observation_reindex_receipts', []).extend(reindexed)
