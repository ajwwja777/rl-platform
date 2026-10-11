"""Episode-local exact-input memoization for the fixed plug_v3 exploration server.

The server uses fixed key 42; reuse is restricted to this registered frozen
collection entry and on-demand replay building. No cross-Episode cache, nearby
state reuse, cached live anchors, altered window coverage, or precision change.
"""
from __future__ import annotations
import copy
import hashlib
import json
import os
import time
from .input_audit import conditional_input_receipt


def enabled(driver):
    return (os.environ.get("COBOT_DEPLOYMENT_MODEL_ID") == "plug-v3-hil8-explore"
            and os.environ.get("COBOT_RLT_RAW_OBSERVATION_CONTRACT") == "trace"
            and os.environ.get("COBOT_RLT_REPLAY_EXACT_CACHE", "1") == "1"
            and getattr(driver._env, "cobot_task2_contract", False)
            and driver._env_config.step_trace_stride == 0)


class ExactFeatureCache:
    def __init__(self, provider):
        self.provider = provider
        self.cache = {}
        self.hits = self.requests = 0
        self.hash_sec = self.inference_sec = 0.0

    def get_features(self, observation):
        return self.get_or_compute(observation, lambda: self.provider.get_features(observation))

    def get_or_compute(self, observation, compute):
        started = time.perf_counter()
        # Include every key, array dtype/shape/value and RTC prefix. Never key
        # by proprio alone, image timestamp alone, or observation list index.
        receipt = conditional_input_receipt(observation)
        key = hashlib.sha256(json.dumps(receipt, sort_keys=True, allow_nan=False).encode()).digest()
        self.hash_sec += time.perf_counter() - started
        if key in self.cache:
            self.hits += 1
            return copy.deepcopy(self.cache[key])
        started = time.perf_counter()
        value = compute()
        self.inference_sec += time.perf_counter() - started
        self.requests += 1
        self.cache[key] = copy.deepcopy(value)
        return value

    def summary(self):
        return dict(exact_cache_hits=self.hits, feature_requests=self.requests,
                    feature_hash_sec=self.hash_sec, feature_inference_sec=self.inference_sec)


def build(driver, raw_episode, original):
    if not enabled(driver):
        return original(driver, raw_episode)
    # Wrap only Replay lookup, never the live planner's shared feature provider.
    own_lookup = driver.__dict__.get('_feature_payload_for_observation')
    original_lookup = driver._feature_payload_for_observation
    cache = ExactFeatureCache(None)
    started = time.perf_counter()
    def lookup(raw, index, indexed_cache, stats):
        if int(index) in indexed_cache:
            return original_lookup(raw, index, indexed_cache, stats)
        value = cache.get_or_compute(raw.observations[int(index)],
                    lambda: original_lookup(raw, index, indexed_cache, stats))
        indexed_cache[int(index)] = value
        return value
    driver._feature_payload_for_observation = lookup
    try:
        transitions, stats = original(driver, raw_episode)
        stats.update(cache.summary())
        stats['replay_build_sec'] = time.perf_counter() - started
        print('COBOT_REPLAY_BUILD ' + json.dumps(dict(episode_id=raw_episode.episode_id,
              transitions=len(transitions), **cache.summary(),
              replay_build_sec=stats['replay_build_sec']), sort_keys=True), flush=True)
        return transitions, stats
    finally:
        if own_lookup is None:
            del driver._feature_payload_for_observation
        else:
            driver._feature_payload_for_observation = own_lookup
