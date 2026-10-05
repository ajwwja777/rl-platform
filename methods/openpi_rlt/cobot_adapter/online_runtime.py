"""Narrow runtime patch for upstream's single-arm action representation.

The public online runtime hard-codes ``:6`` as its joint group. Cobot uses
``[left 6 joints, left gripper, right 6 joints, right gripper]``. This module
keeps the fixed upstream checkout untouched and replaces only that adapter at
process startup.
"""

from __future__ import annotations

import os
import sys
from methods.openpi_rlt.cobot_adapter.actor_pinning import EpisodeActorPin
import time

import jax.numpy as jnp
import numpy as np
import rlt_online_rl.action_representation as upstream_actions

_ORIGINAL_ADAPTER = upstream_actions.ActionRepresentationAdapter
_ORIGINAL_JAX_DENORMALIZE = upstream_actions.jax_denormalize_to_abs_chunk
_JOINT_INDICES = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
_PATCH_INSTALLED = False


def _mark_cobot_replay_finalized(env) -> None:
    if getattr(env, "cobot_task2_contract", False):
        env.mark_replay_finalized()


def _mark_previous_step_terminal(raw_episode, success: int) -> None:
    if not raw_episode.steps:
        return
    last_step = raw_episode.steps[-1]
    last_step.done = True
    last_step.reward = float(success)
    last_step.success = int(success)
    if raw_episode.chunks:
        raw_episode.chunks[-1].done = True
        raw_episode.chunks[-1].success = int(success)


class CobotPhaseAwareActorClient:
    """Use the frozen reference policy until the episode phase is online."""

    def __init__(self, actor_client, env) -> None:
        self._actor_client = actor_client
        self._env = env

    def infer(self, request):
        phase = str(self._env.current_phase_name()).split(":", 1)[0]
        if phase == "warmup":
            from rlt_online_rl.inference import ActorResponse
            from rlt_online_rl.replay import TransitionSource

            return ActorResponse(
                refined_chunk=np.asarray(request.ref_chunk, dtype=np.float32),
                actor_param_version=-1,
                request_id=request.request_id,
                timestamp=time.time(),
                source=int(TransitionSource.BASE),
            )
        return self._actor_client.infer(request)

    def get_actor_param_version(self) -> int:
        return self._actor_client.get_actor_param_version()


def _broadcast_state0(state0: np.ndarray, chunk: np.ndarray) -> np.ndarray:
    state = np.asarray(state0, dtype=np.float32)
    while state.ndim < chunk.ndim:
        state = np.expand_dims(state, axis=-2)
    return state


class CobotBimanualActionRepresentationAdapter(_ORIGINAL_ADAPTER):
    """Upstream-compatible adapter with both Cobot joint groups in delta form."""

    def _abs_to_delta_chunk(self, chunk_abs: np.ndarray, state0: np.ndarray) -> np.ndarray:
        chunk = np.asarray(chunk_abs, dtype=np.float32)
        state = np.asarray(state0, dtype=np.float32)
        if chunk.shape[-1] != 14 or state.shape[-1] != 14:
            return super()._abs_to_delta_chunk(chunk, state)
        state = _broadcast_state0(state, chunk)
        delta = chunk.copy()
        zero_rows = np.all(np.isclose(chunk, 0.0), axis=-1, keepdims=True)
        delta[..., _JOINT_INDICES] = chunk[..., _JOINT_INDICES] - state[..., _JOINT_INDICES]
        return np.where(zero_rows, 0.0, delta)

    def _delta_to_abs_chunk(self, chunk_delta: np.ndarray, state0: np.ndarray) -> np.ndarray:
        chunk = np.asarray(chunk_delta, dtype=np.float32)
        state = np.asarray(state0, dtype=np.float32)
        if chunk.shape[-1] != 14 or state.shape[-1] != 14:
            return super()._delta_to_abs_chunk(chunk, state)
        state = _broadcast_state0(state, chunk)
        absolute = chunk.copy()
        absolute[..., _JOINT_INDICES] = chunk[..., _JOINT_INDICES] + state[..., _JOINT_INDICES]
        return absolute


def cobot_jax_denormalize_to_abs_chunk(
    chunk_norm: jnp.ndarray,
    state0: jnp.ndarray,
    q01: jnp.ndarray,
    q99: jnp.ndarray,
    *,
    action_representation: str,
) -> jnp.ndarray:
    """JAX inverse used by the actor loss, extended to the right arm."""
    if chunk_norm.shape[-1] != 14 or state0.shape[-1] != 14:
        return _ORIGINAL_JAX_DENORMALIZE(
            chunk_norm,
            state0,
            q01,
            q99,
            action_representation=action_representation,
        )
    chunk = upstream_actions.jax_quantile_denormalize(chunk_norm, q01, q99)
    if action_representation == "abs":
        return chunk
    state = jnp.asarray(state0, dtype=jnp.float32)
    while state.ndim < chunk.ndim:
        state = jnp.expand_dims(state, axis=-2)
    return chunk.at[..., _JOINT_INDICES].add(state[..., _JOINT_INDICES])


def install_bimanual_runtime_patch() -> None:
    """Install the Cobot adapter before importing any Machine B role."""
    global _PATCH_INSTALLED
    if not _PATCH_INSTALLED:
        upstream_actions.ActionRepresentationAdapter = CobotBimanualActionRepresentationAdapter
        upstream_actions.jax_denormalize_to_abs_chunk = cobot_jax_denormalize_to_abs_chunk
        _PATCH_INSTALLED = True

    from rlt_online_rl import inference

    if not getattr(inference.EnvDriver._append_raw_chunk, "_cobot_terminal_last_action", False):
        original_append_raw_chunk = inference.EnvDriver._append_raw_chunk

        def _append_raw_chunk_with_terminal_last_action(driver, raw_episode, **kwargs):
            if (
                getattr(driver._env, "cobot_task2_contract", False)
                and kwargs.get("done")
                and not kwargs.get("trace_records")
            ):
                # The outcome can arrive after an operator pause, without any
                # new action.  Upstream always appends a chunk, even for an
                # empty trace; that chunk's step_start has no replay position
                # and crashes replay window construction.  Attribute the
                # result to the last executed action and omit the empty chunk.
                _mark_previous_step_terminal(raw_episode, int(kwargs.get("success", 0)))
                return kwargs["observation_idx"]
            return original_append_raw_chunk(driver, raw_episode, **kwargs)

        _append_raw_chunk_with_terminal_last_action._cobot_terminal_last_action = True
        inference.EnvDriver._append_raw_chunk = _append_raw_chunk_with_terminal_last_action

    if os.environ.get("COBOT_RLT_PIN_ACTOR_PER_EPISODE", "0") == "1" and not getattr(
        inference.ActorService.infer, "_cobot_episode_actor_pin", False
    ):

        def _infer_with_episode_pin(service, request):
            with service._lock:
                pin = getattr(service, "_cobot_episode_actor_pin_state", None)
                if pin is None:
                    pin = EpisodeActorPin()
                    service._cobot_episode_actor_pin_state = pin
                actor_params, actor_version = pin.select(
                    episode_id=request.episode_id,
                    latest_params=service._actor_params,
                    latest_version=service._actor_version,
                )
                infer_rng = None
                if actor_params is not None and not request.deterministic:
                    service._rng, infer_rng = inference.jax.random.split(service._rng)
            if actor_params is None:
                return inference.ActorResponse(
                    refined_chunk=np.asarray(request.ref_chunk, dtype=np.float32),
                    actor_param_version=actor_version,
                    request_id=request.request_id,
                    timestamp=time.time(),
                    source=int(inference.TransitionSource.BASE),
                )
            model_ref_chunk = np.asarray(request.ref_chunk, dtype=np.float32)
            if service._action_adapter is not None:
                model_ref_chunk = service._action_adapter.normalize_ref_chunk(
                    model_ref_chunk, request.proprio
                )
            refined_chunk = service._wrapper.infer(
                actor_params,
                request.z_rl,
                request.proprio,
                model_ref_chunk,
                rng=infer_rng,
                deterministic=request.deterministic,
            )
            if service._action_adapter is not None:
                refined_chunk = service._action_adapter.denormalize_to_abs_chunk(
                    refined_chunk, request.proprio
                )
            return inference.ActorResponse(
                refined_chunk=refined_chunk,
                actor_param_version=actor_version,
                request_id=request.request_id,
                timestamp=time.time(),
                source=int(inference.TransitionSource.RL),
            )

        _infer_with_episode_pin._cobot_episode_actor_pin = True
        inference.ActorService.infer = _infer_with_episode_pin

    if not getattr(inference.EnvDriver.__init__, "_cobot_phase_aware_actor", False):
        original_driver_init = inference.EnvDriver.__init__

        def _driver_init_with_phase_actor(driver, *args, **kwargs):
            original_driver_init(driver, *args, **kwargs)
            if hasattr(driver._env, "current_phase_name") and not isinstance(
                driver._actor_client, CobotPhaseAwareActorClient
            ):
                driver._actor_client = CobotPhaseAwareActorClient(driver._actor_client, driver._env)

        _driver_init_with_phase_actor._cobot_phase_aware_actor = True
        inference.EnvDriver.__init__ = _driver_init_with_phase_actor

    if not getattr(inference.ActorClient.get_actor_param_version, "_cobot_retryable_timeout", False):
        original_get_version = inference.ActorClient.get_actor_param_version

        def _get_actor_param_version(client):
            try:
                return original_get_version(client)
            except TimeoutError as error:
                raise RuntimeError("actor_service version request timed out") from error

        _get_actor_param_version._cobot_retryable_timeout = True
        inference.ActorClient.get_actor_param_version = _get_actor_param_version

    if not getattr(inference.EnvDriver._execute_chunk, "_cobot_trace_fallback", False):
        original_execute_chunk = inference.EnvDriver._execute_chunk

        def _execute_chunk_with_trace(driver, observation, policy_planner):
            if getattr(driver._env, "cobot_task2_contract", False):
                return driver._env.execute_chunk(observation, policy_planner)
            if hasattr(driver._env, "execute_chunk"):
                return original_execute_chunk(driver, observation, policy_planner)

            plan = policy_planner(observation, 0)
            action_chunk = np.asarray(plan.action_chunk, dtype=np.float32)
            ref_chunk = np.asarray(plan.ref_chunk, dtype=np.float32)
            horizon = min(int(driver._env_config.chunk_exec_horizon), action_chunk.shape[0])
            period = 1.0 / max(float(driver._env_config.control_frequency_hz), 1e-6)
            rewards: list[float] = []
            step_trace: list[dict] = []
            current_observation = observation
            done = False
            final_info: dict = {}
            for index in range(horizon):
                tick_start = time.perf_counter()
                next_observation, reward, terminated, truncated, step_info = driver._env.step(action_chunk[index])
                done = bool(terminated or truncated)
                rewards.append(float(reward))
                final_info = dict(step_info)
                step_trace.append(
                    {
                        "observation": current_observation,
                        "action": action_chunk[index].copy(),
                        "ref_action": ref_chunk[index].copy(),
                        "reward": float(reward),
                        "next_observation": next_observation,
                        "human_controlled": False,
                        "source": int(plan.source),
                        "actor_param_version": int(plan.actor_param_version),
                        "done": done,
                    }
                )
                current_observation = next_observation
                remaining = period - (time.perf_counter() - tick_start)
                if remaining > 0:
                    time.sleep(remaining)
                if done:
                    break
            final_info.update(
                {
                    "step_trace": step_trace,
                    "source": int(plan.source),
                    "intervention_flag": False,
                    "chunk_start_features": plan.start_features,
                    "policy_anchor_offsets": [],
                    "policy_anchor_features": [],
                }
            )
            return current_observation, rewards, done, final_info

        _execute_chunk_with_trace._cobot_trace_fallback = True
        inference.EnvDriver._execute_chunk = _execute_chunk_with_trace

    if not getattr(inference.EnvDriver._build_episode_replay, "_cobot_replay_gate", False):
        original_build_episode_replay = inference.EnvDriver._build_episode_replay

        def _build_episode_replay_with_gate(driver, raw_episode):
            if getattr(driver._env, "cobot_task2_contract", False):
                allowed = getattr(driver._env, "replay_commit_allowed", lambda: False)()
                if not allowed:
                    return [], {
                        "raw_step_count": len(raw_episode.steps),
                        "raw_chunk_count": len(raw_episode.chunks),
                        "replay_transition_count": 0,
                        "replay_skipped_reason": "cobot_episode_not_eligible",
                    }
            return original_build_episode_replay(driver, raw_episode)

        _build_episode_replay_with_gate._cobot_replay_gate = True
        inference.EnvDriver._build_episode_replay = _build_episode_replay_with_gate

    if not getattr(inference.EnvDriver._persist_raw_episode, "_cobot_task5_raw_source", False):
        original_persist_raw_episode = inference.EnvDriver._persist_raw_episode

        def _persist_raw_episode_with_task5_source(driver, raw_episode, *, episode_id, started_at):
            if getattr(driver._env, "cobot_task2_contract", False):
                allowed = getattr(driver._env, "persist_upstream_raw_episode", lambda: True)()
                if not allowed:
                    return None
            return original_persist_raw_episode(
                driver,
                raw_episode,
                episode_id=episode_id,
                started_at=started_at,
            )

        _persist_raw_episode_with_task5_source._cobot_task5_raw_source = True
        inference.EnvDriver._persist_raw_episode = _persist_raw_episode_with_task5_source

    if not getattr(inference.EnvDriver.run_episode, "_cobot_session_finalize", False):
        original_run_episode = inference.EnvDriver.run_episode

        def _run_episode_with_session_finalize(driver, episode_id):
            if getattr(driver._env, "cobot_task2_contract", False):
                driver._env._trace_replay_episode_id = int(episode_id)
                io = getattr(driver._env, "_io", None)
                if hasattr(io, "_trace_replay_episode_id"):
                    io._trace_replay_episode_id = int(episode_id)
            result = original_run_episode(driver, episode_id)
            _mark_cobot_replay_finalized(driver._env)
            return result

        _run_episode_with_session_finalize._cobot_session_finalize = True
        inference.EnvDriver.run_episode = _run_episode_with_session_finalize

    for module_name in ("rlt_online_rl.inference", "rlt_online_rl.trainer"):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        if hasattr(module, "ActionRepresentationAdapter"):
            module.ActionRepresentationAdapter = CobotBimanualActionRepresentationAdapter
        if hasattr(module, "jax_denormalize_to_abs_chunk"):
            module.jax_denormalize_to_abs_chunk = cobot_jax_denormalize_to_abs_chunk

    from .execution_runtime import install as install_optional_execution
    install_optional_execution()
