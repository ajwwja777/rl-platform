import json

import numpy as np
import pytest


def _write_stats(path) -> None:
    path.write_text(
        json.dumps(
            {
                "norm_stats": {
                    "actions": {
                        "q01": [-1.0] * 14,
                        "q99": [1.0] * 14,
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def test_runtime_patch_converts_both_arms_and_keeps_grippers_absolute(tmp_path) -> None:
    from rlt_online_rl.config import RLTOnlineRLConfig

    from methods.openpi_rlt.cobot_adapter.online_runtime import (
        install_bimanual_runtime_patch,
    )

    install_bimanual_runtime_patch()
    from rlt_online_rl.action_representation import ActionRepresentationAdapter

    stats_path = tmp_path / "stats.json"
    _write_stats(stats_path)
    config = RLTOnlineRLConfig(
        action_dim=14,
        proprio_dim=14,
        action_representation="delta_chunk",
        action_norm_stats_path=str(stats_path),
    )
    adapter = ActionRepresentationAdapter.from_config(config)
    assert adapter is not None
    state = np.array([1, 2, 3, 4, 5, 6, 0.01, 11, 12, 13, 14, 15, 16, 0.02], dtype=np.float32)
    absolute = np.array(
        [[1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 0.04, 10.9, 11.8, 12.7, 13.6, 14.5, 15.4, 0.05]],
        dtype=np.float32,
    )

    normalized = adapter.normalize_chunk(absolute, state)
    restored = adapter.denormalize_to_abs_chunk(normalized, state)

    np.testing.assert_allclose(restored, absolute, atol=2e-6)
    np.testing.assert_allclose(normalized[0, 7:13], [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6], atol=1e-6)
    np.testing.assert_allclose(normalized[0, [6, 13]], absolute[0, [6, 13]], atol=1e-6)


def test_runtime_patch_updates_trainer_jax_inverse_for_both_arms() -> None:
    from methods.openpi_rlt.cobot_adapter.online_runtime import (
        install_bimanual_runtime_patch,
    )

    install_bimanual_runtime_patch()
    from rlt_online_rl import trainer

    state = np.arange(14, dtype=np.float32)
    delta = np.zeros((1, 2, 14), dtype=np.float32)
    q01 = np.zeros(14, dtype=np.float32)
    q99 = np.zeros(14, dtype=np.float32)

    absolute = np.asarray(
        trainer.jax_denormalize_to_abs_chunk(
            delta,
            state[None, :],
            q01,
            q99,
            action_representation="delta_chunk",
        )
    )

    np.testing.assert_allclose(absolute[0, :, :6], np.broadcast_to(state[:6], (2, 6)), atol=1e-6)
    np.testing.assert_allclose(absolute[0, :, 7:13], np.broadcast_to(state[7:13], (2, 6)), atol=1e-6)
    np.testing.assert_allclose(absolute[0, :, [6, 13]], np.zeros((2, 2), dtype=np.float32), atol=1e-6)


def test_actor_version_socket_timeout_is_retryable_runtime_error(monkeypatch) -> None:
    """Catches an uncaught slow-start timeout aborting upstream's readiness loop."""
    from methods.openpi_rlt.cobot_adapter.online_runtime import (
        install_bimanual_runtime_patch,
    )

    install_bimanual_runtime_patch()
    from rlt_online_rl import inference

    def _timeout(*_args, **_kwargs):
        raise TimeoutError("slow actor startup")

    monkeypatch.setattr(inference.urllib_request, "urlopen", _timeout)
    client = inference.ActorClient("http://127.0.0.1:1", timeout_sec=0.01, max_retries=0)

    with pytest.raises(RuntimeError, match="actor_service version"):
        client.get_actor_param_version()


def test_patched_generic_chunk_fallback_emits_replay_step_trace() -> None:
    """Catches upstream local-debug episodes finalizing with no raw replay steps."""
    from methods.openpi_rlt.cobot_adapter.online_runtime import (
        install_bimanual_runtime_patch,
    )

    install_bimanual_runtime_patch()
    from rlt_online_rl.config import EnvDriverConfig, RLTOnlineRLConfig
    from rlt_online_rl.inference import ChunkFeatures, EnvDriver, PolicyPlan
    from rlt_online_rl.replay import TransitionSource

    class _Env:
        def __init__(self):
            self.step_id = 0

        def step(self, action):
            self.step_id += 1
            observation = {"state": np.full(14, self.step_id, dtype=np.float32)}
            return observation, 0.0, self.step_id >= 2, False, {"success": int(self.step_id >= 2)}

    config = RLTOnlineRLConfig(action_dim=14, proprio_dim=14, chunk_len=2)
    driver = EnvDriver(
        env=_Env(),
        feature_provider=object(),
        actor_client=object(),
        replay_client=object(),
        rl_config=config,
        env_config=EnvDriverConfig(chunk_exec_horizon=2),
    )
    features = ChunkFeatures(
        z_rl=np.zeros(2048, dtype=np.float32),
        proprio=np.zeros(14, dtype=np.float32),
        ref_chunk=np.zeros((2, 14), dtype=np.float32),
    )

    def _planner(_observation, _local_step):
        return PolicyPlan(
            action_chunk=np.ones((2, 14), dtype=np.float32),
            ref_chunk=np.zeros((2, 14), dtype=np.float32),
            source=int(TransitionSource.BASE),
            start_features=features,
        )

    _, rewards, done, info = driver._execute_chunk({"state": np.zeros(14, dtype=np.float32)}, _planner)

    assert rewards == [0.0, 0.0]
    assert done is True
    assert len(info["step_trace"]) == 2
    assert info["step_trace"][-1]["done"] is True
    assert info["step_trace"][0]["source"] == int(TransitionSource.BASE)
    np.testing.assert_array_equal(info["step_trace"][0]["action"], np.ones(14, dtype=np.float32))
    np.testing.assert_array_equal(info["step_trace"][0]["ref_action"], np.zeros(14, dtype=np.float32))


def test_env_driver_never_uses_untrained_actor_during_warmup() -> None:
    from methods.openpi_rlt.cobot_adapter.online_runtime import (
        install_bimanual_runtime_patch,
    )

    install_bimanual_runtime_patch()
    from rlt_online_rl.config import EnvDriverConfig, RLTOnlineRLConfig
    from rlt_online_rl.inference import ActorRequest, EnvDriver
    from rlt_online_rl.replay import TransitionSource

    class _Env:
        phase = "warmup"

        def current_phase_name(self):
            return self.phase

    class _Actor:
        calls = 0

        def infer(self, _request):
            self.calls += 1
            raise AssertionError("warmup must not call the trainable actor")

        def get_actor_param_version(self):
            return 9

    env = _Env()
    actor = _Actor()
    driver = EnvDriver(
        env=env,
        feature_provider=object(),
        actor_client=actor,
        replay_client=object(),
        rl_config=RLTOnlineRLConfig(action_dim=14, proprio_dim=14, chunk_len=2),
        env_config=EnvDriverConfig(chunk_exec_horizon=2),
    )
    request = ActorRequest(
        z_rl=np.zeros(2048, dtype=np.float32),
        proprio=np.zeros(14, dtype=np.float32),
        ref_chunk=np.ones((2, 14), dtype=np.float32),
        request_id="warmup",
        episode_id=0,
        step_id=0,
        deterministic=False,
    )

    response = driver._actor_client.infer(request)

    assert actor.calls == 0
    assert response.source == int(TransitionSource.BASE)
    assert response.actor_param_version == -1
    np.testing.assert_array_equal(response.refined_chunk, request.ref_chunk)


def test_patched_driver_calls_cobot_chunk_contract_with_current_observation() -> None:
    from methods.openpi_rlt.cobot_adapter.online_runtime import (
        install_bimanual_runtime_patch,
    )

    install_bimanual_runtime_patch()
    from rlt_online_rl.config import EnvDriverConfig, RLTOnlineRLConfig
    from rlt_online_rl.inference import EnvDriver

    expected = {"state": np.zeros(14, dtype=np.float32)}

    class _CobotEnv:
        cobot_task2_contract = True

        def current_phase_name(self):
            return "warmup"

        def execute_chunk(self, observation, policy_planner):
            assert observation is expected
            assert callable(policy_planner)
            return observation, [0.0], True, {"step_trace": []}

    driver = EnvDriver(
        env=_CobotEnv(),
        feature_provider=object(),
        actor_client=object(),
        replay_client=object(),
        rl_config=RLTOnlineRLConfig(action_dim=14, proprio_dim=14, chunk_len=2),
        env_config=EnvDriverConfig(chunk_exec_horizon=2),
    )

    result = driver._execute_chunk(expected, lambda *_args: None)

    assert result[0] is expected
    assert result[1] == [0.0]
    assert result[2] is True


def test_patched_driver_skips_replay_when_cobot_episode_is_not_eligible() -> None:
    from methods.openpi_rlt.cobot_adapter.online_runtime import install_bimanual_runtime_patch

    install_bimanual_runtime_patch()
    from rlt_online_rl.config import EnvDriverConfig, RLTOnlineRLConfig
    from rlt_online_rl.inference import EnvDriver, RawEpisodeTrace

    class _CobotEnv:
        cobot_task2_contract = True

        def current_phase_name(self):
            return "warmup"

        def replay_commit_allowed(self):
            return False

    driver = EnvDriver(
        env=_CobotEnv(),
        feature_provider=object(),
        actor_client=object(),
        replay_client=object(),
        rl_config=RLTOnlineRLConfig(action_dim=14, proprio_dim=14, chunk_len=2),
        env_config=EnvDriverConfig(chunk_exec_horizon=2),
    )
    raw = RawEpisodeTrace(episode_id=1, chunk_len=2, observations=[], steps=[], chunks=[], policy_start_steps=[])

    transitions, stats = driver._build_episode_replay(raw)

    assert transitions == []
    assert stats["replay_skipped_reason"] == "cobot_episode_not_eligible"


def test_cobot_post_episode_helper_advances_session_after_replay_finalize() -> None:
    from methods.openpi_rlt.cobot_adapter.online_runtime import _mark_cobot_replay_finalized

    class _CobotEnv:
        cobot_task2_contract = True

        def __init__(self):
            self.calls = 0

        def mark_replay_finalized(self):
            self.calls += 1

    env = _CobotEnv()
    _mark_cobot_replay_finalized(env)
    assert env.calls == 1


def test_non_cobot_post_episode_helper_is_noop() -> None:
    from methods.openpi_rlt.cobot_adapter.online_runtime import _mark_cobot_replay_finalized

    class _Env:
        def mark_replay_finalized(self):
            raise AssertionError("non-Cobot env must not be touched")

    _mark_cobot_replay_finalized(_Env())


def test_cobot_uses_task5_instead_of_duplicate_upstream_raw_pickle() -> None:
    from methods.openpi_rlt.cobot_adapter.online_runtime import install_bimanual_runtime_patch

    install_bimanual_runtime_patch()
    from rlt_online_rl.inference import EnvDriver

    class _CobotEnv:
        cobot_task2_contract = True

        @staticmethod
        def persist_upstream_raw_episode():
            return False

    driver = object.__new__(EnvDriver)
    driver._env = _CobotEnv()

    assert driver._persist_raw_episode(object(), episode_id=0, started_at=0.0) is None
