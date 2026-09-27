import threading

import numpy as np
import pytest


def test_actor_params_are_pinned_for_one_episode() -> None:
    from methods.openpi_rlt.cobot_adapter.actor_pinning import EpisodeActorPin

    pin = EpisodeActorPin()
    params_v1 = {"value": 1}
    params_v2 = {"value": 2}

    first = pin.select(episode_id=10, latest_params=params_v1, latest_version=1)
    same = pin.select(episode_id=10, latest_params=params_v2, latest_version=2)
    next_episode = pin.select(episode_id=11, latest_params=params_v2, latest_version=2)

    assert first == (params_v1, 1)
    assert same == (params_v1, 1)
    assert next_episode == (params_v2, 2)


def test_actor_pin_rejects_missing_episode_id() -> None:
    from methods.openpi_rlt.cobot_adapter.actor_pinning import EpisodeActorPin

    with pytest.raises(ValueError, match="episode_id"):
        EpisodeActorPin().select(episode_id=None, latest_params={}, latest_version=0)


def test_actor_pin_rejects_stale_episode_request() -> None:
    from methods.openpi_rlt.cobot_adapter.actor_pinning import EpisodeActorPin

    pin = EpisodeActorPin()
    pin.select(episode_id=10, latest_params={}, latest_version=1)
    with pytest.raises(RuntimeError, match="stale actor request"):
        pin.select(episode_id=9, latest_params={}, latest_version=2)


def test_installed_actor_service_uses_new_snapshot_only_on_next_episode(monkeypatch) -> None:
    from methods.openpi_rlt.cobot_adapter.online_runtime import (
        install_bimanual_runtime_patch,
    )

    monkeypatch.setenv("COBOT_RLT_PIN_ACTOR_PER_EPISODE", "1")
    install_bimanual_runtime_patch()
    from rlt_online_rl import inference

    class _Wrapper:
        def infer(self, params, *_args, **_kwargs):
            return np.full((2, 14), params["value"], dtype=np.float32)

    service = object.__new__(inference.ActorService)
    service._lock = threading.Lock()
    service._actor_params = {"value": 1.0}
    service._actor_version = 1
    service._rng = inference.jax.random.PRNGKey(0)
    service._action_adapter = None
    service._wrapper = _Wrapper()

    def request(episode_id: int):
        return inference.ActorRequest(
            z_rl=np.zeros(2048, dtype=np.float32),
            proprio=np.zeros(14, dtype=np.float32),
            ref_chunk=np.zeros((2, 14), dtype=np.float32),
            request_id=str(episode_id),
            episode_id=episode_id,
            step_id=0,
            deterministic=True,
        )

    first = service.infer(request(10))
    service._actor_params = {"value": 2.0}
    service._actor_version = 2
    same_episode = service.infer(request(10))
    next_episode = service.infer(request(11))

    assert first.actor_param_version == 1
    assert same_episode.actor_param_version == 1
    assert next_episode.actor_param_version == 2
    np.testing.assert_array_equal(same_episode.refined_chunk, np.ones((2, 14), dtype=np.float32))
    np.testing.assert_array_equal(next_episode.refined_chunk, np.full((2, 14), 2.0, dtype=np.float32))
