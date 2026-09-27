import numpy as np
import pytest


def test_bimanual_smoothness_penalizes_both_arms_and_first_action() -> None:
    from methods.openpi_rlt.cobot_adapter.smoothness_loss import (
        numpy_smoothness_components,
    )

    target = np.zeros((1, 4, 14), dtype=np.float32)
    pred = target.copy()
    pred[:, :, 0] = [0.2, 0.4, 0.6, 0.8]
    pred[:, :, 7] = [0.3, 0.6, 0.9, 1.2]
    state = np.zeros((1, 14), dtype=np.float32)

    metrics = numpy_smoothness_components(pred, target, state)

    assert metrics["left_velocity"] > 0
    assert metrics["right_velocity"] > metrics["left_velocity"]
    assert metrics["state_first"] > 0
    assert metrics["acceleration"] < 1e-12


def test_smoothness_keeps_gripper_metric_separate() -> None:
    from methods.openpi_rlt.cobot_adapter.smoothness_loss import (
        numpy_smoothness_components,
    )

    target = np.zeros((1, 3, 14), dtype=np.float32)
    pred = target.copy()
    pred[:, :, 6] = [0.0, 0.01, 0.02]
    pred[:, :, 13] = [0.0, -0.02, -0.04]

    metrics = numpy_smoothness_components(pred, target, np.zeros((1, 14), dtype=np.float32))

    assert metrics["left_velocity"] == 0.0
    assert metrics["right_velocity"] == 0.0
    assert metrics["gripper_velocity"] > 0.0


def test_jax_and_numpy_smoothness_components_agree() -> None:
    pytest.importorskip("jax")
    from methods.openpi_rlt.cobot_adapter.smoothness_loss import (
        jax_smoothness_components,
        numpy_smoothness_components,
    )

    rng = np.random.default_rng(11)
    pred = rng.normal(size=(2, 5, 14)).astype(np.float32)
    target = rng.normal(size=(2, 5, 14)).astype(np.float32)
    state = rng.normal(size=(2, 14)).astype(np.float32)

    numpy_metrics = numpy_smoothness_components(pred, target, state)
    jax_metrics = jax_smoothness_components(pred, target, state)

    for name, expected in numpy_metrics.items():
        np.testing.assert_allclose(np.asarray(jax_metrics[name]), expected, rtol=1e-5)


def test_learner_patch_replaces_only_actor_update() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    optax = pytest.importorskip("optax")
    from types import SimpleNamespace

    from methods.openpi_rlt.cobot_adapter.learner_patch import (
        install_symmetric_smoothness_patch,
    )

    original = lambda: None
    fake = SimpleNamespace(update_actor=original, jax=jax, jnp=jnp, optax=optax)
    install_symmetric_smoothness_patch(fake)

    assert fake.update_actor is not original
    assert fake.update_actor._cobot_symmetric_smoothness is True


def test_r2_patch_runs_one_real_14d_actor_update(monkeypatch) -> None:
    jax = pytest.importorskip("jax")
    monkeypatch.setenv("COBOT_RLT_R2_SMOOTHNESS", "1")
    from methods.openpi_rlt.cobot_adapter.online_runtime import (
        install_bimanual_runtime_patch,
    )

    install_bimanual_runtime_patch()
    from rlt_online_rl import trainer
    from rlt_online_rl.config import RLTOnlineRLConfig

    config = RLTOnlineRLConfig(
        action_dim=14,
        chunk_len=4,
        z_dim=5,
        proprio_dim=14,
        actor_hidden_dim=16,
        critic_hidden_dim=16,
        actor_num_layers=1,
        critic_num_layers=1,
        actor_update_period=1,
    )
    state, actor, critic = trainer.init_train_state(config, rng=jax.random.PRNGKey(0))
    batch_size = 2
    batch = {
        "z_rl": np.ones((batch_size, 5), dtype=np.float32),
        "proprio": np.zeros((batch_size, 14), dtype=np.float32),
        "ref_chunk": np.zeros((batch_size, 4, 14), dtype=np.float32),
        "action_chunk": np.zeros((batch_size, 4, 14), dtype=np.float32),
        "rewards": np.zeros((batch_size, 4), dtype=np.float32),
        "done": np.zeros((batch_size,), dtype=np.float32),
        "next_z_rl": np.ones((batch_size, 5), dtype=np.float32),
        "next_proprio": np.zeros((batch_size, 14), dtype=np.float32),
        "next_ref_chunk": np.zeros((batch_size, 4, 14), dtype=np.float32),
        "source_chunk": np.zeros((batch_size, 4), dtype=np.uint8),
    }
    batch = {name: jax.numpy.asarray(value) for name, value in batch.items()}
    _state, metrics = trainer.train_step(
        state,
        batch,
        actor=actor,
        critic=critic,
        rl_config=config,
        delta_weight=1.0,
    )

    assert int(metrics["did_actor_update"]) == 1
    assert np.isfinite(float(metrics["delta_penalty"]))
