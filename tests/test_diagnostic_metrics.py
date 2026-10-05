"""Logging must not alter native loss, gradients, optimizer state or RNG."""
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from rlt_online_rl import trainer
from rlt_online_rl.config import RLTOnlineRLConfig
from rlt_online_rl.networks import compute_critic_loss
from methods.openpi_rlt.cobot_adapter.diagnostic_metrics import install_diagnostic_metrics_patch


@pytest.mark.parametrize('positive', [False, True])
def test_diagnostics_preserve_update_and_do_not_call_unrewarded_failures(positive):
    cfg = RLTOnlineRLConfig(z_dim=8, proprio_dim=7, action_dim=7, chunk_len=10,
                           actor_hidden_dim=16, critic_hidden_dim=16)
    state, actor, critic = trainer.init_train_state(cfg, rng=jax.random.PRNGKey(9))
    batch = {k: jnp.zeros(shape) for k, shape in {
        'z_rl':(3,8), 'proprio':(3,7), 'action_chunk':(3,10,7),
        'ref_chunk':(3,10,7), 'next_z_rl':(3,8), 'next_proprio':(3,7),
        'next_ref_chunk':(3,10,7), 'rewards':(3,10), 'done':(3,)}.items()}
    if positive:
        batch['rewards'] = batch['rewards'].at[0,-1].set(1)
        batch['done'] = batch['done'].at[0].set(1)
    module = SimpleNamespace(compute_critic_loss=compute_critic_loss)
    install_diagnostic_metrics_patch(module)
    args = (critic, state.critic_params, actor, state.target_actor_params,
            state.target_critic_params, batch['z_rl'], batch['proprio'],
            batch['action_chunk'], batch['rewards'], batch['done'],
            batch['next_z_rl'], batch['next_proprio'], batch['next_ref_chunk'],
            cfg.gamma, jax.random.PRNGKey(5))
    old_loss, _ = compute_critic_loss(*args)
    new_loss, metrics = module.compute_critic_loss(*args)
    np.testing.assert_array_equal(old_loss, new_loss)
    assert int(metrics['rewarded_chunk_count']) == int(positive)
    assert int(metrics['unrewarded_chunk_count']) == 3-int(positive)
    assert np.isnan(metrics['q1_rewarded_chunk_mean']) == (not positive)
    original = trainer.compute_critic_loss
    try:
        trainer.compute_critic_loss = compute_critic_loss
        before, _ = trainer.update_critic(state, batch, actor, critic, cfg)
        trainer.compute_critic_loss = module.compute_critic_loss
        after, _ = trainer.update_critic(state, batch, actor, critic, cfg)
    finally:
        trainer.compute_critic_loss = original
    for a,b in zip(jax.tree_util.tree_leaves(before),jax.tree_util.tree_leaves(after)):
        np.testing.assert_array_equal(a,b)


def test_installation_is_idempotent():
    module=SimpleNamespace(compute_critic_loss=compute_critic_loss)
    install_diagnostic_metrics_patch(module)
    first=module.compute_critic_loss
    install_diagnostic_metrics_patch(module)
    assert module.compute_critic_loss is first


def test_unknown_credit_target_is_not_mislabeled_as_native_td():
    module=SimpleNamespace(compute_critic_loss=lambda *args: None)
    with pytest.raises(ValueError,match='native TD'):install_diagnostic_metrics_patch(module)


def test_spawn_bootstrap_keeps_default_off_and_installs_explicit_optin(monkeypatch):
    from methods.openpi_rlt.cobot_adapter.process_bootstrap import initialize_process
    monkeypatch.setattr(trainer,'compute_critic_loss',compute_critic_loss)
    monkeypatch.delenv('COBOT_RLT_DIAGNOSTIC_METRICS',raising=False)
    initialize_process()
    assert trainer.compute_critic_loss is compute_critic_loss
    monkeypatch.setenv('COBOT_RLT_DIAGNOSTIC_METRICS','1')
    initialize_process()
    assert trainer.compute_critic_loss._cobot_diagnostic_metrics
