"""Reproducible openpi-RLT Stage-1 configs for Cobot ``in_the_pot``."""

import math
from pathlib import Path

from openpi.models import pi0_config
from openpi.training import config as training_config
from openpi.training import weight_loaders

from methods.openpi_rlt.cobot_adapter.upstream_data import CobotLeRobotDataConfig


def build_stage1_config(
    *,
    exp_name: str,
    dataset_repo_id: str,
    base_params: str,
    assets_base_dir: str | Path,
    checkpoint_base_dir: str | Path,
    batch_size: int = 32,
    num_train_steps: int = 5_000,
    num_workers: int = 8,
    fsdp_devices: int = 1,
    rlt_alpha: float = 1.0,
    seed: int = 42,
    wandb_enabled: bool = False,
    debug_model: bool = False,
) -> training_config.TrainConfig:
    """Build an official-default π0.5 RLT config with Cobot data semantics."""
    if not exp_name:
        raise ValueError("exp_name must be non-empty")
    if not dataset_repo_id:
        raise ValueError("dataset_repo_id must be non-empty")
    if not base_params:
        raise ValueError("base_params must be non-empty")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_train_steps <= 0:
        raise ValueError("num_train_steps must be positive")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    if fsdp_devices <= 0:
        raise ValueError("fsdp_devices must be positive")
    if not math.isfinite(rlt_alpha) or rlt_alpha < 0:
        raise ValueError("rlt_alpha must be finite and non-negative")

    config_name = "cobot_rlt_pi05_only" if rlt_alpha == 0.0 else "cobot_rlt_pi05_joint"
    if debug_model:
        config_name += "_debug"
    model_variant = "dummy" if debug_model else "gemma_2b"
    expert_variant = "dummy" if debug_model else "gemma_300m"
    weight_loader = (
        weight_loaders.NoOpWeightLoader()
        if debug_model
        else weight_loaders.CheckpointWeightLoader(base_params)
    )
    rlt_dim = 64 if debug_model else 2_048
    return training_config.TrainConfig(
        name=config_name,
        project_name="cobot-realworld-rl",
        exp_name=exp_name,
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            discrete_state_input=True,
            paligemma_variant=model_variant,
            action_expert_variant=expert_variant,
        ),
        data=CobotLeRobotDataConfig(repo_id=dataset_repo_id),
        weight_loader=weight_loader,
        assets_base_dir=str(assets_base_dir),
        checkpoint_base_dir=str(checkpoint_base_dir),
        seed=seed,
        batch_size=batch_size,
        num_workers=num_workers,
        num_train_steps=num_train_steps,
        save_interval=1_000,
        keep_period=5_000,
        wandb_enabled=wandb_enabled,
        fsdp_devices=fsdp_devices,
        rlt_num_tokens=1,
        rlt_num_layers=2,
        rlt_embed_dim=rlt_dim,
        rlt_input_dim=rlt_dim,
        rlt_alpha=rlt_alpha,
    )
