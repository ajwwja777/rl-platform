from pathlib import Path

import pytest


def test_joint_stage1_config_matches_official_pi05_rlt_defaults(tmp_path: Path) -> None:
    """Catches changing the model, RLT, optimizer, or ownership-critical paths silently."""
    from methods.openpi_rlt.stage1_config import build_stage1_config

    config = build_stage1_config(
        exp_name="r1-smoke",
        dataset_repo_id="legacy40-v2.1",
        base_params="/models/pi05_base/params",
        assets_base_dir=tmp_path / "assets",
        checkpoint_base_dir=tmp_path / "checkpoints",
        batch_size=8,
        num_train_steps=5000,
        num_workers=4,
        fsdp_devices=4,
        rlt_alpha=1.0,
    )

    assert config.name == "cobot_rlt_pi05_joint"
    assert config.project_name == "cobot-realworld-rl"
    assert config.exp_name == "r1-smoke"
    assert config.model.pi05 is True
    assert config.model.action_dim == 32
    assert config.model.action_horizon == 50
    assert config.model.discrete_state_input is True
    assert config.data.repo_id == "legacy40-v2.1"
    assert config.weight_loader.params_path == "/models/pi05_base/params"
    assert config.batch_size == 8
    assert config.num_train_steps == 5000
    assert config.num_workers == 4
    assert config.fsdp_devices == 4
    assert config.seed == 42
    assert config.rlt_num_tokens == 1
    assert config.rlt_num_layers == 2
    assert config.rlt_embed_dim == 2048
    assert config.rlt_input_dim == 2048
    assert config.rlt_alpha == 1.0
    assert config.lr_schedule.warmup_steps == 1000
    assert config.lr_schedule.peak_lr == 2.5e-5
    assert config.save_interval == 1000
    assert config.keep_period == 5000
    assert config.wandb_enabled is False
    assert config.assets_base_dir == str(tmp_path / "assets")
    assert config.checkpoint_base_dir == str(tmp_path / "checkpoints")


def test_rlt_only_stage1_config_has_distinct_name_and_frozen_vla_mode(tmp_path: Path) -> None:
    """Catches confusing the RLT-only health run with the joint R1 experiment."""
    from methods.openpi_rlt.stage1_config import build_stage1_config

    config = build_stage1_config(
        exp_name="rlt-only-smoke",
        dataset_repo_id="legacy40-v2.1",
        base_params="gs://openpi-assets/checkpoints/pi05_base/params",
        assets_base_dir=tmp_path / "assets",
        checkpoint_base_dir=tmp_path / "checkpoints",
        rlt_alpha=0.0,
    )

    assert config.name == "cobot_rlt_pi05_only"
    assert config.rlt_alpha == 0.0


def test_debug_stage1_config_uses_dummy_model_without_loading_base(tmp_path: Path) -> None:
    """Catches a smoke run accidentally loading the multi-gigabyte production model."""
    from openpi.training.weight_loaders import NoOpWeightLoader

    from methods.openpi_rlt.stage1_config import build_stage1_config

    config = build_stage1_config(
        exp_name="real-batch-smoke",
        dataset_repo_id="legacy40-v2.1",
        base_params="unused-for-debug",
        assets_base_dir=tmp_path / "assets",
        checkpoint_base_dir=tmp_path / "checkpoints",
        batch_size=1,
        num_train_steps=1,
        num_workers=0,
        rlt_alpha=1.0,
        debug_model=True,
    )

    assert config.name == "cobot_rlt_pi05_joint_debug"
    assert config.model.paligemma_variant == "dummy"
    assert config.model.action_expert_variant == "dummy"
    assert config.rlt_embed_dim == 64
    assert config.rlt_input_dim == 64
    assert isinstance(config.weight_loader, NoOpWeightLoader)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("batch_size", 0),
        ("num_train_steps", 0),
        ("num_workers", -1),
        ("fsdp_devices", 0),
        ("rlt_alpha", -0.1),
    ],
)
def test_stage1_config_rejects_invalid_run_parameters(tmp_path: Path, field: str, value: float) -> None:
    """Catches invalid resource or loss parameters before a GPU allocation starts."""
    from methods.openpi_rlt.stage1_config import build_stage1_config

    kwargs = {
        "exp_name": "bad",
        "dataset_repo_id": "legacy40-v2.1",
        "base_params": "/models/pi05_base/params",
        "assets_base_dir": tmp_path / "assets",
        "checkpoint_base_dir": tmp_path / "checkpoints",
    }
    kwargs[field] = value

    with pytest.raises(ValueError):
        build_stage1_config(**kwargs)
