"""Faithful Yyshadow Stage-1 config for plug_v3 single-right-arm data."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from typing_extensions import override

import openpi.models.model as model_api
from openpi import transforms
from openpi.models import pi0_config
from openpi.policies import agilexbag_image_policy
from openpi.training import config as training_config
from openpi.training import weight_loaders

from methods.openpi_rlt.stage1_entry import configure_lerobot_video_backend


RIGHT_ARM_DIM = 7
RIGHT_ARM_DELTA_MASK = transforms.make_bool_mask(6, -1)


@dataclasses.dataclass(frozen=True)
class RightArmLeRobotDataConfig(training_config.DataConfigFactory):
    """Map Cobot camera keys and 7D right-arm vectors to upstream Agilex inputs."""

    base_config: training_config.DataConfig | None = dataclasses.field(
        default_factory=lambda: training_config.DataConfig(prompt_from_task=True)
    )

    @override
    def create(
        self,
        assets_dirs: Path,
        model_config: model_api.BaseModelConfig,
    ) -> training_config.DataConfig:
        configure_lerobot_video_backend()
        repack = transforms.Group(
            inputs=[
                transforms.RepackTransform(
                    {
                        "images": {
                            "base_0_rgb": "observation.images.cam_high",
                            "left_wrist_0_rgb": "observation.images.cam_left_wrist",
                            "right_wrist_0_rgb": "observation.images.cam_right_wrist",
                        },
                        "state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        data = transforms.Group(
            inputs=[
                agilexbag_image_policy.AgilexBagImageInputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                ),
                transforms.DeltaActions(RIGHT_ARM_DELTA_MASK),
            ],
            outputs=[
                transforms.AbsoluteActions(RIGHT_ARM_DELTA_MASK),
                agilexbag_image_policy.AgilexBagImageOutputs(action_dim=RIGHT_ARM_DIM),
            ],
        )
        model = training_config.ModelTransformFactory()(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack,
            data_transforms=data,
            model_transforms=model,
            action_sequence_keys=("action",),
            prompt_from_task=True,
        )


def build_config(
    *,
    dataset_repo_id: str,
    base_params: str,
    project_root: str | Path,
    exp_name: str,
    batch_size: int = 32,
    num_workers: int = 8,
    fsdp_devices: int = 4,
    num_train_steps: int = 5000,
    seed: int = 42,
) -> training_config.TrainConfig:
    """Match upstream rlt_pi05_agilexbag_image_delta_joint except hardware sharding."""

    project = Path(project_root).resolve()
    return training_config.TrainConfig(
        name="plug_v3_yyshadow_rlt_pi05_delta_joint",
        project_name="cobot-realworld-rl",
        exp_name=exp_name,
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            discrete_state_input=True,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
        ),
        data=RightArmLeRobotDataConfig(repo_id=dataset_repo_id),
        weight_loader=weight_loaders.CheckpointWeightLoader(base_params),
        assets_base_dir=str(project / "outputs" / "rlt" / "plug_v3_yyshadow" / "assets"),
        checkpoint_base_dir=str(project / "models" / "rlt" / "plug_v3_yyshadow" / "stage1-training"),
        seed=seed,
        batch_size=batch_size,
        num_workers=num_workers,
        num_train_steps=num_train_steps,
        save_interval=1000,
        keep_period=5000,
        wandb_enabled=False,
        fsdp_devices=fsdp_devices,
        rlt_num_tokens=1,
        rlt_num_layers=2,
        rlt_embed_dim=2048,
        rlt_input_dim=2048,
        rlt_alpha=1.0,
    )
