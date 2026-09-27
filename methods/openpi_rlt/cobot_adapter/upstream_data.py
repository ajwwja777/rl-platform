"""Cobot-specific LeRobot transforms layered on the fixed openpi-RLT upstream."""

import dataclasses
import pathlib

from typing_extensions import override

import openpi.models.model as model_api
from openpi import transforms
from openpi.policies import agilexbag_image_policy
from openpi.training import config as training_config

from methods.openpi_rlt.stage1_entry import configure_lerobot_video_backend


PHYSICAL_ACTION_DIM = 14
_BIMANUAL_JOINT_DELTA_MASK = transforms.make_bool_mask(6, -1, 6, -1)


def build_cobot_transform_groups(
    model_config: model_api.BaseModelConfig,
) -> tuple[transforms.Group, transforms.Group]:
    """Build the physical-key repack and bimanual action transforms."""
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
            transforms.DeltaActions(_BIMANUAL_JOINT_DELTA_MASK),
        ],
        outputs=[
            transforms.AbsoluteActions(_BIMANUAL_JOINT_DELTA_MASK),
            agilexbag_image_policy.AgilexBagImageOutputs(action_dim=PHYSICAL_ACTION_DIM),
        ],
    )
    return repack, data


@dataclasses.dataclass(frozen=True)
class CobotLeRobotDataConfig(training_config.DataConfigFactory):
    """LeRobot v2.1 config for Cobot three-camera, bimanual 14D data."""

    base_config: training_config.DataConfig | None = dataclasses.field(
        default_factory=lambda: training_config.DataConfig(prompt_from_task=True)
    )

    @override
    def create(
        self,
        assets_dirs: pathlib.Path,
        model_config: model_api.BaseModelConfig,
    ) -> training_config.DataConfig:
        configure_lerobot_video_backend()
        repack, data = build_cobot_transform_groups(model_config)
        model = training_config.ModelTransformFactory()(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack,
            data_transforms=data,
            model_transforms=model,
            action_sequence_keys=("action",),
            prompt_from_task=True,
        )
