"""Task-local left arm and two wrist ROS boundary; no middle camera subscription."""
import os
import numpy as np
from methods.openpi_rlt.plug_v3_yyshadow.right_arm_env import RightArmPolicyRuntime,RightArmCobotOnlineEnv
from methods.openpi_rlt.cobot_adapter.cobot_ros1 import RosTask2IO,CobotIOSample,CAMERA_TOPICS
from methods.openpi_rlt.cobot_adapter.trace import ControlSource

class LeftArmPolicyRuntime(RightArmPolicyRuntime):
    def control_source(self,policy_source):
        if self.snapshot().expert_mask[0]:return ControlSource.HUMAN
        source=ControlSource(policy_source)
        if source not in (ControlSource.BASE,ControlSource.RL):raise ValueError('left policy source required')
        return source

class LeftArmRosTask2IO(RosTask2IO):
    camera_topics={key:CAMERA_TOPICS[key] for key in ('cam_left_wrist','cam_right_wrist')}
    @staticmethod
    def _build_observation(*,left_state,right_state,images,prompt):
        if set(images)!={'cam_left_wrist','cam_right_wrist'}:raise ValueError('only wrist images allowed')
        return dict(state=np.concatenate([left_state,right_state]),images={'left_wrist_0_rgb':images['cam_left_wrist'],'right_wrist_0_rgb':images['cam_right_wrist']},prompt=prompt)
    @staticmethod
    def _left(sample):
        obs=dict(sample.observation);full=np.asarray(obs['state'],np.float32)
        if full.shape!=(14,):raise ValueError('full feedback14 required before left projection')
        obs['state']=full[:7].copy()
        return CobotIOSample(observation=obs,mode=sample.mode,outcome=sample.outcome,paused=sample.paused,timestamp=sample.timestamp,io_evidence=getattr(sample,'io_evidence',None))
    def sample(self):return self._left(super().sample())
    def sample_control(self,*,right_arm_only=False,**kwargs):
        # Legacy async caller requests a 7D sample using this name. This class
        # always projects LEFT and never samples right-only before projection.
        return self._left(super().sample_control(right_arm_only=False,**kwargs))
    def request_home(self):
        # Never inherit the old right-task fallback pose for a new left task.
        if not os.environ.get('COBOT_RLT_HOME_POSE','').strip():
            raise RuntimeError('Left-task home pose has not been registered')
        return super().request_home()
    def publish_policy_action(self,action):
        target=np.asarray(action,np.float32)
        if target.shape!=(7,) or not np.isfinite(target).all():raise ValueError('finite left7 target required')
        with self._condition:
            if self._paused or self._mode!='policy' or self._shadow_mode:return False
            self._publishers['left'].publish(self._joint_message(target));return True

class LeftArmCobotOnlineEnv(RightArmCobotOnlineEnv):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._runtime=LeftArmPolicyRuntime(joint_step_limit=float(kwargs['joint_step_limit']),gripper_step_limit=float(kwargs['gripper_step_limit']),hold_gripper=False)
        self._active_arm='left'


def validate_environment(environment):
    from .dataset import PROMPT
    required = {
        'COBOT_RLT_TASK_CONTRACT': 'realtime_expo_left_2cam_10hz',
        'COBOT_RLT_PROMPT': PROMPT,
        'COBOT_RLT_CONTROL_HZ': '10',
        'COBOT_RLT_LOGICAL_HZ': '10',
        'COBOT_RLT_CHUNK_EXEC_HORIZON': '10',
        'COBOT_RLT_HIL_SAMPLING': 'logical10',
        'COBOT_RLT_HIL_TARGET': 'coordinator_command',
        'COBOT_RLT_RAW_OBSERVATION_CONTRACT': 'trace',
    }
    for key, value in required.items():
        if environment.get(key) != value:
            raise ValueError(f'{key} must be {value!r} for the left-arm task')
    if environment.get('COBOT_RLT_COLLECTION_PHASE','online') not in ('warmup','online'):
        raise ValueError('Native collection phase must be warmup or online; frozen is a load mode')
    if environment.get('COBOT_RLT_HOME_AFTER_TERMINAL','0') != '0' or environment.get('COBOT_RLT_ENABLE_ROBOT_RESET','0') != '0':
        raise ValueError('Automatic home/reset not enabled for the new left-arm task')


def create_left_arm_online_env():
    """Reuse audited recorder/session boundaries with explicit left-task I/O."""
    validate_environment(os.environ)
    from methods.openpi_rlt.cobot_adapter.online_runtime import install_bimanual_runtime_patch
    from methods.openpi_rlt.cobot_adapter import cobot_ros1
    install_bimanual_runtime_patch()
    original_io, original_env = cobot_ros1.RosTask2IO, cobot_ros1.CobotOnlineEnv
    cobot_ros1.RosTask2IO, cobot_ros1.CobotOnlineEnv = LeftArmRosTask2IO, LeftArmCobotOnlineEnv
    try:
        return cobot_ros1.create_cobot_online_env()
    finally:
        cobot_ros1.RosTask2IO, cobot_ros1.CobotOnlineEnv = original_io, original_env
