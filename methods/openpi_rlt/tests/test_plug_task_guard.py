"""Wrong-task fallback must fail before any field I/O can be constructed."""
import pytest
from methods.openpi_rlt.plug_v3_yyshadow import right_arm_env


@pytest.mark.parametrize("prompt", [None, "", "  ", "Open the pot lid, put the object into the pot, then close the lid."])
def test_invalid_task_instruction_fails_before_io(monkeypatch, prompt):
    if prompt is None:
        monkeypatch.delenv("COBOT_RLT_PROMPT", raising=False)
    else:
        monkeypatch.setenv("COBOT_RLT_PROMPT", prompt)
    from methods.openpi_rlt.cobot_adapter import cobot_ros1
    monkeypatch.setattr(cobot_ros1, "create_cobot_online_env", lambda: pytest.fail("must not reach field factory"))
    with pytest.raises(ValueError, match="explicit plug task"):
        right_arm_env.create_right_arm_online_env()


@pytest.mark.parametrize("task_id", [None, "in_the_pot"])
def test_session_without_plug_identity_is_rejected(task_id):
    environment={"COBOT_RLT_PROMPT":"Insert the held plug into the socket.","COBOT_RLT_SESSION_UI":"1"}
    if task_id is not None:environment["COBOT_RLT_TASK_ID"]=task_id
    with pytest.raises(ValueError, match="TASK_ID"):
        right_arm_env.validate_plug_task_environment(environment)


def test_explicit_plug_factory_preserves_environment_and_shared_classes(monkeypatch):
    from methods.openpi_rlt.cobot_adapter import cobot_ros1, online_runtime
    monkeypatch.setenv("COBOT_RLT_PROMPT", "Insert the held plug into the socket.")
    monkeypatch.setenv("COBOT_RLT_TASK_ID", "plug_insertion")
    monkeypatch.setenv("COBOT_RLT_SESSION_UI", "1")
    monkeypatch.setattr(online_runtime, "install_bimanual_runtime_patch", lambda: None)
    original_io,original_env=cobot_ros1.RosTask2IO,cobot_ros1.CobotOnlineEnv
    marker=object()
    def factory():
        assert cobot_ros1.RosTask2IO is right_arm_env.RightArmRosTask2IO
        assert cobot_ros1.CobotOnlineEnv is right_arm_env.RightArmCobotOnlineEnv
        assert right_arm_env.os.environ["COBOT_RLT_PROMPT"] == "Insert the held plug into the socket."
        return marker
    monkeypatch.setattr(cobot_ros1, "create_cobot_online_env", factory)
    assert right_arm_env.create_right_arm_online_env() is marker
    assert cobot_ros1.RosTask2IO is original_io and cobot_ros1.CobotOnlineEnv is original_env
    assert right_arm_env.os.environ["COBOT_RLT_TASK_ID"] == "plug_insertion"
