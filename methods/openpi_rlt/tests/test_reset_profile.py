import pytest


def test_unverified_reset_profile_cannot_be_armed() -> None:
    from methods.openpi_rlt.cobot_adapter.reset_profile import ResetProfile

    profile = ResetProfile(
        verified=False,
        gripper_mode="hold",
        retreat_waypoints=((0.0,) * 14,),
        home_pose="plug_preinsert_v1",
    )
    with pytest.raises(RuntimeError, match="not onsite-verified"):
        profile.assert_auto_reset_safe()


def test_verified_reset_requires_hold_and_waypoint() -> None:
    from methods.openpi_rlt.cobot_adapter.reset_profile import ResetProfile

    with pytest.raises(ValueError, match="hold"):
        ResetProfile(True, "open", ((0.0,) * 14,), "plug_preinsert_v1")
    with pytest.raises(ValueError, match="retreat"):
        ResetProfile(True, "hold", (), "plug_preinsert_v1")

