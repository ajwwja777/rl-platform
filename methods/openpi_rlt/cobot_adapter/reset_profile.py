"""Pure validation for project-private plug reset profiles."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResetProfile:
    verified: bool
    gripper_mode: str
    retreat_waypoints: tuple[tuple[float, ...], ...]
    home_pose: str

    def __post_init__(self) -> None:
        if self.gripper_mode != "hold":
            raise ValueError("plug reset must hold the gripper")
        if self.verified and not self.retreat_waypoints:
            raise ValueError("verified plug reset requires a retreat waypoint")
        if any(len(waypoint) != 14 for waypoint in self.retreat_waypoints):
            raise ValueError("every retreat waypoint must be 14D")
        if not self.home_pose:
            raise ValueError("home_pose is required")

    def assert_auto_reset_safe(self) -> None:
        if not self.verified:
            raise RuntimeError("reset profile is not onsite-verified")

