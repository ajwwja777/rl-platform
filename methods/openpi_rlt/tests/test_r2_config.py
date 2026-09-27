from pathlib import Path

import pytest


def test_checked_r2_config_has_bounded_warmup_and_safe_reset() -> None:
    from methods.openpi_rlt.cobot_adapter.r2_config import load_r2_config

    path = Path(__file__).parents[1] / "configs" / "plug_insertion_r2.yaml"
    config = load_r2_config(path)

    assert config.mode == "eval"
    assert config.active_arm in {"left", "right"}
    assert config.explore_gripper is False
    assert 0 < config.warmup.min_updates <= config.warmup.max_updates <= 2_000
    assert config.reset.auto_reset is False
    assert config.reset.verified is False


def test_r2_config_rejects_unsafe_auto_reset(tmp_path) -> None:
    from methods.openpi_rlt.cobot_adapter.r2_config import load_r2_config

    path = tmp_path / "bad.yaml"
    path.write_text(
        """
mode: explore
active_arm: left
explore_gripper: false
warmup: {min_transitions: 32, min_updates: 20, max_updates: 100, eval_interval: 10}
reset: {auto_reset: true, verified: false, gripper_mode: hold}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="verified reset"):
        load_r2_config(path)


@pytest.mark.parametrize("mode", ["random", "live", ""])
def test_r2_config_rejects_unknown_mode(tmp_path, mode: str) -> None:
    from methods.openpi_rlt.cobot_adapter.r2_config import load_r2_config

    path = tmp_path / "bad.yaml"
    path.write_text(
        f"""
mode: {mode!r}
active_arm: left
explore_gripper: false
warmup: {{min_transitions: 32, min_updates: 20, max_updates: 100, eval_interval: 10}}
reset: {{auto_reset: false, verified: false, gripper_mode: hold}}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="mode"):
        load_r2_config(path)

