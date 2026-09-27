import pytest


def test_online_freeze_manifest_only_trains_actor_and_critic() -> None:
    from methods.openpi_rlt.cobot_adapter.freeze_policy import build_freeze_manifest

    names = ["vlm/block0", "rlt_encoder/token", "actor/layer0", "critic/q1"]
    manifest = build_freeze_manifest(names, stage="online_actor_critic")

    assert manifest["trainable"] == ["actor/layer0", "critic/q1"]
    assert manifest["frozen"] == ["rlt_encoder/token", "vlm/block0"]


def test_freeze_manifest_fails_on_unclassified_parameter() -> None:
    from methods.openpi_rlt.cobot_adapter.freeze_policy import build_freeze_manifest

    with pytest.raises(ValueError, match="unclassified"):
        build_freeze_manifest(["mystery/value"], stage="online_actor_critic")

