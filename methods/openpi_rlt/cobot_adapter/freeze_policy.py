"""Fail-closed parameter ownership rules for Stage 1 and online Stage 2."""

from __future__ import annotations


def build_freeze_manifest(parameter_names, *, stage: str) -> dict[str, list[str] | str]:
    names = sorted({str(name) for name in parameter_names})
    if not names:
        raise ValueError("parameter inventory cannot be empty")
    if stage == "joint_stage1":
        return {"stage": stage, "trainable": names, "frozen": [], "unclassified": []}
    if stage != "online_actor_critic":
        raise ValueError("unknown freeze stage")
    trainable = [name for name in names if name.startswith(("actor/", "critic/"))]
    frozen = [name for name in names if name.startswith(("vlm/", "rlt_encoder/", "rl_token/"))]
    classified = set(trainable) | set(frozen)
    unclassified = [name for name in names if name not in classified]
    if unclassified:
        raise ValueError(f"unclassified parameters: {unclassified}")
    if not trainable or not frozen:
        raise ValueError("online freeze requires non-empty trainable and frozen parameter sets")
    return {"stage": stage, "trainable": trainable, "frozen": frozen, "unclassified": []}

