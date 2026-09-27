from __future__ import annotations

from types import SimpleNamespace


def test_machine_a_module_is_configured_for_cobot_14d_and_registered() -> None:
    from methods.openpi_rlt.cobot_adapter.machine_a import configure_serve_module

    module = SimpleNamespace(
        PROPRIO_DIM=7,
        ACTION_DIM=7,
        CHUNK_LEN=50,
        _config=SimpleNamespace(_CONFIGS_DICT={}),
    )
    config = SimpleNamespace(name="cobot_rlt_pi05_joint")
    module.RLTPolicy = SimpleNamespace(COMPILED_BATCH_SIZES=[1, 2, 4, 8, 16])

    configure_serve_module(module, config)

    assert module.PROPRIO_DIM == 14
    assert module.ACTION_DIM == 14
    assert module.CHUNK_LEN == 50
    assert module._config._CONFIGS_DICT[config.name] is config
    assert module.RLTPolicy.COMPILED_BATCH_SIZES == [1]


def test_machine_a_contract_rejects_single_arm_metadata() -> None:
    from methods.openpi_rlt.cobot_adapter.machine_a import validate_machine_a_metadata

    metadata = {
        "has_rl_token": True,
        "z_dim": 2048,
        "proprio_dim": 7,
        "action_dim": 7,
        "chunk_len": 50,
    }

    try:
        validate_machine_a_metadata(metadata)
    except ValueError as error:
        assert "proprio_dim" in str(error)
    else:
        raise AssertionError("single-arm Machine A metadata was accepted")


def test_machine_a_contract_accepts_stage1_cobot_shapes() -> None:
    from methods.openpi_rlt.cobot_adapter.machine_a import validate_machine_a_metadata

    metadata = {
        "has_rl_token": True,
        "z_dim": 2048,
        "proprio_dim": 14,
        "action_dim": 14,
        "chunk_len": 50,
    }

    assert validate_machine_a_metadata(metadata) == metadata
