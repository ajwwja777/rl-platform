import os
from pathlib import Path
import types

import pytest


def test_validate_dataset_root_requires_complete_lerobot_layout(tmp_path: Path) -> None:
    """Catches starting a run against a metadata-only or partial dataset root."""
    from methods.openpi_rlt.stage1_entry import validate_dataset_root

    root = tmp_path / "legacy40-v2.1"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text("{}")

    with pytest.raises(ValueError, match="data"):
        validate_dataset_root(root)

    (root / "data").mkdir()
    with pytest.raises(ValueError, match="parquet"):
        validate_dataset_root(root)
    (root / "data" / "episode_000000.parquet").write_bytes(b"parquet")
    with pytest.raises(ValueError, match="videos"):
        validate_dataset_root(root)

    (root / "videos").mkdir()
    with pytest.raises(ValueError, match="mp4"):
        validate_dataset_root(root)
    (root / "videos" / "episode_000000.mp4").write_bytes(b"mp4")
    assert validate_dataset_root(root) == root.resolve()


def test_prepare_environment_routes_all_runtime_writes_to_project(tmp_path: Path, monkeypatch) -> None:
    """Catches dependencies silently writing caches or temporary files to home or /tmp."""
    from methods.openpi_rlt.stage1_entry import prepare_environment

    dataset = tmp_path / "datasets" / "legacy40-v2.1"
    project = tmp_path / "project"
    dataset.mkdir(parents=True)
    prepare_environment(dataset, project)

    assert os.environ["HF_LEROBOT_HOME"] == str(dataset.parent.resolve())
    assert os.environ["HF_HOME"] == str((project / "cache/huggingface").resolve())
    assert os.environ["OPENPI_DATA_HOME"] == str((project / "cache/openpi").resolve())
    assert os.environ["XDG_CACHE_HOME"] == str((project / "cache/xdg").resolve())
    assert os.environ["JAX_COMPILATION_CACHE_DIR"] == str((project / "cache/jax").resolve())
    assert os.environ["WANDB_DIR"] == str((project / "cache/wandb").resolve())
    assert os.environ["TMPDIR"] == str((project / "cache/tmp").resolve())
    for value in (
        "HF_HOME",
        "OPENPI_DATA_HOME",
        "XDG_CACHE_HOME",
        "JAX_COMPILATION_CACHE_DIR",
        "WANDB_DIR",
        "TMPDIR",
    ):
        assert Path(os.environ[value]).is_dir()


def test_configure_lerobot_video_backend_forces_pyav() -> None:
    """Catches silently selecting TorchCodec on hosts without system FFmpeg."""
    from lerobot.common.datasets import lerobot_dataset

    from methods.openpi_rlt.stage1_entry import configure_lerobot_video_backend

    original = lerobot_dataset.get_safe_default_codec
    try:
        configure_lerobot_video_backend()
        assert lerobot_dataset.get_safe_default_codec() == "pyav"
    finally:
        lerobot_dataset.get_safe_default_codec = original


def test_redirect_upstream_jax_cache_is_narrow_and_restored(tmp_path: Path) -> None:
    """Catches the upstream hard-coded ~/.cache/jax path escaping the project boundary."""
    from methods.openpi_rlt.stage1_entry import redirect_upstream_jax_cache

    original = Path
    module = types.SimpleNamespace(epath=types.SimpleNamespace(Path=original))
    target = (tmp_path / "jax").resolve()

    with redirect_upstream_jax_cache(module, target):
        assert module.epath.Path("~/.cache/jax") == target
        assert module.epath.Path("relative/file") == original("relative/file")

    assert module.epath.Path is original


def test_redirect_does_not_mutate_the_shared_epath_module(tmp_path: Path) -> None:
    """Catches breaking Orbax isinstance checks by replacing global epath.Path."""
    from methods.openpi_rlt.stage1_entry import redirect_upstream_jax_cache

    shared_epath = types.SimpleNamespace(Path=Path, sentinel=object())
    module = types.SimpleNamespace(epath=shared_epath)

    with redirect_upstream_jax_cache(module, tmp_path / "jax"):
        assert shared_epath.Path is Path
        assert module.epath is not shared_epath
        assert module.epath.sentinel is shared_epath.sentinel

    assert module.epath is shared_epath


def test_run_upstream_train_passes_config_with_project_jax_cache(tmp_path: Path) -> None:
    """Catches bypassing the fixed upstream main or leaking its JAX cache to home."""
    from methods.openpi_rlt.stage1_entry import run_upstream_train

    seen = {}
    module = types.SimpleNamespace(epath=types.SimpleNamespace(Path=Path))

    def main(config):
        seen["config"] = config
        seen["cache"] = module.epath.Path("~/.cache/jax")

    module.main = main
    marker = object()
    target = tmp_path / "cache/jax"

    run_upstream_train(module, marker, target)

    assert seen == {"config": marker, "cache": target.resolve()}
    assert module.epath.Path is Path
