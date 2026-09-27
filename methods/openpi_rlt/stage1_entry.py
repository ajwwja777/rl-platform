"""Boundary checks shared by the project-owned Stage-1 entrypoints."""

from collections.abc import Iterator
from contextlib import contextmanager
import os
from pathlib import Path
from typing import Any


def configure_lerobot_video_backend() -> None:
    """Force the verified PyAV decoder instead of LeRobot's package-presence heuristic."""
    from lerobot.common.datasets import lerobot_dataset
    from lerobot.common.datasets import video_utils

    def use_pyav() -> str:
        return "pyav"

    # LeRobot imports this symbol into lerobot_dataset, so patch both references.
    video_utils.get_safe_default_codec = use_pyav
    lerobot_dataset.get_safe_default_codec = use_pyav


def validate_dataset_root(dataset_root: str | Path) -> Path:
    """Require a local, materialized LeRobot dataset rather than a partial view."""
    root = Path(dataset_root).resolve()
    info = root / "meta" / "info.json"
    if not info.is_file():
        raise ValueError(f"LeRobot metadata is missing: {info}")
    data = root / "data"
    if not data.is_dir():
        raise ValueError(f"LeRobot data directory is missing: {data}")
    if next(data.rglob("*.parquet"), None) is None:
        raise ValueError(f"LeRobot parquet files are missing below: {data}")
    videos = root / "videos"
    if not videos.is_dir():
        raise ValueError(f"LeRobot videos directory is missing: {videos}")
    if next(videos.rglob("*.mp4"), None) is None:
        raise ValueError(f"LeRobot mp4 files are missing below: {videos}")
    return root


def prepare_environment(dataset_root: str | Path, project_root: str | Path) -> None:
    """Route dataset lookup and writable runtime caches into registered roots."""
    dataset = Path(dataset_root).resolve()
    project = Path(project_root).resolve()
    routes = {
        "HF_LEROBOT_HOME": dataset.parent,
        "HF_HOME": project / "cache" / "huggingface",
        "OPENPI_DATA_HOME": project / "cache" / "openpi",
        "XDG_CACHE_HOME": project / "cache" / "xdg",
        "JAX_COMPILATION_CACHE_DIR": project / "cache" / "jax",
        "WANDB_DIR": project / "cache" / "wandb",
        "TMPDIR": project / "cache" / "tmp",
    }
    for variable, path in routes.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)


@contextmanager
def redirect_upstream_jax_cache(module: Any, cache_dir: str | Path) -> Iterator[None]:
    """Narrowly redirect the upstream ``~/.cache/jax`` literal for one call."""
    target = Path(cache_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    original_epath = module.epath
    original_path = original_epath.Path

    def redirected_path(value: Any, *args: Any, **kwargs: Any):
        if value == "~/.cache/jax":
            return original_path(target)
        return original_path(value, *args, **kwargs)

    class EPathProxy:
        Path = staticmethod(redirected_path)

        def __getattr__(self, name: str) -> Any:
            return getattr(original_epath, name)

    module.epath = EPathProxy()
    try:
        yield
    finally:
        module.epath = original_epath


def run_upstream_train(module: Any, config: Any, cache_dir: str | Path) -> None:
    """Run the fixed upstream main while containing its hard-coded cache path."""
    with redirect_upstream_jax_cache(module, cache_dir):
        module.main(config)
