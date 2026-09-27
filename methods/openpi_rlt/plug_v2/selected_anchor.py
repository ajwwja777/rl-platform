from __future__ import annotations

import json
from pathlib import Path
from typing import Any, SupportsIndex

import numpy as np


class SelectedAnchorDataset:
    """Expose only manifest-approved full-horizon anchors from a LeRobot dataset."""

    def __init__(self, dataset: Any, manifest_path: Path | str, *, split: str):
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if manifest.get("status") != "frozen":
            raise ValueError("selected-anchor manifest must be frozen")
        if split not in {"train", "val", "test"}:
            raise ValueError(f"unsupported split: {split}")
        self._dataset = dataset
        self._indices: list[int] = []
        global_start = 0
        total_frames = 0
        for episode in manifest.get("episodes", []):
            frame_count = int(episode["frame_count"])
            if episode.get("split") == split:
                for local_frame in episode.get("anchor_frames", []):
                    local_frame = int(local_frame)
                    if not 0 <= local_frame < frame_count:
                        raise ValueError("anchor frame outside release episode")
                    self._indices.append(global_start + local_frame)
            global_start += frame_count
            total_frames += frame_count
        if total_frames != len(dataset):
            raise ValueError(
                f"manifest/dataset frame count mismatch: manifest={total_frames} dataset={len(dataset)}"
            )
        if not self._indices:
            raise ValueError(f"split {split} has no approved anchors")

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: SupportsIndex) -> Any:
        source_index = self._indices[index.__index__()]
        item = self._dataset[source_index]
        for key, value in item.items():
            if key.endswith("_is_pad") and np.asarray(value, dtype=np.bool_).any():
                raise RuntimeError(f"approved anchor returned padded action sequence: {key}")
        return item


def install_selected_anchor_patch(data_loader_module: Any, manifest_path: Path | str, *, split: str) -> None:
    """Patch the fixed upstream factory at runtime without modifying upstream source."""

    manifest_path = Path(manifest_path).resolve()
    marker = (str(manifest_path), split)
    installed = getattr(data_loader_module, "_cobot_selected_anchor_patch", None)
    if installed is not None:
        if installed != marker:
            raise RuntimeError(f"selected-anchor patch already installed with {installed}")
        return
    original = data_loader_module.create_torch_dataset

    def create_selected_dataset(*args: Any, **kwargs: Any) -> SelectedAnchorDataset:
        base = original(*args, **kwargs)
        return SelectedAnchorDataset(base, manifest_path, split=split)

    data_loader_module.create_torch_dataset = create_selected_dataset
    data_loader_module._cobot_selected_anchor_patch = marker
