"""Build an auditable transition plan from a frozen segmented-teach release."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def build_transition_plan(manifest_path: str | Path, *, chunk_len: int, stride: int) -> dict:
    path = Path(manifest_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "frozen":
        raise ValueError("dataset manifest must be frozen before transition planning")
    if chunk_len <= 0 or stride <= 0:
        raise ValueError("chunk_len and stride must be positive")
    transitions: list[dict] = []
    used_episodes: set[str] = set()
    for episode in payload.get("episodes", []):
        if episode.get("status") != "complete":
            raise ValueError("incomplete/deleted episodes cannot enter a training release")
        episode_id = str(episode.get("episode_uuid", ""))
        frame_count = int(episode.get("frame_count", 0))
        source_sha = str(episode.get("source_sha256", ""))
        if not episode_id or frame_count <= 0 or len(source_sha) != 64:
            raise ValueError("episode provenance is incomplete")
        for segment_id, segment in enumerate(episode.get("approved_segments", []), start=1):
            start = int(segment["start_frame"])
            end = int(segment["end_frame"])
            if start < 0 or end >= frame_count or end < start:
                raise ValueError("approved segment is outside episode bounds")
            last_start = end - chunk_len + 1
            if last_start < start:
                continue
            used_episodes.add(episode_id)
            for frame in range(start, last_start + 1, stride):
                transitions.append(
                    {
                        "episode_uuid": episode_id,
                        "segment_id": segment_id,
                        "start_frame": frame,
                        "end_frame": frame + chunk_len - 1,
                        "source_sha256": source_sha,
                        "source": "expert",
                    }
                )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "schema_version": 1,
        "status": "planned-not-materialized",
        "dataset_id": payload.get("dataset_id"),
        "task_prompt": payload.get("task_prompt"),
        "source_manifest": str(path.resolve()),
        "source_manifest_sha256": digest,
        "chunk_len": chunk_len,
        "stride": stride,
        "unique_episodes": len(used_episodes),
        "transition_count": len(transitions),
        "transitions": transitions,
    }

