#!/usr/bin/env python3
"""Convert an audited plug_v3 expert selection to 7D LeRobot v2.1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from functools import partial
from importlib.metadata import version
from pathlib import Path

import h5py
import numpy as np


CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
ACTIVE = slice(7, 14)
TASK = "Insert the held plug into the socket."


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def deterministic_split(uuids: list[str], val_count: int, seed: int) -> dict[str, str]:
    if not 0 < val_count < len(uuids):
        raise ValueError("val-count must be between zero and the accepted episode count")
    ranked = sorted(uuids, key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).digest())
    validation = set(ranked[-val_count:])
    return {value: "val" if value in validation else "train" for value in uuids}


def load_selection(path: Path, raw_root: Path) -> tuple[dict, list[dict]]:
    selection = json.loads(path.read_text())
    if int(selection.get("schema_version", 0)) < 2:
        raise ValueError("quality selection schema v2 or newer is required")
    if Path(selection["data_root"]).resolve() != raw_root:
        raise ValueError("selection data_root does not match --raw-root")
    complete = sorted(raw_root.glob("episode_*.hdf5"))
    if len(complete) != int(selection["raw_episode_count"]):
        raise ValueError("raw episode inventory changed after quality audit")
    accepted = [item for item in selection["episodes"] if item["decision"].startswith("accept")]
    if len(accepted) != int(selection["accepted_count"]):
        raise ValueError("accepted count mismatch")
    if {Path(item["source"]).resolve() for item in selection["episodes"]} != set(complete):
        raise ValueError("selection does not cover the complete raw inventory")
    return selection, accepted


def source_sidecar(raw_root: Path, uuid: str) -> Path:
    path = raw_root / ".segments" / uuid / "sidecar.json"
    if not path.is_file():
        raise ValueError(f"missing sidecar: {uuid}")
    return path


def verify_release(raw_root: Path, output: Path, records: list[dict]) -> dict:
    import av
    import pyarrow.parquet as pq

    decoded_frames = 0
    worst_rgb_mae = 0.0
    for item in records:
        release_index = int(item["release_episode_index"])
        start = int(item["frame_start"])
        end = int(item["frame_end_exclusive"])
        count = end - start
        source = raw_root / item["source_hdf5"]
        parquet = output / f"data/chunk-{release_index // 1000:03d}/episode_{release_index:06d}.parquet"
        table = pq.read_table(parquet)
        if len(table) != count:
            raise ValueError(f"parquet frame count mismatch: {release_index}")
        with h5py.File(source, "r") as episode:
            expected_state = np.asarray(episode["observations/qpos"][start:end, ACTIVE], dtype=np.float32)
            expected_action = np.asarray(episode["action"][start:end, ACTIVE], dtype=np.float32)
            if not np.array_equal(np.stack(table["observation.state"].to_pylist()), expected_state):
                raise ValueError(f"state mismatch: {release_index}")
            if not np.array_equal(np.stack(table["action"].to_pylist()), expected_action):
                raise ValueError(f"action mismatch: {release_index}")
            for camera in CAMERAS:
                video_path = output / (
                    f"videos/chunk-{release_index // 1000:03d}/"
                    f"observation.images.{camera}/episode_{release_index:06d}.mp4"
                )
                sample_indices = {0, count // 2, count - 1}
                video_count = 0
                with av.open(str(video_path)) as container:
                    stream = container.streams.video[0]
                    if stream.width != 640 or stream.height != 480 or float(stream.average_rate) != 30:
                        raise ValueError(f"video contract mismatch: {release_index} {camera}")
                    for frame in container.decode(stream):
                        if video_count in sample_indices:
                            actual = frame.to_ndarray(format="rgb24")
                            expected = episode[f"observations/images/{camera}"][start + video_count]
                            mae = float(np.abs(actual.astype(np.float32) - expected.astype(np.float32)).mean())
                            worst_rgb_mae = max(worst_rgb_mae, mae)
                            if mae > 5.0:
                                raise ValueError(f"video quality mismatch: {release_index} {camera} {mae}")
                        video_count += 1
                if video_count != count:
                    raise ValueError(f"decoded video frame count mismatch: {release_index} {camera}")
                decoded_frames += video_count
        if sha256(source) != item["source_hdf5_sha256"]:
            raise ValueError(f"source changed during conversion: {source}")
        sidecar = raw_root / item["sidecar"]
        if sha256(sidecar) != item["sidecar_sha256"]:
            raise ValueError(f"sidecar changed during conversion: {sidecar}")
    return {
        "status": "passed",
        "parquet_vectors": "all_7d_frames_exact",
        "video_frame_count": decoded_frames,
        "max_sample_rgb_mae": worst_rgb_mae,
        "source_hashes_rechecked": True,
    }


def convert(args: argparse.Namespace) -> None:
    raw_root = args.raw_root.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if version("lerobot") != "0.1.0":
        raise RuntimeError(f"requires lerobot==0.1.0, found {version('lerobot')}")
    selection, accepted = load_selection(args.selection.resolve(strict=True), raw_root)
    split = deterministic_split([item["episode_uuid"] for item in accepted], args.val_count, args.seed)

    from lerobot.common.datasets import lerobot_dataset
    from lerobot.common.datasets.video_utils import encode_video_frames

    lerobot_dataset.encode_video_frames = partial(encode_video_frames, vcodec="h264", crf=10)
    names = ["right_j1", "right_j2", "right_j3", "right_j4", "right_j5", "right_j6", "right_gripper"]
    features = {
        "observation.state": {"dtype": "float32", "shape": (7,), "names": names},
        "action": {"dtype": "float32", "shape": (7,), "names": names},
    }
    for camera in CAMERAS:
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        }
    dataset = lerobot_dataset.LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=30,
        root=output,
        robot_type="cobot_right_arm",
        features=features,
        use_videos=True,
        image_writer_threads=4,
    )

    records: list[dict] = []
    try:
        for release_index, item in enumerate(accepted):
            source = Path(item["source"]).resolve(strict=True)
            start = int(item["selected_frame_start"])
            end = int(item["selected_frame_end_exclusive"])
            sidecar = source_sidecar(raw_root, item["episode_uuid"])
            with h5py.File(source, "r") as episode:
                if str(episode.attrs["episode_uuid"]) != item["episode_uuid"]:
                    raise ValueError(f"UUID mismatch: {source}")
                if episode["action"].shape[1:] != (14,) or episode["observations/qpos"].shape[1:] != (14,):
                    raise ValueError(f"raw vector contract changed: {source}")
                for frame in range(start, end):
                    value = {
                        f"observation.images.{camera}": np.asarray(
                            episode[f"observations/images/{camera}"][frame]
                        )
                        for camera in CAMERAS
                    }
                    value.update(
                        {
                            "observation.state": np.asarray(episode["observations/qpos"][frame, ACTIVE], dtype=np.float32),
                            "action": np.asarray(episode["action"][frame, ACTIVE], dtype=np.float32),
                            "task": args.task,
                        }
                    )
                    dataset.add_frame(value)
            dataset.save_episode()
            records.append(
                {
                    "release_episode_index": release_index,
                    "source_episode_index": int(item["episode_index"]),
                    "episode_uuid": item["episode_uuid"],
                    "split": split[item["episode_uuid"]],
                    "frame_start": start,
                    "frame_end_exclusive": end,
                    "frame_count": end - start,
                    "source_hdf5": source.name,
                    "source_hdf5_sha256": sha256(source),
                    "sidecar": sidecar.relative_to(raw_root).as_posix(),
                    "sidecar_sha256": sha256(sidecar),
                }
            )
            print("CONVERT", release_index, source.name, start, end, flush=True)
    finally:
        dataset.stop_image_writer()

    validation = verify_release(raw_root, output, records)
    manifest = {
        "schema_version": "plug-v3-yyshadow-expert-lerobot-v1",
        "status": "validated",
        "repo_id": args.repo_id,
        "task": args.task,
        "source_selection": str(args.selection.resolve()),
        "source_selection_sha256": sha256(args.selection.resolve()),
        "source_raw_episode_count": int(selection["raw_episode_count"]),
        "accepted_episode_count": len(records),
        "rejected_episode_count": int(selection["rejected_count"]),
        "training_frame_count": sum(item["frame_count"] for item in records),
        "action_dim": 7,
        "proprio_dim": 7,
        "active_indices_in_raw14": list(range(7, 14)),
        "camera_views": list(CAMERAS),
        "split_seed": args.seed,
        "split_counts": {
            "train": sum(item["split"] == "train" for item in records),
            "val": sum(item["split"] == "val" for item in records),
        },
        "episodes": records,
        "validation": validation,
    }
    write_new_json(output / "conversion_manifest.json", manifest)
    write_new_json(
        output / "splits.json",
        {name: [item["release_episode_index"] for item in records if item["split"] == name] for name in ("train", "val")},
    )
    print(json.dumps({key: manifest[key] for key in ("accepted_episode_count", "rejected_episode_count", "training_frame_count", "split_counts")}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", default="jiaan/plug_v3_yyshadow_demonstrations")
    parser.add_argument("--task", default=TASK)
    parser.add_argument("--val-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    convert(parser.parse_args())


if __name__ == "__main__":
    main()
