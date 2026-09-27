#!/usr/bin/env python3
"""Reproduce the 14D chunk-relative action stats used by online RLT."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


JOINTS = np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
GRIPPERS = np.asarray([6, 13])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--chunk-len", type=int, default=10)
    args = parser.parse_args()
    paths = sorted(glob.glob(str(args.dataset / "data" / "**" / "*.parquet"), recursive=True))
    if not paths:
        raise FileNotFoundError(f"no episode parquet files below {args.dataset}")
    chunks: list[np.ndarray] = []
    window_count = 0
    for path in paths:
        table = pq.read_table(path, columns=["observation.state", "action"])
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        starts = list(range(0, len(action) - args.chunk_len + 1, args.chunk_len))
        terminal = len(action) - args.chunk_len
        if terminal >= 0 and terminal not in starts:
            starts.append(terminal)
        for start in starts:
            chunk = action[start : start + args.chunk_len].copy()
            chunk[:, JOINTS] -= state[start, JOINTS]
            chunks.append(chunk)
            window_count += 1
    values = np.concatenate(chunks, axis=0)
    stats = {
        "norm_stats": {
            "actions": {
                "mean": np.mean(values, axis=0, dtype=np.float64).tolist(),
                "std": np.std(values, axis=0, dtype=np.float64).tolist(),
                "q01": np.quantile(values, 0.01, axis=0).tolist(),
                "q99": np.quantile(values, 0.99, axis=0).tolist(),
            }
        },
        "provenance": {
            "dataset": args.dataset.name,
            "episodes": len(paths),
            "chunk_len": args.chunk_len,
            "window_stride": args.chunk_len,
            "terminal_window": True,
            "windows": window_count,
            "action_vectors": int(values.shape[0]),
            "joint_delta_indices": JOINTS.tolist(),
            "absolute_gripper_indices": GRIPPERS.tolist(),
        },
    }
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
