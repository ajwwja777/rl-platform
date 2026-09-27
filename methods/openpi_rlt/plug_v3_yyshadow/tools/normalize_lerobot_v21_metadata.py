#!/usr/bin/env python3
"""Normalize LeRobot v2.1 vector metadata for datasets<=3.6 readers."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pyarrow.parquet as pq


VECTOR_KEYS = ("observation.state", "action")


def normalize(path: Path, *, expected_dim: int) -> bool:
    table = pq.read_table(path)
    metadata = dict(table.schema.metadata or {})
    if b"huggingface" not in metadata:
        raise ValueError(f"missing Hugging Face schema metadata: {path}")
    payload = json.loads(metadata[b"huggingface"].decode("utf-8"))
    features = payload["info"]["features"]
    changed = False
    for key in VECTOR_KEYS:
        feature = features[key]
        if int(feature.get("length", -1)) != expected_dim:
            raise ValueError(f"{path}: {key} length is not {expected_dim}")
        kind = feature.get("_type")
        if kind == "List":
            feature["_type"] = "Sequence"
            changed = True
        elif kind != "Sequence":
            raise ValueError(f"{path}: unexpected {key} feature type {kind!r}")
    if not changed:
        return False

    codecs = {
        pq.ParquetFile(path).metadata.row_group(row).column(column).compression.lower()
        for row in range(pq.ParquetFile(path).metadata.num_row_groups)
        for column in range(pq.ParquetFile(path).metadata.num_columns)
    }
    if len(codecs) != 1:
        raise ValueError(f"mixed parquet compression is unsupported: {path}: {codecs}")
    metadata[b"huggingface"] = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    table = table.replace_schema_metadata(metadata)
    temporary = path.with_name(path.name + ".metadata-fix.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    pq.write_table(table, temporary, compression=next(iter(codecs)))
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--expected-dim", type=int, default=7)
    args = parser.parse_args()
    root = args.dataset_root.resolve(strict=True)
    paths = sorted((root / "data").rglob("*.parquet"))
    if not paths:
        raise ValueError(f"no parquet files below {root / 'data'}")
    changed = sum(normalize(path, expected_dim=args.expected_dim) for path in paths)
    if list((root / "data").rglob("*.metadata-fix.tmp")):
        raise RuntimeError("temporary metadata files remain")
    print(json.dumps({"parquet_count": len(paths), "normalized_count": changed}))


if __name__ == "__main__":
    main()
