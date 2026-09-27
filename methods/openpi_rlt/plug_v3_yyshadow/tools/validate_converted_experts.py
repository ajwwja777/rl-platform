#!/usr/bin/env python3
"""Revalidate a converted plug_v3 dataset and seal its compatibility metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import pyarrow.parquet as pq


TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
from convert_experts import sha256, verify_release  # noqa: E402


def payload_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != "conversion_manifest.json"):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def verify_parquet_metadata(root: Path, expected_dim: int) -> int:
    paths = sorted((root / "data").rglob("*.parquet"))
    for path in paths:
        metadata = pq.read_schema(path).metadata or {}
        payload = json.loads(metadata[b"huggingface"].decode("utf-8"))
        features = payload["info"]["features"]
        for key in ("observation.state", "action"):
            feature = features[key]
            if feature.get("_type") != "Sequence" or int(feature.get("length", -1)) != expected_dim:
                raise ValueError(f"incompatible metadata: {path}: {key}: {feature}")
    return len(paths)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--expected-dim", type=int, default=7)
    args = parser.parse_args()
    raw_root = args.raw_root.resolve(strict=True)
    root = args.dataset_root.resolve(strict=True)
    manifest_path = root / "conversion_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    parquet_count = verify_parquet_metadata(root, args.expected_dim)
    validation = verify_release(raw_root, root, manifest["episodes"])
    manifest["validation"] = validation
    manifest["compatibility"] = {
        "lerobot": "0.1.0",
        "format": "v2.1",
        "huggingface_vector_type": "Sequence",
        "validated_reader": "datasets==3.6.0, pyarrow==20.0.0",
        "parquet_count": parquet_count,
    }
    manifest["payload_tree_sha256"] = payload_tree_sha256(root)
    temporary = manifest_path.with_suffix(".json.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, manifest_path)
    print(json.dumps({
        "parquet_count": parquet_count,
        "payload_tree_sha256": manifest["payload_tree_sha256"],
        "validation": validation,
    }, indent=2))


if __name__ == "__main__":
    main()
