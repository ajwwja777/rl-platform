#!/usr/bin/env python3
"""Restore the frozen RLT runtime without changing packages or starting services."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()

def relocate(envs):
    stage1 = envs / "stage1"
    base = envs / "python311"
    config = stage1 / "pyvenv.cfg"
    lines = config.read_text().splitlines()
    config.write_text("\n".join("home = " + str(base / "bin") if line.startswith("home =") else line for line in lines) + "\n")
    for name in ("python", "python3", "python3.11"):
        binary = stage1 / "bin" / name
        if not binary.is_symlink():
            raise RuntimeError("Expected a snapshot interpreter symlink: " + str(binary))
        binary.unlink()
        binary.symlink_to(base / "bin/python3.11")
    site = stage1 / "lib/python3.11/site-packages"
    for name, source in (
        ("_editable_impl_openpi.pth", ROOT / "third_party/openpi-rlt/src"),
        ("_editable_impl_openpi_client.pth", ROOT / "integrations/stage1-client/src"),
    ):
        if not source.is_dir():
            raise RuntimeError("Restore the pinned source first: " + str(source))
        (site / name).write_text(str(source) + "\n")

def verify(envs):
    environment = os.environ.copy()
    environment.update(JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    environment.pop("PYTHONPATH", None)
    online = "import sys,numpy,jax,flax,optax,openpi_client; print(sys.version); print(numpy.__version__,jax.__version__,flax.__version__,optax.__version__)"
    paths = [str(envs / "machine-a-py311-overlay"), str(ROOT), str(ROOT / "third_party/openpi-rlt/src")]
    stage1 = "import sys;sys.path[:0]=" + repr(paths) + ";import torch,jax,openpi,openpi_client;from openpi.models import pi0;from openpi.training import config;print(torch.__version__,jax.__version__);print(pi0.__file__)"
    for name, code in (("online", online), ("stage1", stage1)):
        subprocess.run([str(envs / name / "bin/python"), "-I", "-c", code], check=True, env=environment)
    print("Runtime imports passed; model readiness and robot acceptance are separate checks.")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materials", type=Path)
    parser.add_argument("--destination", type=Path, required=True, help="Empty environment directory, normally <project>/envs")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    destination = args.destination.resolve()
    if not args.verify_only:
        if not args.materials:
            parser.error("--materials is required when restoring")
        if destination.exists() and any(destination.iterdir()):
            parser.error("Destination must be empty; existing environments are never replaced")
        manifest = json.loads((ROOT / "configs/assets/runtime_environment.json").read_text())
        archive = args.materials / manifest["archive"]
        if digest(archive) != manifest["sha256"]:
            raise ValueError("Runtime archive SHA256 mismatch")
        destination.mkdir(parents=True, exist_ok=True)
        subprocess.run(["tar", "-xzf", str(archive.resolve()), "--no-same-owner", "--strip-components=1", "-C", str(destination)], check=True)
        relocate(destination)
    verify(destination)

if __name__ == "__main__":
    main()
