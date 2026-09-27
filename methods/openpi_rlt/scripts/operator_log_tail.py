#!/usr/bin/env python3
"""Follow the combined supervisor log while hiding routine polling noise."""

from __future__ import annotations

import argparse
import time
from pathlib import Path


def visible_line(line: str) -> bool:
    return '"GET /api/session HTTP/' not in line and "Waiting for warmup replay_size=" not in line


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    with args.path.open("r", encoding="utf-8", errors="replace") as stream:
        stream.seek(0, 2)
        while True:
            line = stream.readline()
            if line:
                if visible_line(line):
                    print(line, end="", flush=True)
            else:
                time.sleep(0.1)


if __name__ == "__main__":
    main()
