#!/usr/bin/env python3
"""Best-effort CLI used by the Cobot wrapper before terminating child roles."""

from __future__ import annotations

import argparse
import json

from methods.openpi_rlt.cobot_adapter.session_shutdown import shutdown_via_http


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--timeout-sec", type=float, default=10.0)
    args = parser.parse_args()
    print(json.dumps(shutdown_via_http(args.url, timeout_sec=args.timeout_sec), sort_keys=True))


if __name__ == "__main__":
    main()
