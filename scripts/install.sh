#!/usr/bin/env bash
# Install only the lightweight development/test environment, not GPU runtime environments.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
uv sync --frozen --python 3.11
echo "Development environment installed; stage1/online deployment environments remain separate."
