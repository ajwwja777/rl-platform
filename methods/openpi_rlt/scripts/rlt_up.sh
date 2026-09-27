#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/media/agilex/Getea1/jiaan/projects/cobot-realworld-rl/envs/rlt-online-py310/bin/python"
exec "$PYTHON" "$SCRIPT_DIR/backend_lifecycle.py" up "$@"