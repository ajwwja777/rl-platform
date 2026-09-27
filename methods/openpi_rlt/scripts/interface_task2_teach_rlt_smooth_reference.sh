#!/usr/bin/env bash
set -euo pipefail
export COBOT_RLT_COMMAND_ALPHA="0.35"
export COBOT_RLT_COMMAND_STEP="0.01"
export PYTHONFAULTHANDLER="1"
echo "SMOOTH DEMO: reference; command EMA alpha=0.35; joint command step=0.01 rad; tracking guard retained."
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${HERE}/interface_task2_teach_rlt_live.sh" 4000 "$@"
