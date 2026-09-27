#!/usr/bin/env bash
set -euo pipefail
if [[ "${COBOT_RLT_ALLOW_LEGACY_8017:-0}" != "1" ]]; then
  echo "8017 recorder is retired. Start scripts/start_cobot_data_ui.sh on 8015, select RLT, then use rlt_up.sh." >&2
  exit 2
fi
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK5_CODE="/media/agilex/Getea1/jiaan/projects/cobot-realworld-vla/task5/segmented-teach-v1/code"
PY="/home/agilex/miniconda3/envs/cobot-station/bin/python"
set +u
source /home/agilex/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash
set -u
export PYTHONPATH="${TASK5_CODE}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
exec "$PY" "$ROOT/ensure_rollout_recorder.py" "$@"
