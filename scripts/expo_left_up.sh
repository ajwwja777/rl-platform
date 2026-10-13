#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
model="${1:?model id required}"
case "$model" in expo-plug-left-2cam-rtc-sft-2000) step=2000;; *) echo "Unknown EXPO model" >&2; exit 2;; esac
project=/home/agilex/xiqing/projects/expo-ft
bundle=/home/agilex/xiqing/models/expo-ft/plug_left_2cam/rtc_sft_20261013/step_${step}
[[ -x "$project/.venv/bin/python" && -f "$bundle/norm_stats.json" && -d "$bundle/checkpoints/$step/params" ]] || exit 1
export NO_PROXY=127.0.0.1,localhost
export no_proxy="$NO_PROXY"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
set +u
source /home/agilex/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash
set -u
exec /usr/bin/python3 "$root/methods/expo_ft/cobot_adapter/ros_runtime.py" \
 --bundle "$bundle" --project "$project" --step "$step" --model-id "$model"
