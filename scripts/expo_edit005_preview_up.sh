#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
model="${1:?model id required}"
[[ "$model" == expo-plug-left-2cam-edit005-preview-2000 ]] || exit 2
bundle=/home/agilex/xiqing/models/expo-ft/plug_left_2cam/rtc_sft_20261013/step_2000
project=/home/agilex/xiqing/projects/expo-ft
snapshot=/home/agilex/jiaan/data/expo_ft/plug_left_2cam/native_remote_sft2000_success25_20261014/actors/0
[[ -f "$snapshot/manifest.json" && -f "$bundle/norm_stats.json" ]] || exit 1
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
set +u
source /home/agilex/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash
set -u
exec /usr/bin/python3 "$root/methods/expo_ft/cobot_adapter/frozen_preview.py" runtime  --bundle "$bundle" --project "$project" --native-project "$root/runtime/expo-remote/native-source"  --snapshot "$snapshot" --model-id "$model"
