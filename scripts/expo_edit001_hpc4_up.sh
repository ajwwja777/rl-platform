#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
model="${1:?model id required}"
[[ "$model" == expo-plug-left-2cam-edit001-hpc4-2000 ]] || exit 2
profile="$root/runtime/expo-remote/profile-edit001-hpc4.json"
bundle=/home/agilex/xiqing/models/expo-ft/plug_left_2cam/rtc_sft_20261013/step_2000
project=/home/agilex/xiqing/projects/expo-ft
run_dir=/home/agilex/jiaan/data/expo_ft/plug_left_2cam/native_remote_edit001_hpc4_20261018
[[ -f "$profile" && -f "$bundle/norm_stats.json" && -f "$root/runtime/expo-remote/hpc4-ready.json" ]] || exit 1
mkdir -p "$run_dir"
exec 9>"$run_dir/owner.lock"
flock -n 9 || { echo "EXPO remote runtime already owned" >&2; exit 1; }
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
set +u
source /home/agilex/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash
set -u
exec /usr/bin/python3 "$root/methods/expo_ft/cobot_adapter/online_runtime.py" \
 --bundle "$bundle" --project "$project" --run-dir "$run_dir" --model-id "$model" \
 --recipe native_sync --rtc-delay 5 --remote-config "$profile" --gripper-feedback
