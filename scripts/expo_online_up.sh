#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
model="${1:?model id required}"
extra=()
case "$model" in
 expo-plug-left-2cam-online-2000) run_name=online_from_rtc_sft_2000 ;;
 expo-plug-left-2cam-native-sync-2000)
   run_name=native_sync_sft2000_success25_20261014
   extra=(--recipe native_sync --seed-dir /home/agilex/jiaan/data/expo_ft/plug_left_2cam/seed_success25_20261014_v2)
   ;;
 *) exit 2 ;;
esac
project=/home/agilex/xiqing/projects/expo-ft
bundle=/home/agilex/xiqing/models/expo-ft/plug_left_2cam/rtc_sft_20261013/step_2000
run_dir=/home/agilex/jiaan/data/expo_ft/plug_left_2cam/"$run_name"
[[ -x "$project/.venv/bin/python" && -f "$bundle/norm_stats.json" ]] || exit 1
mkdir -p "$run_dir"
exec 9>"$run_dir/owner.lock"
flock -n 9 || { echo "EXPO online run already owned" >&2; exit 1; }
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export XLA_PYTHON_CLIENT_ALLOCATOR=platform XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.98
set +u
source /home/agilex/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash
set -u
exec /usr/bin/python3 "$root/methods/expo_ft/cobot_adapter/online_runtime.py" \
 --bundle "$bundle" --project "$project" --run-dir "$run_dir" --model-id "$model" "${extra[@]}"
