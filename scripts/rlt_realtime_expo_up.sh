#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export COBOT_RLT_PROJECT_ROOT="$ROOT"
source "$ROOT/scripts/environment.sh"
MODE="${1:-frozen}"
[[ "$MODE" == frozen || "$MODE" == online ]] || { echo "frozen or online required" >&2; exit 2; }
MACHINE_PY="$ROOT/envs/stage1/bin/python"
ONLINE_PY="$ROOT/envs/online/bin/python"
BUNDLE="/home/agilex/jiaan/data/rlt/realtime_expo/release_20261015"
export PYTHONPATH="$ROOT:$ROOT/third_party/openpi-rlt/rlt_online_rl/src:$ROOT/third_party/openpi-rlt/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
RLT_CONFIG="$ROOT/runtime/realtime-expo/$MODE.yaml"
"$ONLINE_PY" -m methods.openpi_rlt.realtime_expo.release check "$BUNDLE" "$MODE" "${COBOT_EVAL_SNAPSHOT:?selected Actor required}" "$RLT_CONFIG"
checkpoint="/home/agilex/jiaan/data/rlt/realtime_expo/stage1_4999"
RUN="$ROOT/outputs/rlt/plug_v3_yyshadow"
TASK_RUN="$ROOT/outputs/realtime-expo-runtime/$MODE/online"

MODEL_PORT=8030
SESSION_PORT=8026
export OPENPI_DATA_HOME="$(/usr/bin/python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["tokenizer_home"])' "$ROOT/configs/rlt/plug_v3_yyshadow/manifest.json")"
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
unset COBOT_RLT_SUPPORTED_PROFILE COBOT_RLT_EXPERIMENT_PROFILE COBOT_RLT_BRANCH_CONFIG COBOT_RLT_ONLINE_SEED
export COBOT_RLT_LOGICAL_HZ=10
execution_values=$(PYTHONPATH="$ROOT" /usr/bin/python3 -m methods.openpi_rlt.cobot_adapter.execution_profiles)
# Compile RTC once even when initially disabled; a later paused profile switch is safe.
rtc_overlay="$ROOT/../vla-platform/integrations/cobot/pi05/dagger/common/rtc_overlay"
stage1_rtc_args=(--rtc-overlay "$rtc_overlay" --warmup-rtc)
mkdir -p "$RUN/model-server" "$RUN/logs" "$TASK_RUN/metrics"
port_open() {
  /usr/bin/python3 - "$1" <<'PY'
import socket,sys
s=socket.socket();s.settimeout(.2)
try: ok=s.connect_ex(('127.0.0.1',int(sys.argv[1])))==0
finally:s.close()
raise SystemExit(0 if ok else 1)
PY
}

model_registry="$RUN/model-server/process.json"
model_alive() {
  [[ -s "$model_registry" ]] || return 1
  /usr/bin/python3 - "$model_registry" "$checkpoint" <<'PY'
import json,os,sys
p=json.load(open(sys.argv[1]));pid=int(p['pid'])
try:
 text=open(f'/proc/{pid}/stat').read();parts=text[text.rfind(')')+2:].split()
 ok=parts[0]!='Z' and int(parts[19])==int(p['start_ticks']) and p['checkpoint']==sys.argv[2]
except Exception:ok=False
raise SystemExit(0 if ok else 1)
PY
}

if ! model_alive; then
  [[ "${COBOT_RLT_REQUIRE_RESIDENT_STAGE1:-0}" != 1 ]] || { echo "Resident Stage1 is unavailable; recovery will not reload weights" >&2; exit 2; }
  port_open "$MODEL_PORT" && { echo "端口 $MODEL_PORT 被未登记进程占用" >&2; exit 2; }
  log="$RUN/logs/model-$(date -u +%Y%m%dT%H%M%SZ).log"
  export PYTHONPATH="$ROOT/envs/machine-a-py311-overlay:$ROOT:$ROOT/third_party/openpi-rlt/src:$ROOT/third_party/openpi-rlt/scripts${PYTHONPATH:+:$PYTHONPATH}"
  setsid env CUDA_VISIBLE_DEVICES=0 \
    XLA_PYTHON_CLIENT_PREALLOCATE="${COBOT_RLT_STAGE1_PREALLOCATE:-false}" \
    XLA_PYTHON_CLIENT_MEM_FRACTION="${COBOT_RLT_STAGE1_MEMORY_FRACTION:-.72}" \
    JAX_COMPILATION_CACHE_DIR="${COBOT_RLT_STAGE1_CACHE:-$ROOT/runtime/cache/jax/stage1}" \
    PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
    "$MACHINE_PY" -u -m methods.openpi_rlt.realtime_expo.serve_stage1 \
      --project-root "$ROOT" --checkpoint "$checkpoint" --port "$MODEL_PORT" "${stage1_rtc_args[@]}" \
      </dev/null >>"$log" 2>&1 &
  pid=$!
  start_ticks=$(/usr/bin/python3 - "$pid" <<'PY'
import sys
t=open(f'/proc/{sys.argv[1]}/stat').read();print(t[t.rfind(')')+2:].split()[19])
PY
)
  /usr/bin/python3 - "$model_registry" "$pid" "$start_ticks" "$checkpoint" "$log" <<'PY'
import json,os,sys,tempfile
path,pid,ticks,checkpoint,log=sys.argv[1:]
payload={'pid':int(pid),'start_ticks':int(ticks),'checkpoint':checkpoint,'log':log}
tmp=path+'.tmp';open(tmp,'w').write(json.dumps(payload,indent=2)+'\n');os.replace(tmp,path)
PY
  echo "Stage-1 正在加载（首次约数分钟）: $log"
fi

deadline=$((SECONDS+1800))
while ! port_open "$MODEL_PORT"; do
  model_alive || { echo "Stage-1 service exited; inspect registered log" >&2; exit 1; }
  (( SECONDS < deadline )) || { echo "Stage-1 加载超时" >&2; exit 1; }
  sleep 2
done
if [[ ${#stage1_rtc_args[@]} -gt 0 ]]; then
  PYTHONPATH="$ROOT:$ROOT/third_party/openpi-rlt/packages/openpi-client/src" "$MACHINE_PY" - "$MODEL_PORT" <<'PYRTC'
import sys
from openpi_client.websocket_client_policy import WebsocketClientPolicy
policy=WebsocketClientPolicy(host="127.0.0.1",port=int(sys.argv[1]))
try:
 if not policy.get_server_metadata().get("rtc_prefix_supported"):
  raise SystemExit("Loaded Stage1 lacks RTC support. Model retained; select faithful20 or explicitly reload an RTC-capable server.")
finally:
 policy._ws.close()
PYRTC
fi
echo "Stage-1 端口已监听，模型协议由 Session 就绪检查确认: $(basename "$checkpoint")"

port_open 11311 || { echo "ROS master 未启动；先启动机械臂和相机节点" >&2; exit 2; }
/usr/bin/python3 - "$COBOT_RLT_TASK5_URL/api/status" <<'PYRECORDER'
import sys,urllib.request
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
try:
 with opener.open(sys.argv[1],timeout=3) as response:
  if response.status != 200: raise RuntimeError(response.status)
except Exception as exc:
 raise SystemExit("Recorder HTTP endpoint unavailable: "+str(exc))
PYRECORDER
port_open "$SESSION_PORT" && { echo "旧 RLT Session 仍占用 $SESSION_PORT；先 Ctrl-C 旧 rlt_v3_up 终端" >&2; exit 2; }

set +u
source "$TASK5_ROS_SETUP"
set -u
export PYTHONPATH="$ROOT:$ROOT/third_party/openpi-rlt/rlt_online_rl/src:$ROOT/third_party/openpi-rlt/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
export COBOT_RLT_TASK_CONTRACT=realtime_expo_left_2cam_10hz
export COBOT_RLT_PROMPT="Unplug the black plug from the bottom socket of the power strip and insert it into the top socket."
export COBOT_RLT_CONTROL_HZ=10 COBOT_RLT_LOGICAL_HZ=10 COBOT_RLT_CHUNK_EXEC_HORIZON=10
export COBOT_RLT_HIL_SAMPLING=logical10 COBOT_RLT_HIL_TARGET=coordinator_command COBOT_RLT_RAW_OBSERVATION_CONTRACT=trace
export COBOT_RLT_REPLAY_ACTION_PRECISION=float32 COBOT_RLT_INPUT_AUDIT=strict
export COBOT_RLT_HOME_AFTER_TERMINAL=0 COBOT_RLT_ENABLE_ROBOT_RESET=0
unset COBOT_RLT_HOME_POSE COBOT_RLT_HOLD_RIGHT_GRIPPER
export COBOT_RLT_DISABLE_PHASE_CONTROLLER=1 COBOT_RLT_MIN_ONLINE_ACTOR_VERSION=1
export COBOT_RLT_LEARNER_PLATFORM=cpu
export COBOT_RLT_TASK5_DATA_ROOT="/media/agilex/Getea1/jiaan/data/datasets/realtime_expo/recordings/rlt/$MODE"
export COBOT_RLT_TASK_ID=realtime_expo_left_2cam_10hz COBOT_RLT_MODEL_ID=openpi_rlt
export COBOT_RLT_CHECKPOINT_ID="${COBOT_DEPLOYMENT_MODEL_ID:?model identity required}"
export COBOT_RLT_DATASET_ROUND="$MODE" COBOT_RLT_COLLECTION_PHASE="$MODE"
export COBOT_RLT_TASK5_MAX_TIMESTEPS=3000 COBOT_RLT_MAX_EPISODE_STEPS=0
export COBOT_RLT_SESSION_UI=1 COBOT_RLT_SESSION_UI_PORT="$SESSION_PORT" COBOT_RLT_SHADOW=0
export COBOT_RLT_JOINT_STEP_LIMIT=0.03 COBOT_RLT_GRIPPER_STEP_LIMIT=0.004
export COBOT_RLT_TRACE_DIR="$TASK_RUN/traces"
export COBOT_RLT_ACTOR_URL=http://127.0.0.1:9131 COBOT_RLT_REPLAY_URL=http://127.0.0.1:9132
export RLT_OUTPUT_DIR="$TASK_RUN" COBOT_RLT_LEARNER_STATUS_PATH="$TASK_RUN/metrics/learner_status.json"
export RLT_ACTOR_READY_TIMEOUT_SEC=180 RLT_REPLAY_READY_TIMEOUT_SEC=120
unset RLT_ALLOW_UNINITIALIZED_ACTOR
if [[ "$MODE" == frozen ]]; then export RLT_DISABLE_LEARNER=1; else unset RLT_DISABLE_LEARNER; fi
cd "$ROOT"
exec env CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
 "$ONLINE_PY" -u -m methods.openpi_rlt.scripts.online_role --upstream-root "$ROOT/third_party/openpi-rlt" \
 --config "$RLT_CONFIG" --env-factory integrations.cobot_runtime.shared_model_env:create_shared_env
