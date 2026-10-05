#!/usr/bin/env bash
# Start/reuse the frozen plug_v3 Stage-1 server, then run upstream online RLT.
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export COBOT_RLT_PROJECT_ROOT="${COBOT_RLT_PROJECT_ROOT:-$ROOT}"
source "$ROOT/scripts/environment.sh"
# Ctrl-C stops Machine B and the robot Session; the loaded Stage-1 model remains.
set -Eeuo pipefail

ROOT="${COBOT_RLT_PROJECT_ROOT:-$ROOT}"
METHOD="$ROOT/methods/openpi_rlt/plug_v3_yyshadow"
RUN="$ROOT/outputs/rlt/plug_v3_yyshadow"
MANIFEST="$ROOT/configs/rlt/plug_v3_yyshadow/manifest.json"
MACHINE_PY="$ROOT/envs/stage1/bin/python"
ONLINE_PY="$ROOT/envs/online/bin/python"
"$ONLINE_PY" "$ROOT/scripts/preflight.py"
export OPENPI_DATA_HOME="$(/usr/bin/python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["tokenizer_home"])' "$MANIFEST")"
export HF_HUB_OFFLINE=1
MODEL_PORT=8030
SESSION_PORT=8026
MODE=${1:-warmup}

if [[ "$MODE" != reference && "$MODE" != warmup && "$MODE" != frozen && "$MODE" != online ]]; then
  echo "用法: ./scripts/rlt_v3_up.sh [reference|warmup|frozen|online]" >&2
  exit 2
fi
[[ -s "$MANIFEST" ]] || { echo "plug_v3 发布清单不存在: $MANIFEST" >&2; exit 2; }
checkpoint=$(/usr/bin/python3 - "$MANIFEST" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['checkpoint'])
PY
)
[[ -d "$checkpoint/params" ]] || { echo "checkpoint 参数不存在: $checkpoint" >&2; exit 2; }

DATA_PHASE="$MODE"
[[ "$MODE" == reference ]] && DATA_PHASE=warmup
[[ "$MODE" == frozen ]] && DATA_PHASE=online
RLT_CONFIG="$ROOT/configs/rlt/plug_v3_yyshadow/online_rl.yaml"
[[ "$MODE" == frozen ]] && RLT_CONFIG="$ROOT/configs/rlt/plug_v3_yyshadow/online_rl_frozen.yaml"
# Optional project-owned experiment; original model registrations stay unchanged.
EXPERIMENT_RUN=""
unset COBOT_RLT_EXPERIMENT_PROFILE
profile_values=""
if [[ "$MODE" == online ]]; then
  profile_values=$(PYTHONPATH="$ROOT" /usr/bin/python3 -m integrations.cobot_runtime.experiment_profiles "${COBOT_DEPLOYMENT_MODEL_ID:-}")
fi
if [[ -n "$profile_values" ]]; then
  [[ "$MODE" == online ]] || { echo "Experimental profile requires online mode" >&2; exit 2; }
  mapfile -t profile_fields <<< "$profile_values"
  export COBOT_RLT_EXPERIMENT_PROFILE="${profile_fields[0]}"
  RLT_CONFIG="${profile_fields[1]}"
  EXPERIMENT_RUN="${profile_fields[2]}"
  mkdir -p "$EXPERIMENT_RUN/online/metrics"
fi
# An explicit fixed-step collection load forks complete learner state. Rebuilds
# reuse the same branch; the original 5k checkpoint is never a write target.
if [[ "$MODE" == online && -n "${COBOT_RLT_BRANCH_CONFIG:-}" ]]; then
  if [[ -n "${COBOT_RLT_ONLINE_SEED:-}" && ! -f "$COBOT_RLT_SEED_DESTINATION/seed.json" ]]; then
    "$ONLINE_PY" -m integrations.cobot_runtime.online_seed "$COBOT_RLT_ONLINE_SEED" "$COBOT_RLT_SEED_DESTINATION" "$RLT_CONFIG" "$COBOT_RLT_BRANCH_CONFIG" "$COBOT_RLT_BRANCH_RUN" \
      --replay-budget-policy "${COBOT_RLT_SEED_REPLAY_BUDGET_POLICY:-new_arrivals}" \
      --publication-policy "${COBOT_RLT_SEED_PUBLICATION_POLICY:-automatic}"
  fi
  [[ -f "$COBOT_RLT_BRANCH_CONFIG" ]] || { echo "Selected online branch configuration missing" >&2; exit 2; }
  # Explicit choices must also match a reused branch; never silently mutate it.
  if [[ -n "${COBOT_RLT_SEED_PUBLICATION_POLICY:-}" || -n "${COBOT_RLT_SEED_REPLAY_BUDGET_POLICY:-}" ]]; then
    "$ONLINE_PY" - "${COBOT_RLT_SEED_DESTINATION:?Explicit seed policies require a branch destination}/seed.json" "$COBOT_RLT_BRANCH_CONFIG" "${COBOT_RLT_SEED_PUBLICATION_POLICY:-}" "${COBOT_RLT_SEED_REPLAY_BUDGET_POLICY:-}" <<'PYSEEDPOLICY'
import json,sys,yaml
metadata=json.load(open(sys.argv[1]))
config=yaml.safe_load(open(sys.argv[2]))
for field,requested in [('publication_policy',sys.argv[3]),('replay_budget_policy',sys.argv[4])]:
 if requested and metadata.get(field)!=requested:
  raise SystemExit('Selected online branch '+field+' does not match explicit request')
runtime=config['runtime']
served=runtime['actor_service']['snapshot_path']
candidate=runtime['learner_service']['actor_snapshot_path']
if served!=metadata.get('served_actor_snapshot') or candidate!=metadata.get('candidate_actor_snapshot'):
 raise SystemExit('Selected online branch Actor paths do not match seed metadata')
if metadata.get('publication_policy')=='staged' and served==candidate:
 raise SystemExit('Selected staged branch must separate served and candidate Actor')
PYSEEDPOLICY
  fi
  RLT_CONFIG="$COBOT_RLT_BRANCH_CONFIG"
  EXPERIMENT_RUN="$COBOT_RLT_BRANCH_RUN"
fi
# Separate physical publication from the fixed logical Actor/Replay clock.
execution_values=$(PYTHONPATH="$ROOT" /usr/bin/python3 -m methods.openpi_rlt.cobot_adapter.execution_profiles)
stage1_rtc_args=()
if [[ -n "$execution_values" ]]; then
  mapfile -t execution_fields <<< "$execution_values"
  export COBOT_RLT_EXECUTION_PROFILE="${execution_fields[0]}"
  if [[ "${execution_fields[1]}" == 1 ]]; then
    rtc_overlay="$ROOT/../vla-platform/integrations/cobot/pi05/dagger/common/rtc_overlay"
    [[ -f "$rtc_overlay/rtc_openpi/sampler.py" ]] || { echo "Registered VLA RTC overlay is missing: $rtc_overlay" >&2; exit 2; }
    stage1_rtc_args=(--rtc-overlay "$rtc_overlay" --warmup-rtc)
  fi
fi
DATA_ROOT=$(PYTHONPATH="$ROOT" /usr/bin/python3 - "$DATA_PHASE" <<'PYDATA'
import sys
from integrations.cobot_runtime.profile_storage import default_root
print(default_root(sys.argv[1]))
PYDATA
)
mkdir -p "$RUN/model-server" "$RUN/logs" "$RUN/online/metrics" "$DATA_ROOT"

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
    "$MACHINE_PY" -u -m methods.openpi_rlt.plug_v3_yyshadow.serve_stage1 \
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

source "$TASK5_ROS_SETUP"
export PYTHONPATH="$ROOT:$ROOT/third_party/openpi-rlt/rlt_online_rl/src${PYTHONPATH:+:$PYTHONPATH}"
export COBOT_RLT_TRACE_DIR="$COBOT_RLT_TRACE_ROOT/$MODE"
export COBOT_RLT_SHADOW=0
export COBOT_RLT_SESSION_UI=1
export COBOT_RLT_SESSION_UI_PORT="$SESSION_PORT"
export COBOT_RLT_TASK5_URL
export RLT_ACTOR_READY_TIMEOUT_SEC="${RLT_ACTOR_READY_TIMEOUT_SEC:-120}"
export RLT_REPLAY_READY_TIMEOUT_SEC="${RLT_REPLAY_READY_TIMEOUT_SEC:-120}"
export COBOT_RLT_TASK5_DATA_ROOT="$DATA_ROOT"
export COBOT_RLT_TASK_ID=plug_v3_yyshadow
export COBOT_RLT_MODEL_ID=openpi_rlt
export COBOT_RLT_CHECKPOINT_ID="$(basename "$checkpoint")"
if [[ -n "$EXPERIMENT_RUN" ]]; then
  # Actor versions overlap between branches; retain the selected lineage in recordings.
  export COBOT_RLT_CHECKPOINT_ID="$COBOT_DEPLOYMENT_MODEL_ID"
fi
export COBOT_RLT_DATASET_ROUND="$DATA_PHASE"
export COBOT_RLT_COLLECTION_PHASE="$DATA_PHASE"
export COBOT_RLT_WARMUP_MIN_SIZE=600
export COBOT_RLT_TASK5_MAX_TIMESTEPS=3000
export COBOT_RLT_PROMPT="Insert the held plug into the socket."
# Let each terminal UI request decide whether to home.  The runtime remains
# paused while finalization and the requested home operation complete.
export COBOT_RLT_HOME_AFTER_TERMINAL=1
export COBOT_RLT_HOME_TARGET=all
export COBOT_RLT_HOME_POSE=plug2
if [[ "$MODE" == reference ]]; then
  export COBOT_RLT_MIN_ONLINE_ACTOR_VERSION=2147483647
  # Reference is a frozen Stage-1 rollout.  Once replay grows beyond the
  # warmup threshold, the phase controller would otherwise wait forever for
  # a learner that is intentionally disabled in this mode.
  export COBOT_RLT_DISABLE_PHASE_CONTROLLER=1
  # Reference collects replay but must never start the learner.  It uses the
  # frozen Stage-1 ref_chunk for every action.
  export RLT_DISABLE_LEARNER=1
else
  # Upstream updates the actor every two learner steps: 5k warmup -> version 2500.
  export COBOT_RLT_MIN_ONLINE_ACTOR_VERSION=2500
  unset COBOT_RLT_DISABLE_PHASE_CONTROLLER
  unset RLT_DISABLE_LEARNER
fi
if [[ "$MODE" == reference || "$MODE" == warmup ]]; then
  # A reachable actor service with version -1 is expected before warmup has
  # published its first snapshot.  Runtime still falls back to ref_chunk.
  export RLT_ALLOW_UNINITIALIZED_ACTOR=1
else
  unset RLT_ALLOW_UNINITIALIZED_ACTOR
fi
export RLT_OUTPUT_DIR="${EXPERIMENT_RUN:-$RUN}/online"
export COBOT_RLT_LEARNER_STATUS_PATH="$RLT_OUTPUT_DIR/metrics/learner_status.json"
export COBOT_RLT_REPLAY_URL=http://127.0.0.1:9132
export COBOT_RLT_ACTOR_URL=http://127.0.0.1:9131
export COBOT_RLT_CONTROL_HZ=20
export COBOT_RLT_CHUNK_EXEC_HORIZON=10
export COBOT_RLT_JOINT_STEP_LIMIT=0.03
export COBOT_RLT_GRIPPER_STEP_LIMIT=0.004
export COBOT_RLT_MAX_EPISODE_STEPS=0
export PYTHONDONTWRITEBYTECODE=1
# All RLT coordination services are loopback-only.  Never route actor,
# replay, Session or Task5 requests through the workstation HTTP proxy.
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

ENV_FACTORY=methods.openpi_rlt.plug_v3_yyshadow.right_arm_env:create_right_arm_online_env
if [[ "${COBOT_RLT_EVALUATION:-0}" == 1 ]]; then
  "$ONLINE_PY" -m integrations.cobot_runtime.evaluation_env "$MODE" "$COBOT_EVAL_SNAPSHOT" "$COBOT_EVAL_CONFIG"
  RLT_CONFIG="$COBOT_EVAL_CONFIG"
  export RLT_DISABLE_LEARNER=1 COBOT_RLT_DISABLE_PHASE_CONTROLLER=1
  export COBOT_RLT_HOME_AFTER_TERMINAL=0
  export COBOT_RLT_TRACE_DIR="$COBOT_RUNTIME_ROOT/deployment/unused-traces"
  ENV_FACTORY=integrations.cobot_runtime.evaluation_env:create_evaluation_env
fi

# The web owns a single loaded model. Session preparation chooses collection
# or evaluation; the environment latches that choice before each episode.
if [[ "${COBOT_RLT_SHARED_MODEL:-0}" == 1 ]]; then
  if [[ "$MODE" != online ]]; then
    "$ONLINE_PY" -m integrations.cobot_runtime.shared_model_env "$MODE" "$COBOT_EVAL_SNAPSHOT" "$COBOT_EVAL_CONFIG"
    RLT_CONFIG="$COBOT_EVAL_CONFIG"
    export RLT_DISABLE_LEARNER=1 COBOT_RLT_DISABLE_PHASE_CONTROLLER=1
  fi
  export COBOT_RLT_HOME_AFTER_TERMINAL=0
  export COBOT_RLT_TRACE_DIR="$COBOT_RLT_TRACE_ROOT/$MODE"
  ENV_FACTORY=integrations.cobot_runtime.shared_model_env:create_shared_env
fi

echo "RLT $MODE 已启动；操作页沿用 http://127.0.0.1:8015/。Ctrl-C 停止本次 Session，Stage-1 模型保留。"
# Upstream resolves artifact paths relative to the process working directory,
# not relative to the YAML file.  Keep it anchored beside the config so every
# relative models/outputs entry resolves inside this RLT project regardless of
# whether the caller is a terminal or the web console.
cd "$ROOT/configs/rlt/plug_v3_yyshadow"
exec env CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  "$ONLINE_PY" -u -m methods.openpi_rlt.scripts.online_role \
    --upstream-root "$ROOT/third_party/openpi-rlt" \
    --config "$RLT_CONFIG" \
    --env-factory "$ENV_FACTORY"
