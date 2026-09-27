#!/usr/bin/env bash
set -euo pipefail

STEP="${1:-4000}"
shift || true
SHADOW="0"
AUTO_RESET="0"
AUTO_RESET_DELAY=""
RLT_MODE="eval"
SESSION_UI_PORT="8016"
TASK5_URL="http://127.0.0.1:8017"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --shadow)
      SHADOW="1"
      shift
      ;;
    --explore)
      RLT_MODE="explore"
      shift
      ;;
    --auto-reset)
      echo "--auto-reset is not available until a named reset profile passes onsite validation." >&2
      exit 2
      ;;
    --session-ui-port)
      [[ $# -ge 2 ]] || { echo "--session-ui-port requires a port" >&2; exit 2; }
      SESSION_UI_PORT="$2"
      shift 2
      ;;
    --task5-url)
      [[ $# -ge 2 ]] || { echo "--task5-url requires a URL" >&2; exit 2; }
      TASK5_URL="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

RLT_PROJECT_ROOT="/media/agilex/Getea1/jiaan/projects/cobot-realworld-rl"
TASK_PROJECT_ROOT="/media/agilex/Getea1/jiaan/projects/cobot-realworld-vla"
DEPLOYMENT_ROOT="${TASK_PROJECT_ROOT}/deployments/openpi-rlt/plug-insertion-stage1-v2"
RUNTIME_ROOT="${DEPLOYMENT_ROOT}/runtime-overlay"
UPSTREAM_ROOT="${RLT_PROJECT_ROOT}/code/openpi-rlt"
METHOD_ROOT="${RUNTIME_ROOT}/methods/openpi_rlt"
DATASET_ROOT="${TASK_PROJECT_ROOT}/datasets/derived/plug-insertion/plug-insertion-stage1-v2"
CHECKPOINT_ROOT="${TASK_PROJECT_ROOT}/checkpoints/openpi-rlt/cobot_rlt_pi05_joint/plug-rlt-s1-official-e78-h50-s42-20260909/step_${STEP}"
CONFIG="${METHOD_ROOT}/configs/cobot_plug_insertion_online.yaml"
if [[ "${RLT_MODE}" == "eval" ]]; then
  R2_CONFIG="${METHOD_ROOT}/configs/plug_insertion_r2.yaml"
else
  R2_CONFIG="${METHOD_ROOT}/configs/plug_insertion_r2_explore.yaml"
fi
ACTION_STATS="${TASK_PROJECT_ROOT}/assets/openpi-rlt/online/plug-insertion-stage1-v2/action-delta-chunk10.json"
ONLINE_PY="${RLT_PROJECT_ROOT}/envs/rlt-online-py310/bin/python"
MACHINE_A_PY="/home/agilex/junfeng/workspace/pi05_cobot/.venv-server/bin/python"
MACHINE_A_OVERLAY="${RLT_PROJECT_ROOT}/envs/machine-a-py311-overlay"
BASE_PARAMS="/home/agilex/cobot_magic/task3/jiaan/deployments/in_the_pot/pi05/checkpoints/step_2000/params"
ROS_SETUP="/home/agilex/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash"
ALOHA_PYTHON="/home/agilex/miniconda3/envs/aloha/bin/python"
TASK2_HOME_CLI="/home/agilex/cobot_magic/aloha-devel/Piper-AVP-Teleop/multi_arm_launch_tools/task2_homing/task2_home_cli.py"
RUN_ROOT="${TASK_PROJECT_ROOT}/runs/openpi-rlt/plug-insertion-stage1-v2-session-v1"
RUN_RELATIVE="runs/openpi-rlt/plug-insertion-stage1-v2-session-v1"
RUNTIME_CONFIG="${RUN_ROOT}/resolved_online.yaml"
TRACE_DIR="${RUN_ROOT}/raw_cobot_trace"
TASK5_DATA_ROOT="/media/agilex/Getea1/jiaan/data/cobot-realworld-vla/task5/plug-insertion-rlt-v1/raw"
TASK5_START_SCRIPT="${METHOD_ROOT}/scripts/start_rollout_recorder.sh"
SUPERVISOR_LOG="${RUN_ROOT}/logs/interface_$(date +%Y%m%d_%H%M%S).log"
OPERATOR_SHUTDOWN_MARKER="${RUN_ROOT}/.operator-shutdown"
MACHINE_A_PORT=8000
ACTOR_PORT=9101
REPLAY_PORT=9102

for path in "${UPSTREAM_ROOT}" "${METHOD_ROOT}" "${DATASET_ROOT}" "${CHECKPOINT_ROOT}/params" "${CHECKPOINT_ROOT}/assets" "${CONFIG}" "${R2_CONFIG}" "${ACTION_STATS}"; do
  [[ -e "${path}" ]] || { echo "Required asset is missing: ${path}" >&2; exit 1; }
done
for executable in "${ONLINE_PY}" "${MACHINE_A_PY}"; do
  [[ -x "${executable}" ]] || { echo "Python is not executable: ${executable}" >&2; exit 1; }
done
[[ -d "${MACHINE_A_OVERLAY}/pytest" ]] || { echo "Machine A dependency overlay is missing: ${MACHINE_A_OVERLAY}" >&2; exit 1; }
[[ -f "${ROS_SETUP}" ]] || { echo "ROS setup missing: ${ROS_SETUP}" >&2; exit 1; }
[[ -x "${ALOHA_PYTHON}" ]] || { echo "Aloha Python is not executable: ${ALOHA_PYTHON}" >&2; exit 1; }
[[ -f "${TASK2_HOME_CLI}" ]] || { echo "Task2 front home CLI is missing: ${TASK2_HOME_CLI}" >&2; exit 1; }
[[ -f "${TASK5_START_SCRIPT}" ]] || { echo "Task5 start script is missing: ${TASK5_START_SCRIPT}" >&2; exit 1; }

for port in "${ACTOR_PORT}" "${REPLAY_PORT}" "${SESSION_UI_PORT}"; do
  if ss -H -ltn "sport = :${port}" | grep -q .; then
    echo "TCP port ${port} is already in use; refusing to reuse or kill an unknown process." >&2
    exit 1
  fi
done

# Validate the continuous recorder contract before any Machine B/robot service.
bash "${TASK5_START_SCRIPT}" --url "${TASK5_URL}"

mkdir -p \
  "${RUN_ROOT}/logs" \
  "${RUN_ROOT}/metrics" \
  "${RUN_ROOT}/checkpoints" \
  "${RUN_ROOT}/actor_snapshot" \
  "${RUN_ROOT}/replay" \
  "${RUN_ROOT}/wandb" \
  "${TRACE_DIR}" \
  "${TASK5_DATA_ROOT}" \
  "${TASK_PROJECT_ROOT}/cache/tmp"
set +u
source "${ROS_SETUP}"
set -u
export PYTHONPATH="${RUNTIME_ROOT}:${UPSTREAM_ROOT}/rlt_online_rl/src:${PYTHONPATH:-}"
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"
export TMPDIR="${TASK_PROJECT_ROOT}/cache/tmp"
export XDG_CACHE_HOME="${TASK_PROJECT_ROOT}/cache/xdg"
export JAX_COMPILATION_CACHE_DIR="${TASK_PROJECT_ROOT}/cache/jax"
export COBOT_RLT_TRACE_DIR="${TRACE_DIR}"
export COBOT_RLT_SHADOW="${SHADOW}"
# LIVE terminal handling uses the exact pose command already verified onsite
# by the operator.  SHADOW must never invoke a motion command.
export COBOT_RLT_HOME_AFTER_TERMINAL="1"
export COBOT_RLT_HOME_MODE="all"
export COBOT_RLT_HOME_POSE="plug"
if [[ "${SHADOW}" == "1" ]]; then
  export COBOT_RLT_HOME_AFTER_TERMINAL="0"
fi
export COBOT_RLT_PROMPT="Insert the held plug into the socket."
# Automatic next-episode progression remains off: after home, the operator
# resets the object and starts the next Session explicitly from the UI.
export COBOT_RLT_ENABLE_ROBOT_RESET="0"
export COBOT_RLT_AUTO_NEXT_DELAY_SEC="${AUTO_RESET_DELAY}"
export COBOT_RLT_SESSION_UI="1"
export COBOT_RLT_SESSION_UI_HOST="127.0.0.1"
export COBOT_RLT_SESSION_UI_PORT="${SESSION_UI_PORT}"
export COBOT_RLT_TASK5_URL="${TASK5_URL}"
export COBOT_RLT_TASK5_DATA_ROOT="${TASK5_DATA_ROOT}"
export COBOT_RLT_TASK5_MIN_FREE_BYTES="$((30 * 1024 * 1024 * 1024))"
export COBOT_RLT_CHECKPOINT_ID="step_${STEP}"
export COBOT_RLT_TASK_ID="plug_insertion"
export COBOT_RLT_MODEL_ID="openpi_rlt_plug_stage1_v2"
export COBOT_RLT_DATASET_ROUND="plug_online_r1"
export COBOT_RLT_COLLECTION_PHASE="warmup"
# Operator review is required before enabling any learner updates.
export COBOT_RLT_COLLECTION_ONLY="1"
export COBOT_RLT_MAX_EPISODE_STEPS="0"
export COBOT_RLT_TASK5_MAX_TIMESTEPS="3600"
export COBOT_RLT_WARMUP_MIN_SIZE="128"
export COBOT_RLT_MIN_ONLINE_ACTOR_VERSION="100"
export COBOT_RLT_REPLAY_URL="http://127.0.0.1:${REPLAY_PORT}"
export COBOT_RLT_ACTOR_URL="http://127.0.0.1:${ACTOR_PORT}"
export COBOT_RLT_LEARNER_STATUS_PATH="${RUN_ROOT}/metrics/learner_status.json"
export COBOT_RLT_OPERATOR_SHUTDOWN_MARKER="${OPERATOR_SHUTDOWN_MARKER}"
rm -f "${OPERATOR_SHUTDOWN_MARKER}"
"${ONLINE_PY}" "${METHOD_ROOT}/scripts/resolve_online_config.py" \
  --template "${CONFIG}" --project-root "${TASK_PROJECT_ROOT}" \
  --run-relative "${RUN_RELATIVE}" --r2-config "${R2_CONFIG}" \
  --action-stats-path "${ACTION_STATS}" --output "${RUNTIME_CONFIG}" >/dev/null

MACHINE_A_PID=""
REPLAY_PID=""
LEARNER_PID=""
ACTOR_PID=""
ENV_PID=""
TAIL_PID=""
CLEANED="0"

stop_one() {
  local pid="$1"
  [[ -n "${pid}" ]] || return
  kill -0 "${pid}" 2>/dev/null || return
  kill -TERM "${pid}" 2>/dev/null
  for _ in $(seq 1 50); do
    kill -0 "${pid}" 2>/dev/null || { wait "${pid}" 2>/dev/null; return; }
    sleep 0.1
  done
  kill -KILL "${pid}" 2>/dev/null
  wait "${pid}" 2>/dev/null
}

cleanup() {
  [[ "${CLEANED}" == "0" ]] || return
  CLEANED="1"
  set +e
  if curl --noproxy "*" -fsS --max-time 2 "http://127.0.0.1:${SESSION_UI_PORT}/api/session" >/dev/null 2>&1; then
    timeout 20 "${ONLINE_PY}" "${METHOD_ROOT}/scripts/shutdown_session.py" \
      --url "http://127.0.0.1:${SESSION_UI_PORT}" --timeout-sec 15 \
      >>"${SUPERVISOR_LOG}" 2>&1 || \
      echo "Warning: RLT session graceful shutdown did not complete; inspect ${SUPERVISOR_LOG}." >&2
  fi
  rosservice call /task2/policy/set_paused true >/dev/null 2>&1
  touch "${OPERATOR_SHUTDOWN_MARKER}"
  # Learner flushes its final checkpoint/status through replay, so stop in
  # dependency order instead of terminating all services simultaneously.
  for pid in "${TAIL_PID}" "${ENV_PID}" "${ACTOR_PID}" "${LEARNER_PID}" "${REPLAY_PID}"; do
    stop_one "${pid}"
  done
}
trap cleanup EXIT INT TERM

wait_http() {
  local url="$1" pid="$2" label="$3"
  for _ in $(seq 1 300); do
    kill -0 "${pid}" 2>/dev/null || { echo "${label} exited before readiness." >&2; return 1; }
    curl --noproxy "*" -fsS --max-time 1 "${url}" >/dev/null 2>&1 && return 0
    sleep 1
  done
  echo "Timed out waiting for ${label}." >&2
  return 1
}

"${METHOD_ROOT}/scripts/rlt_model_server.sh" start "${STEP}"
free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
[[ "${free_mib}" =~ ^[0-9]+$ ]] || { echo "Cannot read free GPU memory after Machine A preload." >&2; exit 1; }
if (( free_mib < 2000 )); then
  echo "Less than 2000 MiB remains after Machine A preload; refusing to start Machine B." >&2
  exit 1
fi

start_child() {
  setsid "$@" </dev/null >>"${SUPERVISOR_LOG}" 2>&1 &
  LAST_CHILD_PID=$!
}

start_console_child() {
  # Keep the filtered operator view out of the file it is following. Writing
  # it back into SUPERVISOR_LOG creates a recursive feedback loop.
  setsid "$@" </dev/null &
  LAST_CHILD_PID=$!
}

start_child env JAX_PLATFORMS=cpu "${ONLINE_PY}" "${METHOD_ROOT}/scripts/online_role.py" \
  --upstream-root "${UPSTREAM_ROOT}" --r2-config "${R2_CONFIG}" \
  --config "${RUNTIME_CONFIG}" --system.role replay_manager
REPLAY_PID="${LAST_CHILD_PID}"
wait_http "http://127.0.0.1:${REPLAY_PORT}/stats" "${REPLAY_PID}" "replay manager"

start_child env CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  "${ONLINE_PY}" "${METHOD_ROOT}/scripts/online_role.py" \
  --upstream-root "${UPSTREAM_ROOT}" --r2-config "${R2_CONFIG}" \
  --config "${RUNTIME_CONFIG}" --system.role learner_service
LEARNER_PID="${LAST_CHILD_PID}"
for _ in $(seq 1 180); do
  kill -0 "${LEARNER_PID}" 2>/dev/null || { echo "Learner exited; inspect ${SUPERVISOR_LOG}" >&2; exit 1; }
  [[ -f "${COBOT_RLT_LEARNER_STATUS_PATH}" ]] && break
  sleep 1
done
[[ -f "${COBOT_RLT_LEARNER_STATUS_PATH}" ]] || { echo "Learner status was not created." >&2; exit 1; }

start_child env CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  "${ONLINE_PY}" "${METHOD_ROOT}/scripts/online_role.py" \
  --upstream-root "${UPSTREAM_ROOT}" --r2-config "${R2_CONFIG}" \
  --config "${RUNTIME_CONFIG}" --system.role actor_service
ACTOR_PID="${LAST_CHILD_PID}"
wait_http "http://127.0.0.1:${ACTOR_PORT}/version" "${ACTOR_PID}" "actor service"

start_child env JAX_PLATFORMS=cpu "${ONLINE_PY}" "${METHOD_ROOT}/scripts/online_role.py" \
  --upstream-root "${UPSTREAM_ROOT}" --r2-config "${R2_CONFIG}" \
  --config "${RUNTIME_CONFIG}" --system.role env_driver \
  --env-factory methods.openpi_rlt.cobot_adapter.cobot_ros1:create_cobot_online_env
ENV_PID="${LAST_CHILD_PID}"

for service in /task2/policy/arm /task2/policy/set_paused /task2/policy/chunk_ready; do
  for _ in $(seq 1 120); do
    kill -0 "${ENV_PID}" 2>/dev/null || { echo "Env driver exited; inspect ${SUPERVISOR_LOG}" >&2; exit 1; }
    rosservice info "${service}" >/dev/null 2>&1 && break
    sleep 1
  done
  rosservice info "${service}" >/dev/null 2>&1 || { echo "ROS service not ready: ${service}" >&2; exit 1; }
done
wait_http "http://127.0.0.1:${SESSION_UI_PORT}/api/session" "${ENV_PID}" "RLT session UI"

echo
echo "================ Cobot RLT 已就绪，当前暂停 ================"
echo "Machine A: plug Stage-1 checkpoint step_${STEP}; Machine B: actor/critic/replay/learner。"
echo "任一后臂示教按钮会暂停策略并接管；释放后保持暂停，点击继续才 fresh replan。"
echo "Task5 已连接；RLT Session 会自动开始/停止每条录制。"
echo "Collection only: automatic warmup and online updates are DISABLED."
echo "Task5 continuous recorder: 8017; segmented teach: 8015; RLT UI: 8016."
if [[ "${SHADOW}" == "1" ]]; then
  echo "当前为 SHADOW：不会创建 /task2/policy/joint_* publisher。"
else
  echo "当前为 LIVE：确认急停、CAN、相机、夹爪和工作区均已就绪。"
fi
echo "不设 600-step 终局；由操作员在网页选择成功/失败/放弃。"
if [[ "${SHADOW}" == "1" ]]; then
  echo "SHADOW 不执行自动 home；终局后仍保持暂停。"
else
  echo "成功/失败/放弃后先固化数据，再执行已验证的 all --pose plug 归位并保持暂停。"
fi
echo "归位后由现场人员复位物体，再在网页开始下一轮；不会自动连续启动。"
echo "当前 RLT 模式：${RLT_MODE}（eval 无探索；--explore 才启用受控 chunk 级探索）。"
echo "Machine A 常驻供下次秒级复用；全部结束后运行 ./stop_rlt_model_server.sh ${STEP} 释放显存。"
echo "RLT 操作页：http://127.0.0.1:${SESSION_UI_PORT}/"
echo "网页操作：开始 Session → 成功/失败/放弃 → 场景复位 → 开始下一轮。"
echo "日志：${SUPERVISOR_LOG}"
echo "============================================================="
read -r -p "确认无误后按 Enter 仅完成操作员 arm（不会开始 rollout）: " _

until rosservice call /task2/policy/arm >/dev/null 2>&1; do
  echo "Task2 正在示教或不在 policy 模式；释放示教按钮后自动继续等待……"
  sleep 0.5
done
start_console_child "${ONLINE_PY}" "${METHOD_ROOT}/scripts/operator_log_tail.py" "${SUPERVISOR_LOG}"
TAIL_PID="${LAST_CHILD_PID}"
echo "Cobot RLT 已 arm 且保持暂停。现在打开 RLT 页面并点击‘开始 Session’。"
echo "Ctrl-C 将暂停策略并只停止本脚本创建的进程。"

monitor_role() {
  local label="$1" pid="$2" state status
  kill -0 "${pid}" 2>/dev/null || state="dead"
  state="${state:-$(ps -o stat= -p "${pid}" 2>/dev/null | tr -d ' ')}"
  if [[ -z "${state}" || "${state}" == Z* || "${state}" == "dead" ]]; then
    set +e
    wait "${pid}" 2>/dev/null
    status=$?
    set -e
    echo "Machine B role exited unexpectedly: ${label} status=${status}; policy is being paused." >&2
    return 1
  fi
}

while monitor_role "env" "${ENV_PID}"; do
  if [[ -f "${OPERATOR_SHUTDOWN_MARKER}" ]]; then
    echo "Session ended; stopping interface roles and keeping Machine A preloaded."
    exit 0
  fi
  monitor_role "learner" "${LEARNER_PID}" || exit 1
  monitor_role "actor" "${ACTOR_PID}" || exit 1
  monitor_role "replay" "${REPLAY_PID}" || exit 1
  sleep 1
done

set +e
wait "${ENV_PID}" 2>/dev/null
ENV_STATUS=$?
set -e
exit "${ENV_STATUS}"
