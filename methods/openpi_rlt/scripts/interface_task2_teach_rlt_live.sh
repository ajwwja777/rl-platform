#!/usr/bin/env bash
set -euo pipefail

STEP="${1:-4999}"
shift || true
SHADOW="0"
AUTO_RESET="0"
AUTO_RESET_DELAY=""
SESSION_UI_PORT="8016"
TASK5_URL="http://127.0.0.1:8015"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --shadow)
      SHADOW="1"
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

PROJECT_ROOT="/media/agilex/Getea1/jiaan/projects/cobot-realworld-rl"
UPSTREAM_ROOT="${PROJECT_ROOT}/code/openpi-rlt"
METHOD_ROOT="${PROJECT_ROOT}/methods/openpi_rlt"
DATASET_ROOT="${PROJECT_ROOT}/datasets/canonical/in-the-pot/legacy40-v2.1"
CHECKPOINT_ROOT="${PROJECT_ROOT}/checkpoints/openpi-rlt/cobot_rlt_pi05_joint/r1-joint-legacy40-v2.1-b32w16-s42-20260904/${STEP}"
CONFIG="${METHOD_ROOT}/configs/cobot_in_the_pot_online.yaml"
ONLINE_PY="${PROJECT_ROOT}/envs/rlt-online-py310/bin/python"
MACHINE_A_PY="/home/agilex/junfeng/workspace/pi05_cobot/.venv-server/bin/python"
MACHINE_A_OVERLAY="${PROJECT_ROOT}/envs/machine-a-py311-overlay"
BASE_PARAMS="/home/agilex/cobot_magic/task3/jiaan/deployments/in_the_pot/pi05/checkpoints/step_2000/params"
ROS_SETUP="/home/agilex/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash"
ALOHA_PYTHON="/home/agilex/miniconda3/envs/aloha/bin/python"
TASK2_HOME_CLI="/home/agilex/cobot_magic/aloha-devel/Piper-AVP-Teleop/multi_arm_launch_tools/task2_homing/task2_home_cli.py"
RUN_ROOT="${PROJECT_ROOT}/runs/openpi-rlt/online-r1-legacy40-v2.1-session-v2"
RUN_RELATIVE="runs/openpi-rlt/online-r1-legacy40-v2.1-session-v2"
RUNTIME_CONFIG="${RUN_ROOT}/resolved_online.yaml"
TRACE_DIR="${RUN_ROOT}/raw_cobot_trace"
TASK5_DATA_ROOT="/home/agilex/cobot_magic/task3/jiaan/realworld_rl/data/task5-rlt-r1"
SUPERVISOR_LOG="${RUN_ROOT}/logs/interface_$(date +%Y%m%d_%H%M%S).log"
OPERATOR_SHUTDOWN_MARKER="${RUN_ROOT}/.operator-shutdown"
MACHINE_A_PORT=8000
ACTOR_PORT=9101
REPLAY_PORT=9102

for path in "${UPSTREAM_ROOT}" "${METHOD_ROOT}" "${DATASET_ROOT}" "${CHECKPOINT_ROOT}/params" "${CHECKPOINT_ROOT}/assets"; do
  [[ -e "${path}" ]] || { echo "Required asset is missing: ${path}" >&2; exit 1; }
done
for executable in "${ONLINE_PY}" "${MACHINE_A_PY}"; do
  [[ -x "${executable}" ]] || { echo "Python is not executable: ${executable}" >&2; exit 1; }
done
[[ -d "${MACHINE_A_OVERLAY}/pytest" ]] || { echo "Machine A dependency overlay is missing: ${MACHINE_A_OVERLAY}" >&2; exit 1; }
[[ -f "${CONFIG}" ]] || { echo "Online config missing: ${CONFIG}" >&2; exit 1; }
[[ -f "${ROS_SETUP}" ]] || { echo "ROS setup missing: ${ROS_SETUP}" >&2; exit 1; }
[[ -x "${ALOHA_PYTHON}" ]] || { echo "Aloha Python is not executable: ${ALOHA_PYTHON}" >&2; exit 1; }
[[ -f "${TASK2_HOME_CLI}" ]] || { echo "Task2 front home CLI is missing: ${TASK2_HOME_CLI}" >&2; exit 1; }

for port in "${ACTOR_PORT}" "${REPLAY_PORT}" "${SESSION_UI_PORT}"; do
  if ss -H -ltn "sport = :${port}" | grep -q .; then
    echo "TCP port ${port} is already in use; refusing to reuse or kill an unknown process." >&2
    exit 1
  fi
done

curl --noproxy "*" -fsS --max-time 3 "${TASK5_URL}/healthz" | grep -q '"status":"ok"' || {
  echo "Task5 v1 is not ready at ${TASK5_URL}; start and check Task5 before RLT session mode." >&2
  exit 1
}

mkdir -p \
  "${RUN_ROOT}/logs" \
  "${RUN_ROOT}/metrics" \
  "${RUN_ROOT}/checkpoints" \
  "${RUN_ROOT}/actor_snapshot" \
  "${RUN_ROOT}/replay" \
  "${RUN_ROOT}/wandb" \
  "${TRACE_DIR}" \
  "${TASK5_DATA_ROOT}" \
  "${PROJECT_ROOT}/cache/tmp"
set +u
source "${ROS_SETUP}"
set -u
export PYTHONPATH="${PROJECT_ROOT}:${UPSTREAM_ROOT}/rlt_online_rl/src:${PYTHONPATH:-}"
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"
export TMPDIR="${PROJECT_ROOT}/cache/tmp"
export XDG_CACHE_HOME="${PROJECT_ROOT}/cache/xdg"
export JAX_COMPILATION_CACHE_DIR="${PROJECT_ROOT}/cache/jax"
export COBOT_RLT_TRACE_DIR="${TRACE_DIR}"
export COBOT_RLT_SHADOW="${SHADOW}"
export COBOT_RLT_HOME_AFTER_TERMINAL="1"
if [[ "${SHADOW}" == "1" ]]; then
  export COBOT_RLT_HOME_AFTER_TERMINAL="0"
fi
export COBOT_RLT_PROMPT="Open the pot lid, put the object into the pot, then close the lid."
export COBOT_RLT_ENABLE_ROBOT_RESET="${AUTO_RESET}"
export COBOT_RLT_AUTO_NEXT_DELAY_SEC="${AUTO_RESET_DELAY}"
export COBOT_RLT_SESSION_UI="1"
export COBOT_RLT_SESSION_UI_HOST="127.0.0.1"
export COBOT_RLT_SESSION_UI_PORT="${SESSION_UI_PORT}"
export COBOT_RLT_TASK5_URL="${TASK5_URL}"
export COBOT_RLT_TASK5_DATA_ROOT="${TASK5_DATA_ROOT}"
export COBOT_RLT_TASK5_MIN_FREE_BYTES="$((30 * 1024 * 1024 * 1024))"
export COBOT_RLT_CHECKPOINT_ID="step_${STEP}"
export COBOT_RLT_TASK_ID="in_the_pot"
export COBOT_RLT_MODEL_ID="openpi_rlt"
export COBOT_RLT_DATASET_ROUND="online_r1_session_v2"
export COBOT_RLT_COLLECTION_PHASE="warmup"
export COBOT_RLT_MAX_EPISODE_STEPS="0"
export COBOT_RLT_TASK5_MAX_TIMESTEPS="3600"
export COBOT_RLT_WARMUP_MIN_SIZE="600"
export COBOT_RLT_MIN_ONLINE_ACTOR_VERSION="250"
export COBOT_RLT_REPLAY_URL="http://127.0.0.1:${REPLAY_PORT}"
export COBOT_RLT_ACTOR_URL="http://127.0.0.1:${ACTOR_PORT}"
export COBOT_RLT_LEARNER_STATUS_PATH="${RUN_ROOT}/metrics/learner_status.json"
export COBOT_RLT_OPERATOR_SHUTDOWN_MARKER="${OPERATOR_SHUTDOWN_MARKER}"
rm -f "${OPERATOR_SHUTDOWN_MARKER}"
"${ONLINE_PY}" "${METHOD_ROOT}/scripts/resolve_online_config.py" \
  --template "${CONFIG}" --project-root "${PROJECT_ROOT}" \
  --run-relative "${RUN_RELATIVE}" --output "${RUNTIME_CONFIG}" >/dev/null

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
  --upstream-root "${UPSTREAM_ROOT}" --config "${RUNTIME_CONFIG}" --system.role replay_manager
REPLAY_PID="${LAST_CHILD_PID}"
wait_http "http://127.0.0.1:${REPLAY_PORT}/stats" "${REPLAY_PID}" "replay manager"

start_child env CUDA_VISIBLE_DEVICES=0 "${ONLINE_PY}" "${METHOD_ROOT}/scripts/online_role.py" \
  --upstream-root "${UPSTREAM_ROOT}" --config "${RUNTIME_CONFIG}" --system.role learner_service
LEARNER_PID="${LAST_CHILD_PID}"
for _ in $(seq 1 180); do
  kill -0 "${LEARNER_PID}" 2>/dev/null || { echo "Learner exited; inspect ${SUPERVISOR_LOG}" >&2; exit 1; }
  [[ -f "${COBOT_RLT_LEARNER_STATUS_PATH}" ]] && break
  sleep 1
done
[[ -f "${COBOT_RLT_LEARNER_STATUS_PATH}" ]] || { echo "Learner status was not created." >&2; exit 1; }

start_child env CUDA_VISIBLE_DEVICES=0 "${ONLINE_PY}" "${METHOD_ROOT}/scripts/online_role.py" \
  --upstream-root "${UPSTREAM_ROOT}" --config "${RUNTIME_CONFIG}" --system.role actor_service
ACTOR_PID="${LAST_CHILD_PID}"
wait_http "http://127.0.0.1:${ACTOR_PORT}/version" "${ACTOR_PID}" "actor service"

start_child env JAX_PLATFORMS=cpu "${ONLINE_PY}" "${METHOD_ROOT}/scripts/online_role.py" \
  --upstream-root "${UPSTREAM_ROOT}" --config "${RUNTIME_CONFIG}" --system.role env_driver \
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
echo "Machine A: R1 joint checkpoint step_${STEP}; Machine B: actor/critic/replay/learner。"
echo "任一后臂示教按钮会暂停策略并接管；释放后丢弃旧 chunk 并 fresh replan。"
echo "Task5 已连接；RLT Session 会自动开始/停止每条录制。"
echo "8015 仅是 Task5 recorder 后端；RLT 操作只使用 8016 页面。"
if [[ "${SHADOW}" == "1" ]]; then
  echo "当前为 SHADOW：不会创建 /task2/policy/joint_* publisher。"
else
  echo "当前为 LIVE：确认急停、CAN、相机、夹爪和工作区均已就绪。"
fi
echo "不设 600-step 终局；由操作员在网页选择成功/失败/放弃。"
if [[ "${SHADOW}" == "1" ]]; then
  echo "SHADOW 不执行自动 home；终局后仍保持暂停。"
else
  echo "成功/失败/放弃后先固化数据，再自动执行 task2_home_cli.py front。"
fi
echo "front home 完成后人工复位物体，再在网页点击开始下一轮。"
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
