#!/usr/bin/env bash
set -euo pipefail
umask 077

# Local lifecycle probes must never be routed through a configured HTTP proxy.
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="${NO_PROXY}"

ACTION="${1:-status}"
STEP="${2:-4999}"
PROJECT_ROOT="/media/agilex/Getea1/jiaan/projects/cobot-realworld-rl"
UPSTREAM_ROOT="${PROJECT_ROOT}/code/openpi-rlt"
METHOD_ROOT="${PROJECT_ROOT}/methods/openpi_rlt"
DATASET_ROOT="${PROJECT_ROOT}/datasets/canonical/in-the-pot/legacy40-v2.1"
CHECKPOINT_ROOT="${PROJECT_ROOT}/checkpoints/openpi-rlt/cobot_rlt_pi05_joint/r1-joint-legacy40-v2.1-b32w16-s42-20260904/${STEP}"
BASE_PARAMS="/home/agilex/cobot_magic/task3/jiaan/deployments/in_the_pot/pi05/checkpoints/step_2000/params"
MACHINE_A_PY="/home/agilex/junfeng/workspace/pi05_cobot/.venv-server/bin/python"
MACHINE_A_OVERLAY="${PROJECT_ROOT}/envs/machine-a-py311-overlay"
STATE_ROOT="${PROJECT_ROOT}/runs/openpi-rlt/machine-a-server"
PID_FILE="${STATE_ROOT}/machine_a.pid"
SPEC_FILE="${STATE_ROOT}/machine_a.spec"
LOG_DIR="${STATE_ROOT}/logs"
PORT=8000

mkdir -p "${STATE_ROOT}" "${LOG_DIR}" "${PROJECT_ROOT}/cache/tmp" "${PROJECT_ROOT}/cache/xdg" "${PROJECT_ROOT}/cache/jax"

read_pid() {
  [[ -f "${PID_FILE}" ]] || return 1
  local pid
  pid="$(tr -d '[:space:]' <"${PID_FILE}")"
  [[ "${pid}" =~ ^[0-9]+$ ]] || return 1
  printf '%s\n' "${pid}"
}

owned_pid() {
  local pid="$1" command
  [[ -r "/proc/${pid}/cmdline" ]] || return 1
  command="$(tr '\0' ' ' <"/proc/${pid}/cmdline")"
  [[ "${command}" == *"serve_machine_a.py"* && "${command}" == *"${CHECKPOINT_ROOT}"* && "${command}" == *"--port ${PORT}"* ]]
}

metadata_ok() {
  timeout 4 "${MACHINE_A_PY}" -c "from openpi_client.websocket_client_policy import WebsocketClientPolicy; m=WebsocketClientPolicy('127.0.0.1',${PORT}).get_server_metadata(); assert m['has_rl_token'] and m['action_dim']==14 and m['proprio_dim']==14 and m['chunk_len']==50" >/dev/null 2>&1
}

status_server() {
  local pid
  pid="$(read_pid)" || return 1
  kill -0 "${pid}" 2>/dev/null || return 1
  owned_pid "${pid}" || return 1
  [[ -f "${SPEC_FILE}" ]] || return 1
  [[ "$(<"${SPEC_FILE}")" == "step=${STEP} checkpoint=${CHECKPOINT_ROOT} port=${PORT}" ]] || return 1
  metadata_ok || return 1
  echo "RLT Machine A ready: PID=${pid} step_${STEP} port=${PORT}"
}

start_server() {
  for path in "${UPSTREAM_ROOT}" "${METHOD_ROOT}" "${DATASET_ROOT}" "${CHECKPOINT_ROOT}/params" "${CHECKPOINT_ROOT}/assets"; do
    [[ -e "${path}" ]] || { echo "Required Machine A asset is missing: ${path}" >&2; exit 1; }
  done
  [[ -x "${MACHINE_A_PY}" ]] || { echo "Machine A Python is not executable: ${MACHINE_A_PY}" >&2; exit 1; }
  [[ -d "${MACHINE_A_OVERLAY}/pytest" ]] || { echo "Machine A overlay is missing: ${MACHINE_A_OVERLAY}" >&2; exit 1; }
  if status_server; then
    echo "Reusing the exact preloaded RLT model; no checkpoint restore is needed."
    return
  fi
  if pid="$(read_pid 2>/dev/null)" && kill -0 "${pid}" 2>/dev/null; then
    echo "Registered Machine A PID ${pid} is live but does not match step_${STEP}; stop it explicitly first." >&2
    exit 1
  fi
  rm -f "${PID_FILE}" "${SPEC_FILE}"
  if ss -H -ltn "sport = :${PORT}" | grep -q .; then
    echo "TCP port ${PORT} is occupied by an unregistered process; refusing to reuse or stop it." >&2
    exit 1
  fi
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
  [[ "${free_mib}" =~ ^[0-9]+$ ]] || { echo "Cannot read free GPU memory." >&2; exit 1; }
  if (( free_mib < 20000 )); then
    echo "Cold RLT model preload requires at least 20000 MiB free GPU; found ${free_mib}." >&2
    exit 1
  fi

  log_file="${LOG_DIR}/machine_a_step_${STEP}_$(date +%Y%m%d_%H%M%S).log"
  nohup env \
    PYTHONPATH="${MACHINE_A_OVERLAY}:${PROJECT_ROOT}:${UPSTREAM_ROOT}/rlt_online_rl/src:${PYTHONPATH:-}" \
    NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}" no_proxy="127.0.0.1,localhost,${no_proxy:-}" \
    TMPDIR="${PROJECT_ROOT}/cache/tmp" XDG_CACHE_HOME="${PROJECT_ROOT}/cache/xdg" \
    JAX_COMPILATION_CACHE_DIR="${PROJECT_ROOT}/cache/jax" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.72 \
    "${MACHINE_A_PY}" "${METHOD_ROOT}/scripts/serve_machine_a.py" \
      --project-root "${PROJECT_ROOT}" --upstream-root "${UPSTREAM_ROOT}" \
      --dataset-root "${DATASET_ROOT}" --base-params "${BASE_PARAMS}" \
      --checkpoint-dir "${CHECKPOINT_ROOT}" --port "${PORT}" \
      </dev/null >>"${log_file}" 2>&1 &
  pid=$!
  printf '%s\n' "${pid}" >"${PID_FILE}.tmp" && mv "${PID_FILE}.tmp" "${PID_FILE}"
  printf 'step=%s checkpoint=%s port=%s\n' "${STEP}" "${CHECKPOINT_ROOT}" "${PORT}" >"${SPEC_FILE}.tmp" && mv "${SPEC_FILE}.tmp" "${SPEC_FILE}"
  echo "Cold-loading RLT Machine A PID=${pid}; first restore can take about 9 minutes."
  echo "Machine A log: ${log_file}"
  for _ in $(seq 1 900); do
    kill -0 "${pid}" 2>/dev/null || {
      rm -f "${PID_FILE}" "${SPEC_FILE}"
      echo "Machine A exited during preload; inspect ${log_file}" >&2
      exit 1
    }
    if metadata_ok; then
      echo "RLT Machine A preload complete: PID=${pid} step_${STEP}"
      return
    fi
    sleep 1
  done
  echo "Timed out waiting for RLT Machine A; it remains registered as PID=${pid}." >&2
  exit 1
}

stop_server() {
  local pid
  pid="$(read_pid)" || { echo "No registered RLT Machine A server."; return; }
  if ! kill -0 "${pid}" 2>/dev/null; then
    rm -f "${PID_FILE}" "${SPEC_FILE}"
    echo "Removed stale RLT Machine A registration."
    return
  fi
  owned_pid "${pid}" || { echo "PID ${pid} is not the registered step_${STEP} Machine A; refusing to kill it." >&2; exit 1; }
  kill -TERM "${pid}"
  for _ in $(seq 1 100); do
    kill -0 "${pid}" 2>/dev/null || break
    sleep 0.1
  done
  if kill -0 "${pid}" 2>/dev/null; then kill -KILL "${pid}"; fi
  rm -f "${PID_FILE}" "${SPEC_FILE}"
  echo "Stopped registered RLT Machine A PID=${pid}; GPU memory released."
}

case "${ACTION}" in
  start) start_server ;;
  status) status_server ;;
  stop) stop_server ;;
  *) echo "Usage: $0 {start|status|stop} [step]" >&2; exit 2 ;;
esac
