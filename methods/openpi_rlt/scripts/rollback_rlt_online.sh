#!/usr/bin/env bash
set -euo pipefail
ROOT=/media/agilex/Getea1/jiaan/projects/cobot-realworld-rl
DEPLOY=/media/agilex/Getea1/jiaan/projects/cobot-realworld-vla/deployments/openpi-rlt/plug-insertion-stage1-v2
for port in 8016 9101; do
  if ss -H -ltn "sport = :${port}" | grep -q .; then
    echo "End the RLT Session and wait for interface exit before rollback." >&2
    exit 1
  fi
done
exec "${ROOT}/envs/rlt-online-py310/bin/python" "${DEPLOY}/runtime-overlay/methods/openpi_rlt/scripts/online_cycle_worker.py" \
  --run "${ROOT}/runs/plug-online-warmup-r1" --warmup "${ROOT}/runs/plug-warmup-20260911-r1" \
  --upstream "${ROOT}/code/openpi-rlt" --rollback
