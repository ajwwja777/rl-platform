#!/usr/bin/env bash
set -euo pipefail

STEP="${1:-4999}"
PROJECT_ROOT="/media/agilex/Getea1/jiaan/projects/cobot-realworld-rl"
exec "${PROJECT_ROOT}/methods/openpi_rlt/scripts/rlt_model_server.sh" stop "${STEP}"
