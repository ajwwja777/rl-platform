#!/usr/bin/env bash
RLT_ENV_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
eval "$(PYTHONPATH="$RLT_ENV_ROOT" python3 -m integrations.cobot_runtime.shell_environment)"
export PYTHONPATH="$RLT_ENV_ROOT:$RLT_ENV_ROOT/../cobot-control/src${PYTHONPATH:+:$PYTHONPATH}"
