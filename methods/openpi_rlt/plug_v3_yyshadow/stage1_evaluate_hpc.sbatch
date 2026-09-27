#!/bin/bash
#SBATCH --job-name=plug-v3-eval
#SBATCH --partition=acd_u
#SBATCH --qos=formal-user
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/data/user/jhe724/jiaan/research-workspace/projects/cobot-realworld-rl/runs/plug_v3_yyshadow/logs/eval-%j.out
#SBATCH --error=/data/user/jhe724/jiaan/research-workspace/projects/cobot-realworld-rl/runs/plug_v3_yyshadow/logs/eval-%j.err

set -euo pipefail
project=/data/user/jhe724/jiaan/research-workspace/projects/cobot-realworld-rl
python="$project/envs/rlt-stage1-py311-relay-v1/bin/python"
upstream="$project/code/openpi-rlt"
dataset="$project/data/rlt/plug_v3_yyshadow/plug_v3_yyshadow_demonstrations"
entry="$project/methods/openpi_rlt/plug_v3_yyshadow/evaluate_stage1.py"
root="$project/runs/plug_v3_yyshadow/evaluation_snapshots"
out="$project/runs/plug_v3_yyshadow/evaluation"
mkdir -p "$out" "$project/cache/tmp"
export PYTHONPATH="$project:$upstream/src:$upstream/scripts${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.88

for step in "$@"; do
  checkpoint="$root/$step"
  [[ -d "$checkpoint/params" ]] || { echo "missing snapshot $step" >&2; exit 2; }
  "$python" "$entry" \
    --project-root "$project" --upstream-root "$upstream" \
    --dataset-root "$dataset" --checkpoint "$checkpoint" \
    --output "$out/step_${step}.json" --batch-size 8 --batches 8
done
