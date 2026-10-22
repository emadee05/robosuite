#!/usr/bin/env bash
# Uniform vs SGS on the same configuration pool, several seeds, all in tmux.
#
#   bash scripts/run_comparison.sh [TOTAL_STEPS] [CUBE_RANGE] [N_CONFIGS] [N_SEEDS]
#
# Each run gets its own window in the tmux session "sgs":
#   tmux attach -t sgs           # watch; Ctrl-b n cycles windows, Ctrl-b d detaches
#   tmux kill-session -t sgs     # cancel all of them
#
# Each run uses --num-envs worker processes. With 3 seeds x 2 samplers x 64 envs
# that is 384 processes on a 256-core box (~1.5x oversubscribed); lower the seed
# count or --num-envs if that is too much.
set -euo pipefail
cd "$(dirname "$0")/.."

STEPS=${1:-5000000}
RANGE=${2:-0.20}
NCFG=${3:-64}
SEEDS=${4:-3}

for ((seed = 0; seed < SEEDS; seed++)); do
  for sampler in uniform sgs; do
    bash scripts/launch.sh "cmp_${sampler}_r${RANGE}_s${seed}" \
      --sampler "$sampler" --seed "$seed" \
      --cube-range "$RANGE" --n-configs "$NCFG" --config-seed 12345 \
      --num-envs 64 --num-steps 64 --total-steps "$STEPS" --eval-every 25
  done
done

echo
echo "all launched. when they finish:"
echo "  .venv/bin/python scripts/plot.py runs/cmp_*"
