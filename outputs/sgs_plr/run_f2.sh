#!/bin/bash
# Phase 1 step 1.5: uniform / PLR / SGS per start / SGS per cell on pool f2, 3 seeds x 1M.
# One batch per seed (all 4 samplers concurrently, so contention is equal across samplers).
cd "$(dirname "$0")/.."
POOL=stack_omnireset_pools/f2
run() {  # name, extra args...
  local name=$1; shift
  PYTHONPATH=. ../robosuite/.venv/bin/python -m sgs.train_stack_ladder --pool $POOL "$@" \
    --steps 1000000 --eval-every 100000 --name $name 2>&1 | grep --line-buffered -v "WARNING\|INFO" &
}
for seed in 0 1 2; do
  run f2_uniform_s$seed  --sampler uniform --seed $seed
  run f2_plr_s$seed      --sampler plr     --seed $seed
  run f2_sgs_s$seed      --sampler sgs     --seed $seed
  run f2_sgscells_s$seed --sampler sgs --sgs-cells --seed $seed
  wait
  echo "SEED $seed DONE"
done
echo "F2 DONE"
