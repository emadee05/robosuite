#!/bin/bash
# Fresh-start-per-reset comparison on pool f3 (10,368 training starts, each used about once in 1M).
# uniform: least-used start; PLR: standard (half the draws go to unseen starts);
# SGS: success tracked per cell (coarse 21 / fine ~60), least-used start inside the chosen cell.
# One batch per seed with all 4 variants running together.
cd "$(dirname "$0")/.."
POOL=stack_omnireset_pools/f3
run() {  # name, extra args...
  local name=$1; shift
  PYTHONPATH=. ../robosuite/.venv/bin/python -m sgs.train_stack_ladder --pool $POOL "$@" \
    --steps 1000000 --eval-every 100000 --name $name 2>&1 | grep --line-buffered -v "WARNING\|INFO" &
}
for seed in ${SEEDS:-0 1 2}; do
  run f3_uniform_s$seed      --sampler uniform --fresh --seed $seed
  run f3_plr_s$seed          --sampler plr --seed $seed
  run f3_sgscells_s$seed     --sampler sgs --sgs-cells --fresh --seed $seed
  run f3_sgsfine_s$seed      --sampler sgs --sgs-cells --fresh --cell-field cell_fine --seed $seed
  wait
  echo "SEED $seed DONE"
done
echo "F3 DONE"
