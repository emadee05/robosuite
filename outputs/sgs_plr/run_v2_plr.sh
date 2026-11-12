#!/bin/bash
# Phase 1 step 4: PLR on the v2 scripted ladder, same settings as v2_uniform/v2_sgs (3 seeds, 300k).
cd "$(dirname "$0")/.."
for seed in 0 1 2; do
  PYTHONPATH=. ../robosuite/.venv/bin/python -m sgs.train_stack_ladder --pool stack_task_ladders/v2 \
    --sampler plr --seed $seed --steps 300000 --eval-every 25000 --name v2_plr_s$seed 2>&1 \
    | grep --line-buffered -v "WARNING\|INFO" &
done
wait
echo "V2 PLR DONE"
