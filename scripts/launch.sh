#!/usr/bin/env bash
# Launch a training run inside tmux so it can be watched and cancelled.
#
#   bash scripts/launch.sh <run-name> [args passed to sgs.train ...]
#
# Every run becomes one window of the tmux session "sgs":
#
#   tmux attach -t sgs          # watch (detach again with Ctrl-b d)
#   tmux list-windows -t sgs    # what is running
#   Ctrl-b n / Ctrl-b p         # next / previous run
#   Ctrl-c inside a window      # cancel that one run
#   tmux kill-session -t sgs    # cancel everything
#
# The window stays open after the run exits (remain-on-exit), so a crash is
# still readable. Output is also teed to runs/<run-name>.out.
set -euo pipefail
cd "$(dirname "$0")/.."

SESSION=${SGS_TMUX_SESSION:-sgs}
NAME=${1:?usage: launch.sh <run-name> [sgs.train args...]}
shift

if tmux has-session -t "$SESSION" 2>/dev/null \
   && tmux list-windows -t "$SESSION" -F '#W' | grep -qx "$NAME"; then
  echo "error: window '$NAME' already exists in session '$SESSION'." >&2
  echo "       pick another --run-name, or: tmux kill-window -t $SESSION:$NAME" >&2
  exit 1
fi

mkdir -p runs
# One GPU only on this box (and the trainer defaults to CPU anyway).
CMD="CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m sgs.train --run-name '$NAME' $* 2>&1 | tee 'runs/$NAME.out'"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux new-window -t "$SESSION" -n "$NAME" -c "$PWD" "bash -lc \"$CMD\""
else
  tmux new-session -d -s "$SESSION" -n "$NAME" -c "$PWD" "bash -lc \"$CMD\""
  tmux set-option -t "$SESSION" history-limit 50000 >/dev/null
fi
tmux set-option -w -t "$SESSION:$NAME" remain-on-exit on >/dev/null

echo "launched '$NAME' in tmux session '$SESSION'"
echo "  watch:  tmux attach -t $SESSION"
echo "  log:    runs/$NAME/log.csv"
