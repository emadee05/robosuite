#!/usr/bin/env bash
# Create the virtualenv. Works on macOS and Linux.
#
#   bash setup.sh            # uses uv if present, else python3 -m venv + pip
#
# Afterwards, check it works:
#   .venv/bin/python scripts/bench.py --envs 1 4 8 --steps 200
set -euo pipefail
cd "$(dirname "$0")"

if command -v uv >/dev/null 2>&1; then
  uv venv --python 3.11 .venv
  VIRTUAL_ENV="$PWD/.venv" uv pip install -r requirements.txt
else
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r requirements.txt
fi

echo
.venv/bin/python - <<'PY'
import os, platform
os.environ["MUJOCO_GL"] = "disable"
import mujoco, robosuite, torch
print(f"platform     {platform.system()} {platform.machine()}, {os.cpu_count()} cores")
print(f"robosuite    {robosuite.__version__}")
print(f"mujoco       {mujoco.__version__}")
print(f"torch        {torch.__version__}")
print("\nsetup ok. next:  .venv/bin/python scripts/bench.py --envs 1 4 8 --steps 200")
PY
