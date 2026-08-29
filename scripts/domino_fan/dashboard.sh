#!/usr/bin/env bash
# Open the dashboard: every run, its rung, its score, and its video.
#
#   scripts/domino_fan/dashboard.sh            # http://localhost:8765
#   scripts/domino_fan/dashboard.sh 9000       # another port
#
# From there you can start a rung with the "run rung" buttons and watch
# it live - the row appears as soon as the run makes its log dir, and
# the page refreshes itself. One run at a time; use a row's kill button
# to stop one.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python"; [ -x "$PY" ] || PY="python3"
PORT="${1:-8765}"
echo "dashboard: http://localhost:$PORT   (ctrl-c to stop)"
exec $PY scripts/log_viewer.py --port "$PORT"
