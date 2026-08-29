#!/usr/bin/env bash
# Clear domino-fan runs so the dashboard shows only the ladder.
#
#   scripts/domino_fan/reset_runs.sh          # what would be removed
#   scripts/domino_fan/reset_runs.sh --yes    # actually remove it
#
# Removes logs/, videos/, results/ and saved_approaches/ entries for
# every domino-fan experiment. The debugging runs from developing this
# env (domino-fan-v2, -w40, -corr, ...) clutter the dashboard and are
# not results; the ladder runs are cheap to regenerate with run_rung.sh.
set -euo pipefail
cd "$(dirname "$0")/../.."

DRY=1; [ "${1:-}" = "--yes" ] && DRY=0

targets=()
while IFS= read -r d; do targets+=("$d"); done < <(
  { ls -d logs/*/domino*fan*                 2>/dev/null || true
    ls -d videos/*/domino*fan*               2>/dev/null || true
    ls    results/*domino_fan*               2>/dev/null || true
    ls    saved_approaches/*domino_fan*      2>/dev/null || true
  } | sort -u)

if [ ${#targets[@]} -eq 0 ]; then echo "nothing to remove."; exit 0; fi
printf '%s\n' "${targets[@]}"
echo "---"
if [ "$DRY" = "1" ]; then
  echo "${#targets[@]} paths. Re-run with --yes to remove them."
else
  rm -rf "${targets[@]}"
  echo "removed ${#targets[@]} paths."
fi
