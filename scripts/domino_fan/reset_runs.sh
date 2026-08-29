#!/usr/bin/env bash
# Clear domino-fan runs so the dashboard shows only the ladder.
#
#   scripts/domino_fan/reset_runs.sh                  # dry run
#   scripts/domino_fan/reset_runs.sh --yes            # remove them
#   scripts/domino_fan/reset_runs.sh 'domino-oracle*' # another pattern
#   scripts/domino_fan/reset_runs.sh 'domino-oracle*' --yes
#
# Removes logs/, videos/, results/ and saved_approaches/ entries whose
# experiment id matches the pattern (default: every domino-fan run).
# Ladder runs are cheap to regenerate with run_rung.sh, and the
# debugging runs from developing this env clutter the dashboard without
# being results.
#
# Careful with a wider pattern: a real experiment's run is the only copy
# of its transcripts and fits. The domino_high_friction_turn runs, for
# instance, are the friction system-ID result, not scratch.
set -euo pipefail
cd "$(dirname "$0")/../.."

DRY=1
PATTERN='domino*fan*'
for a in "$@"; do
  case "$a" in
    --yes) DRY=0 ;;
    *)     PATTERN="$a" ;;
  esac
done

targets=()
while IFS= read -r d; do targets+=("$d"); done < <(
  { ls -d logs/*/$PATTERN            2>/dev/null || true
    ls -d videos/*/$PATTERN          2>/dev/null || true
    ls    results/*$PATTERN*         2>/dev/null || true
    ls    saved_approaches/*$PATTERN* 2>/dev/null || true
  } | sort -u)

if [ ${#targets[@]} -eq 0 ]; then echo "nothing to remove."; exit 0; fi
printf '%s\n' "${targets[@]}"
echo "---"
if [ "$DRY" = "1" ]; then
  echo "${#targets[@]} paths match '$PATTERN'. Re-run with --yes to remove."
else
  rm -rf "${targets[@]}"
  echo "removed ${#targets[@]} paths."
fi
