#!/bin/bash
# Watch the redesigned-balloons MB-vs-MF A/B (seeds 0-2). Emit one line per
# run as it goes terminal, then ALL_TRANSIENT_DONE. Covers all terminal states.
declare -A DIR=(
  [MB]=/home/ycliang/predicators/logs/agent_continual/balloons-agent_continual_transient
  [MF]=/home/ycliang/predicators/logs/agent_continual_model_free/balloons-agent_continual_model_free_transient
)
declare -A JOB=( [MB]=22191200 [MF]=22191208 )
declare -A seen
total=6
while true; do
  done=0
  for arm in MB MF; do
    for seed in 0 1 2; do
      jid="${JOB[$arm]}_${seed}"
      st=$(sacct -j "$jid" --format=State -n 2>/dev/null | head -1 | tr -d ' ')
      [ -z "$st" ] && st="PENDING"
      case "$st" in
        COMPLETED|FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|BOOT_FAIL)
          done=$((done+1))
          key="${arm}${seed}"
          if [ -z "${seen[$key]}" ]; then
            seen[$key]="$st"
            d=$(ls -dt "${DIR[$arm]}/seed${seed}"/run_* 2>/dev/null | head -1)
            if [ -n "$d" ] && [ -f "$d/scorecard.json" ]; then
              res=$(python -c "
import json
j=json.load(open('$d/scorecard.json'));t=j.get('totals',{})
print('end=%s won=%s/%s resets=%s steps=%s'%(j.get('end_reason'),t.get('levels_completed'),t.get('levels_total'),t.get('total_resets'),t.get('total_steps')))
" 2>/dev/null)
              echo "$arm seed$seed $st -- $res"
            else
              echo "$arm seed$seed $st -- no scorecard"
            fi
          fi
          ;;
      esac
    done
  done
  [ "$done" -eq "$total" ] && { echo "ALL_TRANSIENT_DONE ($total/$total)"; break; }
  sleep 180
done
