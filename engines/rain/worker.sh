#!/bin/bash
cd ${ZT_WORK:-$(dirname $(dirname $(readlink -f $0)))/}rain
for n in "$@"; do
  a=$(( (n-1)*4320 )); b=$(( n*4320 ))
  out=parts/rain_house_loop_part${n}_of5.mp4
  python3 -I render_rain.py 1 render $a $b $out 19 > parts/log$n.txt 2>&1 || { echo "FAIL $n" >> parts/status.txt; continue; }
  cp $out /home/claude/c4comp/rain-house-loop/
  (
    flock 9
    cd /home/claude/c4comp
    git add rain-house-loop/rain_house_loop_part${n}_of5.mp4
    git commit -q -m "Add rain house loop part $n of 5

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011eudubV1NH8SujXCENxsKQ" 
    git pull -q --rebase origin main; git push -q origin HEAD:main
  ) 9>/tmp/gitpush.lock >> parts/push$n.txt 2>&1
  echo "DONE $n" >> parts/status.txt
done
