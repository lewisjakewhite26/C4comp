#!/bin/bash
# usage: drive.sh N...   (part numbers 1..10)
A=${ZT_WORK:-$(dirname $(dirname $(readlink -f $0)))/}alp
cd $A
for n in "$@"; do
  a=$(( (n-1)*2160 )); b=$(( n*2160 )); f=parts/nordic_loop_part${n}_of10.mp4
  crf=24
  while true; do
    python3 -I render_alp.py 1 render $a $b $f $crf
    sz=$(stat -c %s $f)
    if [ $sz -lt 95000000 ]; then break; fi
    crf=$((crf+2))
  done
  (
    flock 9
    cd /home/claude/c4comp && cp $A/$f nordic-loop/ && git add nordic-loop/$(basename $f) && git commit -q -m "Nordic loop part $n of 10" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011eudubV1NH8SujXCENxsKQ" && git push -q origin main && echo "pushed $n" >> $A/drive.log
  ) 9>/tmp/push.lock
  rm -f $A/$f
done
