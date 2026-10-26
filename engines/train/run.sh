#!/bin/bash
# usage: run.sh part numbers...
cd ${ZT_WORK:-$(dirname $(dirname $(readlink -f $0)))/}train
for n in "$@"; do
  a=$(( (n-1)*4320 )); b=$(( n*4320 ))
  f=out/train_bridge_loop_part${n}_of5.mp4
  python3 -I render4.py $a $b $f && flock /tmp/upl.lock python3 -I upload.py $f > out/up$n.log 2>&1 && rm -f $f
  echo "part $n done $(date +%T)" >> out/progress.log
done
