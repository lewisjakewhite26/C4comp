#!/bin/bash
cd "$(dirname "$0")"
cat jobs.txt | xargs -P 2 -L 1 bash -c '[ -s "$2" ] && [ -f "$2.ok" ] || { python3 render.py $0 $1 $2 > "$2.log" 2>&1 && touch "$2.ok"; }'
echo ALLDONE > all.log
