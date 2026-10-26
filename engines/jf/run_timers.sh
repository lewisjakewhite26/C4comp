cd ${ZT_WORK:-$(dirname $(dirname $(readlink -f $0)))/}jf
until grep -qE "DONE|rror" ../alp/hevc.log; do sleep 20; done
for m in 1 2 5 10; do
  T=$((m*60))
  python3 -I tim.py $T v_${m}min.mp4 || { echo "FAIL video $m"; continue; }
  python3 -I ../jelly/jaudio.py $T a_${m}min.wav || { echo "FAIL audio $m"; continue; }
  ffmpeg -v error -y -i v_${m}min.mp4 -i a_${m}min.wav -map 0:v -map 1:a -c:v copy -af loudnorm=I=-16:TP=-1.5:LRA=11 -ar 48000 -c:a aac -b:a 160k -shortest -movflags +faststart jellyfish_timer_${m}min.mp4 && rm -f v_${m}min.mp4 a_${m}min.wav
  echo "done $m $(stat -c %s jellyfish_timer_${m}min.mp4)"
done
echo ALLDONE
