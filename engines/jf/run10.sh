cd ${ZT_WORK:-$(dirname $(dirname $(readlink -f $0)))/}jf
CRF=25 python3 -I tim.py 600 v_10min.mp4 || { echo FAIL video; exit 1; }
python3 -I ../jelly/jaudio.py 600 a_10min.wav || { echo FAIL audio; exit 1; }
ffmpeg -v error -y -i v_10min.mp4 -i a_10min.wav -map 0:v -map 1:a -c:v copy -af loudnorm=I=-16:TP=-1.5:LRA=11 -ar 48000 -c:a aac -b:a 160k -shortest -movflags +faststart jellyfish_timer_10min.mp4 && rm -f v_10min.mp4 a_10min.wav
echo "done 10 $(stat -c %s jellyfish_timer_10min.mp4)"
