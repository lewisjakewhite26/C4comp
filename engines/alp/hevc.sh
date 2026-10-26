set -e
cd ${ZT_WORK:-$(dirname $(dirname $(readlink -f $0)))/}alp
ffmpeg -v error -y -i nordic_meadow_loop_15min.mp4 -c:v libx265 -preset medium -b:v 830k -x265-params pass=1:log-level=error:stats=x265.log -an -f null /dev/null
ffmpeg -v error -y -i nordic_meadow_loop_15min.mp4 -c:v libx265 -preset medium -b:v 830k -x265-params pass=2:log-level=error:stats=x265.log -tag:v hvc1 -pix_fmt yuv420p -movflags +faststart -an nordic_meadow_loop_15min_hevc.mp4
echo DONE
