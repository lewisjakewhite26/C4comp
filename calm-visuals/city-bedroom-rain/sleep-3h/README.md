# City bedroom rain, 3-hour sleep video

Four 3-minute loops (calm, heavy rain, wind, storm) that start and end on the same frame, shuffled in `video_order.txt`, plus 3 hours of matching audio in three one-hour parts (thunder is timed to each loop's lightning).

Join on a computer with ffmpeg installed:

    ./make_3h.sh

Windows (Command Prompt, in this folder):

    ffmpeg -f concat -safe 0 -i video_order.txt -f concat -safe 0 -i audio_order.txt -map 0:v -map 1:a -c copy zonelytime_city_rain_3h.mp4

Result: 3:00:00, 1080p, about 1.7 GB, ready for YouTube. Mark as altered/synthetic content, not made for kids.
