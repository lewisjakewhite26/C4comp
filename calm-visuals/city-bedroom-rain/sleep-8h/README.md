# City bedroom rain, 8-hour sleep video

Uses the four loops in `../sleep-3h/` (keep both folders side by side), shuffled across 160 loops, plus 8 hours of audio in eight one-hour parts.

    ./make_8h.sh

Windows (Command Prompt, in this folder):

    ffmpeg -f concat -safe 0 -i video_order.txt -f concat -safe 0 -i audio_order.txt -map 0:v -map 1:a -c copy zonelytime_city_rain_8h.mp4

Result: 8:00:00, 1080p, about 4.5 GB. YouTube accepts files this size from a computer.
