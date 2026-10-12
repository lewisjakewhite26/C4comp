#!/bin/bash
# Builds the 8-hour video (about 4.5 GB). Needs ffmpeg. Keep this folder next to ../sleep-3h (it uses its loops).
set -e
ffmpeg -f concat -safe 0 -i video_order.txt -f concat -safe 0 -i audio_order.txt -map 0:v -map 1:a -c copy -movflags +faststart zonelytime_city_rain_8h.mp4
