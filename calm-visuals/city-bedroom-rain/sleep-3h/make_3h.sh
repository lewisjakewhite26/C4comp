#!/bin/bash
# Builds the 3-hour video (about 1.7 GB). Needs ffmpeg. Run inside this folder.
set -e
ffmpeg -f concat -safe 0 -i video_order.txt -f concat -safe 0 -i audio_order.txt -map 0:v -map 1:a -c copy -movflags +faststart zonelytime_city_rain_3h.mp4
