#!/bin/bash
# Builds the full 1-hour video from the 3-minute loop + 1-hour audio. Needs ffmpeg. No re-encoding, takes a couple of minutes.
# Usage: ./make_1hr.sh      (run inside this folder)
ffmpeg -stream_loop 19 -i rainy_window_loop_3min_1080p.mp4 -i rainy_window_audio_1hr.m4a -map 0:v -map 1:a -c copy -t 3600 -movflags +faststart rainy_window_1hr_1080p.mp4
