#!/bin/sh
# Adds the ambient track to either silent video without re-encoding the picture.
# Usage: ./add_audio.sh guide   (or: clean)
V=${1:-guide}
ffmpeg -y -i breathing_orb_loop_3min_${V}_1080p.mp4 -i ambient_breathing.m4a -map 0:v -map 1:a -c copy -t 182 -movflags +faststart breathing_orb_3min_${V}_with_audio.mp4
