# Rainy window, 1 hour

- `rainy_window_loop_3min_1080p.mp4`: seamless 3-minute video loop (1920x1080, 24fps, silent).
- `rainy_window_audio_1hr.m4a`: one hour of non-repeating rain and thunder (thunder is timed to the lightning flashes in the loop).
- `make_1hr.sh`: joins them into the 1-hour video (about 1.4 GB, no re-encode).

Windows without bash: `ffmpeg -stream_loop 19 -i rainy_window_loop_3min_1080p.mp4 -i rainy_window_audio_1hr.m4a -map 0:v -map 1:a -c copy -t 3600 rainy_window_1hr_1080p.mp4`
